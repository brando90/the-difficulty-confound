"""Experiment 13, arm 13-GPU (PROTOCOL Sections 2.6-2.8 and 7): check-only, the learning-rate pilot, pilot selection,
one grid shard per call, and the merge into results/gpu. run.py --arm gpu dispatches here.

A shard trains one run (size, target, alpha, seed) on one GPU with bfloat16 autocast, evaluates every checkpoint in
float64 with deterministic algorithms on the same device, and writes OUT/shards/<%05d>/ with a DONE file (written
last) holding the SHA-256 of every file it wrote. Training documents come from 8 CPU workers (NumPy PCG64 seeded by
(seed, concept, j)), so the stream is the one hmm.stream_batches gives and does not depend on the GPU type.
Concepts and references are built once by --check-only into prep/gpu (atomic: a rebuild must reproduce the same
bytes) and read by every shard; each shard records their SHA-256.
"""
import collections
import concurrent.futures as cf
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # deterministic float64 matmuls on CUDA

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hmm as H  # noqa: E402
import model as M  # noqa: E402
import run as R  # noqa: E402

SIZES = ['S', 'M', 'L']
GPU_SOURCES = R.SOURCES + ['gpu_arm.py']
PREP_FILES = ['tasks.json', 'acceptance_log.json', 'oracle.npz', 'mixture.npz', 'posterior16.json']


def load_config():
    c = R.load_config('gpu')
    c['workers'] = c.get('workers', 4)
    return c


# ---------------------------------------------------------------- grid (PROTOCOL Sections 2.6, 2.7, 7)

def model_config(c, size):
    i = SIZES.index(size)
    layers, d, heads = c['sizes'][size]
    return dict(n_layers=layers, d_model=d, n_heads=heads, d_mlp=4 * d, vocab=c['vocab'],
                max_positions=c['max_positions'], expected_params=c['expected_params'][i],
                expected_nonembedding=c['expected_nonembedding'][i])


def checkpoints_for(c, size, T_list=None):
    """The configured checkpoints with 6 * N_total * T <= flop_cap; a capped run ends at the largest power of 2 under
    the cap (L: 2^22, 2^24, 2^26, 2^28, 2^29)."""
    n = c['expected_params'][SIZES.index(size)]
    T_list = c['T_checkpoints'] if T_list is None else T_list
    Ts = [T for T in T_list if 6 * n * T <= c['flop_cap']]
    if len(Ts) < len(T_list):
        Ts.append(2 ** int(math.floor(math.log2(c['flop_cap'] / (6 * n)))))
    return Ts


def shard_coords(c, i):
    """Shard id = 120 i_N + 24 i_n + 12 i_target + 3 i_alpha + i_seed."""
    if not 0 <= i < c['n_shards']:
        raise ValueError(f'shard {i} outside 0..{c["n_shards"] - 1}')
    i_N, r = divmod(i, 120)
    i_n, r = divmod(r, 24)
    i_t, r = divmod(r, 12)
    i_a, i_s = divmod(r, 3)
    return dict(shard=i, size=SIZES[i_N], level=c['levels'][i_n], target_index=i_t, alpha=c['alpha_grid'][i_a],
                seed=c['seeds'][i_s])


def targets_at(tasks, n):
    return sorted([t for t in tasks if t['role'] == 'target' and t['K'] == n], key=lambda t: t['task_id'])


def shard_job(c, tasks, i):
    job = shard_coords(c, i)
    job['target_id'] = targets_at(tasks, job['level'])[job['target_index']]['task_id']
    return job


def pilot_job(c, tasks, j):
    """Pilot job J = 3 i_size + i_lr: seed 13919, the n = 32 pilot concept, alpha 0.4, T to pilot_T."""
    if not 0 <= j < 3 * len(c['pilot_lr_multipliers']):
        raise ValueError(f'pilot job {j} outside 0..8')
    i_size, i_lr = divmod(j, len(c['pilot_lr_multipliers']))
    pl = [t for t in tasks if t['role'] == 'pilot' and t['K'] == c['heldout_level']]
    return dict(job=j, size=SIZES[i_size], lr_multiplier=c['pilot_lr_multipliers'][i_lr], level=c['heldout_level'],
                target_id=pl[0]['task_id'], alpha=c['pilot_alpha'], seed=c['pilot_seed'])


def run_name(job):
    return f"{R.run_name(job['seed'], job['target_id'], job['alpha'])}_{job['size']}"


# ---------------------------------------------------------------- the model on a device (same weights as model.LM)

def block_forward(b, x, fast):
    B, L, d = x.shape
    q, k, v = b.qkv(b.ln1(x)).split(d, dim=-1)
    q, k, v = (t.view(B, L, b.h, d // b.h).transpose(1, 2) for t in (q, k, v))
    if fast:
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:  # model.Block's arithmetic, with the mask on the input's device
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // b.h)
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        a = att.masked_fill(~mask, float('-inf')).softmax(-1) @ v
    x = x + b.proj(a.transpose(1, 2).reshape(B, L, d))
    return x + b.fc2(F.gelu(b.fc1(b.ln2(x))))


def lm_forward(m, x, fast=False):
    """model.LM.forward on any device; fast=True uses fused causal attention (training only)."""
    h = m.tok(x) + m.pos(torch.arange(x.shape[1], device=x.device))
    for b in m.blocks:
        h = block_forward(b, h, fast)
    return m.ln_f(h) @ m.tok.weight.T


@torch.no_grad()
def code_lengths(model, docs, bos, device, batch=128):
    """Per-token -log2 p(x_t | x_<t) (B, L) in float64 on a float64 copy of the model, deterministic algorithms."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        m64 = copy.deepcopy(model).double().eval().to(device)
        out = []
        for s in range(0, len(docs), batch):
            x, y = M.inputs_targets(docs[s:s + batch], bos)
            lp = F.log_softmax(lm_forward(m64, x.to(device)), -1)
            out.append((-lp.gather(-1, y.to(device)[..., None])[..., 0] / math.log(2.0)).cpu().numpy())
        del m64
        return np.concatenate(out).astype(np.float64)
    finally:
        torch.use_deterministic_algorithms(prev)


def device_or_die(allow_cpu):
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap < (8, 0):
            raise SystemExit(f'GPU compute capability {cap} < 8.0: bfloat16 needs >= 8.0')
        return torch.device('cuda'), dict(gpu=torch.cuda.get_device_name(0), capability=list(cap),
                                          cuda=torch.version.cuda, driver=_driver())
    if allow_cpu:
        return torch.device('cpu'), dict(gpu=None, capability=None, cuda=None, driver=None)
    raise SystemExit('CUDA is required for 13-GPU shards and pilot jobs (TDC_ALLOW_CPU=1 is for tests only)')


def _driver():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                       text=True).strip().splitlines()[0]
    except Exception:
        return None


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get('TDC_COMMIT', 'unknown')


def runtime_info():
    import scipy
    return dict(host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, git_revision=git_revision(),
                sources={s: R.sha256_file(HERE / s) for s in GPU_SOURCES})


# ---------------------------------------------------------------- the training stream on CPU workers

_D = {}


def _data_init(defs, seed, target_cid, dist_cids, alpha, n_docs, L, batch_docs):
    torch.set_num_threads(1)
    _D.update(cm={int(k): H.from_json(v) for k, v in defs.items()}, seed=seed, L=L, Bd=batch_docs)
    _D['cids'], _D['js'] = H.schedule(seed, target_cid, dist_cids, alpha, n_docs)


def _data_batch(step):
    """Batch `step` of the stream, identical to hmm.stream_batches (as int16)."""
    s, Bd, L = step * _D['Bd'], _D['Bd'], _D['L']
    cc, jj = _D['cids'][s:s + Bd], _D['js'][s:s + Bd]
    out = np.empty((len(cc), L), np.int16)
    for cid in np.unique(cc):
        m = cc == cid
        out[m] = H.sample_docs(_D['cm'][int(cid)], _D['seed'], int(cid), jj[m], L)
    return out


class Stream:
    """Batches 0..n_steps-1 in order from a pool of CPU workers, at most `depth` in flight."""

    def __init__(self, workers, initargs, n_steps, depth):
        self.pool = mp.get_context('spawn').Pool(workers, initializer=_data_init, initargs=initargs)
        self.q, self.next, self.n, self.depth = collections.deque(), 0, n_steps, depth

    def get(self):
        while len(self.q) < self.depth and self.next < self.n:
            self.q.append(self.pool.apply_async(_data_batch, (self.next,)))
            self.next += 1
        return self.q.popleft().get()

    def close(self):
        self.pool.terminate()
        self.pool.join()


def stream_args(c, tasks, job, n_docs):
    target = next(t for t in tasks if t['task_id'] == job['target_id'])
    defs = {str(t['cid']): t['definition'] for t in tasks}
    dist = [d['cid'] for d in R.distractors(tasks)]
    return (defs, job['seed'], target['cid'], dist, job['alpha'], n_docs, c['doc_len'], c['batch_docs'])


# ---------------------------------------------------------------- one run

def make_optimizer(c, model, lr):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    kw = dict(fused=True) if next(model.parameters()).is_cuda else {}
    return torch.optim.AdamW([dict(params=decay, weight_decay=c['weight_decay']),
                              dict(params=no_decay, weight_decay=0.0)], lr=lr, betas=tuple(c['betas']), **kw)


def train_run(c, tasks, refs, job, lr, T_checkpoints, d, device, log):
    """Train one run, evaluating at every checkpoint; checkpoints go to d/checkpoints/<run>/T_<T>.npz."""
    t0 = time.time()
    target = next(t for t in tasks if t['task_id'] == job['target_id'])
    dist = R.distractors(tasks)
    L, Bd, V = c['doc_len'], c['batch_docs'], c['V']
    n_docs = T_checkpoints[-1] // L
    n_steps = n_docs // Bd
    args = stream_args(c, tasks, job, n_docs)
    stream = Stream(c['data_workers'], args, n_steps, depth=4 * c['data_workers'])
    cids, _ = H.schedule(job['seed'], target['cid'], args[3], job['alpha'], n_docs)
    torch.manual_seed(job['seed'])
    model = M.build_model(model_config(c, job['size'])).to(device)
    opt = make_optimizer(c, model, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c['warmup_steps']))
    ck_dir = d / 'checkpoints' / run_name(job)
    ck_dir.mkdir(parents=True)
    oc = refs['oracle'][job['target_id']]
    mc = refs['mixture'][f"{job['target_id']}|{job['alpha']:.2f}"]
    docs = refs['docs'][job['target_id']]
    n_ne = M.nonembedding_count(model)
    sha = hashlib.sha256()
    rows, perpos, drows, counts, losses = [], [], [], {}, []
    t_eval = t_wait = 0.0
    amp = device.type == 'cuda'
    nonfinite_steps = 0
    model.train()
    t_loop = time.time()
    try:
        for step in range(n_steps):
            ts = time.time()
            batch = stream.get()
            sha.update(batch.tobytes())
            t_wait += time.time() - ts
            x, y = M.inputs_targets(batch, V)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = lm_forward(model, x, fast=True)
            loss = F.cross_entropy(logits.float().reshape(-1, V + 1), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
            opt.step()
            sched.step()
            if (step + 1) % 1024 == 0 or step == 0:
                lv = float(loss.item())
                nonfinite_steps += 0 if math.isfinite(lv) else 1
                losses.append([step + 1, lv])
                t_train = time.time() - t_loop - t_eval
                log(f'step {step + 1}/{n_steps} loss {lv / math.log(2):.4f} bits '
                    f'{(step + 1) * Bd * L / max(t_train, 1e-9):.0f} tok/s, data wait {t_wait:.0f}s')
            T = (step + 1) * Bd * L
            if T in T_checkpoints:
                ts = time.time()
                m_codes = code_lengths(model, docs, V, device)
                M.save_npz(model, ck_dir / f'T_{T}.npz')
                m_doc = (step + 1) * Bd
                counts[str(T)] = {str(int(cid)): int((cids[:m_doc] == cid).sum()) for cid in np.unique(cids[:m_doc])}
                for s in R.SLICES:
                    sm = R.slice_mask(c, s)
                    mb, ob, xb = m_codes[:, sm].mean(), oc[:, sm].mean(), mc[:, sm].mean()
                    per_doc = (m_codes[:, sm] - oc[:, sm]).mean(1)
                    rows.append(dict(seed=job['seed'], target_id=job['target_id'], K=target['K'],
                                     K_bits=target['K_bits'], alpha=job['alpha'], N=n_ne, T=T, step=step + 1,
                                     slice=s, excess_bits=float(mb - ob), ident_bits=float(xb - ob),
                                     learn_bits=float(mb - xb), model_bits=float(mb), oracle_bits=float(ob),
                                     mixture_bits=float(xb), n_tokens=int(sm.sum() * len(docs)),
                                     excess_se=float(per_doc.std(ddof=1) / math.sqrt(len(per_doc))),
                                     size=job['size']))
                perpos.append(dict(T=T, model=m_codes.mean(0).tolist()))
                with torch.no_grad():  # training-distribution loss on distractors (not gated; float32)
                    model.eval()
                    for dd in dist:
                        x, y = M.inputs_targets(refs['docs'][dd['task_id']], V)
                        lp = F.log_softmax(lm_forward(model, x.to(device)).float(), -1)
                        nll = float((-lp.gather(-1, y.to(device)[..., None])).mean()) / math.log(2.0)
                        drows.append(dict(seed=job['seed'], target_id=job['target_id'], alpha=job['alpha'], T=T,
                                          distractor_id=dd['task_id'], model_bits=nll,
                                          oracle_bits=float(refs['oracle'][dd['task_id']].mean()), size=job['size']))
                    model.train()
                t_eval += time.time() - ts
                log(f'checkpoint T={T}: excess {rows[-1]["excess_bits"]:.4f} bits (all), learn '
                    f'{rows[-1]["learn_bits"]:.4f}, eval {time.time() - ts:.0f}s')
    finally:
        stream.close()
    if amp:
        torch.cuda.synchronize()
    t_train = time.time() - t_loop - t_eval  # includes waiting for data (data_wait_seconds)
    return dict(run=run_name(job), job=job, lr=lr, seed=job['seed'], target_id=job['target_id'], alpha=job['alpha'],
                size=job['size'], N=n_ne, N_total=M.param_count(model), T_checkpoints=T_checkpoints, rows=rows,
                per_position=perpos, distractor_rows=drows, counts=counts, stream_sha256=sha.hexdigest(),
                data_order_seed=[int(job['seed']), 7777], n_docs=int(n_docs), losses=losses,
                nonfinite_loss_checks=nonfinite_steps,
                train_seconds=round(t_train, 3), eval_seconds=round(t_eval, 3), data_wait_seconds=round(t_wait, 3),
                tokens_per_second_train=T_checkpoints[-1] / max(t_train, 1e-9),
                wall_seconds=round(time.time() - t0, 3),
                finite=bool(all(math.isfinite(r['model_bits']) for r in rows)))


# ---------------------------------------------------------------- prep (concepts and references, built once)

def prep_hashes(prep):
    files = PREP_FILES + sorted(f'data/{p.name}' for p in (prep / 'data').glob('*.npz'))
    return {f: R.sha256_file(prep / f) for f in files}


def load_prep(prep):
    prep = Path(prep)
    missing = [f for f in PREP_FILES if not (prep / f).exists()]
    if missing:
        raise SystemExit(f'{prep} lacks {missing}: run run.py --arm gpu --check-only first')
    tasks = json.loads((prep / 'tasks.json').read_text())
    return tasks, R.load_refs(prep, tasks), prep_hashes(prep)


def finish_dir(d, extra):
    """Write DONE last, holding the SHA-256 of every file the job wrote."""
    files = {p.relative_to(d).as_posix(): R.sha256_file(p) for p in sorted(d.rglob('*')) if p.is_file()}
    R.atomic_write_text(d / 'DONE', json.dumps(dict(files=files, **extra), indent=1))


def run_job(c, prep, d, job, lr, T_checkpoints, kind, allow_cpu):
    if (d / 'DONE').exists():
        print(f'{kind} {d.name} already DONE', flush=True)
        return
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f'{d} exists without DONE (a crash): move it aside, log it in CKPT, then rerun')
    tasks, refs, ph = load_prep(prep)
    device, hw = device_or_die(allow_cpu)
    d.mkdir(parents=True, exist_ok=True)
    started = time.strftime('%m-%d-%Y %H:%M:%S')
    log = R.log_to(d / 'train.log')
    log(f'{kind} {d.name}: {json.dumps(job)} lr {lr:g} T {T_checkpoints} on {hw["gpu"] or "cpu"}')
    rec = train_run(c, tasks, refs, job, lr, T_checkpoints, d, device, log)
    rec.update(kind=kind, hardware=hw, runtime=runtime_info(), prep_sha256=ph, started=started)
    R.atomic_write_text(d / 'run.json', json.dumps(rec))
    log(f'finished in {rec["wall_seconds"]:.0f}s ({rec["tokens_per_second_train"]:.0f} training tokens/s); writing DONE')
    finish_dir(d, dict(kind=kind, job=job, lr=lr, run=rec['run']))


def run_shard(c, prep, out, i, allow_cpu, T_override=None):
    if c.get('lr_selected') is None:
        raise SystemExit('gpu.lr_selected is unset: run the GPU pilot and make the "Experiment 13 GPU pilot '
                         'disclosure" commit first')
    tasks = json.loads((Path(prep) / 'tasks.json').read_text()) if (Path(prep) / 'tasks.json').exists() else None
    if tasks is None:
        raise SystemExit(f'{prep}/tasks.json missing: run run.py --arm gpu --check-only first')
    job = shard_job(c, tasks, i)
    lr = c['lr'][SIZES.index(job['size'])] * c['lr_selected'][job['size']]
    Ts = T_override or checkpoints_for(c, job['size'])
    run_job(c, prep, Path(out) / 'shards' / f'{i:05d}', job, lr, Ts, 'shard', allow_cpu)


def run_pilot(c, prep, out, j, allow_cpu, T_override=None):
    tasks = json.loads((Path(prep) / 'tasks.json').read_text()) if (Path(prep) / 'tasks.json').exists() else None
    if tasks is None:
        raise SystemExit(f'{prep}/tasks.json missing: run run.py --arm gpu --check-only first')
    job = pilot_job(c, tasks, j)
    lr = c['lr'][SIZES.index(job['size'])] * job['lr_multiplier']
    Ts = T_override or [T for T in c['T_checkpoints'] if T <= c['pilot_T']]
    run_job(c, prep, Path(out) / f'{j:05d}', job, lr, Ts, 'pilot', allow_cpu)


# ---------------------------------------------------------------- pilot selection (PROTOCOL Section 2.8)

def select_rule(c, results):
    """Per size, the multiplier with the lowest pilot excess at pilot_T; ties within pilot_tie_bits go to 1x
    (the best multiplier is replaced by 1x when 1x is within pilot_tie_bits of it). Nonfinite runs never win."""
    chosen = {}
    for size in SIZES:
        rr = [r for r in results if r['size'] == size]
        ok = [r for r in rr if math.isfinite(r['excess_bits'])]
        if not ok:
            raise SystemExit(f'every pilot run of size {size} is nonfinite')
        best = min(ok, key=lambda r: r['excess_bits'])
        one = next((r for r in ok if r['lr_multiplier'] == 1), None)
        if one is not None and one['excess_bits'] - best['excess_bits'] <= c['pilot_tie_bits']:
            best = one
        chosen[size] = best['lr_multiplier']
    return chosen


def pilot_select(c, pilot_dir):
    pilot_dir = Path(pilot_dir)
    results = []
    for j in range(3 * len(c['pilot_lr_multipliers'])):
        d = pilot_dir / f'{j:05d}'
        if not (d / 'DONE').exists():
            raise SystemExit(f'pilot job {j} is not DONE ({d})')
        rec = json.loads((d / 'run.json').read_text())
        row = next(r for r in rec['rows'] if r['slice'] == 'all' and r['T'] == c['pilot_T'])
        results.append(dict(job=j, size=rec['size'], lr_multiplier=rec['job']['lr_multiplier'], lr=rec['lr'],
                            excess_bits=row['excess_bits'], learn_bits=row['learn_bits'], ident_bits=row['ident_bits'],
                            excess_se=row['excess_se'], finite=rec['finite'], gpu=rec['hardware']['gpu'],
                            wall_seconds=rec['wall_seconds'], tokens_per_second_train=rec['tokens_per_second_train']))
    sel = dict(lr_selected=select_rule(c, results), rule=f'lowest excess at T={c["pilot_T"]}, ties within '
               f'{c["pilot_tie_bits"]} bits go to 1x', results=results)
    R.atomic_write_text(pilot_dir / 'pilot_selection.json', json.dumps(sel, indent=1))
    for r in results:
        print(f"job {r['job']} {r['size']} x{r['lr_multiplier']:<4} lr {r['lr']:.2e} excess {r['excess_bits']:.5f} "
              f"+- {r['excess_se']:.5f} bits ({r['gpu']}, {r['wall_seconds']:.0f}s)")
    print('gpu.lr_selected =', json.dumps(sel['lr_selected']))
    return sel


# ---------------------------------------------------------------- check-only (PROMPT step 1, GPU arm)

def _schedule_check(args):
    seed, cid, dist, alpha, n_docs = args
    cids, _ = H.schedule(seed, cid, dist, alpha, n_docs)
    cum = np.cumsum(cids == cid)
    num = int(round(alpha * 100))
    exact = bool(np.array_equal(cum, (np.arange(1, n_docs + 1) * num) // 100))
    spread = 0
    if alpha < 1:
        dc = [int((cids == x).sum()) for x in dist]
        spread = max(dc) - min(dc)
    return exact, spread


def build_prep(c, prep, log):
    """Build concepts and references into a fresh folder, then install it atomically as prep; an existing prep must
    have identical bytes (so a concurrent reader never sees a partial file)."""
    prep = Path(prep)
    tmp = prep.parent / f'{prep.name}.tmp-{os.getpid()}'
    shutil.rmtree(tmp, ignore_errors=True)
    tasks, acc_log = R.prepare(c, tmp, log)
    if prep.exists():
        a, b = prep_hashes(prep), prep_hashes(tmp)
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise SystemExit(f'rebuilt references differ from {prep}: {diff[:5]} (numpy version? see CKPT)')
        shutil.rmtree(tmp)
        log(f'{prep} already present and identical to the rebuild')
    else:
        os.replace(tmp, prep)
        log(f'installed {prep}')
    return tasks, acc_log


def check_only(c, prep, log):
    t0 = time.time()
    tasks, acc_log = build_prep(c, prep, log)
    rec = dict(arm='gpu', acceptance=acc_log, concepts=[], checks={})
    for t in tasks:
        rec['concepts'].append(dict(task_id=t['task_id'], role=t['role'], K=t['K'], K_bits=t['K_bits'],
                                    process_rank=t['process_rank'], entropy_rate=round(t['entropy_rate'], 4),
                                    entropy_rate_se=round(t['entropy_rate_se'], 4),
                                    candidates_tried=t['acceptance']['candidates_tried_at_level']))
        log(f"{t['task_id']:14s} {t['role']:10s} n={t['K']:3d} K_bits={t['K_bits']:5d} rank={t['process_rank']} "
            f"h={t['entropy_rate']:.4f}+-{t['entropy_rate_se']:.4f}")
    rec['checks']['params'] = {}
    for s in SIZES:
        m = M.build_model(model_config(c, s))  # asserts both counts against config
        rec['checks']['params'][s] = [M.param_count(m), M.nonembedding_count(m)]
    rec['checks']['T_checkpoints'] = {s: checkpoints_for(c, s) for s in SIZES}
    assert rec['checks']['T_checkpoints']['L'][-1] == 2 ** 29, rec['checks']['T_checkpoints']
    # alpha schedules: exact floor(m * alpha) target documents after every prefix up to the largest T
    nd = max(c['T_checkpoints']) // c['doc_len']
    dist_ids = [d['cid'] for d in R.distractors(tasks)]
    jobs = [(c['seeds'][0], t['cid'], dist_ids, a, nd) for t in tasks if t['role'] != 'distractor'
            for a in c['alpha_grid']]
    ts = time.time()
    with cf.ProcessPoolExecutor(c['workers']) as ex:
        res = list(ex.map(_schedule_check, jobs))
    assert all(r[0] for r in res), 'an alpha schedule is not exact'
    rec['checks']['alpha_schedule_exact'] = dict(schedules=len(jobs), docs_each=nd, seconds=round(time.time() - ts, 1))
    rec['checks']['max_distractor_count_spread'] = max(r[1] for r in res)
    log(f'alpha schedules exact ({len(jobs)} x {nd} documents, {time.time() - ts:.0f}s)')
    # disjointness: eval documents vs the first 2^16 training documents of every concept and every seed
    eval_h = {}
    for t in tasks:
        d = np.load(Path(prep) / 'data' / f"{t['task_id']}_test.npz")['docs']
        for x in d:
            eval_h[H.doc_hash(x)] = t['task_id']
    seeds = list(c['seeds']) + [c['pilot_seed']]
    jobs = [(t['definition'], s, t['cid'], 2 ** 16, c['doc_len']) for t in tasks for s in seeds]
    ts = time.time()
    hits = 0
    with cf.ProcessPoolExecutor(c['workers']) as ex:
        for hs in ex.map(R._first_doc_hashes, jobs):
            hits += len(hs & eval_h.keys())
    assert hits == 0, f'{hits} eval documents equal training documents'
    rec['checks']['eval_train_disjoint'] = dict(eval_docs=len(eval_h), training_docs_checked=len(jobs) * 2 ** 16,
                                                hits=hits, seconds=round(time.time() - ts, 1))
    log(f'eval/train disjoint ({len(jobs) * 2 ** 16} training documents, {time.time() - ts:.0f}s)')
    shards = [shard_job(c, tasks, i) for i in range(c['n_shards'])]
    keys = {(j['size'], j['target_id'], j['alpha'], j['seed']) for j in shards}
    assert len(keys) == c['n_shards'] == 3 * len(c['levels']) * c['targets_per_level'] * len(c['alpha_grid']) * \
        len(c['seeds'])
    rec['checks']['expected_runs'] = len(shards)
    rec['shard_map'] = dict(formula='120*i_N + 24*i_level + 12*i_target + 3*i_alpha + i_seed',
                            examples={str(i): shards[i] for i in (0, 1, 3, 12, 24, 119, 120, 239, 240, 359)},
                            pilot_jobs=[pilot_job(c, tasks, j) for j in range(9)])
    fl = {s: 6 * c['expected_params'][i] * checkpoints_for(c, s)[-1] for i, s in enumerate(SIZES)}
    rec['time_estimate'] = dict(flops_per_run=fl, a100_hours_protocol_section_7=250,
                                note='S about 0.3 h, M and L about 0.8 h per run on an A100 (PROTOCOL Section 7)')
    o = np.load(Path(prep) / 'oracle.npz')
    mx = np.load(Path(prep) / 'mixture.npz')
    rec['eps_id_all'] = {k: float((mx[k] - o[k.split('|')[0]]).mean()) for k in mx.files}
    rec['prep_sha256'] = prep_hashes(Path(prep))
    rec['source_sha256'] = {s: R.sha256_file(HERE / s) for s in GPU_SOURCES}
    rec['seconds'] = round(time.time() - t0, 1)
    R.atomic_write_text(HERE / 'results_check_gpu.json', json.dumps(rec, indent=1))
    log(json.dumps(rec['checks']))
    return rec


# ---------------------------------------------------------------- merge (launch/README.md step 7)

def merge(c, out, prep, dest):
    """Collect the DONE shards into dest (results/gpu) without checkpoints; checkpoint SHA-256s come from DONE."""
    out, prep, dest = Path(out), Path(prep), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'{dest} exists and is not empty')
    ph = prep_hashes(prep)
    recs, shards, ck_sha = [], [], {}
    for i in range(c['n_shards']):
        d = out / 'shards' / f'{i:05d}'
        if not (d / 'DONE').exists():
            raise SystemExit(f'shard {i} is not DONE')
        done = json.loads((d / 'DONE').read_text())
        if done['files']['run.json'] != R.sha256_file(d / 'run.json'):
            raise SystemExit(f'shard {i}: run.json does not match its DONE hash')
        rec = json.loads((d / 'run.json').read_text())
        if rec['prep_sha256'] != ph:
            raise SystemExit(f'shard {i} used different concepts or references than {prep}')
        recs.append(rec)
        ver = json.loads((d / 'VERIFIED').read_text()) if (d / 'VERIFIED').exists() else None
        shards.append(dict(shard=i, run=rec['run'], job=rec['job'], lr=rec['lr'], hardware=rec['hardware'],
                           runtime={k: v for k, v in rec['runtime'].items() if k != 'sources'},
                           sources=rec['runtime']['sources'], wall_seconds=rec['wall_seconds'],
                           train_seconds=rec['train_seconds'], eval_seconds=rec['eval_seconds'],
                           tokens_per_second_train=rec['tokens_per_second_train'], finite=rec['finite'],
                           done_sha256=R.sha256_file(d / 'DONE'), verified=ver))
        ck_sha[rec['run']] = {k: v for k, v in done['files'].items() if k.startswith('checkpoints/')}
    srcs = {json.dumps(s['sources'], sort_keys=True) for s in shards}
    if len(srcs) != 1:
        raise SystemExit('shards ran different code (sources differ between shards)')
    dest.mkdir(parents=True, exist_ok=True)
    for name in PREP_FILES:
        shutil.copy2(prep / name, dest / name)
    shutil.copytree(prep / 'data', dest / 'data')
    recs.sort(key=lambda r: (SIZES.index(r['size']), r['seed'], r['target_id'], r['alpha']))
    cols = ['seed', 'target_id', 'K', 'K_bits', 'alpha', 'size', 'N', 'T', 'step', 'slice', 'excess_bits',
            'ident_bits', 'learn_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'n_tokens', 'excess_se']
    with open(dest / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in recs:
            for row in r['rows']:
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    with open(dest / 'per_position.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'target_id', 'alpha', 'size', 'T'] + [f'p{i + 1}' for i in range(c['doc_len'])])
        for r in recs:
            for pp in r['per_position']:
                w.writerow([r['seed'], r['target_id'], r['alpha'], r['size'], pp['T']] + [repr(x) for x in pp['model']])
    with open(dest / 'distractor_loss.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['seed', 'target_id', 'alpha', 'size', 'T', 'distractor_id', 'model_bits', 'oracle_bits'])
        w.writeheader()
        for r in recs:
            for row in r['distractor_rows']:
                w.writerow(row)
    mixes = {r['run']: dict(target_id=r['target_id'], alpha=r['alpha'], seed=r['seed'], size=r['size'],
                            counts=r['counts'], data_order_seed=r['data_order_seed'], stream_sha256=r['stream_sha256'],
                            n_docs=r['n_docs'], T_checkpoints=r['T_checkpoints']) for r in recs}
    R.atomic_write_text(dest / 'mixes.json', json.dumps(mixes, indent=0, sort_keys=True))
    R.atomic_write_text(dest / 'shards.json', json.dumps(shards, indent=1))
    R.atomic_write_text(dest / 'checkpoint_sha256.json', json.dumps(ck_sha, indent=1, sort_keys=True))
    sel = out / 'pilot' / 'pilot_selection.json'
    man = dict(sources=shards[0]['sources'], prep_sha256=ph, checkpoint_root=str(out / 'shards'),
               grid=dict(runs=len(recs), diverged=[r['run'] for r in recs if not r['finite']],
                         verified=sum(1 for s in shards if s['verified']),
                         gpu_hours=round(sum(r['wall_seconds'] for r in recs) / 3600, 2),
                         gpus=sorted({r['hardware']['gpu'] or 'cpu' for r in recs})),
               lr_selected=c['lr_selected'], pilot_selection=json.loads(sel.read_text()) if sel.exists() else None)
    R.atomic_write_text(dest / 'manifest.json', json.dumps(man, indent=1))
    print(f'merged {len(recs)} shards ({man["grid"]["verified"]} VERIFIED) into {dest}', flush=True)


# ---------------------------------------------------------------- CLI (called from run.py)

def main(args):
    c = load_config()
    if args.workers:
        c['workers'] = args.workers
    allow_cpu = os.environ.get('TDC_ALLOW_CPU') == '1'
    prep = Path(args.prep) if args.prep else HERE / 'prep' / 'gpu'
    if args.check_only:
        prep.parent.mkdir(parents=True, exist_ok=True)
        check_only(c, prep, R.log_to(prep.parent / 'check_gpu.log'))
    elif args.pilot_select:
        pilot_select(c, args.pilot_select)
    elif args.merge:
        merge(c, args.merge, prep, Path(args.dest) if args.dest else HERE / 'results' / 'gpu')
    elif args.pilot:
        if args.pilot_job is None or not args.output:
            raise SystemExit('--pilot needs --pilot-job J and --output DIR for the GPU arm')
        run_pilot(c, prep, args.output, args.pilot_job, allow_cpu)
    elif args.shard is not None:
        if not args.output:
            raise SystemExit('--shard needs --output DIR')
        run_shard(c, prep, args.output, args.shard, allow_cpu)
    else:
        raise SystemExit('13-GPU: give --check-only, --pilot --pilot-job J, --pilot-select DIR, --shard I or --merge DIR')
