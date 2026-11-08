"""Experiment 25 runner: check-only, the GPU smoke, one shard per call, merge.

Usage (from the repository root, venv python):
  run25.py --check-only [--prep DIR] [--workers W]   concepts, pools, references into prep (CPU); invariant checks
  run25.py --smoke I --output DIR                     shard I's job to 2^22 into DIR/smoke/shards/<%05d>/ (GPU)
  run25.py --shard I --output DIR                     shard I (0..2879) into DIR/shards/<%05d>/ (GPU)
  run25.py --merge DIR [--dest DIR]                   collect DONE shards into results/ (no checkpoints)
Model, data stream, optimizer and float64 evaluation are Experiment 13-GPU's (gpu_arm.py), as in Experiment 24. The
only training change is the masked arm: the loss averages over the target documents' tokens only.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import concepts25 as C  # noqa: E402  (puts Experiments 13 and 24 on the path)

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import gpu_arm as G  # noqa: E402
import hmm as H  # noqa: E402
import model as M  # noqa: E402
import run as R13  # noqa: E402

SOURCES = ['expt_v1/config.json', 'expt_v1/concepts25.py', 'expt_v1/run25.py']
E13_SOURCES = ['hmm.py', 'model.py', 'run.py', 'gpu_arm.py']
E24_SOURCES = ['concepts24.py', 'config.json']
SLICES = ['early', 'middle', 'late', 'all']


def sources_sha():
    out = {s: C.sha256_file(HERE.parent / s) for s in SOURCES}
    out.update({f'e13/{s}': C.sha256_file(C.E13 / s) for s in E13_SOURCES})
    out.update({f'e24/{s}': C.sha256_file(C.E24 / s) for s in E24_SOURCES})
    return out


# ---------------------------------------------------------------- grid

def cells(cfg):
    lows = [a for a in cfg['grid']['alphas'] if a < 1]
    return [('none', 1.0)] + [(arm, a) for arm in cfg['grid']['arms'] for a in lows]


def dims(cfg):
    g = cfg['grid']
    return [len(g['sizes']), 3 * len(cfg['concepts']['levels']), len(cells(cfg)), len(g['lr_multipliers']),
            len(g['seeds'])]


def n_shards(cfg):
    return int(np.prod(dims(cfg)))


def shard_coords(cfg, i):
    """Shard id = (((i_size * 6 + i_target) * 16 + i_cell) * 5 + i_lr) * 3 + i_seed; targets ordered by (n, task_id)."""
    if not 0 <= i < n_shards(cfg):
        raise ValueError(f'shard {i} outside 0..{n_shards(cfg) - 1}')
    g = cfg['grid']
    idx = np.unravel_index(i, dims(cfg))
    arm, alpha = cells(cfg)[idx[2]]
    return dict(shard=i, size=g['sizes'][idx[0]], target_index=int(idx[1]), arm=arm, alpha=alpha,
                lr_multiplier=g['lr_multipliers'][idx[3]], seed=g['seeds'][idx[4]])


def shard_index(cfg, size, target_index, arm, alpha, lr_multiplier, seed):
    g = cfg['grid']
    return int(np.ravel_multi_index((g['sizes'].index(size), target_index, cells(cfg).index((arm, alpha)),
                                     g['lr_multipliers'].index(lr_multiplier), g['seeds'].index(seed)), dims(cfg)))


def shard_job(cfg, tasks, i):
    job = shard_coords(cfg, i)
    t = C.targets(tasks)[job['target_index']]
    job.update(target_id=t['task_id'], level=t['K'])
    return job


def smoke_shards(cfg):
    """A cheap and an expensive code path: S, first target, alpha = 1, lowest learning rate; M, last target, masked
    arm at the lowest alpha, highest learning rate."""
    g = cfg['grid']
    lo = min(a for a in g['alphas'] if a < 1)
    return [shard_index(cfg, 'S', 0, 'none', 1.0, g['lr_multipliers'][0], g['seeds'][0]),
            shard_index(cfg, 'M', 3 * len(cfg['concepts']['levels']) - 1, 'masked', lo, g['lr_multipliers'][-1],
                        g['seeds'][0])]


def run_name(job):
    return f"{job['seed']}_{job['target_id']}_{job['arm']}_{job['alpha']:.2f}_x{job['lr_multiplier']:g}_{job['size']}"


def pool_kind(cfg, job):
    return cfg['grid']['pool_of_arm'][job['arm']] if job['alpha'] < 1 else None


def model_config(cfg, size):
    m = cfg['model']
    layers, d, heads = m['sizes'][size]
    return dict(n_layers=layers, d_model=d, n_heads=heads, d_mlp=4 * d, vocab=m['vocab'],
                max_positions=m['max_positions'], expected_params=m['expected_params'][size],
                expected_nonembedding=m['expected_nonembedding'][size])


def train_config(cfg):
    m = cfg['model']
    return dict(weight_decay=m['weight_decay'], betas=m['betas'])


def stream_inputs(cfg, tasks, job, n_docs):
    """(Stream initargs, schedule cids, target, pool) for a job; identical in training and verification."""
    target = next(t for t in tasks if t['task_id'] == job['target_id'])
    pk = pool_kind(cfg, job)
    pool = C.pool_of(tasks, job['target_id'], pk) if pk else []
    L, Bd = cfg['concepts']['doc_len'], cfg['model']['batch_docs']
    defs = {str(t['cid']): t['definition'] for t in [target] + pool}
    dist = [p['cid'] for p in pool]
    cids, _ = H.schedule(job['seed'], target['cid'], dist, job['alpha'], n_docs)
    return (defs, job['seed'], target['cid'], dist, job['alpha'], n_docs, L, Bd), cids, target, pool


def mixture_ref(refs, job, pk, oc):
    """The identification reference: the exact Bayes mixture of the training distribution. The masked arm learns from
    target documents only, so its reference is the oracle (identification floor 0)."""
    if job['alpha'] >= 1 or job['arm'] == 'masked':
        return oc
    return refs['mixture'][f"{job['target_id']}|{pk}|{job['alpha']:.2f}"]


# ---------------------------------------------------------------- one run

def batch_loss(logits, y, V, rows=None):
    """Mean next-token cross-entropy (nats). With rows (a boolean mask over the batch's documents), the mean is over
    those documents' tokens only, so the other documents give no gradient (the masked arm)."""
    if rows is not None:
        logits, y = logits[rows], y[rows]
    return F.cross_entropy(logits.float().reshape(-1, V + 1), y.reshape(-1))


def train_run(cfg, tasks, refs, job, lr, T_checkpoints, d, device, log):
    t0 = time.time()
    L, Bd, V = cfg['concepts']['doc_len'], cfg['model']['batch_docs'], cfg['concepts']['V_model_tokens']
    n_docs = T_checkpoints[-1] // L
    n_steps = n_docs // Bd
    args, cids, target, pool = stream_inputs(cfg, tasks, job, n_docs)
    pk = pool_kind(cfg, job)
    stream = G.Stream(cfg['model']['data_workers'], args, n_steps, depth=4 * cfg['model']['data_workers'])
    masked = job['arm'] == 'masked' and job['alpha'] < 1
    is_target = torch.from_numpy(cids == target['cid'])
    torch.manual_seed(job['seed'])
    model = M.build_model(model_config(cfg, job['size'])).to(device)
    opt = G.make_optimizer(train_config(cfg), model, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg['model']['warmup_steps']))
    ck_dir = d / 'checkpoints' / run_name(job)
    ck_dir.mkdir(parents=True)
    oc = refs['oracle'][job['target_id']]
    mc = mixture_ref(refs, job, pk, oc)
    docs = refs['docs'][job['target_id']]
    n_ne = M.nonembedding_count(model)
    sha = hashlib.sha256()
    rows, perpos, drows, counts, losses, T_done = [], [], [], {}, [], []
    diverged_at, steps_run = None, 0
    t_eval = t_wait = 0.0
    amp = device.type == 'cuda'
    model.train()
    t_loop = time.time()
    try:
        for step in range(n_steps):
            ts = time.time()
            batch = stream.get()
            sha.update(batch.tobytes())
            steps_run = step + 1
            t_wait += time.time() - ts
            x, y = M.inputs_targets(batch, V)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = G.lm_forward(model, x, fast=True)
            rows_t = is_target[step * Bd:(step + 1) * Bd].to(device, non_blocking=True) if masked else None
            loss = batch_loss(logits, y, V, rows_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['model']['grad_clip'])
            opt.step()
            sched.step()
            if (step + 1) % 1024 == 0 or step == 0:
                lv = float(loss.item())
                losses.append([step + 1, lv])
                log(f'step {step + 1}/{n_steps} loss {lv / math.log(2):.4f} bits')
            T = (step + 1) * Bd * L
            if T in T_checkpoints:
                ts = time.time()
                m_codes = G.code_lengths(model, docs, V, device)
                if not np.all(np.isfinite(m_codes)):  # diverged: stop; the run never crosses after this point
                    diverged_at = T
                    log(f'nonfinite code lengths at T=2^{int(math.log2(T))}: stopping')
                    break
                M.save_npz(model, ck_dir / f'T_{T}.npz')
                T_done.append(T)
                m_doc = (step + 1) * Bd
                counts[str(T)] = {str(int(c)): int((cids[:m_doc] == c).sum()) for c in np.unique(cids[:m_doc])}
                for s in SLICES:
                    lo, hi = cfg['eval']['slices'][s]
                    sm = np.zeros(L, bool)
                    sm[lo - 1:hi] = True
                    mb, ob, xb = m_codes[:, sm].mean(), oc[:, sm].mean(), mc[:, sm].mean()
                    per_doc = (m_codes[:, sm] - oc[:, sm]).mean(1)
                    rows.append(dict(seed=job['seed'], size=job['size'], target_id=job['target_id'], K=target['K'],
                                     arm=job['arm'], alpha=job['alpha'], lr_multiplier=job['lr_multiplier'], N=n_ne,
                                     T=T, step=step + 1, slice=s, excess_bits=float(mb - ob),
                                     ident_bits=float(xb - ob), learn_bits=float(mb - xb), model_bits=float(mb),
                                     oracle_bits=float(ob), mixture_bits=float(xb), n_tokens=int(sm.sum() * len(docs)),
                                     excess_se=float(per_doc.std(ddof=1) / math.sqrt(len(per_doc)))))
                perpos.append(dict(T=T, model=m_codes.mean(0).tolist()))
                with torch.no_grad():  # loss on this run's own distractors (reported, float32)
                    model.eval()
                    for p in pool:
                        xx, yy = M.inputs_targets(refs['docs'][p['task_id']], V)
                        lp = F.log_softmax(G.lm_forward(model, xx.to(device)).float(), -1)
                        nll = float((-lp.gather(-1, yy.to(device)[..., None])).mean()) / math.log(2.0)
                        drows.append(dict(seed=job['seed'], size=job['size'], target_id=job['target_id'],
                                          arm=job['arm'], alpha=job['alpha'], lr_multiplier=job['lr_multiplier'], T=T,
                                          distractor_id=p['task_id'], model_bits=nll,
                                          oracle_bits=float(refs['oracle'][p['task_id']].mean())))
                    model.train()
                t_eval += time.time() - ts
                log(f'checkpoint T=2^{int(math.log2(T))}: learn {rows[-1]["learn_bits"]:.4f} bits, '
                    f'excess {rows[-1]["excess_bits"]:.4f}')
    finally:
        stream.close()
    if amp:
        torch.cuda.synchronize()
    t_train = time.time() - t_loop - t_eval
    return dict(run=run_name(job), job=job, lr=lr, N=n_ne, N_total=M.param_count(model), T_checkpoints=T_checkpoints,
                T_done=T_done, diverged_at=diverged_at, pool=[p['task_id'] for p in pool], pool_kind=pk, rows=rows,
                per_position=perpos, distractor_rows=drows, counts=counts,
                stream_sha256=sha.hexdigest(), steps_run=steps_run,
                data_order_seed=[int(job['seed']), 7777], n_docs=int(n_docs), losses=losses,
                train_seconds=round(t_train, 3), eval_seconds=round(t_eval, 3), data_wait_seconds=round(t_wait, 3),
                wall_seconds=round(time.time() - t0, 3), finite=diverged_at is None)


def runtime_info():
    import scipy
    return dict(host=socket.gethostname(), python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, git_revision=G.git_revision(), sources=sources_sha())


def load_prep(prep):
    prep = Path(prep)
    missing = [f for f in C.PREP_FILES if not (prep / f).exists()]
    if missing:
        raise SystemExit(f'{prep} lacks {missing}: run run25.py --check-only first')
    tasks = json.loads((prep / 'tasks.json').read_text())
    return tasks, C.load_refs(prep, tasks), C.prep_hashes(prep)


def run_job(cfg, prep, d, job, lr, Ts, kind, allow_cpu):
    if (d / 'DONE').exists():
        print(f'{kind} {d.name} already DONE', flush=True)
        return
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f'{d} exists without DONE (a crash): move it aside, log it in CKPT, then rerun')
    tasks, refs, ph = load_prep(prep)
    device, hw = G.device_or_die(allow_cpu)
    d.mkdir(parents=True, exist_ok=True)
    log = R13.log_to(d / 'train.log')
    log(f'{kind} {d.name}: {json.dumps(job)} lr {lr:g} T {Ts[0]}..{Ts[-1]} on {hw["gpu"] or "cpu"}')
    rec = train_run(cfg, tasks, refs, job, lr, Ts, d, device, log)
    rec.update(kind=kind, hardware=hw, runtime=runtime_info(), prep_sha256=ph,
               started=time.strftime('%m-%d-%Y %H:%M:%S'))
    R13.atomic_write_text(d / 'run.json', json.dumps(rec))
    log(f'finished in {rec["wall_seconds"]:.0f}s; writing DONE')
    G.finish_dir(d, dict(kind=kind, job=job, lr=lr, run=rec['run']))


def job_lr(cfg, job):
    return cfg['model']['lr_table'][job['size']] * job['lr_multiplier']


def run_shard(cfg, prep, out, i, allow_cpu, T_override=None):
    tasks = json.loads((Path(prep) / 'tasks.json').read_text())
    job = shard_job(cfg, tasks, i)
    run_job(cfg, prep, Path(out) / 'shards' / f'{i:05d}', job, job_lr(cfg, job),
            T_override or cfg['grid']['T_checkpoints'], 'shard', allow_cpu)


def run_smoke(cfg, prep, out, i, allow_cpu):
    """Shard i's job to 2^22 into OUT/smoke/shards/ (a GPU code-path check; never analysed)."""
    tasks = json.loads((Path(prep) / 'tasks.json').read_text())
    job = shard_job(cfg, tasks, i)
    run_job(cfg, prep, Path(out) / 'smoke' / 'shards' / f'{i:05d}', job, job_lr(cfg, job),
            cfg['smoke']['T_checkpoints'], 'smoke', allow_cpu)


# ---------------------------------------------------------------- check-only

def _schedule_check(args):
    seed, cid, dist, alpha, n_docs, Bd = args
    cids, _ = H.schedule(seed, cid, dist, alpha, n_docs)
    num = int(round(alpha * 100))
    exact = bool(np.array_equal(np.cumsum(cids == cid), (np.arange(1, n_docs + 1) * num) // 100))
    dc = [int((cids == x).sum()) for x in dist]
    per_batch = (cids == cid).reshape(-1, Bd).sum(1)
    return exact, max(dc) - min(dc), int(per_batch.min())


def check_only(cfg, prep, workers, log):
    t0 = time.time()
    tasks, acc = C.build_prep(cfg, prep, log)
    rec = dict(acceptance=acc, checks={})
    tg = C.targets(tasks)
    assert [t['K'] for t in tg] == [8, 8, 8, 32, 32, 32], [t['K'] for t in tg]
    ref24 = {t['task_id']: t for t in json.loads((C.ROOT / cfg['source']['exp24_tasks']).read_text())}
    same = 0
    for t in tasks:
        if t['similarity'] == 'iid':
            continue
        r = ref24[t['task_id']]
        assert r['cid'] == t['cid'] and r['definition'] == t['definition'], t['task_id']
        h = H.from_json(t['definition'])
        assert t['process_rank'] == t['K'] == h['n'], t['task_id']
        assert len(H.encode(h)) == t['K_bits'] == H.K_bits(h['n'], h['V'], h['q']), t['task_id']
        same += 1
    rec['checks']['identical_to_exp24'] = same
    log(f'{same} targets and disjoint distractors identical to Experiment 24 (definitions and concept ids)')
    cc = cfg['concepts']
    byid = {t['task_id']: t for t in tasks}
    iid = [t for t in tasks if t['similarity'] == 'iid']
    disj = [t for t in tasks if t['similarity'] == 'disjoint']
    assert len(disj) == len(iid) == len(tg) * cc['pool_size']
    for t in iid:
        g, h = H.from_json(t['definition']), H.from_json(byid[t['parent_id']]['definition'])
        assert g['n'] == 1 and int(g['emit'].sum()) == 2 ** cc['iid_q'] and g['etok'].min() >= 150
        assert C.iid_matches_parent(g, h, cc['doc_len'], cc['iid_tv_max']), t['task_id']
    for t in disj:
        assert H.from_json(t['definition'])['etok'].min() >= 150 and t['emission_overlap_with_target'] == 0
    rec['checks']['iid'] = dict(n=len(iid), tv_max=max(t['unigram_tv'] for t in iid),
                                support=[min(t['support'] for t in iid), max(t['support'] for t in iid)],
                                entropy_iid=float(np.mean([t['entropy_rate'] for t in iid])),
                                entropy_parent=float(np.mean([t['parent_entropy_rate'] for t in iid])))
    for s in ('S', 'M'):
        m = M.build_model(model_config(cfg, s))
        rec['checks'][f'params_{s}'] = [M.param_count(m), M.nonembedding_count(m)]
    nd = cfg['grid']['T_checkpoints'][-1] // cc['doc_len']
    Bd = cfg['model']['batch_docs']
    jobs = [(cfg['grid']['seeds'][0], t['cid'], [p['cid'] for p in C.pool_of(tasks, t['task_id'], pk)], a, nd, Bd)
            for t in tg for pk in C.POOLS for a in cfg['grid']['alphas'] if a < 1]
    with cf.ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_schedule_check, jobs))
    assert all(r[0] for r in res), 'an alpha schedule is not exact'
    assert min(r[2] for r in res) >= 1, 'a batch without target documents (the masked arm needs one)'
    rec['checks']['alpha_schedules'] = dict(n=len(jobs), docs_each=nd, max_distractor_spread=max(r[1] for r in res),
                                            min_target_docs_per_batch=min(r[2] for r in res))
    log(f'alpha schedules exact ({len(jobs)} x {nd}); every batch has a target document')
    eval_h = {}
    for t in tasks:
        for x in np.load(Path(prep) / 'data' / f"{t['task_id']}_test.npz")['docs']:
            eval_h[H.doc_hash(x)] = t['task_id']
    jobs = [(t['definition'], s, t['cid'], 2 ** 16, cc['doc_len']) for t in tasks for s in cfg['grid']['seeds']]
    hits = 0
    with cf.ProcessPoolExecutor(workers) as ex:
        for hs in ex.map(R13._first_doc_hashes, jobs):
            hits += len(hs & eval_h.keys())
    assert hits == 0, f'{hits} evaluation documents equal training documents'
    rec['checks']['eval_train_disjoint'] = dict(eval_docs=len(eval_h), training_docs_checked=len(jobs) * 2 ** 16)
    log(f'eval/train disjoint ({len(jobs) * 2 ** 16} training documents)')
    sh = [shard_job(cfg, tasks, i) for i in range(n_shards(cfg))]
    keys = {(j['size'], j['target_id'], j['arm'], j['alpha'], j['lr_multiplier'], j['seed']) for j in sh}
    assert len(keys) == n_shards(cfg) == 2880
    assert all(shard_index(cfg, j['size'], j['target_index'], j['arm'], j['alpha'], j['lr_multiplier'], j['seed'])
               == j['shard'] for j in sh)
    rec['checks']['shards'] = n_shards(cfg)
    rec['checks']['smoke_shards'] = smoke_shards(cfg)
    rec['eps_id_mean'] = C.mean_eps_id(prep)
    rec['posterior16_mean'] = {pk: float(np.mean([v for k, v in json.loads((Path(prep) / 'posterior16.json')
                                                                           .read_text()).items() if f'|{pk}|' in k]))
                               for pk in C.POOLS}
    rec['prep_sha256'] = C.prep_hashes(prep)
    rec['sources'] = sources_sha()
    rec['seconds'] = round(time.time() - t0, 1)
    R13.atomic_write_text(HERE / 'results_check.json', json.dumps(rec, indent=1))
    log(json.dumps(rec['checks']))
    log('eps_id mean by pool and alpha: ' + json.dumps(rec['eps_id_mean']))
    return rec


# ---------------------------------------------------------------- merge

METRIC_COLS = ['seed', 'size', 'target_id', 'K', 'arm', 'alpha', 'lr_multiplier', 'N', 'T', 'step', 'slice',
               'excess_bits', 'ident_bits', 'learn_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'n_tokens',
               'excess_se']


def merge(cfg, out, prep, dest):
    out, prep, dest = Path(out), Path(prep), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f'{dest} exists and is not empty')
    ph = C.prep_hashes(prep)
    recs, shards, ck = [], [], {}
    for i in range(n_shards(cfg)):
        d = out / 'shards' / f'{i:05d}'
        if not (d / 'DONE').exists():
            raise SystemExit(f'shard {i} is not DONE')
        done = json.loads((d / 'DONE').read_text())
        if done['files']['run.json'] != C.sha256_file(d / 'run.json'):
            raise SystemExit(f'shard {i}: run.json does not match DONE')
        rec = json.loads((d / 'run.json').read_text())
        if rec['prep_sha256'] != ph:
            raise SystemExit(f'shard {i} used different concepts or references')
        recs.append(rec)
        ver = json.loads((d / 'VERIFIED').read_text()) if (d / 'VERIFIED').exists() else None
        shards.append(dict(shard=i, run=rec['run'], job=rec['job'], lr=rec['lr'], hardware=rec['hardware'],
                           sources=rec['runtime']['sources'], wall_seconds=rec['wall_seconds'], finite=rec['finite'],
                           diverged_at=rec['diverged_at'], done_sha256=C.sha256_file(d / 'DONE'), verified=ver))
        ck[rec['run']] = {k: v for k, v in done['files'].items() if k.startswith('checkpoints/')}
    if len({json.dumps(s['sources'], sort_keys=True) for s in shards}) != 1:
        raise SystemExit('shards ran different code')
    dest.mkdir(parents=True, exist_ok=True)
    for name in C.PREP_FILES:
        shutil.copy2(prep / name, dest / name)
    shutil.copytree(prep / 'data', dest / 'data')
    with open(dest / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, METRIC_COLS)
        w.writeheader()
        for r in recs:
            for row in r['rows']:
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    with open(dest / 'runs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'size', 'target_id', 'K', 'arm', 'alpha', 'lr_multiplier', 'diverged_at', 'T_last',
                    'wall_seconds'])
        for r in recs:
            j = r['job']
            w.writerow([j['seed'], j['size'], j['target_id'], j['level'], j['arm'], j['alpha'], j['lr_multiplier'],
                        r['diverged_at'] or '', r['T_done'][-1] if r['T_done'] else '', r['wall_seconds']])
    with open(dest / 'per_position.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'size', 'target_id', 'arm', 'alpha', 'lr_multiplier', 'T'] + [f'p{i + 1}' for i in range(128)])
        for r in recs:
            j = r['job']
            for pp in r['per_position']:
                w.writerow([j['seed'], j['size'], j['target_id'], j['arm'], j['alpha'], j['lr_multiplier'], pp['T']]
                           + [repr(x) for x in pp['model']])
    with open(dest / 'distractor_loss.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['seed', 'size', 'target_id', 'arm', 'alpha', 'lr_multiplier', 'T', 'distractor_id',
                               'model_bits', 'oracle_bits'])
        w.writeheader()
        for r in recs:
            w.writerows(r['distractor_rows'])
    R13.atomic_write_text(dest / 'mixes.json', json.dumps(
        {r['run']: dict(job=r['job'], pool=r['pool'], counts=r['counts'], stream_sha256=r['stream_sha256'],
                        n_docs=r['n_docs']) for r in recs}, indent=0, sort_keys=True))
    R13.atomic_write_text(dest / 'shards.json', json.dumps(shards, indent=1))
    R13.atomic_write_text(dest / 'checkpoint_sha256.json', json.dumps(ck, indent=1, sort_keys=True))
    man = dict(sources=shards[0]['sources'], prep_sha256=ph,
               grid=dict(runs=len(recs), diverged=[r['run'] for r in recs if not r['finite']],
                         verified=sum(1 for s in shards if s['verified']),
                         gpu_hours=round(sum(r['wall_seconds'] for r in recs) / 3600, 2),
                         gpus=sorted({r['hardware']['gpu'] or 'cpu' for r in recs})))
    R13.atomic_write_text(dest / 'manifest.json', json.dumps(man, indent=1))
    print(f'merged {len(recs)} shards ({man["grid"]["verified"]} VERIFIED) into {dest}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--shard', type=int)
    ap.add_argument('--smoke', type=int)
    ap.add_argument('--merge')
    ap.add_argument('--output')
    ap.add_argument('--prep')
    ap.add_argument('--dest')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    cfg = C.load_config()
    prep = Path(a.prep) if a.prep else HERE / 'prep'
    allow_cpu = os.environ.get('TDC_ALLOW_CPU') == '1'
    if a.check_only:
        prep.parent.mkdir(parents=True, exist_ok=True)
        check_only(cfg, prep, a.workers, R13.log_to(prep.parent / 'check25.log'))
    elif a.merge:
        merge(cfg, a.merge, prep, Path(a.dest) if a.dest else HERE / 'results')
    elif a.shard is not None:
        run_shard(cfg, prep, a.output, a.shard, allow_cpu)
    elif a.smoke is not None:
        run_smoke(cfg, prep, a.output, a.smoke, allow_cpu)
    else:
        ap.error('nothing to do')


if __name__ == '__main__':
    main()
