"""Experiment 24 runner: check-only, the learning-rate pilot, pilot selection, one shard per call, merge.

Usage (from the repository root, venv python):
  run24.py --check-only [--prep DIR] [--workers W]      concepts, pools, references into prep (CPU); invariant checks
  run24.py --pilot-job J --output DIR                   pilot job J (0..5) into DIR/pilot/<%05d>/ (GPU)
  run24.py --pilot-select DIR                            frozen pilot rule; prints lr_selected
  run24.py --shard I --output DIR                        shard I (0..917) into DIR/shards/<%05d>/ (GPU)
  run24.py --merge DIR [--dest DIR]                      collect DONE shards into results/ (no checkpoints)
Training reuses Experiment 13-GPU's model, data stream, optimizer and float64 evaluation (gpu_arm.py), unchanged.
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
E13 = HERE.parent.parent / '13_ginc_hmm_complexity_vs_alignment' / 'expt_v1'
sys.path.insert(0, str(E13))
sys.path.insert(0, str(HERE))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import concepts24 as C  # noqa: E402
import gpu_arm as G  # noqa: E402
import hmm as H  # noqa: E402
import model as M  # noqa: E402
import run as R13  # noqa: E402

SOURCES = ['expt_v1/config.json', 'expt_v1/concepts24.py', 'expt_v1/run24.py']
E13_SOURCES = ['hmm.py', 'model.py', 'run.py', 'gpu_arm.py']
SLICES = ['early', 'middle', 'late', 'all']
N_CELLS = 17  # alpha = 1 (no distractors) + 4 similarity levels x 4 alphas < 1


def sources_sha():
    out = {s: C.sha256_file(HERE.parent / s) for s in SOURCES}
    out.update({f'e13/{s}': C.sha256_file(E13 / s) for s in E13_SOURCES})
    return out


# ---------------------------------------------------------------- grid

def cells(cfg):
    sims = cfg['concepts']['similarity_levels']
    lows = [a for a in cfg['grid']['alphas'] if a < 1]
    return [('none', 1.0)] + [(s, a) for s in sims for a in lows]


def targets(tasks):
    return sorted([t for t in tasks if t['role'] == 'target'], key=lambda t: (t['K'], t['task_id']))


def n_shards(cfg):
    return len(cfg['grid']['sizes']) * 3 * len(cfg['concepts']['levels']) * N_CELLS * len(cfg['grid']['seeds'])


def shard_coords(cfg, i):
    """Shard id = ((i_size * 9 + i_target) * 17 + i_cell) * 3 + i_seed; targets ordered by (n, task_id)."""
    if not 0 <= i < n_shards(cfg):
        raise ValueError(f'shard {i} outside 0..{n_shards(cfg) - 1}')
    rest, i_seed = divmod(i, len(cfg['grid']['seeds']))
    rest, i_cell = divmod(rest, N_CELLS)
    i_size, i_t = divmod(rest, 3 * len(cfg['concepts']['levels']))
    sim, alpha = cells(cfg)[i_cell]
    return dict(shard=i, size=cfg['grid']['sizes'][i_size], target_index=i_t, similarity=sim, alpha=alpha,
                seed=cfg['grid']['seeds'][i_seed])


def shard_job(cfg, tasks, i):
    job = shard_coords(cfg, i)
    t = targets(tasks)[job['target_index']]
    job.update(target_id=t['task_id'], level=t['K'])
    return job


def pilot_job(cfg, tasks, j):
    p = cfg['pilot']
    sizes, mults = cfg['grid']['sizes'], p['lr_multipliers']
    if not 0 <= j < len(sizes) * len(mults):
        raise ValueError(f'pilot job {j} outside 0..{len(sizes) * len(mults) - 1}')
    i_size, i_m = divmod(j, len(mults))
    pt = next(t for t in tasks if t['role'] == 'pilot' and t['K'] == p['level'])
    return dict(job=j, size=sizes[i_size], lr_multiplier=mults[i_m], target_id=pt['task_id'], level=pt['K'],
                similarity=p['similarity'], alpha=p['alpha'], seed=p['seed'])


def run_name(job):
    return f"{job['seed']}_{job['target_id']}_{job['similarity']}_{job['alpha']:.2f}_{job['size']}"


def model_config(cfg, size):
    m = cfg['model']
    layers, d, heads = m['sizes'][size]
    return dict(n_layers=layers, d_model=d, n_heads=heads, d_mlp=4 * d, vocab=m['vocab'],
                max_positions=m['max_positions'], expected_params=m['expected_params'][size],
                expected_nonembedding=m['expected_nonembedding'][size])


def train_config(cfg):
    """The keys gpu_arm.make_optimizer reads."""
    m = cfg['model']
    return dict(weight_decay=m['weight_decay'], betas=m['betas'])


# ---------------------------------------------------------------- one run

def train_run(cfg, tasks, refs, job, lr, T_checkpoints, d, device, log):
    t0 = time.time()
    target = next(t for t in tasks if t['task_id'] == job['target_id'])
    pool = C.pool_of(tasks, job['target_id'], job['similarity']) if job['alpha'] < 1 else []
    L, Bd, V = cfg['concepts']['doc_len'], cfg['model']['batch_docs'], cfg['concepts']['V_model_tokens']
    n_docs = T_checkpoints[-1] // L
    n_steps = n_docs // Bd
    defs = {str(t['cid']): t['definition'] for t in [target] + pool}
    dist = [p['cid'] for p in pool]
    args = (defs, job['seed'], target['cid'], dist, job['alpha'], n_docs, L, Bd)
    stream = G.Stream(cfg['model']['data_workers'], args, n_steps, depth=4 * cfg['model']['data_workers'])
    cids, _ = H.schedule(job['seed'], target['cid'], dist, job['alpha'], n_docs)
    torch.manual_seed(job['seed'])
    model = M.build_model(model_config(cfg, job['size'])).to(device)
    opt = G.make_optimizer(train_config(cfg), model, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg['model']['warmup_steps']))
    ck_dir = d / 'checkpoints' / run_name(job)
    ck_dir.mkdir(parents=True)
    oc = refs['oracle'][job['target_id']]
    mc = oc if job['alpha'] >= 1 else refs['mixture'][f"{job['target_id']}|{job['similarity']}|{job['alpha']:.2f}"]
    docs = refs['docs'][job['target_id']]
    n_ne = M.nonembedding_count(model)
    sha = hashlib.sha256()
    rows, perpos, drows, counts, losses = [], [], [], {}, []
    t_eval = t_wait = 0.0
    amp = device.type == 'cuda'
    nonfinite = 0
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
                logits = G.lm_forward(model, x, fast=True)
            loss = F.cross_entropy(logits.float().reshape(-1, V + 1), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['model']['grad_clip'])
            opt.step()
            sched.step()
            if (step + 1) % 1024 == 0 or step == 0:
                lv = float(loss.item())
                nonfinite += 0 if math.isfinite(lv) else 1
                losses.append([step + 1, lv])
                log(f'step {step + 1}/{n_steps} loss {lv / math.log(2):.4f} bits')
            T = (step + 1) * Bd * L
            if T in T_checkpoints:
                ts = time.time()
                m_codes = G.code_lengths(model, docs, V, device)
                M.save_npz(model, ck_dir / f'T_{T}.npz')
                m_doc = (step + 1) * Bd
                counts[str(T)] = {str(int(c)): int((cids[:m_doc] == c).sum()) for c in np.unique(cids[:m_doc])}
                for s in SLICES:
                    lo, hi = cfg['eval']['slices'][s]
                    sm = np.zeros(L, bool)
                    sm[lo - 1:hi] = True
                    mb, ob, xb = m_codes[:, sm].mean(), oc[:, sm].mean(), mc[:, sm].mean()
                    per_doc = (m_codes[:, sm] - oc[:, sm]).mean(1)
                    rows.append(dict(seed=job['seed'], size=job['size'], target_id=job['target_id'], K=target['K'],
                                     K_bits=target['K_bits'], similarity=job['similarity'], alpha=job['alpha'], N=n_ne,
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
                                          similarity=job['similarity'], alpha=job['alpha'], T=T,
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
                pool=[p['task_id'] for p in pool], rows=rows, per_position=perpos, distractor_rows=drows,
                counts=counts, stream_sha256=sha.hexdigest(), data_order_seed=[int(job['seed']), 7777],
                n_docs=int(n_docs), losses=losses, nonfinite_loss_checks=nonfinite,
                train_seconds=round(t_train, 3), eval_seconds=round(t_eval, 3), data_wait_seconds=round(t_wait, 3),
                wall_seconds=round(time.time() - t0, 3),
                finite=bool(all(math.isfinite(r['model_bits']) for r in rows)))


def runtime_info():
    import scipy
    return dict(host=socket.gethostname(), python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, git_revision=G.git_revision(), sources=sources_sha())


def load_prep(prep):
    prep = Path(prep)
    missing = [f for f in C.PREP_FILES if not (prep / f).exists()]
    if missing:
        raise SystemExit(f'{prep} lacks {missing}: run run24.py --check-only first')
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


def run_shard(cfg, prep, out, i, allow_cpu, T_override=None):
    if cfg.get('lr_selected') is None:
        raise SystemExit('lr_selected is unset: run the pilot and make the "Experiment 24 pilot disclosure" commit first')
    tasks = json.loads((Path(prep) / 'tasks.json').read_text())
    job = shard_job(cfg, tasks, i)
    lr = cfg['model']['lr_table'][job['size']] * cfg['lr_selected'][job['size']]
    run_job(cfg, prep, Path(out) / 'shards' / f'{i:05d}', job, lr, T_override or cfg['grid']['T_checkpoints'],
            'shard', allow_cpu)


def run_pilot(cfg, prep, out, j, allow_cpu, T_override=None):
    tasks = json.loads((Path(prep) / 'tasks.json').read_text())
    job = pilot_job(cfg, tasks, j)
    lr = cfg['model']['lr_table'][job['size']] * job['lr_multiplier']
    run_job(cfg, prep, Path(out) / 'pilot' / f'{j:05d}', job, lr, T_override or cfg['pilot']['T_checkpoints'],
            'pilot', allow_cpu)


SMOKE_T = [1048576, 2097152, 4194304]


def run_smoke(cfg, prep, out, i, allow_cpu):
    """Shard i's job at 2x the table learning rate, T to 2^22, into OUT/smoke/shards/ (a GPU code-path check before the
    pilot; never analysed)."""
    tasks = json.loads((Path(prep) / 'tasks.json').read_text())
    job = shard_job(cfg, tasks, i)
    lr = cfg['model']['lr_table'][job['size']] * 2
    run_job(cfg, prep, Path(out) / 'smoke' / 'shards' / f'{i:05d}', job, lr, SMOKE_T, 'smoke', allow_cpu)


def select_rule(cfg, results):
    """Per size, the multiplier with the lowest pilot excess at pilot T (all positions); the tie multiplier (2x, the
    Experiment 13-GPU choice) wins when it is within tie_bits of the best; nonfinite runs never win."""
    p = cfg['pilot']
    chosen = {}
    for size in cfg['grid']['sizes']:
        ok = [r for r in results if r['size'] == size and math.isfinite(r['excess_bits'])]
        if not ok:
            raise SystemExit(f'every pilot run of size {size} is nonfinite')
        best = min(ok, key=lambda r: r['excess_bits'])
        tie = next((r for r in ok if r['lr_multiplier'] == p['tie_multiplier']), None)
        if tie is not None and tie['excess_bits'] - best['excess_bits'] <= p['tie_bits']:
            best = tie
        chosen[size] = best['lr_multiplier']
    return chosen


def pilot_select(cfg, out):
    pdir = Path(out) / 'pilot'
    res = []
    for j in range(len(cfg['grid']['sizes']) * len(cfg['pilot']['lr_multipliers'])):
        d = pdir / f'{j:05d}'
        if not (d / 'DONE').exists():
            raise SystemExit(f'pilot job {j} is not DONE')
        rec = json.loads((d / 'run.json').read_text())
        row = next(r for r in rec['rows'] if r['slice'] == 'all' and r['T'] == cfg['pilot']['T'])
        first = next(r for r in rec['rows'] if r['slice'] == 'all' and r['T'] == rec['T_checkpoints'][0])
        res.append(dict(job=j, size=rec['job']['size'], lr_multiplier=rec['job']['lr_multiplier'], lr=rec['lr'],
                        excess_bits=row['excess_bits'] if rec['finite'] else float('nan'),
                        learn_bits=row['learn_bits'], excess_se=row['excess_se'], first_excess=first['excess_bits'],
                        gpu=rec['hardware']['gpu'], wall_seconds=rec['wall_seconds']))
    sel = dict(lr_selected=select_rule(cfg, res), results=res)
    R13.atomic_write_text(pdir / 'pilot_selection.json', json.dumps(sel, indent=1))
    for r in res:
        print(f"job {r['job']} {r['size']} x{r['lr_multiplier']} lr {r['lr']:.1e} excess {r['excess_bits']:.5f}")
    print('lr_selected =', json.dumps(sel['lr_selected']))
    return sel


# ---------------------------------------------------------------- check-only

def _schedule_check(args):
    seed, cid, dist, alpha, n_docs = args
    cids, _ = H.schedule(seed, cid, dist, alpha, n_docs)
    num = int(round(alpha * 100))
    exact = bool(np.array_equal(np.cumsum(cids == cid), (np.arange(1, n_docs + 1) * num) // 100))
    spread = 0
    if alpha < 1 and dist:
        dc = [int((cids == x).sum()) for x in dist]
        spread = max(dc) - min(dc)
    return exact, spread


def check_only(cfg, prep, workers, log):
    t0 = time.time()
    tasks, acc = C.build_prep(cfg, prep, log)
    rec = dict(acceptance={k: v for k, v in acc.items() if k != 'pools'}, checks={})
    tg = targets(tasks)
    assert len(tg) == 9 and [t['K'] for t in tg] == [8, 8, 8, 32, 32, 32, 64, 64, 64], [t['K'] for t in tg]
    for t in tasks:
        h = H.from_json(t['definition'])
        assert t['process_rank'] == t['K'] == h['n'], t['task_id']
        assert len(H.encode(h)) == t['K_bits'] == H.K_bits(h['n'], h['V'], h['q']), t['task_id']
    sims = cfg['concepts']['similarity_levels']
    summ = {}
    for s in sims:
        ps = [t for t in tasks if t['role'] == 'distractor' and t['similarity'] == s and
              any(t['target_id'] == x['task_id'] for x in tg)]
        assert len(ps) == 9 * cfg['concepts']['pool_size'], (s, len(ps))
        summ[s] = dict(shared_fraction=[min(p['shared_fraction'] for p in ps), max(p['shared_fraction'] for p in ps)],
                       emission_overlap=float(np.mean([p['emission_overlap_with_target'] for p in ps])),
                       entropy=[min(p['entropy_rate'] for p in ps), max(p['entropy_rate'] for p in ps)],
                       attempts_max=max(p['attempts'] for p in ps))
    assert summ['disjoint']['emission_overlap'] == 0.0 and summ['disjoint']['shared_fraction'] == [0.0, 0.0]
    rec['checks']['pools'] = summ
    for s in ('S', 'M'):
        m = M.build_model(model_config(cfg, s))
        rec['checks'][f'params_{s}'] = [M.param_count(m), M.nonembedding_count(m)]
    nd = cfg['grid']['T_checkpoints'][-1] // cfg['concepts']['doc_len']
    jobs = [(cfg['grid']['seeds'][0], t['cid'], [p['cid'] for p in C.pool_of(tasks, t['task_id'], s)], a, nd)
            for t in tg for s in sims for a in cfg['grid']['alphas'] if a < 1]
    with cf.ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_schedule_check, jobs))
    assert all(r[0] for r in res), 'an alpha schedule is not exact'
    rec['checks']['alpha_schedules'] = dict(n=len(jobs), docs_each=nd, max_distractor_spread=max(r[1] for r in res))
    log(f'alpha schedules exact ({len(jobs)} x {nd})')
    eval_h = {}
    for t in tasks:
        for x in np.load(Path(prep) / 'data' / f"{t['task_id']}_test.npz")['docs']:
            eval_h[H.doc_hash(x)] = t['task_id']
    seeds = list(cfg['grid']['seeds']) + [cfg['pilot']['seed']]
    jobs = [(t['definition'], s, t['cid'], 2 ** 16, cfg['concepts']['doc_len']) for t in tasks for s in seeds]
    hits = 0
    with cf.ProcessPoolExecutor(workers) as ex:
        for hs in ex.map(R13._first_doc_hashes, jobs):
            hits += len(hs & eval_h.keys())
    assert hits == 0, f'{hits} evaluation documents equal training documents'
    rec['checks']['eval_train_disjoint'] = dict(eval_docs=len(eval_h), training_docs_checked=len(jobs) * 2 ** 16)
    log(f'eval/train disjoint ({len(jobs) * 2 ** 16} training documents)')
    sh = [shard_job(cfg, tasks, i) for i in range(n_shards(cfg))]
    assert len({(j['size'], j['target_id'], j['similarity'], j['alpha'], j['seed']) for j in sh}) == n_shards(cfg) == 918
    rec['checks']['shards'] = n_shards(cfg)
    o, mx = np.load(Path(prep) / 'oracle.npz'), np.load(Path(prep) / 'mixture.npz')
    eid = {}
    for k in mx.files:
        tid, s, a = k.split('|')
        eid.setdefault(s, {}).setdefault(a, []).append(float((mx[k] - o[tid]).mean()))
    rec['eps_id_mean'] = {s: {a: float(np.mean(v)) for a, v in d.items()} for s, d in eid.items()}
    rec['prep_sha256'] = C.prep_hashes(prep)
    rec['sources'] = sources_sha()
    rec['seconds'] = round(time.time() - t0, 1)
    R13.atomic_write_text(HERE / 'results_check.json', json.dumps(rec, indent=1))
    log(json.dumps(rec['checks']))
    log('eps_id mean by similarity and alpha: ' + json.dumps(rec['eps_id_mean']))
    return rec


# ---------------------------------------------------------------- merge

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
                           done_sha256=C.sha256_file(d / 'DONE'), verified=ver))
        ck[rec['run']] = {k: v for k, v in done['files'].items() if k.startswith('checkpoints/')}
    if len({json.dumps(s['sources'], sort_keys=True) for s in shards}) != 1:
        raise SystemExit('shards ran different code')
    dest.mkdir(parents=True, exist_ok=True)
    for name in C.PREP_FILES:
        shutil.copy2(prep / name, dest / name)
    shutil.copytree(prep / 'data', dest / 'data')
    cols = ['seed', 'size', 'target_id', 'K', 'K_bits', 'similarity', 'alpha', 'N', 'T', 'step', 'slice',
            'excess_bits', 'ident_bits', 'learn_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'n_tokens',
            'excess_se']
    with open(dest / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in recs:
            for row in r['rows']:
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    with open(dest / 'per_position.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'size', 'target_id', 'similarity', 'alpha', 'T'] + [f'p{i + 1}' for i in range(128)])
        for r in recs:
            j = r['job']
            for pp in r['per_position']:
                w.writerow([j['seed'], j['size'], j['target_id'], j['similarity'], j['alpha'], pp['T']]
                           + [repr(x) for x in pp['model']])
    with open(dest / 'distractor_loss.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['seed', 'size', 'target_id', 'similarity', 'alpha', 'T', 'distractor_id',
                               'model_bits', 'oracle_bits'])
        w.writeheader()
        for r in recs:
            w.writerows(r['distractor_rows'])
    R13.atomic_write_text(dest / 'mixes.json', json.dumps(
        {r['run']: dict(job=r['job'], pool=r['pool'], counts=r['counts'], stream_sha256=r['stream_sha256'],
                        n_docs=r['n_docs']) for r in recs}, indent=0, sort_keys=True))
    R13.atomic_write_text(dest / 'shards.json', json.dumps(shards, indent=1))
    R13.atomic_write_text(dest / 'checkpoint_sha256.json', json.dumps(ck, indent=1, sort_keys=True))
    sel = out / 'pilot' / 'pilot_selection.json'
    man = dict(sources=shards[0]['sources'], prep_sha256=ph, lr_selected=cfg['lr_selected'],
               pilot_selection=json.loads(sel.read_text()) if sel.exists() else None,
               grid=dict(runs=len(recs), diverged=[r['run'] for r in recs if not r['finite']],
                         verified=sum(1 for s in shards if s['verified']),
                         gpu_hours=round(sum(r['wall_seconds'] for r in recs) / 3600, 2),
                         gpus=sorted({r['hardware']['gpu'] or 'cpu' for r in recs})))
    R13.atomic_write_text(dest / 'manifest.json', json.dumps(man, indent=1))
    print(f'merged {len(recs)} shards ({man["grid"]["verified"]} VERIFIED) into {dest}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--pilot-job', type=int)
    ap.add_argument('--pilot-select')
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
        check_only(cfg, prep, a.workers, R13.log_to(prep.parent / 'check24.log'))
    elif a.pilot_select:
        pilot_select(cfg, a.pilot_select)
    elif a.merge:
        merge(cfg, a.merge, prep, Path(a.dest) if a.dest else HERE / 'results')
    elif a.pilot_job is not None:
        run_pilot(cfg, prep, a.output, a.pilot_job, allow_cpu)
    elif a.shard is not None:
        run_shard(cfg, prep, a.output, a.shard, allow_cpu)
    elif a.smoke is not None:
        run_smoke(cfg, prep, a.output, a.smoke, allow_cpu)
    else:
        ap.error('nothing to do')


if __name__ == '__main__':
    main()
