"""Experiment 17 runner (CPU): concepts, exact references, invariant checks, pilot and gated grid.

Two arms on Experiment 13-small's GINC-style HMMs and model (PROTOCOL.md):
  H (primary)   hard -> easy, same family: training pool 'easy' (8 n=4 HMMs), 'hard' (8 n=32 HMMs) or 'near' (the
                mixed control, 2 HMMs at each n in {4, 8, 16, 32}), all on tokens 0-31; held-out easy (n=4) and hard
                (n=32) targets at alpha in {0, 0.05}.
  D (secondary) disjoint support: pool 'near' (tokens 0-31, the targets' tokens) versus 'far' (the same 8 HMMs with
                every token moved to 32-63); alpha in {0, 0.05, 0.4}, plus alpha = 1 ('none', no pool) shared.
A run is (seed, pool, target, alpha). An alpha = 0 run has no target ('-') and is scored on every target.

Usage (venv python, from expt_v1/):
  run.py --check-only                  concepts + references into prep/, invariant checks, results_check.json
  run.py --pilot                       disclosed pilot (seed 17909, the two pilot concepts) into results_pilot/
  run.py --output results [--max-seconds S]   gated grid, resume-safe (finished runs are atomic and skipped)
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch  # noqa: E402

import hmm as H  # noqa: E402
import model as M  # noqa: E402

SOURCES = ['config.json', 'hmm.py', 'model.py', 'run.py']
ANALYSIS_SOURCES = ['laws.py', 'analyze.py', 'verify.py', 'report.py']
SLICES = ['early', 'middle', 'late', 'all']
POOL_ROLES = {'near': 'near', 'far': 'far', 'easy': 'easy', 'hard': 'hard'}


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_write_text(path, text):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def run_name(seed, pool, target_id, alpha):
    return f'{seed}_{pool}_{target_id}_a{alpha:.2f}'


def log_to(path):
    def log(msg):
        line = f'{datetime.now().strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(path, 'a') as f:
            f.write(line + '\n')
    return log


# ---------------------------------------------------------------- concepts (PROTOCOL Section 2)

def _check12(h):
    if not H.strongly_connected_aperiodic(h):
        return False, 'not strongly connected and aperiodic', None
    r, how = H.process_rank(h)
    if r != h['n']:
        return False, f'process rank {r} < n ({how})', r
    return True, 'ok', r


def needed_per_level(c):
    """Region-A concepts needed per level, in acceptance order: targets, near (= far after the shift), pool, pilot."""
    need = {}
    for n in sorted(set(c['levels_near']) | set(c['levels_targets'])):
        roles = []
        if n in c['levels_targets']:
            roles += ['target'] * c['targets_per_level']
        if n in c['levels_near']:
            roles += ['near'] * c['near_per_level']
        if n == c['easy_level']:
            roles += ['easy'] * c['pool_size']
        if n == c['hard_level']:
            roles += ['hard'] * c['pool_size']
        if n in c['levels_targets']:
            roles += ['pilot']
        need[n] = roles
    return need


def build_concepts(c, log=print):
    """Experiment 13's acceptance procedure on tokens 0-31: per level a fixed candidate list from one PCG64 stream,
    checks 1-2 (strongly connected and aperiodic, process rank = n), entropy band around the median of the levels'
    first 64 passing candidates (bands tried in order), acceptance in generation order. Far concepts are the near
    concepts with every token moved by +32."""
    rng = np.random.default_rng([c['task_sampling_seed'], 17])
    V, q, W, lo = c['V'], c['q'], c['region_width'], c['region_A']
    need = needed_per_level(c)
    levels = sorted(need)
    cands = {n: [H.sample_candidate_region(rng, n, V, q, lo, W) for _ in range(c['max_candidates_per_level'])]
             for n in levels}
    status = {n: {} for n in levels}
    quick = {n: {} for n in levels}

    def chk(n, i):
        if i not in status[n]:
            status[n][i] = _check12(cands[n][i])
        return status[n][i]

    def qh(n, i):
        if i not in quick[n]:
            r = np.random.default_rng([c['task_sampling_seed'], 17, n, i, 1])
            quick[n][i] = H.entropy_rate(cands[n][i], r, c['quick_entropy_seqs'], c['entropy_seq_len'],
                                         c['entropy_burn_in'])[0]
        return quick[n][i]
    calib = []
    for n in levels:
        got = 0
        for i in range(c['max_candidates_per_level']):
            if chk(n, i)[0]:
                calib.append(qh(n, i))
                got += 1
                if got == c['band_calibration_candidates']:
                    break
        log(f'level {n}: {got} calibration candidates')
    h_ref = float(np.median(calib))
    acc, band = None, None
    for b in c['entropy_bands']:
        a = {}
        for n in levels:
            a[n] = []
            for i in range(c['max_candidates_per_level']):
                if len(a[n]) == len(need[n]):
                    break
                if chk(n, i)[0] and abs(qh(n, i) - h_ref) <= b:
                    a[n].append(i)
        if all(len(a[n]) == len(need[n]) for n in levels):
            acc, band = a, b
            break
        log(f'band {b}: short {[(n, len(a[n]), len(need[n])) for n in levels]}')
    if acc is None:
        raise RuntimeError('no entropy band supplies every level within the candidate cap')
    tasks = []
    for n in levels:
        ctr = {}
        for role, i in zip(need[n], acc[n]):
            ctr[role] = ctr.get(role, 0) + 1
            h = cands[n][i]
            variants = [(role, h)] + ([('far', H.shift_tokens(h, c['region_B'] - c['region_A']))]
                                      if role == 'near' else [])
            for rl, hh in variants:
                tag = {'target': 't', 'near': 'near', 'far': 'far', 'easy': 'easy', 'hard': 'hard',
                       'pilot': 'p'}[rl]
                r = np.random.default_rng([c['task_sampling_seed'], 17, n, i, 2])
                hr, se = H.entropy_rate(hh, r, c['entropy_seqs'], c['entropy_seq_len'], c['entropy_burn_in'])
                pi = H.stationary(hh)
                tasks.append(dict(
                    task_id=f'hmm_n{n:03d}_{tag}{ctr[role]}', cid=len(tasks), role=rl, K=n,
                    K_bits=H.K_bits(n, W, q), definition=H.to_json(hh), process_rank=int(status[n][i][2]),
                    candidate_index=i, region=[int(hh['etok'].min()), int(hh['etok'].max())],
                    entropy_rate=hr, entropy_rate_se=se, quick_entropy_rate=qh(n, i),
                    min_stationary=float(pi.min()),
                    acceptance=dict(band_bits=band, h_ref=h_ref)))
    acc_log = dict(h_ref=h_ref, band_bits=band, n_calibration=len(calib), levels={
        str(n): dict(accepted_indices=acc[n], roles=need[n], checked=len(status[n]),
                     passed_checks_1_2=sum(v[0] for v in status[n].values()), quick_scored=len(quick[n]))
        for n in levels})
    return tasks, acc_log


def by_role(tasks, role):
    return [t for t in tasks if t['role'] == role]


def concept_map(tasks):
    return {t['cid']: H.from_json(t['definition']) for t in tasks}


def prior_for(alpha, D):
    return [alpha] + [(1 - alpha) / D] * D if alpha < 1 else [1.0]


# ---------------------------------------------------------------- the grid (PROTOCOL Section 3)

def grid_runs(c, tasks, seeds=None, targets=None):
    """(seed, pool, target_id, alpha) of every gated run."""
    seeds = c['seeds'] if seeds is None else seeds
    tg = by_role(tasks, 'target') if targets is None else targets
    runs = []
    for s in seeds:
        for t in tg:
            runs.append((s, 'none', t['task_id'], 1.0))
            for pool in ('near', 'far'):
                for a in c['alpha_D']:
                    runs.append((s, pool, t['task_id'], a))
            for pool in ('easy', 'hard'):
                for a in c['alpha_H']:
                    runs.append((s, pool, t['task_id'], a))
        for pool in ('near', 'far', 'easy', 'hard'):
            runs.append((s, pool, '-', 0.0))
    return runs


def arm_H_first(run):
    """Run order: arm H's runs (easy/hard pools, the mixed 'near' control at alpha in {0} + alpha_H, and the
    alpha = 1 curve) before arm D's remaining runs; alpha = 0 runs (scored on every target) first within each."""
    s, pool, tid, a = run
    h = pool in ('easy', 'hard', 'none') or (pool == 'near' and (a == 0 or a in (0.05,)))
    return (not h, tid != '-', s)


def ref_keys(c, tasks):
    """(target_id, pool, alpha) for which a Bayes-mixture reference is needed (alpha > 0 runs)."""
    keys = set()
    for t in by_role(tasks, 'target') + by_role(tasks, 'pilot'):
        keys.add((t['task_id'], 'none', 1.0))
        for pool in ('near', 'far'):
            for a in c['alpha_D']:
                keys.add((t['task_id'], pool, a))
        for pool in ('easy', 'hard'):
            for a in c['alpha_H']:
                keys.add((t['task_id'], pool, a))
    return sorted(keys)


def mix_key(tid, pool, alpha):
    return f'{tid}|{pool}|{alpha:.2f}'


# ---------------------------------------------------------------- references (PROTOCOL Section 4)

def eval_docs(c, h, cid, n_docs):
    return H.sample_docs(h, c['eval_seed'], cid, range(n_docs), c['doc_len'])


def build_references(c, tasks, out):
    """Held-out documents, oracle and Bayes-mixture code lengths (float64) for every target/pilot and (pool, alpha)."""
    cm = concept_map(tasks)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    oracle, mixture, post = {}, {}, {}
    for t in tasks:
        h = cm[t['cid']]
        scored = t['role'] in ('target', 'pilot')
        docs = eval_docs(c, h, t['cid'], c['eval_docs_per_target'] if scored else c['pool_eval_docs'])
        with open(out / 'data' / f"{t['task_id']}_test.npz", 'wb') as f:
            np.savez(f, docs=docs.astype(np.int16))
        oracle[t['task_id']] = H.forward_logprobs(h, docs)
    pool_codes = {}
    for tid, pool, a in ref_keys(c, tasks):
        t = next(x for x in tasks if x['task_id'] == tid)
        oc = oracle[tid]
        if a >= 1:
            mixture[mix_key(tid, pool, a)] = oc.copy()
            post[mix_key(tid, pool, a)] = 1.0
            continue
        docs = np.load(out / 'data' / f'{tid}_test.npz')['docs'].astype(np.int64)
        if (tid, pool) not in pool_codes:
            pool_codes[(tid, pool)] = [H.forward_logprobs(cm[d['cid']], docs) for d in by_role(tasks, pool)]
        codes = [oc] + pool_codes[(tid, pool)]
        pr = prior_for(a, len(codes) - 1)
        mixture[mix_key(tid, pool, a)] = H.mixture_from_codes(codes, pr)
        post[mix_key(tid, pool, a)] = float(H.posterior_after(codes, pr, 16).mean())
    with open(out / 'oracle.npz', 'wb') as f:
        np.savez(f, **oracle)
    with open(out / 'mixture.npz', 'wb') as f:
        np.savez(f, **mixture)
    atomic_write_text(out / 'posterior16.json', json.dumps(post, indent=1, sort_keys=True))


def slice_mask(c, name):
    lo, hi = c['slices'][name]
    m = np.zeros(c['doc_len'], bool)
    m[lo - 1:hi] = True
    return m


# ---------------------------------------------------------------- training (Experiment 13 PROTOCOL Section 2.6)

def make_optimizer(c, model):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=c['weight_decay']),
                              dict(params=no_decay, weight_decay=0.0)], lr=c['lr'], betas=tuple(c['betas']))


def stream_ids(tasks, seed, pool, target_id, alpha, n_docs):
    """Experiment 13's exact Bresenham alpha schedule; the pool is cycled in a seed-permuted order."""
    tmap = {t['task_id']: t for t in tasks}
    pool_cids = [d['cid'] for d in by_role(tasks, pool)] if pool != 'none' else []
    if alpha >= 1:
        cid = tmap[target_id]['cid']
        return np.full(n_docs, cid, np.int64), np.arange(n_docs, dtype=np.int64)
    tcid = tmap[target_id]['cid'] if target_id != '-' else -1
    return H.schedule(seed, tcid, pool_cids, alpha, n_docs)


def scored_targets(tasks, target_id):
    if target_id != '-':
        return [next(t for t in tasks if t['task_id'] == target_id)]
    return by_role(tasks, 'target')


def train_run(c, tasks, run, out, T_checkpoints, refs, scored=None):
    seed, pool, target_id, alpha = run
    torch.set_num_threads(c['threads'])
    t0 = time.time()
    cm = concept_map(tasks)
    name = run_name(seed, pool, target_id, alpha)
    L, Bd, V = c['doc_len'], c['batch_docs'], c['V']
    n_docs = T_checkpoints[-1] // L
    cids, js = stream_ids(tasks, seed, pool, target_id, alpha, n_docs)
    torch.manual_seed(seed)
    model = M.build_model(c['model'])
    opt = make_optimizer(c, model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c['warmup_steps']))
    targets = scored if scored is not None else scored_targets(tasks, target_id)
    stream = hashlib.sha256()
    rows, perpos, prow = [], [], []
    t_train = t_eval = 0.0
    model.train()
    gen = H.stream_batches(cm, seed, cids, js, L, Bd)
    for step in range(n_docs // Bd):
        batch = next(gen)
        stream.update(batch.astype(np.int16).tobytes())
        ts = time.time()
        x, y = M.inputs_targets(batch, V)
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, V + 1), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
        opt.step()
        sched.step()
        t_train += time.time() - ts
        T = (step + 1) * Bd * L
        if T not in T_checkpoints:
            continue
        ts = time.time()
        m_doc = (step + 1) * Bd
        for tg in targets:
            tid = tg['task_id']
            docs, oc = refs['docs'][tid], refs['oracle'][tid]
            mk = mix_key(tid, pool, alpha)
            mc = refs['mixture'][mk] if alpha > 0 else oc  # alpha = 0: no identification term (target unseen)
            m_codes = M.code_lengths(model, docs, V)
            for s in SLICES:
                sm = slice_mask(c, s)
                mb, ob, xb = m_codes[:, sm].mean(), oc[:, sm].mean(), mc[:, sm].mean()
                per_doc = (m_codes[:, sm] - oc[:, sm]).mean(1)
                rows.append(dict(seed=seed, pool=pool, run_target=target_id, target_id=tid, K=tg['K'],
                                 K_bits=tg['K_bits'], alpha=alpha, T=T, step=step + 1, slice=s,
                                 excess_bits=float(mb - ob), ident_bits=float(xb - ob), learn_bits=float(mb - xb),
                                 model_bits=float(mb), oracle_bits=float(ob), mixture_bits=float(xb),
                                 target_docs_seen=int((cids[:m_doc] == tg['cid']).sum()),
                                 excess_se=float(per_doc.std(ddof=1) / math.sqrt(len(per_doc)))))
            perpos.append(dict(target_id=tid, T=T, model=m_codes.mean(0).tolist()))
        if T == T_checkpoints[-1] and pool != 'none':
            with torch.no_grad():  # training-pool loss at the last checkpoint (not gated; float32)
                model.eval()
                for d in by_role(tasks, pool):
                    dd = refs['docs'][d['task_id']]
                    x, y = M.inputs_targets(dd, V)
                    lp = torch.nn.functional.log_softmax(model(x), -1)
                    nll = float((-lp.gather(-1, y[..., None])).mean()) / math.log(2.0)
                    prow.append(dict(pool_id=d['task_id'], model_bits=nll,
                                     oracle_bits=float(refs['oracle'][d['task_id']].mean())))
                model.train()
        t_eval += time.time() - ts
    rec = dict(run=name, seed=seed, pool=pool, target_id=target_id, alpha=alpha, rows=rows, per_position=perpos,
               pool_rows=prow, stream_sha256=stream.hexdigest(), n_docs=int(n_docs),
               docs_per_concept={str(int(k)): int(v) for k, v in zip(*np.unique(cids, return_counts=True))},
               train_seconds=round(t_train, 3), eval_seconds=round(t_eval, 3),
               tokens_per_second_train=T_checkpoints[-1] / t_train, wall_seconds=round(time.time() - t0, 3),
               finite=bool(all(math.isfinite(r['model_bits']) for r in rows)))
    return rec


def load_refs(out, tasks):
    o = np.load(out / 'oracle.npz')
    m = np.load(out / 'mixture.npz')
    docs = {t['task_id']: np.load(out / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
            for t in tasks}
    return dict(oracle={k: o[k] for k in o.files}, mixture={k: m[k] for k in m.files}, docs=docs)


_W = {}


def _init_worker(out):
    torch.set_num_threads(1)
    out = Path(out)
    _W['c'] = load_config()
    _W['tasks'] = json.loads((out / 'tasks.json').read_text())
    _W['refs'] = load_refs(out, _W['tasks'])
    _W['out'] = out


def _work(args):
    run, T_checkpoints, scored_role = args
    tasks = _W['tasks']
    scored = by_role(tasks, scored_role) if (scored_role and run[2] == '-') else None
    rec = train_run(_W['c'], tasks, tuple(run), _W['out'], T_checkpoints, _W['refs'], scored)
    (_W['out'] / 'runs').mkdir(parents=True, exist_ok=True)
    atomic_write_text(_W['out'] / 'runs' / f"{rec['run']}.json", json.dumps(rec))
    return rec['run'], rec['wall_seconds'], rec['tokens_per_second_train']


def run_many(c, out, runs, T_checkpoints, max_seconds, log, scored_role=None):
    """Runs with workers, skipping finished ones; stops scheduling past the budget. Finished runs are written
    atomically, so an interrupted chunk resumes without duplicate work. Order: arm_H_first."""
    state_path = out / 'run_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(chunks=[], elapsed_seconds=0.0)
    start = time.time()
    budget = min(max_seconds, c['max_wall_seconds'] - state['elapsed_seconds'])
    todo = [r for r in runs if not (out / 'runs' / f'{run_name(*r)}.json').exists()]
    todo.sort(key=arm_H_first)
    log(f'{len(runs) - len(todo)} runs done, {len(todo)} to run, budget {budget:.0f}s, workers {c["workers"]}')
    done_now = 0
    with cf.ProcessPoolExecutor(c['workers'], initializer=_init_worker, initargs=(str(out),)) as ex:
        pending, it = set(), iter(todo)
        while True:
            while len(pending) < c['workers'] and time.time() - start < budget:
                nxt = next(it, None)
                if nxt is None:
                    break
                pending.add(ex.submit(_work, (nxt, T_checkpoints, scored_role)))
            if not pending:
                break
            fin, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                name, wall, tps = f.result()
                done_now += 1
                log(f'done {name} {wall:.1f}s train {tps:.0f} tok/s ({done_now}/{len(todo)}, '
                    f'{time.time() - start:.0f}s)')
    el = time.time() - start
    state['elapsed_seconds'] += el
    state['chunks'].append(dict(start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                                seconds=round(el, 1), runs_completed=done_now))
    atomic_write_text(state_path, json.dumps(state, indent=1))
    missing = [run_name(*r) for r in runs if not (out / 'runs' / f'{run_name(*r)}.json').exists()]
    return missing, state


ROW_COLS = ['seed', 'pool', 'run_target', 'target_id', 'K', 'K_bits', 'alpha', 'T', 'step', 'slice', 'excess_bits',
            'ident_bits', 'learn_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'target_docs_seen', 'excess_se']


def aggregate(out, runs):
    recs = []
    for r in runs:
        p = out / 'runs' / f'{run_name(*r)}.json'
        if p.exists():
            recs.append(json.loads(p.read_text()))
    recs.sort(key=lambda r: (r['seed'], r['pool'], r['target_id'], r['alpha']))
    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ROW_COLS)
        w.writeheader()
        for r in recs:
            for row in r['rows']:
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    with open(out / 'pool_loss.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'pool_id', 'model_bits', 'oracle_bits'])
        for r in recs:
            for pr in r['pool_rows']:
                w.writerow([r['run'], pr['pool_id'], repr(pr['model_bits']), repr(pr['oracle_bits'])])
    mixes = {r['run']: dict(stream_sha256=r['stream_sha256'], n_docs=r['n_docs'],
                            docs_per_concept=r['docs_per_concept']) for r in recs}
    atomic_write_text(out / 'mixes.json', json.dumps(mixes, indent=0, sort_keys=True))
    return recs


def write_manifest(out, extra):
    import scipy
    cpu = ''
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        pass
    man = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    man.update(dict(sources={s: sha256_file(HERE / s) for s in SOURCES + ANALYSIS_SOURCES if (HERE / s).exists()},
                    versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                  scipy=scipy.__version__),
                    host=socket.gethostname(), cpu_model=cpu, cpu_count=os.cpu_count()))
    man.update(extra)
    for name in ('results_check.json', 'pilot_summary.json'):
        if (HERE / name).exists():
            man[name.replace('.json', '')] = json.loads((HERE / name).read_text())
    atomic_write_text(out / 'manifest.json', json.dumps(man, indent=1))


# ---------------------------------------------------------------- check-only

def prepare(c, out, log):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks, acc_log = build_concepts(c, log)
    log(f'concepts built in {time.time() - t0:.0f}s, h_ref {acc_log["h_ref"]:.3f}, band +-{acc_log["band_bits"]}')
    atomic_write_text(out / 'tasks.json', json.dumps(tasks, indent=1))
    atomic_write_text(out / 'acceptance_log.json', json.dumps(acc_log, indent=1))
    build_references(c, tasks, out)
    return tasks, acc_log


def copy_prep(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('tasks.json', 'acceptance_log.json', 'oracle.npz', 'mixture.npz', 'posterior16.json'):
        shutil.copy2(src / name, dst / name)
    shutil.copytree(src / 'data', dst / 'data', dirs_exist_ok=True)


def check_only(c, prep, log):
    tasks, acc_log = prepare(c, prep, log)
    rec = dict(acceptance=acc_log, concepts=[], checks={})
    for t in tasks:
        rec['concepts'].append(dict(task_id=t['task_id'], role=t['role'], K=t['K'], K_bits=t['K_bits'],
                                    process_rank=t['process_rank'], region=t['region'],
                                    entropy_rate=round(t['entropy_rate'], 4)))
        log(f"{t['task_id']:16s} {t['role']:7s} n={t['K']:3d} K_bits={t['K_bits']:5d} rank={t['process_rank']} "
            f"tokens {t['region']} h={t['entropy_rate']:.4f}+-{t['entropy_rate_se']:.4f}")
    m = M.build_model(c['model'])
    rec['checks']['params'] = [M.param_count(m), M.nonembedding_count(m)]
    # far = near shifted: same rank, same entropy rate to sampling error, disjoint supports
    near, far = by_role(tasks, 'near'), by_role(tasks, 'far')
    assert len(near) == len(far) == 8
    for a, b in zip(near, far):
        assert a['K'] == b['K'] and a['process_rank'] == b['process_rank']
        assert abs(a['entropy_rate'] - b['entropy_rate']) < 1e-9, (a['task_id'], b['task_id'])
    tg_tokens = set().union(*[set(np.array(t['definition']['etok']).ravel().tolist()) for t in tasks
                              if t['role'] != 'far'])
    far_tokens = set().union(*[set(np.array(t['definition']['etok']).ravel().tolist()) for t in far])
    assert not (tg_tokens & far_tokens) and max(tg_tokens) < 32 <= min(far_tokens)
    rec['checks']['far_is_shifted_near'] = True
    rec['checks']['supports'] = dict(region_A_tokens_used=len(tg_tokens), far_tokens_used=len(far_tokens))
    # alpha schedules exact
    Tmax = max(c['T_checkpoints_primary'][-1], c['T_checkpoints_fallback'][-1])
    nd = Tmax // c['doc_len']
    for s, pool, tid, a in grid_runs(c, tasks, seeds=c['seeds'][:1]):
        cids, js = stream_ids(tasks, s, pool, tid, a, nd)
        if tid != '-' and a < 1:
            tcid = next(t['cid'] for t in tasks if t['task_id'] == tid)
            num = int(round(a * 100))
            assert np.array_equal(np.cumsum(cids == tcid), (np.arange(1, nd + 1) * num) // 100), (tid, pool, a)
        allowed = {t['cid'] for t in by_role(tasks, pool)} if pool != 'none' else set()
        if tid != '-':
            allowed.add(next(t['cid'] for t in tasks if t['task_id'] == tid))
        assert set(np.unique(cids).tolist()) <= allowed, (s, pool, tid, a)
    rec['checks']['alpha_schedule_exact_and_pools_closed'] = True
    # eval documents never appear in training streams (first 2^14 documents of every concept and seed)
    eval_h = set()
    for t in tasks:
        for x in np.load(prep / 'data' / f"{t['task_id']}_test.npz")['docs']:
            eval_h.add(H.doc_hash(x))
    hits = 0
    for t in tasks:
        h = H.from_json(t['definition'])
        for s in list(c['seeds']) + [c['pilot_seed']]:
            d = H.sample_docs(h, s, t['cid'], range(2 ** 14 // 8), c['doc_len'])
            hits += sum(H.doc_hash(x) in eval_h for x in d)
    assert hits == 0
    rec['checks']['eval_train_disjoint'] = dict(hits=hits)
    rec['checks']['expected_runs'] = len(grid_runs(c, tasks))
    o = np.load(prep / 'oracle.npz')
    mx = np.load(prep / 'mixture.npz')
    rec['eps_id_all'] = {k: float((mx[k] - o[k.split('|')[0]]).mean()) for k in mx.files}
    rec['posterior16'] = json.loads((prep / 'posterior16.json').read_text())
    rec['source_sha256'] = {s: sha256_file(HERE / s) for s in SOURCES}
    atomic_write_text(HERE / 'results_check.json', json.dumps(rec, indent=1))
    log(json.dumps(rec['checks']))
    return rec


# ---------------------------------------------------------------- pilot (PROTOCOL Section 3.3)

def pilot_runs(c, tasks):
    pl = by_role(tasks, 'pilot')
    runs = []
    for t in pl:
        runs += [(c['pilot_seed'], 'none', t['task_id'], 1.0), (c['pilot_seed'], 'near', t['task_id'], 0.4),
                 (c['pilot_seed'], 'near', t['task_id'], 0.05), (c['pilot_seed'], 'hard', t['task_id'], 0.05)]
    runs.append((c['pilot_seed'], 'hard', '-', 0.0))
    return runs


def pilot(c, prep, out, log):
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty')
    copy_prep(prep, out)
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = pilot_runs(c, tasks)
    pc = dict(c, max_wall_seconds=10 ** 9)
    missing, state = run_many(pc, out, runs, c['T_checkpoints_primary'], float('inf'), log, scored_role='pilot')
    recs = aggregate(out, runs)
    single = [r for r in recs if r['target_id'] != '-']
    tps = [r['tokens_per_second_train'] for r in single]
    tok_s = float(np.median(tps))
    ev_one = float(np.median([r['eval_seconds'] / len(r['per_position']) for r in single]))
    data_over = float(np.median([(r['wall_seconds'] - r['train_seconds'] - r['eval_seconds']) / r['n_docs']
                                 for r in single]))
    grid = grid_runs(c, tasks)
    n_tg = len(by_role(tasks, 'target'))

    def proj(Ts):
        tot = 0.0
        for s, pool, tid, a in grid:
            n_eval = n_tg if tid == '-' else 1
            tot += Ts[-1] / tok_s + len(Ts) * n_eval * ev_one + Ts[-1] / c['doc_len'] * data_over
        return tot / c['workers'] / 3600
    hrs_p, hrs_f = proj(c['T_checkpoints_primary']), proj(c['T_checkpoints_fallback'])
    use_fallback = hrs_p > c['fallback_if_projected_hours_above']
    Ts = c['T_checkpoints_fallback'] if use_fallback else c['T_checkpoints_primary']
    T_rule = Ts[-2]
    hard = next(t for t in by_role(tasks, 'pilot') if t['K'] == c['hard_level'])
    rows = [row for r in recs if r['target_id'] == hard['task_id'] and r['pool'] == 'near'
            and abs(r['alpha'] - 0.4) < 1e-9 for row in r['rows'] if row['slice'] == 'all' and row['T'] == T_rule]
    learn = rows[0]['learn_bits']
    e0 = c['epsilon_star_start']
    fires = learn > 2 * e0
    eps_star = math.ceil(learn * 100 - 1e-9) / 100 if fires else e0
    summ = dict(seed=c['pilot_seed'], runs=len(recs), missing=missing, wall_seconds=state['elapsed_seconds'],
                train_tokens_per_second_per_worker=dict(median=tok_s, min=min(tps), max=max(tps)),
                eval_seconds_per_target_checkpoint=ev_one, workers=c['workers'],
                projected_hours_primary=hrs_p, projected_hours_fallback=hrs_f, use_fallback=use_fallback,
                T_checkpoints=Ts, eps_rule_T=T_rule, hard_pilot=hard['task_id'],
                hard_learn_bits_near_alpha0_4=learn, eps_star_rule_fires=fires, epsilon_star=eps_star,
                diverged=[r['run'] for r in recs if not r['finite']],
                table=[dict(run=r['run'], target=row['target_id'], K=row['K'], T=row['T'],
                            excess=round(row['excess_bits'], 4), ident=round(row['ident_bits'], 4))
                       for r in recs for row in r['rows'] if row['slice'] == 'all'])
    atomic_write_text(HERE / 'pilot_summary.json', json.dumps(summ, indent=1))
    log(json.dumps({k: v for k, v in summ.items() if k != 'table'}, indent=1))
    return summ


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--output', default=None)
    ap.add_argument('--prep', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--max-seconds', type=float, default=float('inf'))
    args = ap.parse_args()
    c = load_config()
    if args.workers:
        c['workers'] = args.workers
    prep = Path(args.prep) if args.prep else HERE / 'prep'
    if args.check_only:
        prep.mkdir(parents=True, exist_ok=True)
        check_only(c, prep, log_to(prep / 'check.log'))
        return
    if args.pilot:
        out = Path(args.output) if args.output else HERE / 'results_pilot'
        out.mkdir(parents=True, exist_ok=True)
        pilot(c, prep, out, log_to(HERE / 'pilot.log'))
        return
    if not c.get('T_checkpoints') or not c.get('epsilon_star'):
        raise SystemExit('T_checkpoints / epsilon_star unset: run the pilot and freeze first')
    out = Path(args.output) if args.output else HERE / 'results'
    log = log_to(HERE / 'grid.log')
    if not (out / 'tasks.json').exists():
        copy_prep(prep, out)
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = grid_runs(c, tasks)
    missing, state = run_many(c, out, runs, c['T_checkpoints'], args.max_seconds, log)
    recs = aggregate(out, runs)
    write_manifest(out, dict(grid=dict(runs=len(runs), finished=len(recs), missing=missing,
                                       diverged=[r['run'] for r in recs if not r['finite']],
                                       wall_seconds=state['elapsed_seconds'], chunks=state['chunks'])))
    log(f'{len(recs)}/{len(runs)} runs finished; {len(missing)} missing; cumulative {state["elapsed_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
