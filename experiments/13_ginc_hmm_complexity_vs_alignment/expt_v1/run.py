"""Experiment 13 runner (arm 13-small on CPU): concepts, exact references, invariant checks, pilot and gated grid.

Usage (venv python, from expt_v1/):
  run.py --arm small --check-only                 builds concepts and references into prep/small, checks invariants,
                                                  writes results_check_small.json; trains nothing
  run.py --arm small --pilot                      disclosed pilot (seed 13909, 4 pilot concepts, alpha in
                                                  {0.05, 0.4}, T to 2^21) into results_pilot/
  run.py --arm small --output results/small [--max-seconds S]
                                                  gated grid (128 runs, resume-safe: finished runs are atomic and
                                                  skipped); the cumulative wall cap is tracked in run_state.json
  run.py --arm gpu ...                            not run on the CPU box (13-GPU is launched from launch/)
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
ANALYSIS_SOURCES = ['laws.py', 'independent_fit.py', 'analyze.py', 'verify.py', 'report.py']
SLICES = ['early', 'middle', 'late', 'all']


def load_config(arm='small'):
    raw = json.loads((HERE / 'config.json').read_text())
    c = dict(raw['shared'])
    c.update(raw[arm])
    c['arm'] = arm
    return c


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path, text):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def run_name(seed, target_id, alpha):
    return f'{seed}_{target_id}_{alpha:.2f}'


def log_to(path):
    def log(msg):
        line = f'{datetime.now().strftime("%m-%d-%Y %H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(path, 'a') as f:
            f.write(line + '\n')
    return log


# ---------------------------------------------------------------- concepts (PROTOCOL Section 2.1)

def _check12(h):
    if not H.strongly_connected_aperiodic(h):
        return False, 'not strongly connected and aperiodic', None
    r, how = H.process_rank(h)
    if r != h['n']:
        return False, f'process rank {r} < n ({how})', r
    return True, 'ok', r


def build_concepts(c, log=print):
    """Acceptance procedure: per level, a fixed list of max_candidates_per_level candidates drawn level by level from
    the arm's single PCG64 stream; band calibration on each level's first 64 candidates passing checks 1-2; then
    acceptance in generation order. Returns (tasks, acceptance log)."""
    rng = np.random.default_rng([c['task_sampling_seed'], c['arm_index']])
    V, q = c['V'], c['q']
    per_level = c['targets_per_level'] + c['distractors_per_level'] + c['pilot_concepts_per_level']
    cands = {n: [H.sample_candidate(rng, n, V, q) for _ in range(c['max_candidates_per_level'])] for n in c['levels']}
    status = {n: {} for n in c['levels']}  # index -> (passes 1-2, reason, rank)
    quick = {n: {} for n in c['levels']}

    def chk(n, i):
        if i not in status[n]:
            status[n][i] = _check12(cands[n][i])
        return status[n][i]

    def qh(n, i):
        if i not in quick[n]:
            r = np.random.default_rng([c['task_sampling_seed'], c['arm_index'], n, i, 1])
            quick[n][i] = H.entropy_rate(cands[n][i], r, c['quick_entropy_seqs'], c['entropy_seq_len'],
                                         c['entropy_burn_in'])[0]
        return quick[n][i]
    calib = []
    for n in c['levels']:
        got = 0
        for i in range(c['max_candidates_per_level']):
            if chk(n, i)[0]:
                calib.append(qh(n, i))
                got += 1
                if got == c['band_calibration_candidates']:
                    break
        log(f'level {n}: {got} calibration candidates')
    h_ref = float(np.median(calib))

    def accept(band):
        acc = {}
        for n in c['levels']:
            acc[n] = []
            for i in range(c['max_candidates_per_level']):
                if chk(n, i)[0] and abs(qh(n, i) - h_ref) <= band:
                    acc[n].append(i)
                    if len(acc[n]) == per_level:
                        break
        return acc
    band = c['entropy_band_bits']
    acc = accept(band)
    if any(len(v) < per_level for v in acc.values()):
        band = c['entropy_band_fallback']
        log(f'band widened to +-{band}')
        acc = accept(band)
    short = {n: len(v) for n, v in acc.items() if len(v) < per_level}
    if short:
        raise RuntimeError(f'levels cannot supply {per_level} concepts within the candidate cap: {short}')
    tasks = []
    roles = ['target'] * c['targets_per_level'] + ['distractor'] * c['distractors_per_level'] + \
        ['pilot'] * c['pilot_concepts_per_level']
    for n in c['levels']:
        tried = acc[n][-1] + 1
        ctr = {}
        for role, i in zip(roles, acc[n]):
            ctr[role] = ctr.get(role, 0) + 1
            h = cands[n][i]
            tag = {'target': 't', 'distractor': 'd', 'pilot': 'p'}[role]
            r = np.random.default_rng([c['task_sampling_seed'], c['arm_index'], n, i, 2])
            hr, se = H.entropy_rate(h, r, c['entropy_seqs'], c['entropy_seq_len'], c['entropy_burn_in'])
            pi = H.stationary(h)
            tasks.append(dict(
                task_id=f'hmm_n{n:03d}_{tag}{ctr[role]}', cid=len(tasks), family='hmm', role=role, K=n,
                K_bits=H.K_bits(n, V, q), K_bits_canonical=H.K_bits(n, V, q) - math.log2(math.factorial(n)),
                definition=H.to_json(h), process_rank=int(status[n][i][2]), candidate_index=i,
                entropy_rate=hr, entropy_rate_se=se, quick_entropy_rate=qh(n, i), stationary=pi.tolist(),
                visit_tail_exponent=H.visit_tail_exponent(pi), min_stationary=float(pi.min()),
                expected_visits_rarest_per_doc=H.expected_visits_rarest(h, pi, c['doc_len']),
                acceptance=dict(candidates_tried_at_level=tried, band_bits=band, h_ref=h_ref)))
    dist = [t for t in tasks if t['role'] == 'distractor']
    for t in tasks:
        ht = H.from_json(t['definition'])
        t['emission_overlap'] = {d['task_id']: H.emission_overlap(ht, H.from_json(d['definition'])) for d in dist}
    acc_log = dict(h_ref=h_ref, band_bits=band, n_calibration=len(calib), levels={
        str(n): dict(candidates_tried=acc[n][-1] + 1, accepted_indices=acc[n],
                     checked=len(status[n]), passed_checks_1_2=sum(v[0] for v in status[n].values()),
                     reasons={r: sum(1 for v in status[n].values() if v[1].startswith(r))
                              for r in ('ok', 'not strongly', 'process rank')},
                     quick_scored=len(quick[n]),
                     in_band=sum(1 for i, v in quick[n].items() if abs(v - h_ref) <= band))
        for n in c['levels']})
    return tasks, acc_log


def concept_map(tasks):
    return {t['cid']: H.from_json(t['definition']) for t in tasks}


def distractors(tasks):
    return [t for t in tasks if t['role'] == 'distractor']


def prior_for(alpha, D):
    return [alpha] + [(1 - alpha) / D] * D if alpha < 1 else [1.0]


# ---------------------------------------------------------------- references (PROTOCOL Sections 1, 2.5, 3)

def eval_docs(c, h, cid, n_docs):
    return H.sample_docs(h, c['eval_seed'], cid, range(n_docs), c['doc_len'])


def build_references(c, tasks, out, alphas_by_role):
    """Held-out documents, oracle and mixture code lengths (float64) for every target/pilot concept and alpha."""
    cm = concept_map(tasks)
    dist = distractors(tasks)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    oracle, mixture, post = {}, {}, {}
    dist_codes_cache = {}
    for t in tasks:
        h = cm[t['cid']]
        n_docs = c['eval_docs_per_target'] if t['role'] != 'distractor' else c['distractor_eval_docs']
        docs = eval_docs(c, h, t['cid'], n_docs)
        with open(out / 'data' / f"{t['task_id']}_test.npz", 'wb') as f:
            np.savez(f, docs=docs.astype(np.int16))
        oc = H.forward_logprobs(h, docs)
        oracle[t['task_id']] = oc
        if t['role'] == 'distractor':
            continue
        dist_codes_cache[t['task_id']] = [H.forward_logprobs(cm[d['cid']], docs) for d in dist]
        for a in alphas_by_role[t['role']]:
            if a >= 1:
                mixture[f"{t['task_id']}|{a:.2f}"] = oc.copy()
                post[f"{t['task_id']}|{a:.2f}"] = 1.0
                continue
            pr = prior_for(a, len(dist))
            codes = [oc] + dist_codes_cache[t['task_id']]
            mixture[f"{t['task_id']}|{a:.2f}"] = H.mixture_from_codes(codes, pr)
            post[f"{t['task_id']}|{a:.2f}"] = float(H.posterior_after(codes, pr, 16).mean())
    with open(out / 'oracle.npz', 'wb') as f:
        np.savez(f, **oracle)
    with open(out / 'mixture.npz', 'wb') as f:
        np.savez(f, **mixture)
    atomic_write_text(out / 'posterior16.json', json.dumps(post, indent=1, sort_keys=True))
    return oracle, mixture


def slice_mask(c, name):
    lo, hi = c['slices'][name]
    m = np.zeros(c['doc_len'], bool)
    m[lo - 1:hi] = True
    return m


# ---------------------------------------------------------------- training (PROTOCOL Sections 2.4, 2.6)

def make_optimizer(c, model):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=c['weight_decay']),
                              dict(params=no_decay, weight_decay=0.0)], lr=c['lr'], betas=tuple(c['betas']))


def train_run(c, tasks, target_id, alpha, seed, out, T_checkpoints, refs):
    torch.set_num_threads(c['threads'])
    t0 = time.time()
    cm = concept_map(tasks)
    target = next(t for t in tasks if t['task_id'] == target_id)
    dist = distractors(tasks)
    name = run_name(seed, target_id, alpha)
    L, Bd, V = c['doc_len'], c['batch_docs'], c['V']
    n_docs = T_checkpoints[-1] // L
    cids, js = H.schedule(seed, target['cid'], [d['cid'] for d in dist], alpha, n_docs)
    torch.manual_seed(seed)
    model = M.build_model(c['model'])
    opt = make_optimizer(c, model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c['warmup_steps']))
    ck_dir = out / 'checkpoints' / name
    tmp_dir = Path(str(ck_dir) + '.tmp')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    oc = refs['oracle'][target_id]
    mc = refs['mixture'][f'{target_id}|{alpha:.2f}']
    docs = refs['docs'][target_id]
    stream = hashlib.sha256()
    rows, perpos, drows, counts = [], [], [], {}
    t_train = t_eval = t_data = 0.0
    model.train()
    gen = H.stream_batches(cm, seed, cids, js, L, Bd)
    for step in range(n_docs // Bd):
        ts = time.time()
        batch = next(gen)
        stream.update(batch.astype(np.int16).tobytes())
        t_data += time.time() - ts
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
        if T in T_checkpoints:
            ts = time.time()
            m_codes = M.code_lengths(model, docs, V)
            M.save_npz(model, tmp_dir / f'T_{T}.npz')
            m_doc = (step + 1) * Bd
            cnt = {}
            for cid in np.unique(cids[:m_doc]):
                cnt[str(int(cid))] = int((cids[:m_doc] == cid).sum())
            counts[str(T)] = cnt
            for s in SLICES:
                sm = slice_mask(c, s)
                mb, ob, xb = m_codes[:, sm].mean(), oc[:, sm].mean(), mc[:, sm].mean()
                per_doc = (m_codes[:, sm] - oc[:, sm]).mean(1)
                rows.append(dict(seed=seed, target_id=target_id, K=target['K'], K_bits=target['K_bits'],
                                 alpha=alpha, N=M.nonembedding_count(model), T=T, step=step + 1, slice=s,
                                 excess_bits=float(mb - ob), ident_bits=float(xb - ob), learn_bits=float(mb - xb),
                                 model_bits=float(mb), oracle_bits=float(ob), mixture_bits=float(xb),
                                 n_tokens=int(sm.sum() * len(docs)),
                                 excess_se=float(per_doc.std(ddof=1) / math.sqrt(len(per_doc)))))
            perpos.append(dict(T=T, model=m_codes.mean(0).tolist()))
            with torch.no_grad():  # training-distribution loss on distractors (not gated; float32)
                model.eval()
                for d in dist:
                    dd = refs['docs'][d['task_id']]
                    x, y = M.inputs_targets(dd, V)
                    lp = torch.nn.functional.log_softmax(model(x), -1)
                    nll = float((-lp.gather(-1, y[..., None])).mean()) / math.log(2.0)
                    drows.append(dict(seed=seed, target_id=target_id, alpha=alpha, T=T, distractor_id=d['task_id'],
                                      model_bits=nll, oracle_bits=float(refs['oracle'][d['task_id']].mean())))
                model.train()
            t_eval += time.time() - ts
    shutil.rmtree(ck_dir, ignore_errors=True)
    os.replace(tmp_dir, ck_dir)
    rec = dict(run=name, seed=seed, target_id=target_id, alpha=alpha, rows=rows, per_position=perpos,
               distractor_rows=drows, counts=counts, stream_sha256=stream.hexdigest(),
               data_order_seed=[int(seed), 7777], n_docs=int(n_docs),
               train_seconds=round(t_train, 3), eval_seconds=round(t_eval, 3), data_seconds=round(t_data, 3),
               tokens_per_second_train=T_checkpoints[-1] / t_train, wall_seconds=round(time.time() - t0, 3),
               finite=bool(all(math.isfinite(r['model_bits']) for r in rows)))
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / 'runs' / f'{name}.json', json.dumps(rec))
    return rec


def load_refs(out, tasks):
    o = np.load(out / 'oracle.npz')
    m = np.load(out / 'mixture.npz')
    docs = {t['task_id']: np.load(out / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
            for t in tasks}
    return dict(oracle={k: o[k] for k in o.files}, mixture={k: m[k] for k in m.files}, docs=docs)


_W = {}


def _init_worker(out, arm):
    torch.set_num_threads(1)
    out = Path(out)
    _W['c'] = load_config(arm)
    _W['tasks'] = json.loads((out / 'tasks.json').read_text())
    _W['refs'] = load_refs(out, _W['tasks'])
    _W['out'] = out


def _work(args):
    target_id, alpha, seed, T_checkpoints = args
    rec = train_run(_W['c'], _W['tasks'], target_id, alpha, seed, _W['out'], T_checkpoints, _W['refs'])
    return rec['run'], rec['wall_seconds'], rec['tokens_per_second_train']


def run_many(c, out, runs, max_seconds, log):
    """Runs (target_id, alpha, seed, T_checkpoints) with workers, skipping finished ones; stops scheduling past the
    budget. Finished runs are written atomically, so an interrupted chunk resumes without duplicate work."""
    state_path = out / 'run_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(chunks=[], elapsed_seconds=0.0)
    start = time.time()
    budget = min(max_seconds, c['max_wall_seconds'] - state['elapsed_seconds'])
    todo = [r for r in runs if not (out / 'runs' / f'{run_name(r[2], r[0], r[1])}.json').exists()]
    log(f'{len(runs) - len(todo)} runs done, {len(todo)} to run, budget {budget:.0f}s')
    done_now = 0
    with cf.ProcessPoolExecutor(c['workers'], initializer=_init_worker, initargs=(str(out), c['arm'])) as ex:
        pending, it = set(), iter(todo)
        while True:
            while len(pending) < c['workers'] and time.time() - start < budget:
                nxt = next(it, None)
                if nxt is None:
                    break
                pending.add(ex.submit(_work, nxt))
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
    missing = [run_name(s, t, a) for t, a, s, _ in runs if not (out / 'runs' / f'{run_name(s, t, a)}.json').exists()]
    return missing, state


def aggregate(c, out, runs):
    recs = []
    for t, a, s, _ in runs:
        p = out / 'runs' / f'{run_name(s, t, a)}.json'
        if p.exists():
            recs.append(json.loads(p.read_text()))
    recs.sort(key=lambda r: (r['seed'], r['target_id'], r['alpha']))
    cols = ['seed', 'target_id', 'K', 'K_bits', 'alpha', 'N', 'T', 'step', 'slice', 'excess_bits', 'ident_bits',
            'learn_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'n_tokens', 'excess_se']
    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in recs:
            for row in r['rows']:
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    with open(out / 'per_position.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'target_id', 'alpha', 'T'] + [f'p{i + 1}' for i in range(c['doc_len'])])
        for r in recs:
            for pp in r['per_position']:
                w.writerow([r['seed'], r['target_id'], r['alpha'], pp['T']] + [repr(x) for x in pp['model']])
    with open(out / 'distractor_loss.csv', 'w', newline='') as f:
        cols_d = ['seed', 'target_id', 'alpha', 'T', 'distractor_id', 'model_bits', 'oracle_bits']
        w = csv.DictWriter(f, cols_d)
        w.writeheader()
        for r in recs:
            for row in r['distractor_rows']:
                w.writerow(row)
    mixes = {r['run']: dict(target_id=r['target_id'], alpha=r['alpha'], seed=r['seed'], counts=r['counts'],
                            data_order_seed=r['data_order_seed'], stream_sha256=r['stream_sha256'],
                            n_docs=r['n_docs']) for r in recs}
    atomic_write_text(out / 'mixes.json', json.dumps(mixes, indent=0, sort_keys=True))
    return recs


def write_manifest(c, out, extra):
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
    for name in (f'results_check_{c["arm"]}.json', 'pilot_summary.json'):
        if (HERE / name).exists():
            man[name.replace('.json', '')] = json.loads((HERE / name).read_text())
    atomic_write_text(out / 'manifest.json', json.dumps(man, indent=1))


# ---------------------------------------------------------------- check-only

def _first_doc_hashes(args):
    defn, seed, cid, n_docs, L = args
    h = H.from_json(defn)
    hs = set()
    for s in range(0, n_docs, 4096):
        d = H.sample_docs(h, seed, cid, range(s, min(n_docs, s + 4096)), L)
        hs.update(H.doc_hash(x) for x in d)
    return hs


def prepare(c, out, log):
    """Concepts + references into out (prep directory). Deterministic: rebuilt identically on every call."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks, acc_log = build_concepts(c, log)
    log(f'concepts built in {time.time() - t0:.0f}s, h_ref {acc_log["h_ref"]:.3f}, band +-{acc_log["band_bits"]}')
    atomic_write_text(out / 'tasks.json', json.dumps(tasks, indent=1))
    atomic_write_text(out / 'acceptance_log.json', json.dumps(acc_log, indent=1))
    alphas = {'target': c['alpha_grid'], 'pilot': c['alpha_grid']}
    build_references(c, tasks, out, alphas)
    return tasks, acc_log


def copy_prep(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for name in ('tasks.json', 'acceptance_log.json', 'oracle.npz', 'mixture.npz', 'posterior16.json'):
        shutil.copy2(src / name, dst / name)
    shutil.copytree(src / 'data', dst / 'data', dirs_exist_ok=True)


def grid_runs(c, tasks):
    tg = [t for t in tasks if t['role'] == 'target']
    return [(t['task_id'], a, s, c['T_checkpoints']) for s in c['seeds'] for t in tg for a in c['alpha_grid']]


def check_only(c, prep, log):
    tasks, acc_log = prepare(c, prep, log)
    rec = dict(arm=c['arm'], acceptance=acc_log, concepts=[], checks={})
    for t in tasks:
        rec['concepts'].append(dict(task_id=t['task_id'], role=t['role'], K=t['K'], K_bits=t['K_bits'],
                                    process_rank=t['process_rank'], entropy_rate=round(t['entropy_rate'], 4),
                                    entropy_rate_se=round(t['entropy_rate_se'], 4),
                                    candidates_tried=t['acceptance']['candidates_tried_at_level']))
        log(f"{t['task_id']:14s} {t['role']:10s} n={t['K']:3d} K_bits={t['K_bits']:5d} rank={t['process_rank']} "
            f"h={t['entropy_rate']:.4f}+-{t['entropy_rate_se']:.4f}")
    m = M.build_model(c['model'])
    rec['checks']['params'] = [M.param_count(m), M.nonembedding_count(m)]
    # alpha schedules: exact floor(m * alpha) target documents after every prefix, at every checkpoint of both ranges
    Tmax = max(c['T_checkpoints_primary'][-1], c['T_checkpoints_fallback'][-1])
    nd = Tmax // c['doc_len']
    dist_ids = [d['cid'] for d in distractors(tasks)]
    worst = 0
    for t in tasks:
        if t['role'] == 'distractor':
            continue
        for a in c['alpha_grid']:
            cids, _ = H.schedule(c['seeds'][0], t['cid'], dist_ids, a, nd)
            cum = np.cumsum(cids == t['cid'])
            mm = np.arange(1, nd + 1)
            num = int(round(a * 100))
            assert np.array_equal(cum, (mm * num) // 100), (t['task_id'], a)
            if a < 1:
                dc = [int((cids == d).sum()) for d in dist_ids]
                worst = max(worst, max(dc) - min(dc))
    rec['checks']['alpha_schedule_exact'] = True
    rec['checks']['max_distractor_count_spread'] = worst
    # disjointness: eval documents vs the first 2^16 training documents of every concept and every seed
    eval_h = {}
    for t in tasks:
        d = np.load(prep / 'data' / f"{t['task_id']}_test.npz")['docs']
        for x in d:
            eval_h[H.doc_hash(x)] = t['task_id']
    seeds = list(c['seeds']) + [c['pilot_seed']]
    jobs = [(t['definition'], s, t['cid'], 2 ** 16, c['doc_len']) for t in tasks for s in seeds]
    ts = time.time()
    hits = 0
    with cf.ProcessPoolExecutor(c['workers']) as ex:
        for hs in ex.map(_first_doc_hashes, jobs):
            hits += len(hs & eval_h.keys())
    assert hits == 0, f'{hits} eval documents equal training documents'
    rec['checks']['eval_train_disjoint'] = dict(eval_docs=len(eval_h), training_docs_checked=len(jobs) * 2 ** 16,
                                                hits=hits, seconds=round(time.time() - ts, 1))
    rec['checks']['expected_runs'] = len(grid_runs(dict(c, T_checkpoints=c['T_checkpoints_primary']), tasks))
    o = np.load(prep / 'oracle.npz')
    mx = np.load(prep / 'mixture.npz')
    rec['eps_id_all'] = {k: float((mx[k] - o[k.split('|')[0]]).mean()) for k in mx.files}
    rec['source_sha256'] = {s: sha256_file(HERE / s) for s in SOURCES}
    atomic_write_text(HERE / f'results_check_{c["arm"]}.json', json.dumps(rec, indent=1))
    log(json.dumps(rec['checks']))
    return rec


# ---------------------------------------------------------------- pilot (PROTOCOL Section 2.8)

def pilot(c, prep, out, log):
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty')
    copy_prep(prep, out)
    tasks = json.loads((out / 'tasks.json').read_text())
    pl = [t for t in tasks if t['role'] == 'pilot']
    runs = [(t['task_id'], a, c['pilot_seed'], c['T_checkpoints_primary']) for t in pl for a in c['pilot_alphas']]
    pc = dict(c, max_wall_seconds=10 ** 9)
    missing, state = run_many(pc, out, runs, float('inf'), log)
    recs = aggregate(c, out, runs)
    tps = [r['tokens_per_second_train'] for r in recs]
    ev = [r['eval_seconds'] / len(r['per_position']) for r in recs]
    data_s = [r['data_seconds'] / r['n_docs'] for r in recs]
    tok_s = float(np.median(tps))
    ev_ck = float(np.median(ev))
    n_runs = len(grid_runs(dict(c, T_checkpoints=c['T_checkpoints_primary']), tasks))

    def proj(Ts):
        per = Ts[-1] / tok_s + len(Ts) * ev_ck + Ts[-1] / c['doc_len'] * float(np.median(data_s))
        return per, per * math.ceil(n_runs / c['workers']) / 3600
    per_p, hrs_p = proj(c['T_checkpoints_primary'])
    per_f, hrs_f = proj(c['T_checkpoints_fallback'])
    use_fallback = hrs_p > c['fallback_if_projected_hours_above']
    Ts = c['T_checkpoints_fallback'] if use_fallback else c['T_checkpoints_primary']
    T_fit_max = Ts[-2]
    n_max = max(c['levels'])
    hard = next(t for t in pl if t['K'] == n_max)
    rows = [row for r in recs if r['target_id'] == hard['task_id'] and abs(r['alpha'] - 0.4) < 1e-9
            for row in r['rows'] if row['slice'] == 'all' and row['T'] == T_fit_max]
    learn = rows[0]['learn_bits']
    eps_star = c['epsilon_star']
    rule_fires = learn > 2 * eps_star
    eps_star_new = math.ceil(learn * 100 - 1e-9) / 100 if rule_fires else eps_star
    se = [row['excess_se'] for r in recs for row in r['rows'] if row['slice'] == 'all']
    summ = dict(seed=c['pilot_seed'], runs=len(recs), missing=missing, wall_seconds=state['elapsed_seconds'],
                train_tokens_per_second_per_worker=dict(median=tok_s, min=min(tps), max=max(tps)),
                eval_seconds_per_checkpoint=ev_ck,
                projected_hours_primary=hrs_p, projected_hours_fallback=hrs_f,
                projected_seconds_per_run_primary=per_p, projected_seconds_per_run_fallback=per_f,
                use_fallback=use_fallback, T_checkpoints=Ts, largest_fit_T=T_fit_max,
                hard_pilot=hard['task_id'], hard_learn_bits_at_largest_fit_T_alpha0_4=learn,
                eps_star_rule_fires=rule_fires, epsilon_star=eps_star_new,
                eval_se_excess_all=dict(median=float(np.median(se)), max=float(max(se))),
                diverged=[r['run'] for r in recs if not r['finite']],
                table=[dict(run=r['run'], K=row['K'], alpha=r['alpha'], T=row['T'], excess=row['excess_bits'],
                            ident=row['ident_bits'], learn=row['learn_bits'])
                       for r in recs for row in r['rows'] if row['slice'] == 'all'])
    atomic_write_text(HERE / 'pilot_summary.json', json.dumps(summ, indent=1))
    log(json.dumps({k: v for k, v in summ.items() if k != 'table'}, indent=1))
    return summ


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['small', 'gpu'], default='small')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--output', default=None)
    ap.add_argument('--prep', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--max-seconds', type=float, default=float('inf'))
    args = ap.parse_args()
    if args.arm == 'gpu':
        raise SystemExit('13-GPU is launched on GPU nodes from launch/ and is not run on the CPU box '
                         '(13-GPU: pending, not run).')
    c = load_config(args.arm)
    if args.workers:
        c['workers'] = args.workers
    prep = Path(args.prep) if args.prep else HERE / 'prep' / args.arm
    if args.check_only:
        prep.mkdir(parents=True, exist_ok=True)
        check_only(c, prep, log_to(prep / 'check.log'))
        return
    if args.pilot:
        out = Path(args.output) if args.output else HERE / 'results_pilot'
        out.mkdir(parents=True, exist_ok=True)
        pilot(c, prep, out, log_to(HERE / 'pilot.log'))
        return
    if not c.get('T_checkpoints'):
        raise SystemExit('T_checkpoints is unset: run the pilot and freeze first')
    out = Path(args.output) if args.output else HERE / 'results' / args.arm
    log = log_to(HERE / f'grid_{args.arm}.log')
    if not (out / 'tasks.json').exists():
        copy_prep(prep, out)
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = grid_runs(c, tasks)
    missing, state = run_many(c, out, runs, args.max_seconds, log)
    recs = aggregate(c, out, runs)
    write_manifest(c, out, dict(grid=dict(runs=len(runs), finished=len(recs), missing=missing,
                                          diverged=[r['run'] for r in recs if not r['finite']],
                                          wall_seconds=state['elapsed_seconds'], chunks=state['chunks'])))
    log(f'{len(recs)}/{len(runs)} runs finished; {len(missing)} missing; cumulative {state["elapsed_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
