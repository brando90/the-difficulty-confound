"""Experiment 20 runner: build tasks and data, check invariants, run the gated 512-cell LSTM grid.

Same tasks, data, mixes, seeds and checkpoints as Experiment 11; the learner is c['arch'] ('lstm'). Experiment 11's
pilot, learning-rate calibration and exploratory LSTM arm are not rerun (PROTOCOL Section 2.6).

Usage (venv python, from expt_v1/):
  run.py --check-only                      invariant checks + timing probe, prints a JSON receipt
  run.py --output results [--resume] [--max-seconds S]
                                           gated grid; --max-seconds bounds one invocation (chunk); the
                                           cumulative wall cap max_wall_seconds is tracked in run_state.json
  run.py --only SEED:TARGET:ALPHA --output SCRATCH   regenerate one cell (never into results/)
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import itertools
import json
import os
import platform
import shutil
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

import data as D  # noqa: E402
import dfa as A  # noqa: E402
import model as M  # noqa: E402

SOURCES = ['config.json', 'dfa.py', 'data.py', 'model.py', 'run.py', 'laws.py', 'independent_fit.py',
           'analyze.py', 'verify.py', 'report.py']


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cell_name(seed, target_id, alpha):
    return f'{seed}_{target_id}_{alpha:.2f}'


def atomic_write_text(path, text):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- tasks and data

def build_all(c):
    """Tasks, shared test set, pools and labels, all from the task seed."""
    tasks, sel_log = A.select_tasks(c)
    test_bits, test_lens = D.build_test_set(c)
    pools = {}
    for t in tasks:
        b, l = D.build_pool(c, t)
        pools[t['task_id']] = dict(bits=b, length=l, label=D.labels_for(t, b, l),
                                   test_label=D.labels_for(t, test_bits, test_lens))
    return tasks, sel_log, test_bits, test_lens, pools


def write_data(out, tasks, test_bits, test_lens, pools):
    (out / 'data').mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / 'tasks.json', json.dumps(tasks, indent=1))
    for t in tasks:
        p = pools[t['task_id']]
        with open(out / 'data' / f"{t['task_id']}_train.npz", 'wb') as f:
            np.savez(f, bits=p['bits'], length=p['length'], label=p['label'])
        with open(out / 'data' / f"{t['task_id']}_test.npz", 'wb') as f:
            np.savez(f, bits=test_bits, length=test_lens, label=p['test_label'])


def load_data(out):
    tasks = json.loads((out / 'tasks.json').read_text())
    order = sorted(tasks, key=lambda t: t['task_token'])
    tr = [np.load(out / 'data' / f"{t['task_id']}_train.npz") for t in order]
    te = [np.load(out / 'data' / f"{t['task_id']}_test.npz") for t in order]
    return dict(tasks=tasks,
                pool_bits=np.stack([z['bits'] for z in tr]), pool_lens=np.stack([z['length'] for z in tr]),
                pool_labels=np.stack([z['label'] for z in tr]),
                test_bits=te[0]['bits'], test_lens=te[0]['length'],
                test_labels=np.stack([z['label'] for z in te]))


# ---------------------------------------------------------------- training

def train_cell(c, tasks, target_id, alpha, seed, out_dir, n_task_tokens=32, arch=None, data=None,
               T_checkpoints=None, save_checkpoints='all', lr=None):
    """Train one (target, alpha, seed) cell; write checkpoints, predictions and the cell record atomically."""
    if lr is not None:  # pre-freeze learning-rate calibration only
        c = dict(c, lr=lr)
    torch.set_num_threads(c['threads'])
    out_dir = Path(out_dir)
    T_checkpoints = list(T_checkpoints or c['T_checkpoints'])
    data = data if data is not None else load_data(out_dir)
    target = next(t for t in tasks if t['task_id'] == target_id)
    name = cell_name(seed, target_id, alpha)
    t0 = time.time()
    mix = D.assemble_mix(c, tasks, target_id, alpha, seed, T_checkpoints)
    torch.manual_seed(seed)
    arch = arch or c['arch']
    model = M.build(c, n_task_tokens) if arch == c['arch'] else (
        M.build_model(c, n_task_tokens) if arch == 'transformer' else M.build_lstm(c, n_task_tokens))
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], betas=tuple(c['betas']), weight_decay=c['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c['warmup_steps']))
    B = c['batch_size']
    ck_dir = out_dir / 'checkpoints' / name
    tmp_dir = Path(str(ck_dir) + '.tmp')
    if save_checkpoints != 'none':
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
    rows, preds = [], []
    test_labels = data['test_labels'][target['task_token']]
    model.train()
    for step in range(T_checkpoints[-1] // B):
        sl = slice(step * B, (step + 1) * B)
        toks, pidx = mix['task_token'][sl], mix['pool_index'][sl]
        L = int(mix['batch_length'][step])
        x = M.encode_batch(toks, data['pool_bits'][toks, pidx], L)
        y = torch.from_numpy(data['pool_labels'][toks, pidx].astype(np.int64))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
        opt.step()
        sched.step()
        T = (step + 1) * B
        if T in T_checkpoints:
            ev = M.evaluate(model, target['task_token'], data['test_bits'], data['test_lens'], test_labels)
            if save_checkpoints == 'all' or (save_checkpoints == 'last' and T == T_checkpoints[-1]):
                M.export_npz(model, tmp_dir / f'T_{T}.npz')
            cnt = mix['counts'][str(T)]
            tgt = cnt.get(target_id, 0)
            rows.append(dict(seed=seed, target_id=target_id, K=target['K'], K_bits=target['K_bits'], alpha=alpha,
                             T=T, step=step + 1, excess_bits=ev['excess_bits'], accuracy=ev['accuracy'],
                             n_test=len(test_labels), target_examples_seen=tgt,
                             distractor_examples_seen=sum(cnt.values()) - tgt,
                             excess_bits_by_length=ev['excess_bits_by_length'],
                             wall_seconds=round(time.time() - t0, 3)))
            preds.append(ev['per_string_bits'])
    (out_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'predictions' / f'{name}.npz.tmp', 'wb') as f:
        np.savez(f, bits=np.stack(preds).astype(np.float32), T=np.array(T_checkpoints))
    os.replace(out_dir / 'predictions' / f'{name}.npz.tmp', out_dir / 'predictions' / f'{name}.npz')
    if save_checkpoints != 'none':
        shutil.rmtree(ck_dir, ignore_errors=True)
        os.replace(tmp_dir, ck_dir)
    (out_dir / 'cells').mkdir(parents=True, exist_ok=True)
    rec = dict(cell=name, seed=seed, target_id=target_id, alpha=alpha, arch=arch, rows=rows,
               counts=mix['counts'], data_order_seed=mix['data_order_seed'], pool_reuse=mix['pool_reuse'],
               wall_seconds=round(time.time() - t0, 3))
    atomic_write_text(out_dir / 'cells' / f'{name}.json', json.dumps(rec))
    return rec


_W = {}


def _init_worker(out_dir, threads):
    torch.set_num_threads(threads)
    _W['data'] = load_data(Path(out_dir))
    _W['c'] = load_config()


def _work(args):
    target_id, alpha, seed, out_dir, kw = args
    rec = train_cell(_W['c'], _W['data']['tasks'], target_id, alpha, seed, out_dir, data=_W['data'], **kw)
    return rec['cell'], rec['wall_seconds']


def run_cells(c, out, cells, max_seconds, kw, log):
    """Run cells (target_id, alpha, seed) with workers, skipping done ones; stop scheduling past the budget."""
    state_path = out / 'run_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(chunks=[], elapsed_seconds=0.0)
    start = time.time()
    budget = min(max_seconds, c['max_wall_seconds'] - state['elapsed_seconds'])
    todo = [x for x in cells if not (out / 'cells' / f'{cell_name(x[2], x[0], x[1])}.json').exists()]
    log(f'{len(cells) - len(todo)} cells done, {len(todo)} to run, budget {budget:.0f}s')
    done_now = 0
    with cf.ProcessPoolExecutor(c['workers'], initializer=_init_worker, initargs=(str(out), c['threads'])) as ex:
        pending = set()
        it = iter(todo)
        while True:
            while len(pending) < c['workers'] and time.time() - start < budget:
                nxt = next(it, None)
                if nxt is None:
                    break
                pending.add(ex.submit(_work, (*nxt, str(out), kw)))
            if not pending:
                break
            fin, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                name, wall = f.result()
                done_now += 1
                log(f'done {name} {wall:.1f}s ({done_now}/{len(todo)}, {time.time() - start:.0f}s)')
    el = time.time() - start
    state['elapsed_seconds'] += el
    state['chunks'].append(dict(start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                                seconds=round(el, 1), cells_completed=done_now))
    atomic_write_text(state_path, json.dumps(state, indent=1))
    missing = [cell_name(s, t, a) for t, a, s in cells
               if not (out / 'cells' / f'{cell_name(s, t, a)}.json').exists()]
    return missing, state


def grid_cells(c, tasks):
    targets = sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: (-t['K'], t['task_token']))
    return [(t['task_id'], a, s) for t in targets for a in sorted(c['alphas']) for s in c['seeds']]


def aggregate(c, out, cells, prefix=''):
    recs = []
    for t, a, s in cells:
        p = out / 'cells' / f'{cell_name(s, t, a)}.json'
        if p.exists():
            recs.append(json.loads(p.read_text()))
    recs.sort(key=lambda r: (r['seed'], r['target_id'], r['alpha']))
    cols = ['seed', 'target_id', 'K', 'K_bits', 'alpha', 'T', 'step', 'excess_bits', 'accuracy', 'n_test',
            'target_examples_seen', 'distractor_examples_seen', 'excess_bits_by_length', 'wall_seconds']
    with open(out / f'{prefix}metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in recs:
            for row in r['rows']:
                row = dict(row)
                row['excess_bits'] = repr(row['excess_bits'])
                row['excess_bits_by_length'] = json.dumps(row['excess_bits_by_length'])
                w.writerow(row)
    mixes = {r['cell']: dict(target_id=r['target_id'], alpha=r['alpha'], seed=r['seed'],
                             data_order_seed=r['data_order_seed'], counts=r['counts']) for r in recs}
    atomic_write_text(out / f'{prefix}mixes.json', json.dumps(mixes, indent=0, sort_keys=True))
    return len(recs)


def write_manifest(c, out, extra):
    import scipy
    import matplotlib
    cpu = ''
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        pass
    man = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    man.update(dict(
        sources={s: sha256_file(HERE / s) for s in SOURCES if (HERE / s).exists()},
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        host='cpu-box', cpu_model=cpu, cpu_count=os.cpu_count()))
    man.update(extra)
    for name in ('results_check.json', 'pilot_summary.json'):
        if (HERE / name).exists():
            man[name.replace('.json', '')] = json.loads((HERE / name).read_text())
    atomic_write_text(out / 'manifest.json', json.dumps(man, indent=1))


# ---------------------------------------------------------------- check-only

def _probe(args):
    out_dir, target_id, steps = args
    c = load_config()
    torch.set_num_threads(c['threads'])
    data = load_data(Path(out_dir))
    tasks = data['tasks']
    target = next(t for t in tasks if t['task_id'] == target_id)
    mix = D.assemble_mix(c, tasks, target_id, 0.4, c['pilot_seed'])
    torch.manual_seed(0)
    model = M.build(c)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'])
    B = c['batch_size']
    t0 = time.time()
    for step in range(steps):
        sl = slice(step * B, (step + 1) * B)
        toks, pidx = mix['task_token'][sl], mix['pool_index'][sl]
        x = M.encode_batch(toks, data['pool_bits'][toks, pidx], int(mix['batch_length'][step]))
        y = torch.from_numpy(data['pool_labels'][toks, pidx].astype(np.int64))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    step_s = (time.time() - t0) / steps
    t0 = time.time()
    M.evaluate(model, target['task_token'], data['test_bits'], data['test_lens'],
               data['test_labels'][target['task_token']])
    return step_s, time.time() - t0


def check_only(c, scratch):
    t_start = time.time()
    tasks, sel_log, test_bits, test_lens, pools = build_all(c)
    write_data(scratch, tasks, test_bits, test_lens, pools)
    rec = dict(date_utc=datetime.now(timezone.utc).isoformat(), selection_log=sel_log)
    rec['task_table'] = [dict(id=t['task_id'], role=t['role'], n=t['K'], K_bits=t['K_bits'], token=t['task_token'],
                              acceptance=round(t['acceptance_rate'], 4), sensitivity=round(t['sensitivity'], 4),
                              quartile=t['sensitivity_quartile'], tau=round(t['visit_tail_steepness'], 4))
                         for t in tasks]
    ok = {}
    hm = []
    for t in tasks:
        d = A.DFA.from_json(t['definition'])
        h, m = A.hopcroft_minimize(d), A.moore_minimize(d)
        hm.append(h.n == m.n == t['K'] and h.key() == m.key() == d.key())
    ok['hopcroft_equals_moore_all_tasks'] = all(hm)
    keys = []
    for t in tasks:
        d = A.DFA.from_json(t['definition'])
        keys.append((A.canonical_form(d).key(), A.canonical_form(A.complement(d)).key()))
    distinct = all(keys[i][0] != keys[j][0] and keys[i][0] != keys[j][1]
                   for i in range(len(keys)) for j in range(len(keys)) if i != j)
    ok['pairwise_distinct_and_not_complement'] = distinct
    Ls = D.lengths(c)
    rec['test_counts_per_length'] = {int(L): int((test_lens == L).sum()) for L in Ls}
    ok['test_size'] = len(test_bits) == c['test_size']
    pool_counts = {t['task_id']: sorted(set(int((pools[t['task_id']]['length'] == L).sum()) for L in Ls))
                   for t in tasks}
    ok['pool_counts_all_4096'] = all(v == [c['pool_per_length']] for v in pool_counts.values())
    test_set = set(zip(test_bits.tolist(), test_lens.tolist()))
    overlap = sum(len(test_set & set(zip(p['bits'].tolist(), p['length'].tolist()))) for p in pools.values())
    rec['test_strings_in_pools'] = overlap
    ok['zero_test_strings_in_pools'] = overlap == 0
    ok['test_all_in_test_partition'] = all(D.in_test_partition(test_bits[test_lens == L], int(L), c).all()
                                           for L in Ls)
    sched = {}
    t0 = next(t for t in tasks if t['role'] == 'target')
    for a in c['alphas']:
        mix = D.assemble_mix(c, tasks, t0['task_id'], a, c['seeds'][0])
        cnt = [mix['counts'][str(T)].get(t0['task_id'], 0) for T in c['T_checkpoints']]
        dev = max(abs(n - a * T) for n, T in zip(cnt, c['T_checkpoints']))
        dc = [v for k, v in mix['counts'][str(c['T_checkpoints'][-1])].items() if k != t0['task_id']]
        bl = np.unique(mix['batch_length'], return_counts=True)[1]
        sched[str(a)] = dict(target_counts=cnt, max_abs_dev_from_alpha_T=float(dev),
                             distractor_count_range=[min(dc), max(dc)] if dc else None,
                             batches_per_length=sorted(set(bl.tolist())), pool_reuse=mix['pool_reuse'])
    rec['schedule_check'] = sched
    ok['schedule_within_4'] = all(v['max_abs_dev_from_alpha_T'] <= 4 for v in sched.values())
    ok['no_pool_reuse_in_gated_grid'] = all(v['pool_reuse'] == 0 for v in sched.values())
    rec['parameters'] = dict(transformer=M.n_params(M.build_model(c)), lstm=M.n_params(M.build_lstm(c)))
    ok['parameter_counts'] = rec['parameters'] == dict(transformer=c['expected_parameters'],
                                                       lstm=c['lstm']['expected_parameters'])
    targets = [t['task_id'] for t in tasks if t['role'] == 'target'][: c['workers']]
    with cf.ProcessPoolExecutor(c['workers']) as ex:
        probes = list(ex.map(_probe, [(str(scratch), t, 20) for t in targets]))
    step_s = float(np.mean([p[0] for p in probes]))
    eval_s = float(np.mean([p[1] for p in probes]))
    per_cell = (c['T_checkpoints'][-1] // c['batch_size']) * step_s + len(c['T_checkpoints']) * eval_s
    n_cells = len(c['seeds']) * len(c['alphas']) * c['targets_per_level'] * len(c['n_levels'])
    rec['timing_probe'] = dict(concurrent_workers=c['workers'], step_seconds=round(step_s, 4),
                               eval_seconds=round(eval_s, 3), projected_cell_seconds=round(per_cell, 1),
                               projected_grid_hours=round(n_cells * per_cell / c['workers'] / 3600, 2))
    rec['checks'] = ok
    rec['all_checks_pass'] = all(ok.values())
    rec['seconds'] = round(time.time() - t_start, 1)
    return rec


# ---------------------------------------------------------------- pilot

def t_star(Ts, eps, eps_star):
    """First crossing with log-log interpolation; returns (T*, flag) with flag in {'', 'left', 'right'}."""
    Ts, eps = np.asarray(Ts, float), np.asarray(eps, float)
    below = np.flatnonzero(eps <= eps_star)
    if len(below) == 0:
        return float(Ts[-1]), 'right'
    i = int(below[0])
    if i == 0:
        return float(Ts[0]), 'left'
    x0, x1 = np.log2(Ts[i - 1]), np.log2(Ts[i])
    y0, y1 = np.log2(eps[i - 1]), np.log2(eps[i])
    x = x0 + (np.log2(eps_star) - y0) * (x1 - x0) / (y1 - y0)
    return float(2 ** x), ''


def pilot_summary(c, out, step_s):
    rows = {}
    for p in sorted((out / 'cells').glob('*.json')):
        r = json.loads(p.read_text())
        rows[r['target_id']] = {row['T']: row['excess_bits'] for row in r['rows']}
    Ts = c['pilot_T_checkpoints']
    per = {}
    for tid, e in sorted(rows.items()):
        ts, fl = t_star(Ts, [e[T] for T in Ts], 0.1)
        per[tid] = dict(eps={str(T): e[T] for T in Ts}, tstar_0p1=ts, tstar_0p1_flag=fl)
    med = float(np.median([v['tstar_0p1'] for v in per.values()]))
    at_1024 = sum(rows[t][1024] <= 0.1 for t in rows)
    improve = sum(rows[t][131072] < rows[t][65536] for t in rows)
    improve10 = sum(rows[t][131072] < 0.9 * rows[t][65536] for t in rows)
    n_cells = len(c['seeds']) * len(c['alphas']) * c['targets_per_level'] * len(c['n_levels'])
    cell_s = float(np.mean([json.loads(p.read_text())['wall_seconds'] for p in (out / 'cells').glob('*.json')]))
    proj = n_cells * cell_s / 2 / c['workers']  # pilot cells run 2,048 steps, gated cells 1,024
    return dict(seed=c['pilot_seed'], per_target=per, median_tstar_0p1=med,
                P1_fires=med > 2 ** 14, targets_at_or_below_0p1_at_T1024=int(at_1024), P2_fires=at_1024 > 4,
                mean_pilot_cell_seconds=round(cell_s, 1), projected_grid_seconds=round(proj),
                P3_fires=proj > 0.9 * c['max_wall_seconds'],
                targets_improving_2e16_to_2e17=int(improve), targets_improving_by_10pct=int(improve10))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--lstm', action='store_true')
    ap.add_argument('--calibrate-lr', type=float, help='pre-freeze lr calibration: calibration_seed, alpha = 1')
    ap.add_argument('--output', default=str(HERE / 'results'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--only')
    ap.add_argument('--max-seconds', type=float, default=float('inf'))
    args = ap.parse_args()
    c = load_config()
    torch.set_num_threads(c['threads'])
    log = lambda m: print(f'[{datetime.now().strftime("%H:%M:%S")}] {m}', flush=True)  # noqa: E731

    if args.check_only:
        scratch = Path(os.environ.get('TMPDIR', '/tmp')) / 'exp20_check'
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        rec = check_only(c, scratch)
        atomic_write_text(HERE / 'results_check.json', json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ('checks', 'all_checks_pass', 'timing_probe', 'seconds')}, indent=1))
        return 0 if rec['all_checks_pass'] else 1

    if args.only:
        seed, target_id, alpha = args.only.split(':')
        out = Path(args.output).resolve()
        assert out != (HERE / 'results').resolve(), '--only never writes into results/'
        src = HERE / 'results'
        out.mkdir(parents=True, exist_ok=True)
        data = load_data(src)
        rec = train_cell(c, data['tasks'], target_id, float(alpha), int(seed), out, data=data)
        print(json.dumps(dict(cell=rec['cell'], wall_seconds=rec['wall_seconds'])))
        return 0

    if args.pilot or args.calibrate_lr or args.lstm:
        sys.exit('Experiment 20 has no pilot, calibration or separate LSTM arm (PROTOCOL Section 2.6)')
    if args.calibrate_lr:
        args.output = str(HERE / f'results_calib_lr{args.calibrate_lr:g}')
    out = Path(HERE / 'results_pilot' if args.pilot else args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        sys.exit(f'{out} exists and is not empty; pass --resume')
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'tasks.json').exists():
        log('building tasks and data')
        tasks, sel_log, tb, tl, pools = build_all(c)
        write_data(out, tasks, tb, tl, pools)
        atomic_write_text(out / 'selection_log.json', json.dumps(sel_log, indent=1))
    tasks = json.loads((out / 'tasks.json').read_text())
    started = datetime.now(timezone.utc).isoformat()

    if args.pilot:
        cells = [(t['task_id'], 1.0, c['pilot_seed']) for t in tasks if t['role'] == 'target']
        missing, state = run_cells(c, out, cells, args.max_seconds,
                                   dict(T_checkpoints=c['pilot_T_checkpoints']), log)
        aggregate(c, out, cells)
        if not missing:
            s = pilot_summary(c, out, None)
            atomic_write_text(HERE / 'pilot_summary.json', json.dumps(s, indent=1))
            print(json.dumps({k: v for k, v in s.items() if k != 'per_target'}, indent=1))
        return 0

    if args.calibrate_lr:
        cells = [(t['task_id'], 1.0, c['calibration_seed']) for t in tasks if t['role'] == 'target']
        missing, state = run_cells(c, out, cells, args.max_seconds,
                                   dict(lr=args.calibrate_lr, save_checkpoints='none'), log)
        aggregate(c, out, cells)
        log(f'calibration lr {args.calibrate_lr:g}: {len(cells) - len(missing)}/{len(cells)} cells')
        return 0

    if args.lstm:
        lstm_cells = [(t['task_id'], 1.0, s) for t in tasks if t['role'] == 'target' for s in c['seeds']]
        state = json.loads((out / 'run_state.json').read_text())
        left = c['max_wall_seconds'] - state['elapsed_seconds']
        if left < 60 * c['lstm']['min_minutes_left']:
            log(f'LSTM arm skipped: {left:.0f}s left under the cap')
            return 0
        lo = out / 'lstm_arm'
        lo.mkdir(exist_ok=True)
        for p in ('tasks.json', 'data'):
            if not (lo / p).exists():
                os.symlink(out / p, lo / p)
        missing, _ = run_cells(c, lo, lstm_cells, min(args.max_seconds, left),
                               dict(arch='lstm', save_checkpoints='last'), log)
        aggregate(c, lo, lstm_cells)
        log(f'LSTM arm: {len(lstm_cells) - len(missing)}/{len(lstm_cells)} cells')
        return 0

    cells = grid_cells(c, tasks)
    missing, state = run_cells(c, out, cells, args.max_seconds, {}, log)
    n = aggregate(c, out, cells)
    man = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    write_manifest(c, out, dict(start_utc=man.get('start_utc', started), end_utc=datetime.now(timezone.utc).isoformat(),
                                elapsed_seconds=round(state['elapsed_seconds'], 1), chunks=state['chunks'],
                                cells_complete=n, cells_missing=missing))
    log(f'{n}/{len(cells)} cells complete; {len(missing)} missing')
    return 0


if __name__ == '__main__':
    sys.exit(main())
