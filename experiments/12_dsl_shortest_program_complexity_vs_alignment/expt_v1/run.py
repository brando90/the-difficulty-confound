"""Experiment 12 runner: BFS, task selection and data, invariant checks, the disclosed pilot, the gated 640-cell grid
and the secondary arms (PROTOCOL Sections 2 and 7).

Usage (venv python, from expt_v1/):
  run.py --check-only                      BFS, selection, split, data, covariates, proxies, invariant checks and a
                                           timing probe; writes the JSON receipt results_check.json, no training
  run.py --pilot                           disclosed pilot (seed 12909, alpha = 1, 20 targets, T to 2^17) into
                                           results_pilot/; prints the P1-P4 verdicts
  run.py --output results [--resume] [--max-seconds S]
                                           gated grid; --max-seconds bounds one invocation (chunk); the cumulative
                                           wall cap max_wall_seconds is tracked in run_state.json
  run.py --rep768 --output results --resume   secondary repetition arm (only if 60 minutes remain under the cap)
  run.py --mlp --output results --resume      exploratory MLP arm (only if 10 minutes remain under the cap)
  run.py --only ARM:SEED:TARGET:ALPHA --output SCRATCH   one cell for determinism checks (never into results/)
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
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
import dsl  # noqa: E402
import model as M  # noqa: E402
import proxies as PX  # noqa: E402

SOURCES = ['config.json', 'dsl.py', 'proxies.py', 'data.py', 'model.py', 'run.py', 'laws.py', 'independent_fit.py',
           'analyze.py', 'verify.py', 'report.py']
ARMS = {'main': '', 'rep768': 'rep768_arm', 'mlp': 'mlp_arm'}
METRIC_COLS = ['arm', 'seed', 'target_id', 'K', 'K_bits', 'alpha', 'T', 'step', 'excess_bits', 'accuracy',
               'bit_accuracy', 'n_test', 'excess_bits_by_output_bit', 'train_bits', 'train_accuracy',
               'target_examples_seen', 'distractor_examples_seen', 'target_passes', 'wall_seconds']


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


def log_fn():
    return lambda m: print(f'[{datetime.now().strftime("%H:%M:%S")}] {m}', flush=True)  # noqa: E731


# ---------------------------------------------------------------- tasks and data

def build_all(c, cache=None):
    """BFS, split and the 200 selected tasks, all from the task seed. Returns (tasks, selection_log, split, bfs)."""
    bfs = dsl.load_or_run_bfs(c, cache)
    split = D.make_split(c)
    tasks, sel_log = dsl.select_tasks(c, bfs, split['train_idx'], PX.proxies)
    sel_log['bfs'] = dict(level_counts=bfs['level_counts'], seconds=bfs['seconds'], peak_rss_gb=bfs['peak_rss_gb'],
                          from_cache=bfs['from_cache'])
    return tasks, sel_log, split, bfs


def write_data(c, out, tasks, split, sel_log):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / 'tasks.json', json.dumps(tasks, indent=1))
    atomic_write_text(out / 'split.json', json.dumps(dict(perm=split['perm'].tolist(), attempts=split['attempts'],
                                                          train_bit_counts=split['train_bit_counts'])))
    atomic_write_text(out / 'selection_log.json', json.dumps(sel_log, indent=1))
    atomic_write_text(out / 'bfs_counts.json', json.dumps(sel_log['bfs'], indent=1))
    D.build_data(c, tasks, split, out)


def load_data(out):
    """Only what a worker needs: the 40 trained truth tables (indexed by task token), the tasks and the split."""
    out = Path(out)
    tasks = json.loads((out / 'tasks.json').read_text())
    sp = json.loads((out / 'split.json').read_text())
    trained = sorted((t for t in tasks if t['task_token'] is not None), key=lambda t: t['task_token'])
    assert [t['task_token'] for t in trained] == list(range(len(trained)))
    return dict(tasks=[{k: v for k, v in t.items() if k != 'truth_table'} | dict(truth_table=None) for t in tasks],
                tables=np.array([t['truth_table'] for t in trained], np.uint16),
                perm=np.array(sp['perm'], np.uint16))


# ---------------------------------------------------------------- training

def train_cell(c, tasks, target_id, alpha, seed, out_dir, n_task_tokens=40, arch='transformer', n_train=512,
               save_checkpoints=True, T_checkpoints=None, data=None, arm='main'):
    """Train one (target, alpha, seed) cell; write checkpoints, predictions and the cell record atomically.

    torch.manual_seed(seed) runs before the model is built, so every cell of a seed shares its initialization.
    """
    torch.set_num_threads(c['threads'])
    out_dir = Path(out_dir)
    T_checkpoints = list(T_checkpoints or c['T_checkpoints'])
    if data is None:
        split = D.make_split(c)
        trained = sorted((t for t in tasks if t['task_token'] is not None), key=lambda t: t['task_token'])
        data = dict(tables=np.array([t['truth_table'] for t in trained], np.uint16), perm=split['perm'])
    target = next(t for t in tasks if t['task_id'] == target_id)
    tok = int(target['task_token'])
    name = cell_name(seed, target_id, alpha)
    t0 = time.time()
    mix = D.assemble_mix(c, tasks, target_id, alpha, seed, n_train=n_train, T_checkpoints=T_checkpoints)
    perm = data['perm'].astype(np.int64)
    train_x, test_x = perm[:n_train], perm[n_train:]
    x_all = train_x[mix['input_index']]
    y_all = data['tables'][mix['task_token'], x_all]
    X_all = M.encode(mix['task_token'], D.to_bits(x_all))
    Y_all = torch.from_numpy(D.to_bits(y_all).astype(np.int64))
    tab = data['tables'][tok].astype(np.int64)
    ev_sets = dict(test=(D.to_bits(test_x), D.to_bits(tab[test_x])), train=(D.to_bits(train_x), D.to_bits(tab[train_x])))
    torch.manual_seed(seed)
    model = M.build_model(c, n_task_tokens) if arch == 'transformer' else M.build_mlp(c, n_task_tokens)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], betas=tuple(c['betas']), weight_decay=c['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c['warmup_steps']))
    B = c['batch_size']
    ck_dir = out_dir / 'checkpoints' / name
    tmp_dir = Path(str(ck_dir) + '.tmp')
    if save_checkpoints:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
    rows, p_test, p_train = [], [], []
    model.train()
    for step in range(T_checkpoints[-1] // B):
        sl = slice(step * B, (step + 1) * B)
        logits = model(X_all[sl])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), Y_all[sl].reshape(-1)) * dsl.WORD_BITS
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
        opt.step()
        sched.step()
        T = (step + 1) * B
        if T in T_checkpoints:
            ev = M.evaluate(model, ev_sets['test'][0], tok, ev_sets['test'][1])
            tr = M.evaluate(model, ev_sets['train'][0], tok, ev_sets['train'][1])
            if save_checkpoints:
                M.export_npz(model, tmp_dir / f'T_{T}.npz')
            cnt = mix['counts'][str(T)]
            tgt = cnt.get(target_id, 0)
            rows.append(dict(arm=arm, seed=seed, target_id=target_id, K=target['K'], K_bits=target['K_bits'],
                             alpha=alpha, T=T, step=step + 1, excess_bits=ev['excess_bits'], accuracy=ev['accuracy'],
                             bit_accuracy=ev['bit_accuracy'], n_test=len(test_x),
                             excess_bits_by_output_bit=ev['excess_bits_by_output_bit'], train_bits=tr['excess_bits'],
                             train_accuracy=tr['accuracy'], target_examples_seen=tgt,
                             distractor_examples_seen=sum(cnt.values()) - tgt,
                             target_passes=mix['target_passes'][str(T)], wall_seconds=round(time.time() - t0, 3)))
            p_test.append(ev['per_example_bits'])
            p_train.append(tr['per_example_bits'])
    (out_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    tmp = out_dir / 'predictions' / f'{name}.npz.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, test=np.stack(p_test).astype(np.float32), train=np.stack(p_train).astype(np.float32),
                 T=np.array(T_checkpoints))
    os.replace(tmp, out_dir / 'predictions' / f'{name}.npz')
    if save_checkpoints:
        shutil.rmtree(ck_dir, ignore_errors=True)
        os.replace(tmp_dir, ck_dir)
    (out_dir / 'cells').mkdir(parents=True, exist_ok=True)
    rec = dict(cell=name, arm=arm, seed=seed, target_id=target_id, alpha=alpha, arch=arch, n_train=n_train, rows=rows,
               counts=mix['counts'], target_passes=mix['target_passes'], data_order_seed=mix['data_order_seed'],
               wall_seconds=round(time.time() - t0, 3))
    atomic_write_text(out_dir / 'cells' / f'{name}.json', json.dumps(rec))  # the done marker, written last
    return rec


_W = {}


def _init_worker(data_dir, threads):
    torch.set_num_threads(threads)
    _W['data'] = load_data(Path(data_dir))
    _W['c'] = load_config()


def _work(args):
    target_id, alpha, seed, out_dir, kw = args
    rec = train_cell(_W['c'], _W['data']['tasks'], target_id, alpha, seed, out_dir, data=_W['data'], **kw)
    return rec['cell'], rec['wall_seconds']


def run_cells(c, out, cells, max_seconds, kw, log, state_dir=None, data_dir=None):
    """Run cells (target_id, alpha, seed) with 4 workers, skipping done ones; stop scheduling past the budget.

    The cumulative wall clock of every chunk and arm is kept in <state_dir>/run_state.json.
    """
    state_path = Path(state_dir or out) / 'run_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(chunks=[], elapsed_seconds=0.0)
    start = time.time()
    budget = min(max_seconds, c['max_wall_seconds'] - state['elapsed_seconds'])
    todo = [x for x in cells if not (out / 'cells' / f'{cell_name(x[2], x[0], x[1])}.json').exists()]
    log(f'{len(cells) - len(todo)} cells done, {len(todo)} to run, budget {budget:.0f}s')
    done_now = 0
    if todo and budget > 0:
        with cf.ProcessPoolExecutor(c['workers'], initializer=_init_worker,
                                    initargs=(str(data_dir or out), c['threads'])) as ex:
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
    state['chunks'].append(dict(start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(), arm=kw.get('arm'),
                                seconds=round(el, 1), cells_completed=done_now))
    atomic_write_text(state_path, json.dumps(state, indent=1))
    missing = [cell_name(s, t, a) for t, a, s in cells if not (out / 'cells' / f'{cell_name(s, t, a)}.json').exists()]
    return missing, state


def targets_of(tasks):
    return sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: t['task_token'])


def grid_cells(c, tasks):
    """Main grid, longest-first: K = 6 targets first."""
    ts = sorted(targets_of(tasks), key=lambda t: (-t['K'], t['task_token']))
    return [(t['task_id'], a, s) for t in ts for a in sorted(c['alphas']) for s in c['seeds']]


def arm_cells(c, tasks, arm):
    cfg = c[arm]
    return [(t['task_id'], a, s) for t in targets_of(tasks) for a in cfg['alphas'] for s in c['seeds'][: cfg['n_seeds']]]


def arm_kw(c, arm):
    if arm == 'rep768':
        return dict(arm='rep768', n_train=c['rep768']['n_train'], save_checkpoints=False)
    if arm == 'mlp':
        return dict(arm='mlp', arch='mlp', save_checkpoints=False)
    return dict(arm='main')


def read_cells(out, cells):
    recs = []
    for t, a, s in cells:
        p = Path(out) / 'cells' / f'{cell_name(s, t, a)}.json'
        if p.exists():
            recs.append(json.loads(p.read_text()))
    recs.sort(key=lambda r: (r['seed'], r['target_id'], r['alpha']))
    return recs


def aggregate(c, out, tasks):
    """metrics.csv (main rows, then secondary-arm rows, labelled by arm) and mixes.json (per arm, per cell)."""
    out = Path(out)
    mixes, n_rows = {}, {}
    with open(out / 'metrics.csv.tmp', 'w', newline='') as f:
        w = csv.DictWriter(f, METRIC_COLS)
        w.writeheader()
        for arm, sub in ARMS.items():
            cells = grid_cells(c, tasks) if arm == 'main' else arm_cells(c, tasks, arm)
            recs = read_cells(out / sub if sub else out, cells)
            n_rows[arm] = 0
            for r in recs:
                for row in r['rows']:
                    row = dict(row)
                    for k in ('excess_bits', 'train_bits', 'accuracy', 'bit_accuracy', 'train_accuracy',
                              'target_passes'):
                        row[k] = repr(float(row[k]))
                    row['excess_bits_by_output_bit'] = json.dumps(row['excess_bits_by_output_bit'])
                    w.writerow(row)
                    n_rows[arm] += 1
            if recs:
                mixes[arm] = {r['cell']: dict(target_id=r['target_id'], alpha=r['alpha'], seed=r['seed'],
                                              n_train=r['n_train'], data_order_seed=r['data_order_seed'],
                                              counts=r['counts'], target_passes=r['target_passes']) for r in recs}
    os.replace(out / 'metrics.csv.tmp', out / 'metrics.csv')
    atomic_write_text(out / 'mixes.json', json.dumps(mixes, indent=0, sort_keys=True))
    return n_rows


def write_manifest(c, out, extra):
    import scipy
    import matplotlib
    cpu = ''
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        pass
    man = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    try:
        from importlib.metadata import version
        pybdm_v = version('pybdm')
    except Exception:
        pybdm_v = None
    man.update(dict(
        sources={s: sha256_file(HERE / s) for s in SOURCES if (HERE / s).exists()},
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, matplotlib=matplotlib.__version__, pybdm=pybdm_v),
        pybdm_error=PX.PYBDM_ERROR, host=socket.gethostname(), cpu_model=cpu, cpu_count=os.cpu_count()))
    man.update(extra)
    for name in ('results_check.json', 'pilot_summary.json'):
        if (HERE / name).exists():
            rec = json.loads((HERE / name).read_text())
            man[name.replace('.json', '')] = {k: v for k, v in rec.items() if k not in ('task_table', 'per_target')}
    atomic_write_text(out / 'manifest.json', json.dumps(man, indent=1))


# ---------------------------------------------------------------- check-only

def _probe(args):
    data_dir, target_id, steps = args
    c = load_config()
    torch.set_num_threads(c['threads'])
    data = load_data(Path(data_dir))
    tasks = data['tasks']
    tok = next(t['task_token'] for t in tasks if t['task_id'] == target_id)
    mix = D.assemble_mix(c, tasks, target_id, 0.4, c['pilot_seed'])
    perm = data['perm'].astype(np.int64)
    x_all = perm[: c['n_train']][mix['input_index']]
    X = M.encode(mix['task_token'], D.to_bits(x_all))
    Y = torch.from_numpy(D.to_bits(data['tables'][mix['task_token'], x_all]).astype(np.int64))
    torch.manual_seed(0)
    model = M.build_model(c)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'])
    B = c['batch_size']
    t0 = time.time()
    for step in range(steps):
        sl = slice(step * B, (step + 1) * B)
        loss = torch.nn.functional.cross_entropy(model(X[sl]).reshape(-1, 2), Y[sl].reshape(-1)) * 10
        opt.zero_grad()
        loss.backward()
        opt.step()
    step_s = (time.time() - t0) / steps
    t0 = time.time()
    tab = data['tables'][tok].astype(np.int64)
    for xs in (perm[c['n_train']:], perm[: c['n_train']]):
        M.evaluate(model, D.to_bits(xs), tok, D.to_bits(tab[xs]))
    return step_s, time.time() - t0


def g8a_statistic(c, tasks):
    """Spearman(row-major BDM, K) over the 200 tasks with a task bootstrap (resampling tasks within each level)."""
    from scipy.stats import spearmanr
    K = np.array([t['K'] for t in tasks], float)
    v = np.array([t['bdm_rowmajor'] for t in tasks], float)
    est = float(spearmanr(v, K).statistic)
    rng = np.random.default_rng(c['analysis_seed'] + 8)
    lv = [np.flatnonzero(K == k) for k in c['K_levels']]
    draws = np.empty(c['bootstrap_draws'])
    for b in range(len(draws)):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in lv])
        draws[b] = spearmanr(v[idx], K[idx]).statistic
    lo, hi = np.percentile(draws, [2.5, 97.5])
    others = {k: float(spearmanr([t[k] for t in tasks], K).statistic)
              for k in ('bdm_bitplane', 'zlib_bytes', 'lzma_bytes', 'zlib_u16_bytes')}
    g = c['gates']['g8_min_spearman']
    return dict(statistic='Spearman(bdm_rowmajor, K) over 200 tasks', estimate=est, ci_task=[float(lo), float(hi)],
                threshold=f'>= {g} with task-bootstrap lower bound > 0', bdm_source=tasks[0].get('bdm_source'),
                verdict='SUPPORTED' if est >= g and lo > 0 else 'NOT_ESTABLISHED', others_spearman_with_K=others)


def check_only(c, scratch):
    t_start = time.time()
    tasks, sel_log, split, bfs = build_all(c)
    write_data(c, scratch, tasks, split, sel_log)
    rec = dict(date_utc=datetime.now(timezone.utc).isoformat(), selection_log={k: v for k, v in sel_log.items()})
    ok = {}
    rec['bfs'] = sel_log['bfs']
    ok['bfs_level_counts'] = bfs['level_counts'] == c['bfs_level_counts']
    ok['bfs_peak_rss_under_limit'] = bfs['peak_rss_gb'] < c['bfs_ram_limit_gb']
    # brute force for K <= 4
    t0 = time.time()
    bf = dsl.brute_force(4)
    bad = 0
    for h, (K, prog, ns, tab) in bf.items():
        k2, g = dsl.exact_k(tab, bfs)
        bad += not (k2 == K and tuple(int(p) for p in bfs['canonical'][g] if p >= 0) == prog
                    and int(bfs['n_shortest'][g]) == ns)
    rec['brute_force_K_le_4'] = dict(functions=len(bf), mismatches=bad, seconds=round(time.time() - t0, 1))
    ok['brute_force_K_le_4'] = bad == 0 and len(bf) == 1 + sum(c['bfs_level_counts'][:4])
    rec['task_table'] = [dict(id=t['task_id'], role=t['role'], token=t['task_token'], K=t['K'], K_bits=t['K_bits'],
                              program=t['definition']['primitives'], balance=round(t['output_balance'], 4),
                              sensitivity=round(t['sensitivity'], 4), stratum=t['sensitivity_stratum'],
                              affine=t['is_affine'], H_rule=round(t['local_rule_entropy'], 4),
                              n_shortest=t['n_shortest_programs'], bdm=round(t['bdm_rowmajor'], 1),
                              zlib=t['zlib_bytes'], lzma=t['lzma_bytes']) for t in tasks]
    # per-task invariants
    inv = dict(program_reproduces_table=0, exact_k_matches=0, balance_ok=0, depends_ok=0, identifiable=0,
               stratum_type_ok=0)
    for t in tasks:
        tab = np.array(t['truth_table'], np.uint16)
        inv['program_reproduces_table'] += bool(np.array_equal(dsl.apply_program(t['definition']['primitives']), tab))
        inv['exact_k_matches'] += dsl.exact_k(tab, bfs)[0] == t['K']
        inv['balance_ok'] += c['balance_range'][0] <= t['output_balance'] <= c['balance_range'][1]
        inv['depends_ok'] += t['n_depends'] >= c['min_depends']
        inv['identifiable'] += dsl.train_identifiable(tab, t['K'], bfs, split['train_idx'])
        k = int(t['sensitivity_stratum'][1:]) - 1
        st_ok = dsl.stratum_of(t['sensitivity'], c['strata']) == k
        if t['role'] in ('target', 'distractor'):
            st_ok = st_ok and (c['cell_types'][k] == 'affine') == t['is_affine']
        inv['stratum_type_ok'] += st_ok
    rec['task_invariants'] = inv
    ok['task_invariants'] = all(v == len(tasks) for v in inv.values())
    roles = {}
    for t in tasks:
        roles[t['role']] = roles.get(t['role'], 0) + 1
    rec['role_counts'] = roles
    ok['role_counts'] = roles == dict(target=20, distractor=20, probe_pretrain=80, fresh=80)
    tabs = [bytes(np.array(t['truth_table'], np.uint16).tobytes()) for t in tasks]
    comps = [bytes((np.array(t['truth_table'], np.uint16) ^ np.uint16(1023)).tobytes()) for t in tasks]
    ok['pairwise_distinct_and_not_complement'] = len(set(tabs)) == len(tabs) and not (set(tabs) & set(comps))
    rec['split'] = dict(attempts=split['attempts'], train_bit_counts=split['train_bit_counts'])
    ok['split_bit_counts'] = all(c['train_bit_count_range'][0] <= v <= c['train_bit_count_range'][1]
                                 for v in split['train_bit_counts'])
    ok['split_disjoint_cover'] = (len(set(split['train_idx'].tolist()) | set(split['test_idx'].tolist())) == 1024
                                  and not set(split['train_idx'].tolist()) & set(split['test_idx'].tolist()))
    sched = {}
    t0 = targets_of(tasks)[0]
    for a in c['alphas']:
        mix = D.assemble_mix(c, tasks, t0['task_id'], a, c['seeds'][0])
        cnt = [mix['counts'][str(T)].get(t0['task_id'], 0) for T in c['T_checkpoints']]
        dev = max(abs(n - a * T) for n, T in zip(cnt, c['T_checkpoints']))
        dc = [v for k, v in mix['counts'][str(c['T_checkpoints'][-1])].items() if k != t0['task_id']]
        sched[str(a)] = dict(target_counts=cnt, max_abs_dev_from_alpha_T=float(dev),
                             distractor_count_range=[min(dc), max(dc)] if dc else None,
                             target_passes_Tmax=mix['target_passes'][str(c['T_checkpoints'][-1])])
    rec['schedule_check'] = sched
    ok['schedule_within_4'] = all(v['max_abs_dev_from_alpha_T'] <= 4 for v in sched.values())
    ok['distractors_equal_pm1'] = all(v['distractor_count_range'] is None or
                                      v['distractor_count_range'][1] - v['distractor_count_range'][0] <= 1
                                      for v in sched.values())
    rec['parameters'] = dict(transformer=M.n_params(M.build_model(c)), transformer_null_token=M.n_params(
        M.build_model(c, c['n_task_tokens'] + 1)), mlp=M.n_params(M.build_mlp(c)))
    ok['parameter_counts'] = rec['parameters'] == dict(transformer=c['expected_parameters'],
                                                       transformer_null_token=c['expected_parameters_null_token'],
                                                       mlp=c['mlp']['expected_parameters'])
    rec['G8a'] = g8a_statistic(c, tasks)
    rec['bdm_source'] = tasks[0].get('bdm_source')
    rec['pybdm_error'] = PX.PYBDM_ERROR
    try:
        busy = subprocess.run(['pgrep', '-af', 'expt_v1'], capture_output=True, text=True).stdout.strip()
    except Exception:
        busy = ''
    rec['other_expt_processes_during_probe'] = [l for l in busy.splitlines() if 'check-only' not in l][:8]
    targets = [t['task_id'] for t in targets_of(tasks)][: c['workers']]
    with cf.ProcessPoolExecutor(c['workers']) as ex:
        probes = list(ex.map(_probe, [(str(scratch), t, 20) for t in targets]))
    step_s = float(np.mean([p[0] for p in probes]))
    eval_s = float(np.mean([p[1] for p in probes]))
    per_cell = (c['T_checkpoints'][-1] // c['batch_size']) * step_s + len(c['T_checkpoints']) * eval_s
    n_cells = len(c['seeds']) * len(c['alphas']) * len(targets_of(tasks))
    ck_bytes = c['expected_parameters'] * 4
    rec['timing_probe'] = dict(concurrent_workers=c['workers'], step_seconds=round(step_s, 4),
                               eval_seconds=round(eval_s, 3), projected_cell_seconds=round(per_cell, 1),
                               projected_grid_hours=round(n_cells * per_cell / c['workers'] / 3600, 2),
                               projected_checkpoint_gb=round(n_cells * len(c['T_checkpoints']) * ck_bytes / 1e9, 3))
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
    return float(2 ** (x0 + (np.log2(eps_star) - y0) * (x1 - x0) / (y1 - y0))), ''


def pilot_summary(c, out):
    p = c['pilot']
    recs = [json.loads(q.read_text()) for q in sorted((out / 'cells').glob('*.json'))]
    Ts = p['T_checkpoints']
    per = {}
    for r in recs:
        e = {row['T']: row['excess_bits'] for row in r['rows']}
        trb = {row['T']: row['train_bits'] for row in r['rows']}
        ts, fl = t_star(Ts, [e[T] for T in Ts], p['p1_eps_star'])
        per[r['target_id']] = dict(test_bits={str(T): e[T] for T in Ts}, train_bits={str(T): trb[T] for T in Ts},
                                   tstar_1bit=ts, tstar_1bit_flag=fl, wall_seconds=r['wall_seconds'])
    med = float(np.median([v['tstar_1bit'] for v in per.values()]))
    at_p2 = sum(float(v['test_bits'][str(p['p2_T'])]) <= p['p2_eps'] for v in per.values())
    above_p4 = sum(float(v['test_bits'][str(Ts[-1])]) > p['p4_eps'] for v in per.values())
    improve = sum(v['test_bits'][str(Ts[-1])] < v['test_bits'][str(Ts[-2])] for v in per.values())
    improve10 = sum(v['test_bits'][str(Ts[-1])] < 0.9 * v['test_bits'][str(Ts[-2])] for v in per.values())
    n_cells = len(c['seeds']) * len(c['alphas']) * 20
    steps_pilot = Ts[-1] // c['batch_size']
    cell_s = float(np.mean([r['wall_seconds'] for r in recs]))
    step_s = cell_s / steps_pilot
    proj = n_cells * cell_s * (c['T_checkpoints'][-1] / Ts[-1]) / c['workers']
    ck = sorted((out / 'checkpoints').rglob('*.npz'))
    ck_bytes = float(np.mean([q.stat().st_size for q in ck])) if ck else c['expected_parameters'] * 4
    proj_gb = n_cells * len(c['T_checkpoints']) * ck_bytes / 1e9
    Tm = str(c['T_checkpoints'][-1])
    census = dict(memorizing=0, not_fitting=0, plateau=0)
    for v in per.values():
        e16, e14 = v['test_bits'][Tm], v['test_bits'][str(c['plateau_reference_T'])]
        if e16 >= c['plateau_ratio'] * e14 and e16 > c['plateau_min_bits']:
            census['plateau'] += 1
            census['memorizing'] += v['train_bits'][Tm] <= c['memorize_train_bits']
            census['not_fitting'] += v['train_bits'][Tm] > c['not_fit_train_bits']
    return dict(seed=c['pilot_seed'], per_target=per, median_tstar_1bit=med,
                P1_fires=med > p['p1_median_tstar_above'],
                targets_at_or_below_1bit_at_T2048=int(at_p2), P2_fires=at_p2 > p['p2_targets_above'],
                mean_pilot_cell_seconds=round(cell_s, 1), step_seconds_4_workers=round(step_s, 4),
                checkpoint_bytes=int(ck_bytes), projected_grid_seconds=round(proj),
                projected_artifact_gb=round(proj_gb, 3),
                P3_fires=proj > p['p3_fraction'] * c['max_wall_seconds'] or proj_gb > c['artifact_limit_gb'],
                targets_above_3bits_at_2e17=int(above_p4), P4_fires=above_p4 > p['p4_targets_above'],
                targets_improving_2e16_to_2e17=int(improve), targets_improving_by_10pct=int(improve10),
                train_fit_census_at_2e16=census)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--rep768', action='store_true')
    ap.add_argument('--mlp', action='store_true')
    ap.add_argument('--output', default=str(HERE / 'results'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--only')
    ap.add_argument('--max-seconds', type=float, default=float('inf'))
    args = ap.parse_args()
    c = load_config()
    torch.set_num_threads(c['threads'])
    log = log_fn()

    if args.check_only:
        scratch = Path(os.environ.get('TMPDIR', '/tmp')) / 'exp12_check'
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        rec = check_only(c, scratch)
        atomic_write_text(HERE / 'results_check.json', json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ('checks', 'all_checks_pass', 'G8a', 'timing_probe', 'seconds')},
                         indent=1))
        return 0 if rec['all_checks_pass'] else 1

    if args.only:
        arm, seed, target_id, alpha = args.only.split(':')
        out = Path(args.output).resolve()
        res = (HERE / 'results').resolve()
        assert out != res and res not in out.parents, '--only never writes into results/'
        out.mkdir(parents=True, exist_ok=True)
        data = load_data(res)
        rec = train_cell(c, data['tasks'], target_id, float(alpha), int(seed), out, data=data, **arm_kw(c, arm))
        print(json.dumps(dict(cell=rec['cell'], wall_seconds=rec['wall_seconds'])))
        return 0

    out = Path(HERE / 'results_pilot' if args.pilot else args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        sys.exit(f'{out} exists and is not empty; pass --resume')
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'tasks.json').exists():
        log('building BFS, tasks and data')
        tasks, sel_log, split, bfs = build_all(c, out / 'bfs_cache.npz')
        del bfs
        write_data(c, out, tasks, split, sel_log)
    tasks = json.loads((out / 'tasks.json').read_text())
    started = datetime.now(timezone.utc).isoformat()

    if args.pilot:
        cells = [(t['task_id'], 1.0, c['pilot_seed']) for t in targets_of(tasks)]
        missing, state = run_cells(c, out, cells, args.max_seconds,
                                   dict(T_checkpoints=c['pilot']['T_checkpoints'], arm='pilot'), log)
        if not missing:
            s = pilot_summary(c, out)
            atomic_write_text(HERE / 'pilot_summary.json', json.dumps(s, indent=1))
            print(json.dumps({k: v for k, v in s.items() if k != 'per_target'}, indent=1))
        return 0

    for arm in ('rep768', 'mlp'):
        if getattr(args, arm):
            state = json.loads((out / 'run_state.json').read_text())
            left = c['max_wall_seconds'] - state['elapsed_seconds']
            if left < 60 * c[arm]['min_minutes_left']:
                log(f'{arm} arm skipped: {left:.0f}s left under the cap')
                state.setdefault('skipped_arms', []).append(dict(arm=arm, seconds_left=round(left, 1)))
                atomic_write_text(out / 'run_state.json', json.dumps(state, indent=1))
                return 0
            sub = out / ARMS[arm]
            sub.mkdir(exist_ok=True)
            cells = arm_cells(c, tasks, arm)
            missing, _ = run_cells(c, sub, cells, min(args.max_seconds, left), arm_kw(c, arm), log, state_dir=out,
                                   data_dir=out)
            n = aggregate(c, out, tasks)
            log(f'{arm} arm: {len(cells) - len(missing)}/{len(cells)} cells; rows {n}')
            return 0

    cells = grid_cells(c, tasks)
    missing, state = run_cells(c, out, cells, args.max_seconds, dict(arm='main'), log)
    n = aggregate(c, out, tasks)
    man = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    write_manifest(c, out, dict(start_utc=man.get('start_utc', started), end_utc=datetime.now(timezone.utc).isoformat(),
                                elapsed_seconds=round(state['elapsed_seconds'], 1), chunks=state['chunks'],
                                cells_complete=len(cells) - len(missing), cells_missing=missing, metrics_rows=n,
                                split_attempts=json.loads((out / 'split.json').read_text())['attempts']))
    log(f'{len(cells) - len(missing)}/{len(cells)} cells complete; {len(missing)} missing')
    return 0


if __name__ == '__main__':
    sys.exit(main())
