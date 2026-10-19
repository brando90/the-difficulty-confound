"""Replay verification for Experiment 12 (PROTOCOL Section 8). Retrains only the 8 secondary-arm cells of step 7.

  verify.py [--results results] [--workers 4]
Writes results/verification.json with status PASS or FAIL and the counts of every check.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze as AN  # noqa: E402
import data as D  # noqa: E402
import dsl  # noqa: E402
import proxies as PX  # noqa: E402
import run as R  # noqa: E402

TRAINING_SOURCES = ['config.json', 'dsl.py', 'proxies.py', 'data.py', 'model.py', 'run.py']
ANALYSIS_SOURCES = ['laws.py', 'independent_fit.py', 'analyze.py', 'verify.py', 'report.py']


def _replay(args):
    """Reload every checkpoint of one main cell and recompute its test and train losses and predictions."""
    res, name = args
    import torch
    import model as M
    torch.set_num_threads(1)
    if 'data' not in _replay.__dict__:
        _replay.data = R.load_data(Path(res))
        _replay.c = R.load_config()
    data, c = _replay.data, _replay.c
    rec = json.loads((Path(res) / 'cells' / f'{name}.json').read_text())
    tok = next(t['task_token'] for t in data['tasks'] if t['task_id'] == rec['target_id'])
    perm = data['perm'].astype(np.int64)
    tab = data['tables'][tok].astype(np.int64)
    sets = {k: (D.to_bits(x), D.to_bits(tab[x])) for k, x in (('test', perm[c['n_train']:]),
                                                             ('train', perm[:c['n_train']]))}
    pred = np.load(Path(res) / 'predictions' / f'{name}.npz')
    w = dict(bits=0.0, train_bits=0.0, acc=0.0, pred=0.0)
    for i, row in enumerate(rec['rows']):
        m = M.load_npz(M.build_model(c), Path(res) / 'checkpoints' / name / f"T_{row['T']}.npz")
        ev = M.evaluate(m, sets['test'][0], tok, sets['test'][1])
        tr = M.evaluate(m, sets['train'][0], tok, sets['train'][1])
        w['bits'] = max(w['bits'], abs(ev['excess_bits'] - row['excess_bits']))
        w['train_bits'] = max(w['train_bits'], abs(tr['excess_bits'] - row['train_bits']))
        w['acc'] = max(w['acc'], abs(ev['accuracy'] - row['accuracy']), abs(tr['accuracy'] - row['train_accuracy']))
        w['pred'] = max(w['pred'], float(np.max(np.abs(ev['per_example_bits'].astype(np.float32) - pred['test'][i]))),
                        float(np.max(np.abs(tr['per_example_bits'].astype(np.float32) - pred['train'][i]))))
    return name, w, len(rec['rows'])


def _retrain(args):
    """One secondary-arm cell retrained through run.py --only into scratch; bitwise comparison of predictions."""
    res, arm, seed, tid, alpha, scratch = args
    out = Path(scratch) / f'{arm}_{seed}_{tid}_{alpha:.2f}'
    env = dict(os.environ, OMP_NUM_THREADS='1')
    subprocess.run([sys.executable, str(HERE / 'run.py'), '--only', f'{arm}:{seed}:{tid}:{alpha}', '--output',
                    str(out)], check=True, capture_output=True, env=env)
    name = R.cell_name(seed, tid, alpha)
    a = np.load(Path(res) / R.ARMS[arm] / 'predictions' / f'{name}.npz')
    b = np.load(out / 'predictions' / f'{name}.npz')
    same = all(np.array_equal(a[k], b[k]) for k in ('test', 'train', 'T'))
    ra = json.loads((Path(res) / R.ARMS[arm] / 'cells' / f'{name}.json').read_text())['rows']
    rb = json.loads((out / 'cells' / f'{name}.json').read_text())['rows']
    rows_same = [{k: v for k, v in r.items() if k != 'wall_seconds'} for r in ra] == \
                [{k: v for k, v in r.items() if k != 'wall_seconds'} for r in rb]
    return dict(arm=arm, seed=seed, target_id=tid, alpha=alpha, predictions_bitwise_equal=bool(same),
                rows_equal=bool(rows_same))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    res = Path(args.results).resolve()
    c = R.load_config()
    man = json.loads((res / 'manifest.json').read_text())
    tasks = json.loads((res / 'tasks.json').read_text())
    n_tasks = len(tasks)
    V = {}
    ok = {}

    # 1. source and config hashes
    now = {s: R.sha256_file(HERE / s) for s in TRAINING_SOURCES + ANALYSIS_SOURCES}
    ok['1_training_sources_match_manifest'] = all(now[s] == man['sources'][s] for s in TRAINING_SOURCES)
    V['analysis_source_sha256'] = {s: now[s] for s in ANALYSIS_SOURCES}

    # 2. BFS from scratch: level counts, every task's K, canonical program and n_shortest
    bfs = dsl.bfs_enumerate(c)
    V['2_bfs_level_counts'] = bfs['level_counts']
    V['2_bfs_seconds'] = round(bfs['seconds'], 1)
    ok['2_bfs_level_counts'] = bfs['level_counts'] == c['bfs_level_counts']
    good = 0
    for t in tasks:
        tab = np.array(t['truth_table'], np.uint16)
        K, g = dsl.exact_k(tab, bfs)
        prog = [dsl.NAMES[int(p)] for p in bfs['canonical'][g] if p >= 0]
        good += (K == t['K'] and dsl.k_bits(K) == t['K_bits'] and prog == t['definition']['primitives']
                 and int(bfs['n_shortest'][g]) == t['n_shortest_programs']
                 and np.array_equal(dsl.apply_program(prog), tab))
    V['2_tasks_K_program_nshortest'] = f'{good}/{n_tasks}'
    ok['2_tasks_K_program_nshortest'] = good == n_tasks

    # 3. tasks regenerated from the task seed (every filter, covariate and proxy recomputed)
    split = D.make_split(c)
    regen, _ = dsl.select_tasks(c, bfs, split['train_idx'], PX.proxies)
    del bfs
    ok['3_tasks_regenerated_identical'] = json.dumps(regen, sort_keys=True) == json.dumps(tasks, sort_keys=True)
    cov = 0
    for t in tasks:
        tab = np.array(t['truth_table'], np.uint16)
        f = dsl.features(tab, c['rare_rule_count'])
        cov += (abs(f['sensitivity'] - t['sensitivity']) < 1e-12 and
                abs(dsl.fourier_sensitivity(tab) - t['sensitivity']) < 1e-12 and
                f['is_affine'] == t['is_affine'] and f['local_rule_entropy'] == t['local_rule_entropy'])
    V['3_covariates_recomputed'] = f'{cov}/{n_tasks}'
    ok['3_covariates_recomputed'] = cov == n_tasks

    # 4. split, data/ and mixes regenerated
    sp = json.loads((res / 'split.json').read_text())
    ok['4_split_identical'] = sp['perm'] == split['perm'].tolist() and sp['attempts'] == split['attempts']
    tmp = Path(tempfile.mkdtemp(prefix='exp12_verify_'))
    D.build_data(c, tasks, split, tmp)
    same = 0
    for t in tasks:
        for part in ('train', 'test'):
            a = np.load(res / 'data' / f"{t['task_id']}_{part}.npz")
            b = np.load(tmp / 'data' / f"{t['task_id']}_{part}.npz")
            same += sorted(a.files) == sorted(b.files) and all(
                np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a.files)
    V['4_data_files_identical'] = f'{same}/{2 * n_tasks}'
    ok['4_data_files_identical'] = same == 2 * n_tasks
    mixes = json.loads((res / 'mixes.json').read_text())
    mix_ok = n_mix = 0
    for arm, cells in mixes.items():
        for name, m in cells.items():
            mm = D.assemble_mix(c, tasks, m['target_id'], m['alpha'], m['seed'], n_train=m['n_train'])
            n_mix += 1
            mix_ok += (json.loads(json.dumps(mm['counts'])) == m['counts']
                       and list(mm['data_order_seed']) == list(m['data_order_seed'])
                       and json.loads(json.dumps(mm['target_passes'])) == m['target_passes'])
    V['4_mixes_regenerated_identical'] = f'{mix_ok}/{n_mix}'
    ok['4_mixes_regenerated_identical'] = mix_ok == n_mix and n_mix > 0

    # 5-6. every main checkpoint replayed
    names = sorted(mixes['main'])
    W = dict(bits=0.0, train_bits=0.0, acc=0.0, pred=0.0)
    n_rows = 0
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for name, w, n in ex.map(_replay, [(str(res), nm) for nm in names], chunksize=4):
            W = {k: max(W[k], w[k]) for k in W}
            n_rows += n
    with open(res / 'metrics.csv') as f:
        n_main = sum(1 for r in csv.DictReader(f) if r['arm'] == 'main')
    V['5_checkpoints_replayed'] = n_rows
    V['5_main_metrics_rows'] = n_main
    V['5_max_abs_diff_excess_bits'] = W['bits']
    V['5_max_abs_diff_train_bits'] = W['train_bits']
    V['5_max_abs_diff_accuracy'] = W['acc']
    V['6_max_abs_diff_predictions'] = W['pred']
    ok['5_checkpoints_replayed'] = (n_rows == n_main and W['bits'] < 1e-9 and W['train_bits'] < 1e-9
                                    and W['acc'] == 0)
    ok['6_predictions_match'] = W['pred'] == 0

    # 7. 8 secondary-arm cells retrained into scratch, chosen with the analysis seed
    sec = sorted((arm, m['seed'], m['target_id'], m['alpha']) for arm, cells in mixes.items() if arm != 'main'
                 for m in cells.values())
    if sec:
        pick = np.random.default_rng(c['analysis_seed']).choice(len(sec), min(c['secondary_verify_cells'], len(sec)),
                                                                replace=False)
        jobs = [(str(res), *sec[i], str(tmp / 'retrain')) for i in sorted(pick)]
        with cf.ProcessPoolExecutor(args.workers) as ex:
            rr = list(ex.map(_retrain, jobs))
        V['7_secondary_retrained'] = rr
        ok['7_secondary_bitwise_equal'] = all(r['predictions_bitwise_equal'] and r['rows_equal'] for r in rr)
    else:
        V['7_secondary_retrained'] = 'no secondary-arm cells ran'

    # 8-10. T*, both fitters and every gate recomputed from scratch in a scratch copy
    work = tmp / 'analysis'
    work.mkdir()
    for f in ('metrics.csv', 'tasks.json'):
        shutil.copy(res / f, work / f)
    if (res / 'predictions').exists():
        os.symlink(res / 'predictions', work / 'predictions')
    h, fits = AN.stage1(c, work, tasks)
    A2 = AN.stage2(c, work, tasks)
    ok['8_tstar_identical'] = (work / 'tstar.csv').read_bytes() == (res / 'tstar.csv').read_bytes()
    ok['9_forecasts_hash_identical'] = h == json.loads((res / 'fits.json').read_text())['forecasts_frozen_sha256']
    ok['9_fitters_agree'] = fits['all_fitters_agree']
    orig = json.loads((res / 'analysis.json').read_text())
    ok['10_gates_identical'] = A2['gates'] == orig['gates']
    keys = [('G1', 'estimate'), ('G2', 'estimate'), ('G3', 'kappa'), ('G3', 'gamma1'), ('G4', 'estimate_bits'),
            ('G4', 'E_mf'), ('G5', 'estimate'), ('G6', 'a1'), ('G7', 'estimate'), ('G8a', 'estimate'),
            ('G8b', 'estimate')]
    V['10_key_numbers'] = {f'{k}.{f}': [A2[k][f], orig[k][f]] for k, f in keys}
    ok['10_key_numbers_identical'] = all(
        np.isclose(A2[k][f], orig[k][f], rtol=1e-12, atol=0) or (np.isinf(A2[k][f]) and np.isinf(orig[k][f]))
        or (np.isnan(A2[k][f]) and np.isnan(orig[k][f])) for k, f in keys)
    shutil.rmtree(tmp, ignore_errors=True)

    V['checks'] = ok
    V['status'] = 'PASS' if all(ok.values()) else 'FAIL'
    V['seconds'] = round(time.time() - t0, 1)
    (res / 'verification.json').write_text(json.dumps(V, indent=1, default=float))
    print(json.dumps(V, indent=1, default=float))
    return 0 if V['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
