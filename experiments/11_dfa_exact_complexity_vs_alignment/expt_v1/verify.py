"""Replay verification for Experiment 11 (PROTOCOL Section 8). Never retrains.

  verify.py [--results results] [--workers 4]
Writes results/verification.json with status PASS or FAIL and the counts of every check.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze as AN  # noqa: E402
import data as D  # noqa: E402
import dfa as A  # noqa: E402
import run as R  # noqa: E402

TRAINING_SOURCES = ['config.json', 'dfa.py', 'data.py', 'model.py', 'run.py']
ANALYSIS_SOURCES = ['laws.py', 'independent_fit.py', 'analyze.py', 'verify.py', 'report.py']


def _replay(args):
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
    pred = np.load(Path(res) / 'predictions' / f'{name}.npz')
    worst_bits = worst_acc = worst_pred = 0.0
    for i, row in enumerate(rec['rows']):
        m = M.load_npz(M.build_model(c), Path(res) / 'checkpoints' / name / f"T_{row['T']}.npz")
        ev = M.evaluate(m, tok, data['test_bits'], data['test_lens'], data['test_labels'][tok])
        worst_bits = max(worst_bits, abs(ev['excess_bits'] - row['excess_bits']))
        worst_acc = max(worst_acc, abs(ev['accuracy'] - row['accuracy']))
        worst_pred = max(worst_pred, float(np.max(np.abs(ev['per_string_bits'] - pred['bits'][i]))))
    return name, worst_bits, worst_acc, worst_pred, len(rec['rows'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    res = Path(args.results)
    c = R.load_config()
    man = json.loads((res / 'manifest.json').read_text())
    V = {}

    # 1. source and config hashes
    now = {s: R.sha256_file(HERE / s) for s in TRAINING_SOURCES + ANALYSIS_SOURCES}
    V['1_training_sources_match_manifest'] = all(now[s] == man['sources'][s] for s in TRAINING_SOURCES)
    V['analysis_source_sha256'] = {s: now[s] for s in ANALYSIS_SOURCES}

    # 2-3. tasks regenerated, n by Hopcroft and Moore, covariates exact + Monte Carlo
    tasks = json.loads((res / 'tasks.json').read_text())
    regen, _ = A.select_tasks(c)
    V['2_tasks_regenerated_identical'] = json.dumps(regen, sort_keys=True) == json.dumps(tasks, sort_keys=True)
    hm = mc_ok = 0
    for t in tasks:
        d = A.DFA.from_json(t['definition'])
        hm += A.hopcroft_minimize(d).n == A.moore_minimize(d).n == t['K']
        mc = A.monte_carlo_covariates(d, np.random.default_rng(t['task_token']), 20000, c['length_min'],
                                      c['length_max'])
        cov = A.covariates(d, c)
        mc_ok += (abs(mc['acceptance'][0] - cov['acceptance_rate']) < 4 * mc['acceptance'][1] and
                  abs(mc['sensitivity'][0] - cov['sensitivity']) < 4 * mc['sensitivity'][1] and
                  cov['sensitivity'] == t['sensitivity'] and cov['visit_freq'] == t['visit_freq'])
    V['2_hopcroft_moore_n_match'] = f'{hm}/{len(tasks)}'
    V['3_covariates_exact_and_monte_carlo'] = f'{mc_ok}/{len(tasks)}'

    # 4. data and mixes regenerated
    tb, tl = D.build_test_set(c)
    same = 0
    for t in tasks:
        b, l = D.build_pool(c, t)
        ztr = np.load(res / 'data' / f"{t['task_id']}_train.npz")
        zte = np.load(res / 'data' / f"{t['task_id']}_test.npz")
        same += (np.array_equal(ztr['bits'], b) and np.array_equal(ztr['length'], l)
                 and np.array_equal(ztr['label'], D.labels_for(t, b, l)) and np.array_equal(zte['bits'], tb)
                 and np.array_equal(zte['length'], tl) and np.array_equal(zte['label'], D.labels_for(t, tb, tl)))
    V['4_data_regenerated_identical'] = f'{same}/{len(tasks)}'
    mixes = json.loads((res / 'mixes.json').read_text())
    mix_ok = 0
    for name, m in mixes.items():
        mm = D.assemble_mix(c, tasks, m['target_id'], m['alpha'], m['seed'])
        mix_ok += mm['counts'] == m['counts'] and list(mm['data_order_seed']) == m['data_order_seed']
    V['4_mixes_regenerated_identical'] = f'{mix_ok}/{len(mixes)}'

    # 5-6. checkpoints replayed
    names = sorted(mixes)
    wb = wa = wp = 0.0
    n_rows = 0
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for name, b, a, p, n in ex.map(_replay, [(str(res), nm) for nm in names], chunksize=4):
            wb, wa, wp, n_rows = max(wb, b), max(wa, a), max(wp, p), n_rows + n
    V['5_checkpoints_replayed'] = n_rows
    V['5_max_abs_diff_excess_bits'] = wb
    V['5_max_abs_diff_accuracy'] = wa
    V['6_max_abs_diff_predictions'] = wp
    with open(res / 'metrics.csv') as f:
        n_metrics = sum(1 for _ in csv.DictReader(f))
    V['5_metrics_rows'] = n_metrics

    # 7-9. T*, both fitters and gates recomputed from scratch in a scratch copy
    tmp = Path(tempfile.mkdtemp(prefix='exp11_verify_'))
    for f in ('metrics.csv', 'tasks.json'):
        shutil.copy(res / f, tmp / f)
    h, fits = AN.stage1(c, tmp, tasks)
    A2 = AN.stage2(c, tmp, tasks)
    V['7_tstar_identical'] = (tmp / 'tstar.csv').read_bytes() == (res / 'tstar.csv').read_bytes()
    V['8_forecasts_hash_identical'] = h == json.loads((res / 'fits.json').read_text())['forecasts_frozen_sha256']
    V['8_fitters_agree'] = fits['all_fitters_agree']
    orig = json.loads((res / 'analysis.json').read_text())
    V['9_gates_identical'] = A2['gates'] == orig['gates']
    keys = [('G1', 'estimate'), ('G2', 'estimate'), ('G3', 'kappa'), ('G3', 'gamma1'), ('G4', 'estimate_bits'),
            ('G5', 'estimate'), ('G6', 'a1'), ('G7', 'estimate')]
    V['9_key_numbers_identical'] = all(np.isclose(A2[k][f], orig[k][f], rtol=1e-12, atol=0) or
                                       (np.isinf(A2[k][f]) and np.isinf(orig[k][f])) for k, f in keys)
    shutil.rmtree(tmp, ignore_errors=True)

    status = (V['1_training_sources_match_manifest'] and V['2_tasks_regenerated_identical']
              and V['2_hopcroft_moore_n_match'] == f'{len(tasks)}/{len(tasks)}'
              and V['3_covariates_exact_and_monte_carlo'] == f'{len(tasks)}/{len(tasks)}'
              and V['4_data_regenerated_identical'] == f'{len(tasks)}/{len(tasks)}'
              and V['4_mixes_regenerated_identical'] == f'{len(mixes)}/{len(mixes)}'
              and V['5_checkpoints_replayed'] == n_metrics and wb < 1e-9 and wa == 0 and wp == 0
              and V['7_tstar_identical'] and V['8_forecasts_hash_identical'] and V['8_fitters_agree']
              and V['9_gates_identical'] and V['9_key_numbers_identical'])
    V['status'] = 'PASS' if status else 'FAIL'
    V['seconds'] = round(time.time() - t0, 1)
    (res / 'verification.json').write_text(json.dumps(V, indent=1))
    print(json.dumps(V, indent=1))
    return 0 if status else 1


if __name__ == '__main__':
    sys.exit(main())
