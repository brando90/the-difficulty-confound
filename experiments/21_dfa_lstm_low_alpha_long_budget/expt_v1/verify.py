"""Verification for Experiment 21 (PROTOCOL Section 6).

  verify.py [--results results] [--workers 4]
Writes results/verification.json with status PASS or FAIL and the counts of every check. Steps 1-8 decide PASS.
Step 5 retrains one cell from scratch into a temporary folder; nothing else retrains.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
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
ANALYSIS_SOURCES = ['analyze.py', 'verify.py']


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
        m = M.load_npz(M.build(c), Path(res) / 'checkpoints' / name / f"T_{row['T']}.npz")
        ev = M.evaluate(m, tok, data['test_bits'], data['test_lens'], data['test_labels'][tok])
        worst_bits = max(worst_bits, abs(ev['excess_bits'] - row['excess_bits']))
        worst_acc = max(worst_acc, abs(ev['accuracy'] - row['accuracy']))
        worst_pred = max(worst_pred, float(np.max(np.abs(ev['per_string_bits'] - pred['bits'][i]))))
    return name, worst_bits, worst_acc, worst_pred, len(rec['rows'])


def file_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): R.sha256_file(p) for p in [root / 'tasks.json'] + sorted((root / 'data').glob('*.npz'))}


def same(a, b, tol=1e-12):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    res = Path(args.results).resolve()
    c = R.load_config()
    man = json.loads((res / 'manifest.json').read_text())
    e20 = (HERE / c['exp20_results']).resolve()
    V = {}

    # 1. sources
    now = {s: R.sha256_file(HERE / s) for s in TRAINING_SOURCES + ANALYSIS_SOURCES}
    V['1_training_sources_match_manifest'] = all(now[s] == man['sources'][s] for s in TRAINING_SOURCES)
    V['analysis_source_sha256'] = {s: now[s] for s in ANALYSIS_SOURCES}

    # 2. tasks and data: regenerated from the task seed, and byte-identical to Experiment 20's
    tasks = json.loads((res / 'tasks.json').read_text())
    regen, _ = A.select_tasks(c)
    V['2_tasks_regenerated_identical'] = json.dumps(regen, sort_keys=True) == json.dumps(tasks, sort_keys=True)
    hm = sum(A.hopcroft_minimize(A.DFA.from_json(t['definition'])).n ==
             A.moore_minimize(A.DFA.from_json(t['definition'])).n == t['K'] for t in tasks)
    V['2_hopcroft_moore_n_match'] = f'{hm}/{len(tasks)}'
    with tempfile.TemporaryDirectory() as tmp:
        tk, _, tb, tl, pools = R.build_all(c)
        R.write_data(Path(tmp), tk, tb, tl, pools)
        h_regen = file_hashes(tmp)
    h_res, h_20 = file_hashes(res), file_hashes(e20)
    V['2_data_files'] = len(h_res)
    V['2_data_regenerated_identical'] = h_regen == h_res
    V['2_data_identical_to_exp20'] = h_res == h_20

    # 3. mixes: all cells regenerate, no pool reuse, task-slot sequence through 2^16 equals Experiment 20's
    cells = R.grid_cells(c, tasks)
    mixes = json.loads((res / 'mixes.json').read_text())
    c20 = json.loads((e20.parent / 'config.json').read_text())
    ok_counts = ok_reuse = ok_prefix = 0
    T_ref = c['verdict']['T_ref']
    for tid, a, s in cells:
        name = R.cell_name(s, tid, a)
        m = D.assemble_mix(c, tasks, tid, a, s)
        ok_counts += name in mixes and mixes[name]['counts'] == m['counts']
        ok_reuse += m['pool_reuse'] == 0
        m20 = D.assemble_mix(c20, tasks, tid, a, s)
        ok_prefix += bool(np.array_equal(m['task_token'][:T_ref], m20['task_token'][:T_ref]))
    V['3_mixes_regenerated_identical'] = f'{ok_counts}/{len(cells)}'
    V['3_no_pool_reuse'] = f'{ok_reuse}/{len(cells)}'
    V['3_task_slots_through_2^16_equal_exp20'] = f'{ok_prefix}/{len(cells)}'

    # 4. replay every checkpoint
    names = [R.cell_name(s, t, a) for t, a, s in cells]
    with cf.ProcessPoolExecutor(args.workers) as ex:
        rep = list(ex.map(_replay, [(str(res), n) for n in names]))
    V['4_checkpoints_replayed'] = int(sum(r[4] for r in rep))
    V['4_max_abs_diff_excess_bits'] = max(r[1] for r in rep)
    V['4_max_abs_diff_accuracy'] = max(r[2] for r in rep)
    V['4_max_abs_diff_predictions'] = max(r[3] for r in rep)

    # 5. retrain the first cell from scratch: training is bitwise deterministic
    tid, a, s = cells[0]
    with tempfile.TemporaryDirectory() as tmp:
        data = R.load_data(res)
        rec = R.train_cell(c, data['tasks'], tid, a, s, tmp, data=data, save_checkpoints='none')
    orig = json.loads((res / 'cells' / f'{R.cell_name(s, tid, a)}.json').read_text())
    V['5_retrained_cell'] = R.cell_name(s, tid, a)
    V['5_retrain_max_abs_diff_excess_bits'] = max(abs(x['excess_bits'] - y['excess_bits'])
                                                   for x, y in zip(rec['rows'], orig['rows']))

    # 6. metrics.csv equals the cell records
    rows = list(csv.DictReader(open(res / 'metrics.csv')))
    by = {(int(r['seed']), r['target_id'], round(float(r['alpha']), 2), int(r['T'])): float(r['excess_bits'])
          for r in rows}
    n_eq = 0
    for t, a, s in cells:
        for row in json.loads((res / 'cells' / f'{R.cell_name(s, t, a)}.json').read_text())['rows']:
            n_eq += by.get((s, t, round(a, 2), row['T'])) == row['excess_bits']
    V['6_metrics_rows'] = len(rows)
    V['6_metrics_rows_equal_cells'] = n_eq == len(rows) == len(cells) * len(c['T_checkpoints'])

    # 7. analysis recomputed identically, and key numbers recomputed independently (plain means over runs)
    A0 = json.loads((res / 'analysis.json').read_text())
    A1 = json.loads(json.dumps(AN.compute(c, res)))
    V['7_analysis_identical'] = same(A0, A1)
    M = AN.read_metrics(res / 'metrics.csv')
    v = c['verdict']
    indep_ok = True
    for key, x in A0['cells'].items():
        sel = [(k, r) for k, r in M.items() if k[2] == x['alpha'] and next(iter(r.values()))['K'] == x['n']]
        e_ref = np.mean([r[v['T_ref']]['excess_bits'] for _, r in sel])
        e_end = np.mean([r[v['T_end']]['excess_bits'] for _, r in sel])
        cross = sum(r[v['T_end']]['excess_bits'] <= v['cross_eps'] for _, r in sel)
        indep_ok &= (len(sel) == x['runs'] and abs(e_ref - x['e_ref']) < 1e-12 and abs(e_end - x['e_end']) < 1e-12
                     and abs((e_ref - e_end) - x['drop']) < 1e-12 and cross == x['crossers_end']
                     and AN.cell_verdict(x['drop_ci'], cross, v) == x['verdict'])
    V['7_independent_cell_numbers_agree'] = bool(indep_ok)

    # 8. distractor losses (descriptive): replay 8 cells
    dpath = res / 'distractors.json'
    if dpath.exists():
        Dj = json.loads(dpath.read_text())
        pick = names[:: max(1, len(names) // 8)][:8]
        Dr = AN.distractor_losses(c, res, args.workers, set(pick))
        V['8_distractor_cells'] = len(Dj)
        V['8_distractor_replay_max_abs_diff'] = max(abs(Dr[n][T][d] - Dj[n][T][d]) for n in Dr for T in Dr[n]
                                                    for d in Dr[n][T])
    ok = (V['1_training_sources_match_manifest'] and V['2_tasks_regenerated_identical']
          and V['2_hopcroft_moore_n_match'] == f'{len(tasks)}/{len(tasks)}' and V['2_data_regenerated_identical']
          and V['2_data_identical_to_exp20']
          and all(V[k] == f'{len(cells)}/{len(cells)}' for k in
                  ('3_mixes_regenerated_identical', '3_no_pool_reuse', '3_task_slots_through_2^16_equal_exp20'))
          and V['4_checkpoints_replayed'] == len(cells) * len(c['T_checkpoints'])
          and max(V['4_max_abs_diff_excess_bits'], V['4_max_abs_diff_accuracy'],
                  V['4_max_abs_diff_predictions']) < 1e-9
          and V['5_retrain_max_abs_diff_excess_bits'] < 1e-9 and V['6_metrics_rows_equal_cells']
          and V['7_analysis_identical'] and V['7_independent_cell_numbers_agree']
          and (not dpath.exists() or V['8_distractor_replay_max_abs_diff'] < 1e-9))
    V['status'] = 'PASS' if ok else 'FAIL'
    V['seconds'] = round(time.time() - t0, 1)
    (res / 'verification.json').write_text(json.dumps(V, indent=1))
    print(json.dumps(V, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
