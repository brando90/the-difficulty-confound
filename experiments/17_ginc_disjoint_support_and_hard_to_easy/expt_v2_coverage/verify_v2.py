"""Experiment 17b verification (PROTOCOL_exp17b.md Section 6). Exit 0 and 'VERIFY PASS' only if every check passes.

Usage: verify_v2.py [--output results] [--no-replay]
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'expt_v1'
sys.path.insert(0, str(HERE))
import run_v2 as RV  # noqa: E402

R = RV.R
sys.path.insert(0, str(V1))
from verify import cmp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(HERE / 'results'))
    ap.add_argument('--no-replay', action='store_true')
    args = ap.parse_args()
    out = Path(args.output)
    c, c2 = R.load_config(), RV.load_c2()
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
        ok &= bool(cond)
    tasks = json.loads((out / 'tasks.json').read_text())
    v1_tasks = json.loads((V1 / 'results' / 'tasks.json').read_text())
    check('concept ids 0-37 equal Experiment 17', tasks[:len(v1_tasks)] == v1_tasks)
    runs = RV.grid_runs_v2(c, c2, tasks)
    present = [r for r in runs if (out / 'runs' / f'{R.run_name(*r)}.json').exists()]
    check('all gated runs present', len(present) == len(runs) == c2['expected_runs'], f'{len(present)}/{len(runs)}')
    recs = [json.loads((out / 'runs' / f'{R.run_name(*r)}.json').read_text()) for r in runs]
    check('all runs finite', all(r['finite'] for r in recs))
    check('checkpoints as frozen', all(sorted({row['T'] for row in r['rows']}) == c['T_checkpoints'] for r in recs))
    # (a) rebuilt aggregates
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'runs').mkdir()
        for r in runs:
            shutil.copy2(out / 'runs' / f'{R.run_name(*r)}.json', td / 'runs')
        R.aggregate(td, runs)
        for name in ('metrics.csv', 'pool_loss.csv', 'mixes.json'):
            check(f'{name} rebuilt identically', (td / name).read_bytes() == (out / name).read_bytes())
    bad = 0
    for r in recs:
        for row in r['rows']:
            if r['target_id'] != '-' and row['slice'] == 'all':
                m = row['T'] // c['doc_len']
                bad += row['target_docs_seen'] != (m * int(round(r['alpha'] * 100))) // 100
            if r['target_id'] == '-':
                bad += row['target_docs_seen'] != 0
    check('alpha schedule exact in every run', bad == 0, f'{bad} bad rows')
    # (b) analysis re-run
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name in ('metrics.csv', 'pool_loss.csv', 'tasks.json'):
            shutil.copy2(out / name, td / name)
        subprocess.run([sys.executable, str(HERE / 'analyze_v2.py'), '--output', str(td)], check=True,
                       capture_output=True)
        a, b = json.loads((out / 'analysis_v2.json').read_text()), json.loads((td / 'analysis_v2.json').read_text())
        mism = cmp(a, b, 1e-9)
        check('analysis_v2.json reproduced', not mism, '; '.join(mism[:5]))
    # (c) sources and the reused Experiment 17 metrics
    man = json.loads((out / 'manifest.json').read_text())
    check('expt_v1 training sources match manifest',
          all(man['sources'].get(s) == R.sha256_file(V1 / s) for s in R.SOURCES))
    check('expt_v2 sources match manifest',
          all(man['sources_v2'].get(s) == R.sha256_file(HERE / s) for s in RV.SOURCES_V2))
    check('Experiment 17 metrics.csv unchanged', man['v1_metrics_sha256'] == R.sha256_file(V1 / 'results' /
                                                                                          'metrics.csv'))
    # (d) replays: one new-pool run and one reused Experiment 17 hard run, with the extended task list
    if not args.no_replay:
        refs = R.load_refs(out, tasks)
        T1 = c['T_checkpoints'][:1]
        cases = [(out, (c['seeds'][0], c2['pool_order'][0], '-', 0.0)),
                 (V1 / 'results', (c['seeds'][1], 'hard', '-', 0.0))]
        for where, run in cases:
            rec = json.loads((where / 'runs' / f'{R.run_name(*run)}.json').read_text())
            new = R.train_run(c, tasks, run, out, T1, refs)
            old = {(x['target_id'], x['slice']): x['excess_bits'] for x in rec['rows'] if x['T'] == T1[0]}
            nw = {(x['target_id'], x['slice']): x['excess_bits'] for x in new['rows']}
            d = max(abs(old[k] - nw[k]) for k in nw)
            check(f'replay {R.run_name(*run)} ({where.parent.name}) to T={T1[0]}',
                  set(old) == set(nw) and d <= c['verify_tol_bits'], f'max diff {d:.2e} bits')
    # (e) RESULTS tables
    res = HERE.parent / 'RESULTS_exp17b.md'
    if res.exists():
        import report_v2
        gen = report_v2.tables(json.loads((out / 'analysis_v2.json').read_text()), c, c2)
        check('RESULTS tables match analysis_v2.json', gen in res.read_text())
    else:
        check('RESULTS_exp17b.md exists', False)
    print('VERIFY PASS' if ok else 'VERIFY FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
