"""Experiment 17 verification (PROTOCOL Section 6). Exit 0 and 'VERIFY PASS' only if every check passes.

Usage: verify.py [--output results] [--no-replay]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402


def cmp(a, b, tol, path=''):
    """Recursive comparison of two JSON values; returns a list of mismatches."""
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f'{path}: keys differ {sorted(set(a) ^ set(b))[:5]}']
        for k in a:
            bad += cmp(a[k], b[k], tol, f'{path}/{k}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} vs {len(b)}']
        for i, (x, y) in enumerate(zip(a, b)):
            bad += cmp(x, y, tol, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or abs(a - b) <= tol * max(1.0, abs(a))):
            bad.append(f'{path}: {a} vs {b}')
    elif a != b:
        bad.append(f'{path}: {a!r} vs {b!r}')
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(HERE / 'results'))
    ap.add_argument('--no-replay', action='store_true')
    args = ap.parse_args()
    out = Path(args.output)
    c = R.load_config()
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
        ok &= bool(cond)
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = R.grid_runs(c, tasks)
    present = [r for r in runs if (out / 'runs' / f'{R.run_name(*r)}.json').exists()]
    check('all gated runs present', len(present) == len(runs) == 96, f'{len(present)}/{len(runs)}')
    recs = [json.loads((out / 'runs' / f'{R.run_name(*r)}.json').read_text()) for r in runs]
    check('all runs finite', all(r['finite'] for r in recs))
    check('checkpoints as frozen', all(sorted({row['T'] for row in r['rows']}) == c['T_checkpoints'] for r in recs))
    # (a) metrics.csv rebuilt from the run files
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'runs').mkdir()
        for r in runs:
            shutil.copy2(out / 'runs' / f'{R.run_name(*r)}.json', td / 'runs')
        R.aggregate(td, runs)
        for name in ('metrics.csv', 'pool_loss.csv', 'mixes.json'):
            same = (td / name).read_bytes() == (out / name).read_bytes()
            check(f'{name} rebuilt identically', same)
    # alpha schedule: target documents seen equal floor(m alpha) at every checkpoint
    bad = 0
    for r in recs:
        for row in r['rows']:
            if r['target_id'] != '-' and row['slice'] == 'all':
                m = row['T'] // c['doc_len']
                exp = m if r['alpha'] >= 1 else (m * int(round(r['alpha'] * 100))) // 100
                bad += row['target_docs_seen'] != exp
            if r['target_id'] == '-':
                bad += row['target_docs_seen'] != 0
    check('alpha schedule exact in every run', bad == 0, f'{bad} bad rows')
    # (b) analysis re-run
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name in ('metrics.csv', 'pool_loss.csv', 'tasks.json'):
            shutil.copy2(out / name, td / name)
        subprocess.run([sys.executable, str(HERE / 'analyze.py'), '--output', str(td)], check=True,
                       capture_output=True)
        a, b = json.loads((out / 'analysis.json').read_text()), json.loads((td / 'analysis.json').read_text())
        mism = cmp(a, b, 1e-9)
        check('analysis.json reproduced', not mism, '; '.join(mism[:5]))
    # (c) sources
    man = json.loads((out / 'manifest.json').read_text())
    srcs = {s: R.sha256_file(HERE / s) for s in R.SOURCES}
    check('training sources match manifest', all(man['sources'].get(s) == h for s, h in srcs.items()))
    # (d) replay two gated runs to the first checkpoint
    if not args.no_replay:
        refs = R.load_refs(out, tasks)
        T1 = c['T_checkpoints'][:1]
        for run in (runs[0], next(r for r in runs if r[2] == '-' and r[1] == 'hard')):
            rec = json.loads((out / 'runs' / f'{R.run_name(*run)}.json').read_text())
            new = R.train_run(c, tasks, run, out, T1, refs)
            old = {(x['target_id'], x['slice']): x['excess_bits'] for x in rec['rows'] if x['T'] == T1[0]}
            nw = {(x['target_id'], x['slice']): x['excess_bits'] for x in new['rows']}
            d = max(abs(old[k] - nw[k]) for k in nw)
            check(f'replay {R.run_name(*run)} to T={T1[0]}', set(old) == set(nw) and d <= c['verify_tol_bits'],
                  f'max diff {d:.2e} bits')
    # (e) RESULTS tables regenerated
    res = HERE.parent / 'RESULTS_exp17.md'
    if res.exists():
        import report
        gen = report.tables(json.loads((out / 'analysis.json').read_text()), c, tasks)
        check('RESULTS tables match analysis.json', gen in res.read_text())
    print('VERIFY PASS' if ok else 'VERIFY FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
