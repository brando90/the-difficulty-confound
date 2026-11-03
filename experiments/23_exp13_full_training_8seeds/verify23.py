"""Experiment 23 verification (PROTOCOL Section 8). Writes OUT_ROOT/verify_log.json; exit 1 if a gated check fails.

Usage: verify23.py --out OUT_ROOT [--quick]      OUT_ROOT holds e13/ and e17b/ (aggregated and analysed);
                                                 --quick skips rerunning Experiment 13's analysis (about an hour)
"""
import argparse
import csv
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
import run23 as R  # noqa: E402

AGG_FILES = {'e13': ['metrics.csv', 'cooldown.csv', 'val.csv', 'per_position.csv', 'distractor_loss.csv', 'mixes.json'],
             'e17b': ['metrics.csv', 'cooldown.csv', 'val.csv', 'per_position.csv', 'pool_loss_all.csv',
                      'pool_loss.csv', 'mixes.json']}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def completeness(exp, out):
    stop = json.loads((out / 'STOP').read_text())
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = R.grid(exp, tasks)
    miss, nonfinite, bad_ck = [], [], []
    want = [T for T in R.CFG['T_checkpoints'] if T <= stop['T_stop']]
    for r in runs:
        n = R.run_name(exp, r)
        Ts = []
        for j in range(stop['stage'] + 1):
            p = out / 'runs' / n / f'stage_{j}.json'
            if not p.exists():
                miss.append(f'{n}/stage_{j}')
                continue
            rec = json.loads(p.read_text())
            Ts += sorted({x['T'] for x in rec['rows']})
            if not rec['finite']:
                nonfinite.append(n)
        if Ts != want:
            bad_ck.append(n)
    ok = not (miss or nonfinite or bad_ck)
    return dict(ok=ok, note=f'{len(runs)} runs to T_stop {stop["T_stop"]}; missing {len(miss)}, non-finite '
                            f'{len(set(nonfinite))}, wrong checkpoints {len(bad_ck)}', missing=miss[:20])


def rebuild(exp, out):
    before = {f: sha(out / f) for f in AGG_FILES[exp]}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in AGG_FILES[exp] + ['manifest.json']:
            shutil.copy2(out / f, td / f)
        R.cmd_aggregate(exp, out)
        after = {f: sha(out / f) for f in AGG_FILES[exp]}
        shutil.copy2(td / 'manifest.json', out / 'manifest.json')  # keep the original timing record
    diff = [f for f in before if before[f] != after[f]]
    return dict(ok=not diff, note='aggregated files rebuilt byte for byte' if not diff else f'differ: {diff}')


def reload_eval(exp, out, n_runs=4, tol=1e-9):
    stop = json.loads((out / 'STOP').read_text())
    tasks = json.loads((out / 'tasks.json').read_text())
    c = R.base_config(exp)
    refs = R.load_refs(out, tasks)
    runs = R.grid(exp, tasks)
    pick = [runs[i] for i in np.random.default_rng(2323).choice(len(runs), n_runs, replace=False)]
    worst = 0.0
    for r in pick:
        n = R.run_name(exp, r)
        rows = [x for j in range(stop['stage'] + 1)
                for x in json.loads((out / 'runs' / n / f'stage_{j}.json').read_text())['rows'] if x['slice'] == 'all']
        for T in (R.CFG['T_checkpoints'][0], stop['T_stop']):
            m = R.M.load_npz(R.M.build_model(c['model']), out / 'checkpoints' / n / f'T_{T}.npz')
            for tg in R.scored_targets(tasks, r):
                mb = float(R.M.code_lengths(m, refs['docs'][tg['task_id']], c['V']).mean())
                rec = next(x['model_bits'] for x in rows if x['T'] == T and x['target_id'] == tg['task_id'])
                worst = max(worst, abs(mb - rec))
    return dict(ok=worst <= tol, note=f'{n_runs} runs x 2 checkpoints re-evaluated, max |diff| {worst:.2e} bits')


def prep_and_sources(exp, out):
    man = json.loads((out / 'prep_manifest.json').read_text())
    bad = [k for k, h in man.items() if sha(out / k) != h]
    m = json.loads((out / 'manifest.json').read_text())
    bad_src = [s for s, h in m['sources'].items() if (HERE / s).exists() and sha(HERE / s) != h]
    bad_x = [f'{k}/{f}' for k, v in m['source_experiments'].items() for f, h in v.items() if sha(R.SRC[k] / f) != h]
    return dict(ok=not (bad or bad_src or bad_x), note=f'prep files {len(man)}, mismatches {bad[:3]}; sources '
                                                       f'{bad_src}; source experiments {bad_x[:3]}')


def rerun_analysis(exp, out):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        keep = ['metrics.csv', 'tasks.json', 'distractor_loss.csv', 'posterior16.json', 'pool_loss.csv']
        for f in keep:
            if (out / f).exists():
                shutil.copy2(out / f, td / f)
        p = subprocess.run([sys.executable, str(HERE / 'analyze23.py'), '--exp', exp, '--output', str(td)],
                           capture_output=True, text=True)
        if p.returncode:
            return dict(ok=False, note=f'analysis rerun failed: {p.stderr[-500:]}')
        names = ['gates.json', 'eps_star_rule.json', 'eps_1.10/gates.json'] if exp == 'e13' else ['analysis_e17b.json']
        diff = [n for n in names if (td / n).read_text() != (out / n).read_text()]
    return dict(ok=not diff, note='analysis rerun identical' if not diff else f'differ: {diff}')


def replication(exp, out):
    """Source seeds at T <= 2^20 against the source experiments' committed metrics (reported, not gated)."""
    if exp == 'e13':
        src = [R.SRC['exp13'] / 'results' / 'small' / 'metrics.csv']
        key = ('seed', 'target_id', 'alpha', 'T', 'slice')
    else:
        src = [R.SRC['exp17'] / 'results' / 'metrics.csv', R.SRC['exp17b'] / 'results' / 'metrics.csv']
        key = ('seed', 'pool', 'run_target', 'target_id', 'alpha', 'T', 'slice')
    mine = {}
    with open(out / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            mine[tuple(str(float(r[k])) if k == 'alpha' else r[k] for k in key)] = float(r['excess_bits'])
    d = []
    for p in src:
        with open(p) as f:
            for r in csv.DictReader(f):
                k = tuple(str(float(r[k])) if k == 'alpha' else r[k] for k in key)
                if k in mine:
                    d.append(abs(mine[k] - float(r['excess_bits'])))
    d = np.array(d) if d else np.zeros(1)
    return dict(ok=True, note=f'{len(d)} overlapping rows; max |diff| {d.max():.2e}, median {np.median(d):.2e} bits '
                              '(identical streams; differences come only from floating point on other CPUs)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    root = Path(a.out).expanduser()
    checks = {}
    for exp in R.EXPS:
        out = root / exp
        checks[f'{exp}_complete'] = completeness(exp, out)
        checks[f'{exp}_rebuild'] = rebuild(exp, out)
        checks[f'{exp}_reload_eval'] = reload_eval(exp, out)
        checks[f'{exp}_hashes'] = prep_and_sources(exp, out)
        if not (a.quick and exp == 'e13'):
            checks[f'{exp}_analysis_rerun'] = rerun_analysis(exp, out)
        checks[f'{exp}_replication_2^20'] = replication(exp, out)
        for k in [k for k in checks if k.startswith(exp)]:
            print(k, 'PASS' if checks[k]['ok'] else 'FAIL', checks[k]['note'], flush=True)
    ok = all(v['ok'] for v in checks.values())
    (root / 'verify_log.json').write_text(json.dumps(dict(ok=ok, checks=checks), indent=1))
    print('VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
