"""Experiment 18 verification (PROTOCOL Section 8). Prints PASS or the failures, and writes results/verify.json.

Usage (from expt_v1/): verify.py [--results results] [--workers 4] [--retrain 3]
Checks:
  1. the npz files hash to data_sha256.json;
  2. which runs of each phase are present (a missing run is reported; a phase analysed as complete must have all);
  3. per run: checkpoints as configured, aligned / target example counts = floor(T alpha), every task's K recomputed;
  4. per run: the final weights are reloaded and the last checkpoint's evaluation (every head's error, cross-entropy,
     eps, training-distribution error and probe errors) recomputed and compared to 1e-12;
  5. a few runs retrained end to end from seeds: every row identical (determinism);
  6. analyze.py rerun into a scratch file: analysis.json identical; the independent fitter agreed;
  7. report.py rerun into a scratch file: RESULTS markdown identical, so every number in it comes from analysis.json.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import run as R  # noqa: E402
import tasks as TK  # noqa: E402

TOL = 1e-12


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def recompute_last(path):
    """Reload a run's final weights and recompute its last checkpoint's evaluation row."""
    import torch
    from model import PairNet, WIDTH
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    res = json.load(open(path))
    s = res['spec']
    ds = s['dataset']
    d = D.load(ds)
    model = PairNet(d['x_train'].shape[1], WIDTH[ds], {t: TK.n_classes(t) for t in res['run_tasks']}, s['seed'])
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(path), 'ckpt', res['run_id'] + '.pt')))
    model.eval()
    xe, ye, _ = D.eval_pairs(ds)
    lp = R.logits_all(model, xe, d)
    row = res['rows'][-1]
    bad = []
    for t in res['run_tasks']:
        err, ce = R.score(lp[t], TK.label(t, ye[:, 0], ye[:, 1]))
        for k, v in ((f'err__{t}', err), (f'ce_bits__{t}', ce), (f'eps__{t}', err / TK.chance_error(t))):
            if abs(row[k] - v) > TOL:
                bad.append((k, row[k], v))
    if f"traindist_err__{res['run_tasks'][0]}" in row:
        xm, ym = D.traindist_pairs(ds, s.get('mismatch', 'rot90'), s['seed'])
        t = res['run_tasks'][0]
        v = R.score(R.logits_all(model, xm, d)[t], TK.label(t, ym[:, 0], ym[:, 1]))[0]
        if abs(row[f'traindist_err__{t}'] - v) > TOL:
            bad.append(('traindist', row[f'traindist_err__{t}'], v))
    pk = [k for k in row if k.startswith('probe_err__')]
    if pk:
        xp, yp, _ = D.probe_pairs(ds, s['seed'])
        ztr, zte = R.features(model, xp, d), R.features(model, xe, d)
        for k in pk:
            t = k.split('__')[1]
            v = R.probe_err(ztr, TK.label(t, yp[:, 0], yp[:, 1]), zte, TK.label(t, ye[:, 0], ye[:, 1]),
                            TK.n_classes(t), R.CFG['H']['probe_l2'], R.CFG['H']['probe_iters'])
            if abs(row[k] - v) > TOL:
                bad.append((k, row[k], v))
    return res['run_id'], bad


def structural(res):
    s = res['spec']
    bad = []
    Ts = list(R.CFG['T_checkpoints'][s['dataset']])
    got = [r['T'] for r in res['rows'] if r['T'] > 0]
    if got != Ts:
        bad.append(('checkpoints', got))
    for T in Ts:
        if s['arm'] in ('main', 'mis'):
            want = (T * Fraction(str(s['alpha']))).__floor__()
            if res['aligned_count_at_T'][str(T)] != want:
                bad.append(('aligned_count', T))
        if s['arm'] == 'H' and s['mode'] == 'joint':
            want = (T * Fraction(str(s['alpha']))).__floor__()
            if res['target_count_at_T'][str(T)] != want:
                bad.append(('target_count', T))
    for t, inf in res['task_info'].items():
        if abs(inf['K_bits'] - TK.k_bits(t)) > 1e-12:
            bad.append(('K', t))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--retrain', type=int, default=3)
    a = ap.parse_args()
    fails, rep = [], {}
    # 1
    want = json.load(open(os.path.join(HERE, 'data_sha256.json')))['npz']
    for f, h in want.items():
        if sha(os.path.join(D.DATA, f)) != h:
            fails.append(f'data hash {f}')
    # 2-3
    present = {}
    for ph in R.ORDER:
        specs = R.PHASES[ph]()
        have = [s for s in specs if os.path.exists(R.out_path(s))]
        present[ph] = dict(expected=len(specs), present=len(have))
        for s in have:
            b = structural(json.load(open(R.out_path(s))))
            if b:
                fails.append(f'{R.rid(s)} structural {b}')
    rep['runs'] = present
    an = json.load(open(os.path.join(a.results, 'analysis.json')))
    for ds, A in an['main'].items():
        if A['complete'] and present[f'{ds}_main']['present'] != present[f'{ds}_main']['expected']:
            fails.append(f'{ds} analysed as complete but runs missing')
    # 4
    paths = [R.out_path(s) for ph in R.ORDER for s in R.PHASES[ph]() if os.path.exists(R.out_path(s))]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.workers) as P:
        nbad = 0
        for rid, bad in P.imap_unordered(recompute_last, paths, chunksize=4):
            if bad:
                nbad += 1
                fails.append(f'{rid} recompute {bad[:2]}')
    rep['recomputed_last_checkpoint'] = dict(runs=len(paths), mismatched=nbad)
    # 5
    pick = [s for s in R.PHASES['mnist_main']() if s['alpha'] == 0.4 and s['seed'] == R.CFG['seeds'][0]][:1]
    pick += [s for s in R.PHASES['mnist_H']() if s['mode'] == 'joint' and s['seed'] == R.CFG['seeds'][1]][:1]
    pick += [s for s in R.PHASES['mis']() if s['seed'] == R.CFG['seeds'][1]][:1]
    pick = [s for s in pick if os.path.exists(R.out_path(s))][:a.retrain]
    det = []
    for s in pick:
        new = R.train(s)
        old = json.load(open(R.out_path(s)))
        same = new['rows'] == old['rows']
        det.append(dict(run=R.rid(s), identical=same))
        if not same:
            fails.append(f'{R.rid(s)} not deterministic')
    rep['retrained'] = det
    # 6
    tmp = tempfile.mkdtemp()
    t_an = os.path.join(tmp, 'analysis.json')
    subprocess.run([sys.executable, os.path.join(HERE, 'analyze.py'), '--results', a.results, '--out', t_an],
                   check=True, capture_output=True)
    if json.load(open(t_an)) != an:
        fails.append('analyze rerun differs from analysis.json')
    for ds, A in an['main'].items():
        if not A['independent_fit']['agree']:
            fails.append(f'{ds} independent fitter disagrees')
    # 7
    t_md = os.path.join(tmp, 'RESULTS.md')
    subprocess.run([sys.executable, os.path.join(HERE, 'report.py'), '--analysis', os.path.join(a.results, 'analysis.json'),
                    '--md', t_md, '--no-figures'], check=True, capture_output=True)
    md = os.path.join(HERE, '..', 'RESULTS_vision.md')
    if open(t_md).read() != open(md).read():
        fails.append('RESULTS_vision.md differs from a fresh report.py render of analysis.json')
    rep['fails'] = fails
    rep['verdict'] = 'PASS' if not fails else 'FAIL'
    json.dump(rep, open(os.path.join(a.results, 'verify.json'), 'w'), indent=1)
    print(rep['verdict'])
    for f in fails[:30]:
        print(' -', f)


if __name__ == '__main__':
    main()
