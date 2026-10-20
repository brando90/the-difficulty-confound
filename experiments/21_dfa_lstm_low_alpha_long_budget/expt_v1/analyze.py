"""Experiment 21 analysis: floor or late learning at alpha in {0.05, 0.15} (PROTOCOL Sections 4-5).

  analyze.py [--results results]                 writes results/analysis.json (reads metrics.csv, Exp 20 metrics)
  analyze.py [--results results] --distractors   writes results/distractors.json from saved checkpoints

Everything here is fixed at the freeze commit. Unit of resampling: the training seed (8 blocks, each holding the
4 targets of a level), as in Experiments 11 and 20.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def read_metrics(path):
    """{(seed, target_id, alpha): {T: row}} with numeric fields parsed."""
    out = {}
    for r in csv.DictReader(open(path)):
        key = (int(r['seed']), r['target_id'], round(float(r['alpha']), 2))
        out.setdefault(key, {})[int(r['T'])] = dict(K=int(r['K']), excess_bits=float(r['excess_bits']),
                                                     accuracy=float(r['accuracy']),
                                                     target_examples_seen=int(r['target_examples_seen']))
    return out


def t_star(Ts, eps, eps_star):
    """First crossing with log-log interpolation (Experiment 11's rule); (T*, flag), flag 'right' = censored."""
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


def interp_loglog(Ts, eps, T, floor):
    """eps at T by linear interpolation in (log2 T, log2 max(eps, floor)); T must lie inside [Ts[0], Ts[-1]]."""
    x = np.log2(np.asarray(Ts, float))
    y = np.log2(np.maximum(np.asarray(eps, float), floor))
    assert x[0] <= np.log2(T) <= x[-1]
    return float(2 ** np.interp(np.log2(T), x, y))


class Boot:
    """Seed-block bootstrap: W[b, s] = multiplicity of seed s in draw b."""

    def __init__(self, n_seeds, draws, seed):
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=draws).astype(float)
        self.n = n_seeds

    def mean(self, per_seed):
        return self.W @ np.asarray(per_seed, float) / self.n


def ci(x):
    lo, hi = np.percentile(np.asarray(x, float), [2.5, 97.5])
    return [float(lo), float(hi)]


def marginal_floor(data, task_token):
    """Mean over test strings of H(label | length): the loss of a predictor that knows only task and length."""
    y, L = data['test_labels'][task_token].astype(float), data['test_lens']
    h = np.zeros(len(y))
    for l in np.unique(L):
        p = y[L == l].mean()
        h[L == l] = 0.0 if p in (0.0, 1.0) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(h.mean())


def cell_verdict(drop_ci, crossers, v):
    if drop_ci[0] > v['late_min_drop_lb'] and crossers >= v['late_min_crossers']:
        return 'LATE_LEARNING'
    if drop_ci[1] < v['floor_max_drop_ub'] and crossers <= v['floor_max_crossers']:
        return 'FLOOR_HOLDS'
    if drop_ci[0] > v['descend_min_drop_lb']:
        return 'DESCENDING'
    return 'UNRESOLVED'


def overall_verdict(cells, v):
    dec = {k: x['verdict'] for k, x in cells.items() if x['n'] in v['decisive_n']}
    if all(x == 'FLOOR_HOLDS' for x in dec.values()):
        return 'FLOOR'
    if dec.get('n4_a0.15') == 'LATE_LEARNING' and dec.get('n8_a0.15') == 'LATE_LEARNING':
        return 'LATE_LEARNING'
    return 'MIXED'


def compute(c, res):
    import run as R
    v = c['verdict']
    Ts, T_ref, T_end, eps_x, floor = c['T_checkpoints'], v['T_ref'], v['T_end'], v['cross_eps'], c['eps_res']
    data = R.load_data(res)
    tasks = {t['task_id']: t for t in data['tasks']}
    M = read_metrics(res / 'metrics.csv')
    e20 = read_metrics((HERE / c['exp20_results'] / 'metrics.csv').resolve())
    T20 = sorted(next(iter(e20.values())).keys())
    seeds = c['seeds']
    boot = Boot(len(seeds), c['bootstrap_draws'], c['analysis_seed'])
    targets = sorted((t for t in tasks.values() if t['role'] == 'target'), key=lambda t: (t['K'], t['task_token']))
    mfloor = {t['task_id']: marginal_floor(data, t['task_token']) for t in targets}
    runs, cells, missing = [], {}, []
    for a in sorted(c['alphas']):
        for n in c['n_levels']:
            tids = [t['task_id'] for t in targets if t['K'] == n]
            per = []
            for s in seeds:
                for tid in tids:
                    rows = M.get((s, tid, a))
                    if rows is None or sorted(rows) != Ts:
                        missing.append(f'{s}_{tid}_{a:.2f}')
                        continue
                    e = [rows[T]['excess_bits'] for T in Ts]
                    ts, fl = t_star(Ts, e, eps_x)
                    a1 = [e20[(s, tid, 1.0)][T]['excess_bits'] for T in T20]
                    r = dict(seed=s, target_id=tid, n=n, alpha=a, e_ref=rows[T_ref]['excess_bits'],
                             e_mid=rows[2 * T_ref]['excess_bits'], e_end=rows[T_end]['excess_bits'],
                             e20_ref=e20[(s, tid, a)][T_ref]['excess_bits'],
                             tstar=ts, tstar_flag=fl, tstar_target_examples=a * ts,
                             e_alpha1_matched_end=interp_loglog(T20, a1, a * T_end, floor),
                             e_alpha1_matched_ref=interp_loglog(T20, a1, a * T_ref, floor),
                             marginal_floor=mfloor[tid])
                    r['drop'] = r['e_ref'] - r['e_end']
                    r['crossed_end'] = r['e_end'] <= eps_x
                    r['crossed_ref'] = r['e_ref'] <= eps_x
                    r['moved'] = r['drop'] >= v['move_min_drop']
                    r['plateau_end'] = (r['e_end'] >= c['plateau_ratio'] * r['e_ref'] and
                                        r['e_end'] > c['plateau_min_bits'])
                    r['interference_end'] = float(np.log2(max(r['e_end'], floor) / r['e_alpha1_matched_end']))
                    r['interference_ref'] = float(np.log2(max(r['e_ref'], floor) / r['e_alpha1_matched_ref']))
                    per.append(r)
            runs += per
            key = f'n{n}_a{a:.2f}'
            if len(per) != len(seeds) * len(tids):
                cells[key] = dict(n=n, alpha=a, runs=len(per), verdict='INCOMPLETE')
                continue

            def seed_means(f):
                return [np.mean([f(r) for r in per if r['seed'] == s]) for s in seeds]

            out = dict(n=n, alpha=a, runs=len(per))
            for name, f in [('e_ref', lambda r: r['e_ref']), ('e_mid', lambda r: r['e_mid']),
                            ('e_end', lambda r: r['e_end']), ('drop', lambda r: r['drop']),
                            ('e20_ref', lambda r: r['e20_ref']),
                            ('replicate_diff_ref', lambda r: r['e_ref'] - r['e20_ref']),
                            ('interference_end', lambda r: r['interference_end']),
                            ('interference_ref', lambda r: r['interference_ref']),
                            ('end_minus_marginal', lambda r: r['e_end'] - r['marginal_floor']),
                            ('marginal_floor', lambda r: r['marginal_floor'])]:
                sm = seed_means(f)
                out[name] = float(np.mean(sm))
                out[name + '_ci'] = ci(boot.mean(sm))
            out['interference_end_median'] = float(np.median([r['interference_end'] for r in per]))
            out['crossers_end'] = int(sum(r['crossed_end'] for r in per))
            out['crossers_ref'] = int(sum(r['crossed_ref'] for r in per))
            out['moved'] = int(sum(r['moved'] for r in per))
            out['plateau_end'] = int(sum(r['plateau_end'] for r in per))
            out['tstar_censored'] = int(sum(r['tstar_flag'] == 'right' for r in per))
            out['tstar_gm'] = float(np.exp(np.mean(np.log([r['tstar'] for r in per]))))
            out['tstar_target_examples_gm'] = a * out['tstar_gm']
            out['curve_mean'] = {str(T): float(np.mean([M[(r['seed'], r['target_id'], a)][T]['excess_bits']
                                                        for r in per])) for T in Ts}
            out['verdict'] = cell_verdict(out['drop_ci'], out['crossers_end'], v)
            cells[key] = out
    ref_curves = {}
    for a in (0.4, 1.0):
        for n in c['n_levels']:
            tids = [t['task_id'] for t in targets if t['K'] == n]
            ref_curves[f'n{n}_a{a:.2f}'] = {str(T): float(np.mean([e20[(s, tid, a)][T]['excess_bits']
                                                                   for s in seeds for tid in tids])) for T in T20}
    A = dict(experiment=c['experiment'], cells=cells, missing=missing, runs=runs, exp20_reference_curves=ref_curves,
             overall_verdict=overall_verdict(cells, v) if not missing else 'INCOMPLETE',
             verdict_rules=v, n_runs=len(runs))
    dpath = res / 'distractors.json'
    if dpath.exists():
        A['distractors'] = summarize_distractors(c, json.loads(dpath.read_text()))
    return A


# ---------------------------------------------------------------- distractors (descriptive, PROTOCOL Section 4.4)

_D = {}


def _dist_work(args):
    res, name, target_id, Ts = args
    import torch
    import model as M
    import run as R
    torch.set_num_threads(1)
    if 'data' not in _D:
        _D['data'], _D['c'] = R.load_data(Path(res)), load_config()
    data, c = _D['data'], _D['c']
    out = {}
    for T in Ts:
        m = M.load_npz(M.build(c), Path(res) / 'checkpoints' / name / f'T_{T}.npz')
        vals = {}
        for t in data['tasks']:
            if t['role'] != 'distractor':
                continue
            tok = t['task_token']
            vals[t['task_id']] = M.evaluate(m, tok, data['test_bits'], data['test_lens'],
                                            data['test_labels'][tok])['excess_bits']
        out[str(T)] = vals
    return name, out


def distractor_losses(c, res, workers, names=None):
    import run as R
    v = c['verdict']
    cells = []
    for p in sorted((res / 'cells').glob('*.json')):
        r = json.loads(p.read_text())
        if names is None or r['cell'] in names:
            cells.append((str(res), r['cell'], r['target_id'], [v['T_ref'], v['T_end']]))
    with cf.ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_dist_work, cells))


def summarize_distractors(c, D):
    out = {}
    for T in (str(c['verdict']['T_ref']), str(c['verdict']['T_end'])):
        for a in c['alphas']:
            vals = [np.mean(list(d[T].values())) for name, d in D.items() if name.endswith(f'_{a:.2f}')]
            if vals:
                out[f'a{a:.2f}_T{T}'] = dict(mean_distractor_bits=float(np.mean(vals)), cells=len(vals),
                                              min=float(np.min(vals)), max=float(np.max(vals)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--distractors', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    c = load_config()
    res = Path(args.results).resolve()
    if args.distractors:
        D = distractor_losses(c, res, args.workers)
        (res / 'distractors.json').write_text(json.dumps(D, indent=0, sort_keys=True))
        print(json.dumps(summarize_distractors(c, D), indent=1))
        return 0
    A = compute(c, res)
    (res / 'analysis.json').write_text(json.dumps(A, indent=1))
    print(json.dumps(dict(overall=A['overall_verdict'], missing=len(A['missing']),
                          cells={k: (x['verdict'], round(x.get('e_ref', np.nan), 3), round(x.get('e_end', np.nan), 3),
                                     x.get('crossers_end')) for k, x in A['cells'].items()}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
