"""Experiment 17b analysis (PROTOCOL_exp17b.md Section 5): coverage-matched easy pools against Experiment 17's hard
pool on easy and hard targets.

Usage: analyze_v2.py [--output results] [--v1 ../expt_v1/results]
  reads <output>/metrics.csv (the 30 new runs) and <v1>/metrics.csv (reused hard, easy, near, alpha = 1 runs),
  writes <output>/analysis_v2.json
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'expt_v1'
sys.path.insert(0, str(V1))
import analyze as A  # noqa: E402


def two_stage_ratio(num, den, seeds, targets, rng, draws):
    """Ratio of means mean(num)/mean(den) with the same two-stage resampling of (seed, target) pairs."""
    S, Tn = len(seeds), len(targets)
    N = np.array([[num[(s, t)] for t in targets] for s in seeds])
    D = np.array([[den[(s, t)] for t in targets] for s in seeds])
    si = rng.integers(0, S, size=(draws, S))
    ti = rng.integers(0, Tn, size=(draws, S, Tn))
    r = N[si[:, :, None], ti].mean(axis=(1, 2)) / D[si[:, :, None], ti].mean(axis=(1, 2))
    return float(N.mean() / D.mean()), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def pool_loss(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(float(r['model_bits']) - float(r['oracle_bits']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(HERE / 'results'))
    ap.add_argument('--v1', default=str(V1 / 'results'))
    args = ap.parse_args()
    out, v1 = Path(args.output), Path(args.v1)
    c = A.load_config()
    c2 = json.loads((HERE / 'config_v2.json').read_text())
    tasks = json.loads((out / 'tasks.json').read_text())
    rows_v1 = [r for r in A.read_rows(v1) if r['pool'] in c2['reuse_from_v1']]
    rows_v2 = A.read_rows(out)
    assert not ({r['pool'] for r in rows_v2} & set(c2['reuse_from_v1']))
    idx = A.index(rows_v1 + rows_v2)
    seeds = c['seeds']
    Ts = c['T_checkpoints']
    Tmax = Ts[-1]
    tg = [t for t in tasks if t['role'] == 'target']
    easy_t = [t['task_id'] for t in tg if t['K'] == c['easy_level']]
    hard_t = [t['task_id'] for t in tg if t['K'] == c['hard_level']]
    alphas = [0.0] + list(c['alpha_H'])
    pools_new = c2['pool_order']
    all_pools = ['easy'] + pools_new + ['hard', 'near']
    rng = np.random.default_rng(c['analysis_seed'])
    draws = c['bootstrap_draws']
    k_min = c['gate']['min_consistent_pairs']

    def e(s, p, t, a, T=Tmax):
        return idx[(s, p, t, round(a, 2), T)]['excess_bits']
    res = dict(easy_targets=easy_t, hard_targets=hard_t, Tmax=Tmax, cells={}, contrasts={}, gap_closed={},
               vs_easy={}, t_equiv={}, per_target_alpha0={}, gates={})
    for a in alphas:
        for p in all_pools:
            for t in easy_t + hard_t:
                res['cells'][f'{p}|{t}|{a:.2f}'] = {str(s): [e(s, p, t, a, T) for T in Ts] for s in seeds}
    for P in pools_new:
        for a in alphas:
            for which, targets, worse, better in (('easy_targets', easy_t, 'hard', P), ('hard_targets', hard_t, P, 'hard')):
                v = {(s, t): e(s, worse, t, a) - e(s, better, t, a) for s in seeds for t in targets}
                m, ci, Arr = A.two_stage(v, seeds, targets, rng, draws)
                lr = {(s, t): math.log2(e(s, worse, t, a) / e(s, better, t, a)) for s in seeds for t in targets}
                mlr, cilr, _ = A.two_stage(lr, seeds, targets, rng, draws)
                verdict, npos, n = A.sign_verdict(m, ci, Arr, k_min, 'MATCHED_POOL_BETTER', 'MISMATCHED_POOL_BETTER')
                perT = [float(np.mean([e(s, worse, t, a, T) - e(s, better, t, a, T) for s in seeds for t in targets]))
                        for T in Ts]
                means = {p: float(np.mean([e(s, p, t, a) for s in seeds for t in targets])) for p in all_pools}
                res['contrasts'][f'{P}|{which}|{a:.2f}'] = dict(
                    definition=f'excess({worse} pool) - excess({better} pool) at Tmax, bits/token', mean=m, ci95=ci,
                    pairs=Arr.tolist(), n_positive=npos, n=n, verdict=verdict, log2_ratio_mean=mlr,
                    log2_ratio_ci95=cilr, per_T=perT, mean_excess_at_Tmax_by_pool=means)
        # reported: gap closed and easy - P on easy targets
        for a in alphas:
            num = {(s, t): e(s, 'easy', t, a) - e(s, P, t, a) for s in seeds for t in easy_t}
            den = {(s, t): e(s, 'easy', t, a) - e(s, 'hard', t, a) for s in seeds for t in easy_t}
            m, ci, _ = A.two_stage(num, seeds, easy_t, rng, draws)
            res['vs_easy'][f'{P}|{a:.2f}'] = dict(definition='excess(easy) - excess(P) on easy targets at Tmax',
                                                 mean=m, ci95=ci)
            if a == 0.0:
                g, gci = two_stage_ratio(num, den, seeds, easy_t, rng, draws)
                res['gap_closed'][P] = dict(definition='(excess(easy) - excess(P)) / (excess(easy) - excess(hard)), '
                                                       'easy targets, alpha = 0, Tmax', estimate=g, ci95=gci)
        ce = res['contrasts']
        e0, e5 = ce[f'{P}|easy_targets|0.00']['verdict'], ce[f'{P}|easy_targets|{alphas[1]:.2f}']['verdict']
        h1 = 'SUPPORTED' if e0 == e5 == 'MATCHED_POOL_BETTER' else (
            'REFUTED' if e0 == e5 == 'MISMATCHED_POOL_BETTER' else 'NOT_ESTABLISHED')
        h0, h5 = ce[f'{P}|hard_targets|0.00']['verdict'], ce[f'{P}|hard_targets|{alphas[1]:.2f}']['verdict']
        h2 = 'SUPPORTED' if (h1 == 'SUPPORTED' and h0 == h5 == 'MATCHED_POOL_BETTER') else (
            'REFUTED' if h1 == 'REFUTED' else 'NOT_ESTABLISHED')
        q = {'MISMATCHED_POOL_BETTER': 'YES (hard-only still better)',
             'MATCHED_POOL_BETTER': 'NO (the coverage-matched easy pool is better)'}.get(e0, 'NO RELIABLE DIFFERENCE')
        res['gates'][P] = dict(Q_hard_still_beats_easy_zero_shot=q, H1_prime=h1, H2_prime=h2,
                               primary=bool(c2['pools'][P]['primary']))
    for p in all_pools:
        for t in easy_t + hard_t:
            res['per_target_alpha0'][f'{p}|{t}'] = float(np.mean([e(s, p, t, 0.0) for s in seeds]))
    for a in alphas:
        for p in pools_new:
            for t in easy_t + hard_t:
                r = []
                for s in seeds:
                    ce_ = np.array([idx[(s, 'none', t, 1.0, T)]['excess_bits'] for T in Ts])
                    r.append(A.t_equiv(np.array(Ts, float), ce_, e(s, p, t, a)))
                res['t_equiv'][f'{p}|{t}|{a:.2f}'] = r
    pl = pool_loss(out / 'pool_loss.csv')
    res['pool_excess_at_Tmax'] = dict(n=len(pl), mean=float(np.mean(pl)), max=float(np.max(pl)))
    (out / 'analysis_v2.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    for P in pools_new:
        print(P, json.dumps(res['gates'][P]), 'gap closed', round(res['gap_closed'][P]['estimate'], 3))


if __name__ == '__main__':
    main()
