"""Experiment 23 analysis (PROTOCOL Section 5): the frozen Experiment 13 and Experiment 17b analyses on the converged
8-seed grids.

Usage (one experiment per process: the two source analyses share module names):
  analyze23.py --exp e13  --output OUT_E13    epsilon* by the Section 5.1 rule, then Experiment 13's own stage 1
                                              (fits, forecasts SHA-256) and stage 2 (G1-G5); stage 2 again at
                                              epsilon* = 1.10 in OUT_E13/eps_1.10/ for comparison with Exp 13-small
  analyze23.py --exp e17b --output OUT_E17B   Experiment 17b's coverage analysis (Q, H1', H2', gap closed) with
                                              8 seeds and the >= 14/16 pair rule; writes analysis_e17b.json
"""
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
SRC = {k: (HERE / v).resolve() for k, v in CFG['sources'].items()}


# ---------------------------------------------------------------- Experiment 13

def eps_star_rule(rows, ladder, K, alpha, share):
    """Section 5.1: the smallest ladder level that >= share of the (K, alpha) runs reach on epsilon_learn at the largest
    fit checkpoint (T_stop / 2). Uses fit rows only (K = 32 and alpha = 0.4 are fit coordinates, T_stop / 2 a fit T)."""
    Ts = sorted({r['T'] for r in rows})
    Tfit = Ts[-2]
    v = np.array([r['learn_bits'] for r in rows if r['K'] == K and abs(r['alpha'] - alpha) < 1e-9 and r['T'] == Tfit])
    reached = {str(x): float(np.mean(v <= x)) for x in ladder}
    ok = [x for x in ladder if np.mean(v <= x) >= share]
    eps = ok[0] if ok else ladder[-1]
    return eps, dict(T=Tfit, n_runs=int(len(v)), learn_bits=v.tolist(), median=float(np.median(v)),
                     share_reached=reached, epsilon_star=eps, reached_any=bool(ok), min_share=share)


def run_e13(out):
    sys.path.insert(0, str(SRC['exp13']))
    import analyze as A  # Experiment 13's analysis, unchanged
    tasks = json.loads((out / 'tasks.json').read_text())
    c = A.load_config('small')
    e = CFG['e13']
    rows = [r for r in A.read_rows(out, tasks, True) if r['slice'] == 'all']
    seeds = sorted({r['seed'] for r in rows})
    assert seeds == sorted(e['seeds']), seeds
    eps, rule = eps_star_rule(rows, CFG['eps_star_ladder'], e['eps_star_cell']['K'], e['eps_star_cell']['alpha'],
                              CFG['eps_star_min_share'])
    (out / 'eps_star_rule.json').write_text(json.dumps(rule, indent=1))
    print(f'epsilon* = {eps} (rule: {rule["share_reached"]})', flush=True)
    c['epsilon_star'] = eps
    c['eps_curve'] = CFG['eps_curve']
    A.stage1(c, out, tasks)
    A.stage2(c, out, tasks)
    cmp = out / f"eps_{e['eps_star_comparison']:.2f}"
    cmp.mkdir(exist_ok=True)
    for name in ('metrics.csv', 'tasks.json', 'fits.json', 'forecasts.csv', 'distractor_loss.csv', 'posterior16.json'):
        shutil.copy2(out / name, cmp / name)
    c2 = dict(c, epsilon_star=e['eps_star_comparison'])
    A.stage2(c2, cmp, tasks)
    return 0


# ---------------------------------------------------------------- Experiment 17b

def two_stage_ratio(num, den, seeds, targets, rng, draws):
    """Experiment 17b's ratio of means with the same two-stage resampling of (seed, target) pairs (copied unchanged)."""
    S, Tn = len(seeds), len(targets)
    N = np.array([[num[(s, t)] for t in targets] for s in seeds])
    D = np.array([[den[(s, t)] for t in targets] for s in seeds])
    si = rng.integers(0, S, size=(draws, S))
    ti = rng.integers(0, Tn, size=(draws, S, Tn))
    r = N[si[:, :, None], ti].mean(axis=(1, 2)) / D[si[:, :, None], ti].mean(axis=(1, 2))
    return float(N.mean() / D.mean()), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def coverage_analysis(A, rows, tasks, seeds, Ts, pools_new, all_pools, alpha_H, easy_level, hard_level, k_min,
                      analysis_seed, draws, primary, with_t_equiv):
    """Experiment 17b's analysis_v2.main, parametrized by seeds, checkpoints, pools and the pair threshold. Random
    draws are taken in the same order as the original, so on the original inputs it returns analysis_v2.json."""
    idx = A.index(rows)
    Tmax = Ts[-1]
    tg = [t for t in tasks if t['role'] == 'target']
    easy_t = [t['task_id'] for t in tg if t['K'] == easy_level]
    hard_t = [t['task_id'] for t in tg if t['K'] == hard_level]
    alphas = [0.0] + list(alpha_H)
    rng = np.random.default_rng(analysis_seed)

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
        res['gates'][P] = dict(Q_hard_still_beats_easy_zero_shot=q, H1_prime=h1, H2_prime=h2, primary=P == primary)
    for p in all_pools:
        for t in easy_t + hard_t:
            res['per_target_alpha0'][f'{p}|{t}'] = float(np.mean([e(s, p, t, 0.0) for s in seeds]))
    if with_t_equiv:
        for a in alphas:
            for p in pools_new:
                for t in easy_t + hard_t:
                    r = []
                    for s in seeds:
                        ce_ = np.array([idx[(s, 'none', t, 1.0, T)]['excess_bits'] for T in Ts])
                        r.append(A.t_equiv(np.array(Ts, float), ce_, e(s, p, t, a)))
                    res['t_equiv'][f'{p}|{t}|{a:.2f}'] = r
    return res


def pool_excess(path):
    with open(path) as f:
        return [float(r['model_bits']) - float(r['oracle_bits']) for r in csv.DictReader(f)]


def run_e17b(out):
    sys.path.insert(0, str(SRC['exp17']))
    import analyze as A  # Experiment 17's analysis helpers, unchanged
    c = A.load_config()
    e = CFG['e17b']
    tasks = json.loads((out / 'tasks.json').read_text())
    rows = A.read_rows(out)
    seeds = sorted({r['seed'] for r in rows})
    assert seeds == sorted(e['seeds']), seeds
    Ts = sorted({r['T'] for r in rows})
    res = coverage_analysis(A, rows, tasks, seeds, Ts, e['new_pools'], ['easy'] + e['new_pools'] + ['hard'],
                            [e['alpha_H']], c['easy_level'], c['hard_level'], e['min_consistent_pairs'],
                            c['analysis_seed'], c['bootstrap_draws'], e['primary_pool'], with_t_equiv=False)
    res['k_min'] = e['min_consistent_pairs']
    res['seeds'] = seeds
    res['T_checkpoints'] = Ts
    pl = pool_excess(out / 'pool_loss.csv')
    res['pool_excess_at_Tmax'] = dict(n=len(pl), mean=float(np.mean(pl)), max=float(np.max(pl)))
    (out / 'analysis_e17b.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    for P in e['new_pools']:
        print(P, json.dumps(res['gates'][P]), 'gap closed', round(res['gap_closed'][P]['estimate'], 3))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', choices=['e13', 'e17b'], required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    out = Path(a.output).expanduser()
    return run_e13(out) if a.exp == 'e13' else run_e17b(out)


if __name__ == '__main__':
    sys.exit(main())
