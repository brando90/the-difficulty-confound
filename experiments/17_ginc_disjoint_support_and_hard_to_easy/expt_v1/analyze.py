"""Experiment 17 analysis (PROTOCOL Section 5): arm H (hard -> easy) and arm D (disjoint support) gates.

Usage: analyze.py [--output results]   reads results/metrics.csv, writes results/analysis.json
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import laws as L  # noqa: E402

EASY, HARD = 'easy', 'hard'


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def read_rows(out, sl='all'):
    rows = []
    with open(out / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            if r['slice'] != sl:
                continue
            rows.append(dict(seed=int(r['seed']), pool=r['pool'], run_target=r['run_target'],
                             target_id=r['target_id'], K=int(r['K']), K_bits=int(r['K_bits']),
                             alpha=float(r['alpha']), T=int(r['T']), excess_bits=float(r['excess_bits']),
                             ident_bits=float(r['ident_bits']), learn_bits=float(r['learn_bits']),
                             model_bits=float(r['model_bits']), oracle_bits=float(r['oracle_bits'])))
    return rows


def index(rows):
    return {(r['seed'], r['pool'], r['target_id'], round(r['alpha'], 2), r['T']): r for r in rows}


def two_stage(vals, seeds, targets, rng, draws):
    """vals[(seed, target)] -> two-stage bootstrap of the mean (seeds, then targets within seed)."""
    S, Tn = len(seeds), len(targets)
    A = np.array([[vals[(s, t)] for t in targets] for s in seeds])
    si = rng.integers(0, S, size=(draws, S))
    ti = rng.integers(0, Tn, size=(draws, S, Tn))
    d = A[si[:, :, None], ti].mean(axis=(1, 2))
    return float(A.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], A


def sign_verdict(mean, ci, A, k_min, pos_label, neg_label):
    npos, nneg, n = int((A > 0).sum()), int((A < 0).sum()), A.size
    if mean > 0 and ci[0] > 0 and npos >= k_min:
        return pos_label, npos, n
    if mean < 0 and ci[1] < 0 and nneg >= k_min:
        return neg_label, npos, n
    return 'NOT_ESTABLISHED', npos, n


def t_equiv(curve_T, curve_eps, eps):
    """Target-only tokens (alpha = 1 curve, log-log interpolation) that reach excess eps; ('<', T_min) or ('>', T_max)
    outside the curve."""
    lT, le = np.log2(curve_T), np.log2(np.maximum(curve_eps, 1e-12))
    if eps >= curve_eps[0]:
        return '<', float(curve_T[0])
    if eps <= min(curve_eps):
        return '>', float(curve_T[-1])
    for i in range(len(lT) - 1):
        a, b = le[i], le[i + 1]
        if (a - math.log2(eps)) * (b - math.log2(eps)) <= 0 and a != b:
            w = (a - math.log2(eps)) / (a - b)
            return '=', float(2 ** (lT[i] + w * (lT[i + 1] - lT[i])))
    return '?', float('nan')


def mf_tstar(Ts, vals, eps_star):
    """First crossing of eps_star, log-log interpolated: (T*, 'ok' | 'left' | 'right')."""
    if vals[0] <= eps_star:
        return float(Ts[0]), 'left'
    for i in range(1, len(Ts)):
        if vals[i] <= eps_star:
            a, b = math.log2(vals[i - 1]), math.log2(vals[i])
            w = (a - math.log2(eps_star)) / (a - b)
            return float(2 ** (math.log2(Ts[i - 1]) + w * (math.log2(Ts[i]) - math.log2(Ts[i - 1])))), 'ok'
    return float(Ts[-1]), 'right'


def arm_H(c, idx, seeds, Ts, tasks, rng):
    tg = [t for t in tasks if t['role'] == 'target']
    easy_t = [t['task_id'] for t in tg if t['K'] == c['easy_level']]
    hard_t = [t['task_id'] for t in tg if t['K'] == c['hard_level']]
    Tmax = Ts[-1]
    out = dict(easy_targets=easy_t, hard_targets=hard_t, Tmax=Tmax, cells={}, contrasts={}, t_equiv={})
    alphas = [0.0] + list(c['alpha_H'])
    for a in alphas:
        for pool in ('easy', 'hard', 'near'):
            for t in easy_t + hard_t:
                out['cells'][f'{pool}|{t}|{a:.2f}'] = {
                    str(s): [idx[(s, pool, t, round(a, 2), T)]['excess_bits'] for T in Ts] for s in seeds}
    k_min = c['gate']['min_consistent_pairs']
    for a in alphas:
        for which, targets, worse, better in (('easy_targets', easy_t, 'hard', 'easy'),
                                              ('hard_targets', hard_t, 'easy', 'hard')):
            # positive = the matched pool ('better') leaves less excess than the mismatched one ('worse')
            v = {(s, t): idx[(s, worse, t, round(a, 2), Tmax)]['excess_bits'] -
                 idx[(s, better, t, round(a, 2), Tmax)]['excess_bits'] for s in seeds for t in targets}
            m, ci, A = two_stage(v, seeds, targets, rng, c['bootstrap_draws'])
            lr = {(s, t): math.log2(idx[(s, worse, t, round(a, 2), Tmax)]['excess_bits'] /
                                    idx[(s, better, t, round(a, 2), Tmax)]['excess_bits'])
                  for s in seeds for t in targets}
            mlr, cilr, _ = two_stage(lr, seeds, targets, rng, c['bootstrap_draws'])
            verdict, npos, n = sign_verdict(m, ci, A, k_min, 'MATCHED_POOL_BETTER', 'MISMATCHED_POOL_BETTER')
            # every checkpoint (descriptive)
            perT = [float(np.mean([idx[(s, worse, t, round(a, 2), T)]['excess_bits'] -
                                   idx[(s, better, t, round(a, 2), T)]['excess_bits']
                                   for s in seeds for t in targets])) for T in Ts]
            mixed = {p: float(np.mean([idx[(s, p, t, round(a, 2), Tmax)]['excess_bits']
                                       for s in seeds for t in targets])) for p in ('easy', 'hard', 'near')}
            out['contrasts'][f'{which}|{a:.2f}'] = dict(
                definition=f'excess({worse} pool) - excess({better} pool) at Tmax, bits/token', mean=m, ci95=ci,
                pairs=A.tolist(), n_positive=npos, n=n, verdict=verdict, log2_ratio_mean=mlr, log2_ratio_ci95=cilr,
                per_T=perT, mean_excess_at_Tmax_by_pool=mixed)
    # worth of each pool in target-only tokens (alpha = 1 curve of the same seed and target)
    for a in alphas:
        for pool in ('easy', 'hard', 'near'):
            for t in easy_t + hard_t:
                res = []
                for s in seeds:
                    cT = np.array(Ts, float)
                    ce = np.array([idx[(s, 'none', t, 1.0, T)]['excess_bits'] for T in Ts])
                    res.append(t_equiv(cT, ce, idx[(s, pool, t, round(a, 2), Tmax)]['excess_bits']))
                out['t_equiv'][f'{pool}|{t}|{a:.2f}'] = res
    ce, ch = out['contrasts'], out
    e0, e5 = ce['easy_targets|0.00']['verdict'], ce[f'easy_targets|{alphas[1]:.2f}']['verdict']
    if e0 == e5 == 'MATCHED_POOL_BETTER':
        g = 'SUPPORTED'
    elif e0 == e5 == 'MISMATCHED_POOL_BETTER':
        g = 'REFUTED'
    else:
        g = 'NOT_ESTABLISHED'
    ch['H1_hard_does_not_subsume_easy'] = g
    h0, h5 = ce['hard_targets|0.00']['verdict'], ce[f'hard_targets|{alphas[1]:.2f}']['verdict']
    ch['H2_crossover'] = 'SUPPORTED' if (g == 'SUPPORTED' and h0 == h5 == 'MATCHED_POOL_BETTER') else (
        'REFUTED' if g == 'REFUTED' else 'NOT_ESTABLISHED')
    return out


def fit_arm(c, rows, seeds, pool, K_ref, K_grid, T0, eps_star, eps_curve):
    """M_K_both and M_align_eff per seed on the pool's alpha in alpha_D rows plus the shared alpha = 1 rows."""
    res = dict(per_seed={}, E={}, eta_align_eff={})
    fits = {}
    for s in seeds:
        rs = [r for r in rows if r['seed'] == s and r['target_id'] == r['run_target'] and
              ((r['pool'] == pool and r['alpha'] in c['alpha_D']) or r['pool'] == 'none')]
        X = L.design(rs, 'K_bits', K_grid, K_ref, T0, c['eps_res'], True)
        fb = L.fit('M_K_both', X, c)
        fe = L.fit('M_align_eff', X, c)
        r_b, _ = L.scored_residual('M_K_both', fb['theta'], X)
        fits[s] = (fb, X)
        res['per_seed'][str(s)] = dict(M_K_both=fb['params'], M_align_eff=fe['params'], n_rows=len(rs),
                                       rmse_M_K_both=float(np.sqrt(np.mean(r_b ** 2))),
                                       converged=fb['converged'])
        res['eta_align_eff'][str(s)] = fe['params']['eta']
    for eps in sorted(set(eps_curve) | {eps_star}):
        ea, ek = [], []
        for s in seeds:
            fb, X = fits[s]
            a, k = L.elasticities('M_K_both', fb['params'], K_ref, eps, X)
            ea.append(a)
            ek.append(k)
        ea, ek = np.array(ea), np.array(ek)
        res['E'][f'{eps:g}'] = dict(E=float(ea.mean() / ek.mean()) if ek.mean() > 0 else math.inf,
                                    e_alpha=ea.tolist(), e_K=ek.tolist(),
                                    E_per_seed=[float(a / k) if k > 0 else math.inf for a, k in zip(ea, ek)])
    return res, fits


def seed_boot(per_seed_a, per_seed_k, rng, draws):
    a, k = np.array(per_seed_a), np.array(per_seed_k)
    idx = rng.integers(0, len(a), size=(draws, len(a)))
    ma, mk = a[idx].mean(1), k[idx].mean(1)
    d = np.where(mk > 0, ma / np.where(mk > 0, mk, 1), np.inf)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def arm_D(c, rows, idx, seeds, Ts, tasks, rng):
    tg = [t for t in tasks if t['role'] == 'target']
    tids = [t['task_id'] for t in tg]
    Tmax, T0 = Ts[-1], Ts[0]
    K_grid = sorted({float(t['K_bits']) for t in tg})
    K_ref = float(np.exp(np.mean(np.log(K_grid))))
    eps_star = c['epsilon_star']
    out = dict(K_ref_bits=K_ref, eps_star=eps_star)
    # D1: zero-shot, far minus near at Tmax (positive = same-alphabet data buys more)
    v = {(s, t): idx[(s, 'far', t, 0.0, Tmax)]['excess_bits'] - idx[(s, 'near', t, 0.0, Tmax)]['excess_bits']
         for s in seeds for t in tids}
    m, ci, A = two_stage(v, seeds, tids, rng, c['bootstrap_draws'])
    k_min = math.ceil(A.size * c['gate']['min_consistent_pairs'] / c['gate']['pairs'])
    verdict, npos, n = sign_verdict(m, ci, A, k_min, 'SUPPORTED', 'REVERSED')
    out['D1_zero_shot'] = dict(mean=m, ci95=ci, n_positive=npos, n=n, verdict=verdict, pairs=A.tolist(),
                               far_mean=float(np.mean([idx[(s, 'far', t, 0.0, Tmax)]['excess_bits']
                                                       for s in seeds for t in tids])),
                               near_mean=float(np.mean([idx[(s, 'near', t, 0.0, Tmax)]['excess_bits']
                                                        for s in seeds for t in tids])))
    # D at alpha in alpha_D: excess at Tmax far - near
    for a in c['alpha_D']:
        v = {(s, t): idx[(s, 'far', t, round(a, 2), Tmax)]['excess_bits'] -
             idx[(s, 'near', t, round(a, 2), Tmax)]['excess_bits'] for s in seeds for t in tids}
        m, ci, A = two_stage(v, seeds, tids, rng, c['bootstrap_draws'])
        out[f'far_minus_near|{a:.2f}'] = dict(mean=m, ci95=ci, n_positive=int((A > 0).sum()), n=A.size)
    # D2/D3: elasticity ratio per arm
    fitsD = {}
    for pool in ('near', 'far'):
        res, fits = fit_arm(c, rows, seeds, pool, K_ref, K_grid, T0, eps_star, c['eps_curve'])
        e = res['E'][f'{eps_star:g}']
        res['E_star'] = e['E']
        res['E_star_ci95_seed'] = seed_boot(e['e_alpha'], e['e_K'], rng, c['bootstrap_draws'])
        out[pool] = res
        fitsD[pool] = fits
    ef, en = out['far'], out['near']
    ps = ef['E'][f'{eps_star:g}']['E_per_seed']
    if ef['E_star'] > 1 and ef['E_star_ci95_seed'][0] > 1 and all(x > 1 for x in ps):
        g2 = 'SUPPORTED'
    elif ef['E_star'] < 1 and ef['E_star_ci95_seed'][1] < 1 and all(x < 1 for x in ps):
        g2 = 'NOT_ESTABLISHED (reverse: E_far < 1)'
    else:
        g2 = 'NOT_ESTABLISHED'
    out['D2_E_far_above_1'] = g2
    diff = [f - n_ for f, n_ in zip(ps, en['E'][f'{eps_star:g}']['E_per_seed'])]
    dE = ef['E_star'] - en['E_star']
    out['D3_E_far_minus_near'] = dict(
        diff=dE, per_seed=diff,
        min_effect=c['gate']['dE_min'],
        verdict='SUPPORTED' if dE >= c['gate']['dE_min'] and all(x > 0 for x in diff) else
        ('REVERSED' if dE <= -c['gate']['dE_min'] and all(x < 0 for x in diff) else 'NOT_ESTABLISHED'))
    out['eta_near'] = float(np.mean(list(en['eta_align_eff'].values())))
    out['eta_far'] = float(np.mean(list(ef['eta_align_eff'].values())))
    # model-free elasticities at eps_star
    mf = {}
    for pool in ('near', 'far'):
        ea, ek, cens = [], [], 0
        for s in seeds:
            ts = {}
            for t in tids:
                for a in [1.0] + list(c['alpha_D']):
                    p = 'none' if a == 1.0 else pool
                    ts[(t, a)] = mf_tstar(Ts, [idx[(s, p, t, round(a, 2), T)]['excess_bits'] for T in Ts], eps_star)
            for t in tids:
                lo, hi = ts[(t, min(c['alpha_D']))], ts[(t, 1.0)]
                if lo[1] == 'ok' and hi[1] == 'ok':
                    ea.append(math.log(lo[0] / hi[0]) / math.log(1 / min(c['alpha_D'])))
                else:
                    cens += 1
            easy = [t['task_id'] for t in tg if t['K'] == min(t2['K'] for t2 in tg)]
            hard = [t['task_id'] for t in tg if t['K'] == max(t2['K'] for t2 in tg)]
            for te, th in zip(easy, hard):
                lo, hi = ts[(te, 1.0)], ts[(th, 1.0)]
                if lo[1] == 'ok' and hi[1] == 'ok':
                    ek.append(math.log(hi[0] / lo[0]) / math.log(K_grid[-1] / K_grid[0]))
        mf[pool] = dict(e_alpha=float(np.mean(ea)) if ea else None, e_K=float(np.mean(ek)) if ek else None,
                        n_alpha=len(ea), n_K=len(ek), censored_alpha=cens,
                        E_mf=(float(np.mean(ea) / np.mean(ek)) if ea and ek and np.mean(ek) > 0 else None))
    out['model_free'] = mf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(HERE / 'results'))
    args = ap.parse_args()
    c = load_config()
    out = Path(args.output)
    tasks = json.loads((out / 'tasks.json').read_text())
    rows = read_rows(out)
    idx = index(rows)
    seeds = sorted({r['seed'] for r in rows})
    Ts = sorted({r['T'] for r in rows})
    rng = np.random.default_rng(c['analysis_seed'])
    res = dict(seeds=seeds, T_checkpoints=Ts, epsilon_star=c['epsilon_star'], n_rows=len(rows))
    res['H'] = arm_H(c, idx, seeds, Ts, tasks, rng)
    res['D'] = arm_D(c, rows, idx, seeds, Ts, tasks, rng)
    # cell means for the report: every (pool, K, alpha) at every T
    cm = {}
    for r in rows:
        k = f"{r['pool']}|{r['K']}|{r['alpha']:.2f}|{r['T']}"
        cm.setdefault(k, []).append(r['excess_bits'])
    res['cell_means'] = {k: float(np.mean(v)) for k, v in sorted(cm.items())}
    pl = []
    with open(out / 'pool_loss.csv') as f:
        for r in csv.DictReader(f):
            pl.append(float(r['model_bits']) - float(r['oracle_bits']))
    res['pool_excess_at_Tmax'] = dict(mean=float(np.mean(pl)), max=float(np.max(pl)), n=len(pl))
    (out / 'analysis.json').write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(dict(H1=res['H']['H1_hard_does_not_subsume_easy'], H2=res['H']['H2_crossover'],
                          D1=res['D']['D1_zero_shot']['verdict'], D2=res['D']['D2_E_far_above_1'],
                          E_near=res['D']['near']['E_star'], E_far=res['D']['far']['E_star'],
                          D3=res['D']['D3_E_far_minus_near']['verdict']), indent=1))


if __name__ == '__main__':
    main()
