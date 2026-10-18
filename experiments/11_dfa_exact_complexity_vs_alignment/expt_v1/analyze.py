"""Experiment 11 analysis (PROTOCOL Sections 3-5).

  analyze.py --stage 1   reads fit rows only; fits every law per seed (both K units, both fitters), picks the G2 K law
                         by fit-cell RMSE, writes fits.json and forecasts_frozen.json and prints the forecasts' SHA-256
  analyze.py --stage 2   reads everything; held-out scores, T*, multipliers, elasticities, mechanism analyses and
                         gates G1-G7 into tstar.csv and analysis.json
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import independent_fit as IF  # noqa: E402
import laws as L  # noqa: E402

UNITS = ['K_bits', 'K']


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def is_fit(r, c):
    return (int(r['K']) in c['fit_n'] and float(r['alpha']) in c['fit_alphas']
            and int(r['T']) <= c['fit_T_max'])


def read_rows(path, c, which):
    """Rows of metrics.csv; which = 'fit' drops held-out rows at read time, 'all' keeps every row."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r = dict(seed=int(r['seed']), target_id=r['target_id'], K=int(r['K']), K_bits=int(r['K_bits']),
                     alpha=float(r['alpha']), T=int(r['T']), excess_bits=float(r['excess_bits']),
                     accuracy=float(r['accuracy']), excess_bits_by_length=json.loads(r['excess_bits_by_length']))
            if which == 'all' or is_fit(r, c):
                out.append(r)
    return out


def heldout_grid(c, tasks):
    """Coordinates of every held-out point (no observations): K = 16, alpha = 0.15 or T = T_max slices."""
    pts = []
    for t in sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: t['task_token']):
        for a in c['alphas']:
            for T in c['T_checkpoints']:
                r = dict(target_id=t['task_id'], K=t['K'], K_bits=t['K_bits'], alpha=a, T=T, excess_bits=1.0)
                if not is_fit(r, c):
                    pts.append(r)
    return pts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- stage 1

def stage1(c, res, tasks):
    rows = read_rows(res / 'metrics.csv', c, 'fit')
    seeds = sorted({r['seed'] for r in rows})
    grid = heldout_grid(c, tasks)
    tol = c['fit']['agreement_rel_tol']
    fits, forecasts, agreement = {}, {}, []
    for unit in UNITS:
        fits[unit], forecasts[unit] = {}, {}
        Xh = L.design(grid, unit, c)
        for s in seeds:
            rs = [r for r in rows if r['seed'] == s]
            X = L.design(rs, unit, c)
            fs = {}
            for law in L.LAWS:
                a = L.fit(law, X, c)
                b = IF.fit(law, X, c)
                fa, fb = L.predict_eps(law, a['theta'], Xh), L.predict_eps(law, b['theta'], Xh)
                ok = IF.agree(a['cost'], b['cost'], tol) and IF.agree(fa, fb, tol)
                agreement.append(dict(unit=unit, seed=s, law=law, cost_laws=a['cost'], cost_independent=b['cost'],
                                      max_rel_forecast_diff=float(np.max(np.abs(fa - fb) / np.abs(fa))), agree=ok))
                res_fit = L.censored_residual(np.log2(np.maximum(L.predict_eps(law, a['theta'], X), 1e-300)), X) \
                    if law != 'M_align_add' else L.scored_residual(law, a['theta'], X)[0]
                fs[law] = dict(a, n_fit=len(rs), fit_sse_log2=float(np.sum(res_fit ** 2)),
                               independent=dict(theta=b['theta'], cost=b['cost']))
            fits[unit][str(s)] = fs
    # G2 K law: lower pooled fit-cell RMSE (primary unit), chosen before held-out rows are read
    rm = {law: np.sqrt(sum(fits['K_bits'][str(s)][law]['fit_sse_log2'] for s in seeds) /
                       sum(fits['K_bits'][str(s)][law]['n_fit'] for s in seeds)) for law in ('M_K_coef', 'M_K_exp')}
    k_law = min(rm, key=rm.get)
    for unit in UNITS:
        Xh = L.design(grid, unit, c)
        for s in seeds:
            rs = [r for r in rows if r['seed'] == s]
            X = L.design(rs, unit, c)
            fk = fits[unit][str(s)][k_law if unit == 'K_bits' else k_law]
            a = L.fit('M_K_floor', X, c, base=(k_law, np.array(fk['theta'])))
            b = IF.fit('M_K_floor', X, c, base=(k_law, np.array(fk['theta'])))
            fa = L.predict_eps('M_K_floor', a['theta'], Xh, k_law)
            fb = L.predict_eps('M_K_floor', b['theta'], Xh, k_law)
            ok = IF.agree(a['cost'], b['cost'], tol) and IF.agree(fa, fb, tol)
            agreement.append(dict(unit=unit, seed=s, law='M_K_floor', cost_laws=a['cost'],
                                  cost_independent=b['cost'],
                                  max_rel_forecast_diff=float(np.max(np.abs(fa - fb) / np.abs(fa))), agree=ok))
            fits[unit][str(s)]['M_K_floor'] = dict(a, base=k_law, independent=dict(theta=b['theta'], cost=b['cost']))
            forecasts[unit][str(s)] = {law: L.predict_eps(law, fits[unit][str(s)][law]['theta'], Xh, k_law).tolist()
                                       for law in L.LAWS + ['M_K_floor']}
    frozen = dict(points=[{k: p[k] for k in ('target_id', 'K', 'K_bits', 'alpha', 'T')} for p in grid],
                  g2_k_law=k_law, forecasts=forecasts)
    (res / 'forecasts_frozen.json').write_text(json.dumps(frozen))
    h = sha256(res / 'forecasts_frozen.json')
    out = dict(g2_k_law=k_law, g2_fit_rmse=rm, seeds=seeds, fits=fits, fitter_agreement=agreement,
               all_fitters_agree=all(a['agree'] for a in agreement), forecasts_frozen_sha256=h)
    (res / 'fits.json').write_text(json.dumps(out, indent=1))
    return h, out


# ---------------------------------------------------------------- helpers for stage 2

def t_star(Ts, eps, eps_star):
    """First crossing with log-log interpolation; returns (T*, flag) with flag in {'', 'left', 'right'}."""
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


class Boot:
    """Seed-block bootstrap: W[b, s] = multiplicity of seed s in draw b."""

    def __init__(self, n_seeds, draws, seed):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=draws).astype(float)
        self.n = n_seeds
        self.draws = draws

    def mean(self, v):
        return self.W @ np.asarray(v, float) / self.n

    def ratio(self, num, den):
        return (self.W @ np.asarray(num, float)) / (self.W @ np.asarray(den, float))


def ci(x):
    x = np.asarray(x, float)
    x = np.where(np.isposinf(x), 1e300, x)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def rel_gain(sse_a, n_a, sse_b, n_b, boot):
    """1 - RMSE_b / RMSE_a pooled over seeds, with its seed-block interval."""
    est = 1 - np.sqrt(np.sum(sse_b) / np.sum(n_b)) / np.sqrt(np.sum(sse_a) / np.sum(n_a))
    draws = 1 - np.sqrt((boot.W @ sse_b) / (boot.W @ n_b)) / np.sqrt((boot.W @ sse_a) / (boot.W @ n_a))
    return float(est), ci(draws)


def gate_g3(kappa_ci, gamma1_ci, rmse_coef, rmse_exp, exp_gain, exp_gain_ci, c):
    g3a = kappa_ci[0] > 0 and gamma1_ci[0] <= 0 <= gamma1_ci[1] and rmse_coef <= rmse_exp
    g3b = gamma1_ci[1] < 0 and exp_gain >= c['gates']['g3b_min_rel_gain'] and exp_gain_ci[0] > 0
    return ('SUPPORTED' if g3a else 'NOT_ESTABLISHED'), ('SUPPORTED' if g3b else 'NOT_ESTABLISHED')


def verdict(ok):
    return 'SUPPORTED' if ok else 'NOT_ESTABLISHED'


# ---------------------------------------------------------------- stage 2

def stage2(c, res, tasks, two_stage_draws=None):
    fits = json.loads((res / 'fits.json').read_text())
    assert sha256(res / 'forecasts_frozen.json') == fits['forecasts_frozen_sha256'], 'forecasts changed after stage 1'
    frozen = json.loads((res / 'forecasts_frozen.json').read_text())
    rows = read_rows(res / 'metrics.csv', c, 'all')
    seeds = fits['seeds']
    S = len(seeds)
    k_law = fits['g2_k_law']
    boot = Boot(S, c['bootstrap_draws'], c['analysis_seed'])
    Ts = c['T_checkpoints']
    T_max = Ts[-1]
    targets = sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: t['task_token'])
    tids = [t['task_id'] for t in targets]
    tinfo = {t['task_id']: t for t in targets}
    levels = c['n_levels']
    by_level = {n: [i for i, t in enumerate(targets) if t['K'] == n] for n in levels}
    eps = {}
    for r in rows:
        eps[(r['seed'], r['target_id'], r['alpha'], r['T'])] = r['excess_bits']
    A = {}

    # held-out scoring
    obs = {(s, p['target_id'], p['alpha'], p['T']): eps[(s, p['target_id'], p['alpha'], p['T'])]
           for s in seeds for p in frozen['points']}
    slices = dict(all=lambda p: True, K16=lambda p: p['K'] in c['heldout_n'],
                  alpha015=lambda p: p['alpha'] in c['heldout_alphas'], Tmax=lambda p: p['T'] == T_max)
    sse = {u: {} for u in UNITS}
    nn = {u: {} for u in UNITS}
    invalid = {u: {} for u in UNITS}
    for unit in UNITS:
        for law in L.LAWS + ['M_K_floor']:
            for sl, f in slices.items():
                sse[unit][(law, sl)] = np.zeros(S)
                nn[unit][(law, sl)] = np.zeros(S)
            invalid[unit][law] = 0
            for si, s in enumerate(seeds):
                pts = frozen['points']
                pred = np.array(frozen['forecasts'][unit][str(s)][law])
                ob = np.array([obs[(s, p['target_id'], p['alpha'], p['T'])] for p in pts])
                bad = ~np.isfinite(pred) | (pred <= 0)
                invalid[unit][law] += int(bad.sum())
                pred = np.where(bad, c['eps_res'], pred)
                X = dict(y=np.log2(np.maximum(ob, 1e-300)), cens=ob < c['eps_res'], yres=np.log2(c['eps_res']))
                r = L.censored_residual(np.log2(pred), X)
                for sl, f in slices.items():
                    m = np.array([f(p) for p in pts])
                    sse[unit][(law, sl)][si] = np.sum(r[m] ** 2)
                    nn[unit][(law, sl)][si] = m.sum()
    rmse = {u: {law: {sl: float(np.sqrt(sse[u][(law, sl)].sum() / nn[u][(law, sl)].sum())) for sl in slices}
                for law in L.LAWS + ['M_K_floor']} for u in UNITS}
    A['heldout_rmse_log2'] = rmse
    A['heldout_invalid_forecasts'] = invalid
    A['heldout_points_per_seed'] = len(frozen['points'])

    def gain(unit, law_a, law_b, sl='all'):
        return rel_gain(sse[unit][(law_a, sl)], nn[unit][(law_a, sl)], sse[unit][(law_b, sl)], nn[unit][(law_b, sl)],
                        boot)

    # T* table
    tstar = {}
    trows = []
    for s in seeds:
        for tid in tids:
            for a in c['alphas']:
                e = [eps[(s, tid, a, T)] for T in Ts]
                for es in c['eps_star_levels']:
                    v, fl = t_star(Ts, e, es)
                    tstar[(s, tid, a, es)] = (v, fl)
                    trows.append(dict(seed=s, target_id=tid, K=tinfo[tid]['K'], K_bits=tinfo[tid]['K_bits'],
                                      alpha=a, eps_star=es, T_star=v, censored=fl))
    with open(res / 'tstar.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(trows[0]))
        w.writeheader()
        w.writerows(trows)
    eps_p = c['eps_star_primary']

    def lt(s, tid, a, es):
        return np.log2(tstar[(s, tid, a, es)][0])

    # G1: complexity multiplier at alpha = 1
    LT = np.array([[lt(s, tid, 1.0, eps_p) for tid in tids] for s in seeds])  # seeds x targets
    i32, i4 = by_level[levels[-1]], by_level[levels[0]]
    mk_log = LT[:, i32].mean(1) - LT[:, i4].mean(1)
    MK = float(2 ** mk_log.mean())
    MK_seed = ci(2 ** boot.mean(mk_log))
    trng = np.random.default_rng(c['analysis_seed'] + 1)
    D2 = two_stage_draws or c['bootstrap_draws']
    tw = trng.integers(0, 4, size=(D2, len(levels), 4))  # target resample within each level
    W2 = boot.W[:D2]
    lvl_idx = [np.array(by_level[n]) for n in levels]
    mk2 = np.empty(D2)
    for b in range(D2):
        m32 = LT[:, lvl_idx[-1][tw[b, -1]]].mean(1)
        m4 = LT[:, lvl_idx[0][tw[b, 0]]].mean(1)
        mk2[b] = 2 ** (W2[b] @ (m32 - m4) / S)
    MK_two = ci(mk2)
    cens = {fl: sum(tstar[(s, tid, 1.0, eps_p)][1] == fl for s in seeds for tid in tids) for fl in ('left', 'right')}
    g = c['gates']
    A['G1'] = dict(statistic='M_K = GM T*(n=32) / GM T*(n=4) at alpha=1', eps_star=eps_p, estimate=MK,
                   ci_seed=MK_seed, ci_two_stage=MK_two, per_seed=(2 ** mk_log).tolist(), censored_alpha1=cens,
                   threshold=f">= {g['g1_min_multiplier']}, both lower bounds > {g['g1_lb_above']}",
                   verdict=verdict(MK >= g['g1_min_multiplier'] and MK_seed[0] > g['g1_lb_above']
                                   and MK_two[0] > g['g1_lb_above']))

    # G2: K law versus M_align_eff
    gk, gk_ci = gain('K_bits', 'M_align_eff', k_law)
    k16 = rmse['K_bits'][k_law]['K16'] < rmse['K_bits']['M_align_eff']['K16']
    A['G2'] = dict(statistic=f'held-out RMSE reduction of {k_law} over M_align_eff (log2 bits)', k_law=k_law,
                   estimate=gk, ci_seed=gk_ci, rmse_k_law=rmse['K_bits'][k_law]['all'],
                   rmse_align_eff=rmse['K_bits']['M_align_eff']['all'],
                   k16_rmse_k_law=rmse['K_bits'][k_law]['K16'], k16_rmse_align_eff=rmse['K_bits']['M_align_eff']['K16'],
                   invalid_k_law=invalid['K_bits'][k_law],
                   unit_K_states=dict(zip(['estimate', 'ci_seed'], gain('K', 'M_align_eff', k_law))),
                   threshold=f">= {g['g2_min_rel_gain']:.0%}, lower bound > 0, lower on K=16, 0 invalid",
                   verdict=verdict(gk >= g['g2_min_rel_gain'] and gk_ci[0] > 0 and k16 and invalid['K_bits'][k_law] == 0))

    # G3: which law, from M_K_both fit on fit points
    both = [fits['fits']['K_bits'][str(s)]['M_K_both']['params'] for s in seeds]
    kap = np.array([p['kappa'] for p in both])
    g1v = np.array([p['gamma1'] for p in both])
    kap_ci, g1_ci = ci(boot.mean(kap)), ci(boot.mean(g1v))
    eg, eg_ci = gain('K_bits', 'M_K_coef', 'M_K_exp')
    g3a, g3b = gate_g3(kap_ci, g1_ci, rmse['K_bits']['M_K_coef']['all'], rmse['K_bits']['M_K_exp']['all'], eg, eg_ci, c)
    A['G3'] = dict(kappa=float(kap.mean()), kappa_ci=kap_ci, gamma1=float(g1v.mean()), gamma1_ci=g1_ci,
                   rmse_coef=rmse['K_bits']['M_K_coef']['all'], rmse_exp=rmse['K_bits']['M_K_exp']['all'],
                   exp_over_coef_gain=eg, exp_over_coef_gain_ci=eg_ci, G3a=g3a, G3b=g3b,
                   C3='coefficient (MDL)' if g3a == 'SUPPORTED' else 'exponent (quanta)' if g3b == 'SUPPORTED'
                   else 'unresolved')

    # G4: elasticities from M_K_both refit on all points, per seed
    ref = {}
    for unit in UNITS:
        ref[unit] = []
        for s in seeds:
            X = L.design([r for r in rows if r['seed'] == s], unit, c)
            ref[unit].append(L.fit('M_K_both', X, c)['params'])
    Xb = L.design(rows[:1], 'K_bits', c)
    Xn = L.design(rows[:1], 'K', c)
    ln = np.log([n for n in levels])
    lb = np.log([t for t in [20, 56, 144, 352]])
    slope = float(np.polyfit(ln, lb, 1)[0])
    K_center_n = float(np.exp(ln.mean()))

    def E_at(es, unit='K_bits'):
        Xu, kc = (Xb, c['K_bits_center']) if unit == 'K_bits' else (Xn, K_center_n)
        ea, ek = zip(*[L.elasticities('M_K_both', p, kc, es, Xu) for p in ref[unit]])
        ea, ek = np.array(ea), np.array(ek)
        est = ea.mean() / ek.mean() if ek.mean() > 0 else np.inf
        num, den = boot.mean(ea), boot.mean(ek)
        draws = np.where(den > 0, num / np.where(den > 0, den, 1), np.inf)
        return float(est), ci(draws), ea, ek, draws
    E_levels = {}
    for es in c['eps_star_levels']:
        e, eci, ea, ek, _ = E_at(es)
        E_levels[str(es)] = dict(E_bits=e, ci_bits=eci, E_states=e / slope, ci_states=[x / slope for x in eci],
                                 e_alpha=float(ea.mean()), e_K_bits=float(ek.mean()))
    E, E_ci, ea, ek, Ed = E_at(eps_p)
    E_n, E_n_ci = E_at(eps_p, 'K')[:2]
    grid = np.logspace(np.log10(c['eps_curve'][0]), np.log10(c['eps_curve'][1]), int(c['eps_curve'][2]))
    curve = [E_at(float(es))[0] for es in grid]
    flip = None
    for i in range(len(grid) - 1):
        a0, a1 = curve[i] - 1, curve[i + 1] - 1
        if np.isfinite(a0) and np.isfinite(a1) and a0 * a1 < 0:
            x = np.log10(grid[i]) + (np.log10(grid[i + 1]) - np.log10(grid[i])) * (-a0) / (a1 - a0)
            flip = float(10 ** x)
            break
    # model-free E
    def emf(es):
        ba, bk, cfr = [], [], []
        for s in seeds:
            Y, Xa, Xk = [], [], []
            n_c = 0
            for tid in tids:
                for a in c['alphas']:
                    v, fl = tstar.get((s, tid, a, es)) or t_star(Ts, [eps[(s, tid, a, T)] for T in Ts], es)
                    if fl:
                        n_c += 1
                        continue
                    Y.append(np.log2(v))
                    Xa.append(np.log2(1 / a))
                    Xk.append(np.log2(tinfo[tid]['K_bits']))
            cfr.append(n_c / (len(tids) * len(c['alphas'])))
            Z = np.column_stack([np.ones(len(Y)), Xa, Xk])
            beta = np.linalg.lstsq(Z, np.array(Y), rcond=None)[0]
            ba.append(beta[1])
            bk.append(beta[2])
        return np.array(ba), np.array(bk), cfr
    ba, bk, cfr = emf(eps_p)
    mf_eps = eps_p
    if np.median(cfr) > g['g4_mf_censor_fallback']:
        mf_eps = g['g4_mf_fallback_eps']
        ba, bk, cfr2 = emf(mf_eps)
    E_mf = float(ba.mean() / bk.mean()) if bk.mean() > 0 else float('inf')
    A['E'] = dict(levels=E_levels, curve=dict(eps_star=grid.tolist(), E_bits=curve), eps_flip=flip,
                  log_Kbits_on_log_n_slope=slope, E_refit_states=E_n, E_refit_states_ci=E_n_ci,
                  E_mf=E_mf, E_mf_eps_star=mf_eps, E_mf_censored_fraction_per_seed=cfr,
                  E_mf_coef_alpha=float(ba.mean()), E_mf_coef_K=float(bk.mean()),
                  refit_params_bits=ref['K_bits'])
    ok4 = E > g['g4_min_E'] and E_ci[0] > g['g4_min_E'] and E / slope > g['g4_min_E'] and E_ci[0] / slope > \
        g['g4_min_E'] and E_mf > g['g4_min_E']
    A['G4'] = dict(statistic='E = e_alpha / e_K (law-based, M_K_both on all points)', eps_star=eps_p,
                   estimate_bits=E, ci_bits=E_ci, estimate_states=E / slope, ci_states=[x / slope for x in E_ci],
                   E_mf=E_mf, E_mf_eps_star=mf_eps, threshold='E > 1 with lower bound > 1 in both units; E_mf > 1',
                   verdict=verdict(ok4))

    # range ratio R
    lr = np.array([np.mean([lt(s, tid, 0.05, eps_p) - lt(s, tid, 1.0, eps_p) for tid in tids]) for s in seeds])
    A['range_ratio_R'] = dict(estimate=float(2 ** (lr.mean() - mk_log.mean())),
                              ci_seed=ci(2 ** (boot.mean(lr) - boot.mean(mk_log))),
                              alignment_multiplier=float(2 ** lr.mean()), complexity_multiplier=MK)

    # alignment comparison and eta
    ag, ag_ci = gain('K_bits', 'M_align_add', 'M_align_eff')
    eta_eff = np.array([fits['fits']['K_bits'][str(s)]['M_align_eff']['params']['eta'] for s in seeds])
    eta_k = np.array([fits['fits']['K_bits'][str(s)][k_law]['params']['eta'] for s in seeds])
    A['alignment'] = dict(rmse_align_eff=rmse['K_bits']['M_align_eff']['all'],
                          rmse_align_add=rmse['K_bits']['M_align_add']['all'],
                          eff_over_add_gain=ag, eff_over_add_gain_ci=ag_ci,
                          invalid_align_add=invalid['K_bits']['M_align_add'],
                          invalid_align_eff=invalid['K_bits']['M_align_eff'],
                          eta_align_eff=float(eta_eff.mean()), eta_align_eff_ci=ci(boot.mean(eta_eff)),
                          eta_k_law=float(eta_k.mean()), eta_k_law_ci=ci(boot.mean(eta_k)),
                          eta_refit_both=float(ea.mean()), eta_refit_both_ci=ci(boot.mean(ea)))

    # G5: complexity gap at alpha = 1
    Tg = c['gap_reference_T']

    def gap(T, logscale=False):
        f = (lambda v: np.log2(v)) if logscale else (lambda v: v)
        return np.array([np.mean([f(eps[(s, tids[i], 1.0, T)]) for i in i32]) -
                         np.mean([f(eps[(s, tids[i], 1.0, T)]) for i in i4]) for s in seeds])
    g12, g16 = gap(Tg), gap(T_max)
    rho = float(g16.mean() / g12.mean())
    rho_ci = ci(boot.ratio(g16, g12))
    l12, l16 = gap(Tg, True), gap(T_max, True)
    A['G5'] = dict(statistic=f'rho = g({T_max}) / g({Tg}), gap in bits (n=32 minus n=4) at alpha=1',
                   g_ref=float(g12.mean()), g_max=float(g16.mean()), estimate=rho, ci_seed=rho_ci,
                   log2_gap_ref=float(l12.mean()), log2_gap_max=float(l16.mean()),
                   log2_rho=float(l16.mean() / l12.mean()), log2_rho_ci=ci(boot.ratio(l16, l12)),
                   threshold=f"g_ref > 0, rho <= {g['g5_max_ratio']}, upper bound < {g['g5_ub_below']}",
                   verdict=verdict(g12.mean() > 0 and rho <= g['g5_max_ratio'] and rho_ci[1] < g['g5_ub_below']))

    # G6: alignment floor
    a1 = np.array([fits['fits']['K_bits'][str(s)]['M_K_floor']['params']['a1'] for s in seeds])
    fg, fg_ci = gain('K_bits', k_law, 'M_K_floor')
    a1_ci = ci(boot.mean(a1))
    A['G6'] = dict(statistic=f'a1 in M_K_floor = {k_law} + a1 (1 - alpha); held-out gain over {k_law}',
                   a1=float(a1.mean()), a1_ci=a1_ci, estimate=fg, ci_seed=fg_ci,
                   rmse_floor=rmse['K_bits']['M_K_floor']['all'],
                   threshold=f"a1 lower bound > 0 and gain >= {g['g6_min_rel_gain']:.0%} with lower bound > 0",
                   verdict=verdict(a1_ci[0] > 0 and fg >= g['g6_min_rel_gain'] and fg_ci[0] > 0))

    # G7: per-target exponents versus visitation tail steepness
    gam = np.zeros((S, len(tids)))
    for si, s in enumerate(seeds):
        for j, tid in enumerate(tids):
            rs = [dict(K=1, K_bits=1, alpha=1.0, T=T, excess_bits=eps[(s, tid, 1.0, T)]) for T in Ts]
            X = L.design(rs, 'K_bits', c)
            gam[si, j] = L.fit('M0_scale', X, c)['params']['gamma']
    tau = np.array([tinfo[t]['visit_tail_steepness'] for t in tids])
    logk = np.log([tinfo[t]['K_bits'] for t in tids])
    sens = np.array([tinfo[t]['sensitivity'] for t in tids])

    def sp(x, y):
        return spearmanr(x, y).statistic

    def partial(x, y, z):
        rxy, rxz, ryz = sp(x, y), sp(x, z), sp(y, z)
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    gbar = gam.mean(0)
    rs_est = float(sp(gbar, tau))
    sd = np.array([sp(boot.W[b] @ gam / S, tau) for b in range(boot.draws)])
    s2 = np.empty(D2)
    for b in range(D2):
        idx = np.concatenate([lvl_idx[k][tw[b, k]] for k in range(len(levels))])
        s2[b] = sp((W2[b] @ gam / S)[idx], tau[idx])
    A['G7'] = dict(statistic='Spearman(gamma_j, tau_j) over 16 targets', estimate=rs_est, ci_seed=ci(sd),
                   ci_two_stage=ci(s2[np.isfinite(s2)]), partial_given_logK=float(partial(gbar, tau, logk)),
                   gamma_j=dict(zip(tids, gbar.tolist())), tau_j=dict(zip(tids, tau.tolist())),
                   threshold=f">= {g['g7_min_spearman']} with seed lower bound > 0",
                   verdict=verdict(rs_est >= g['g7_min_spearman'] and ci(sd)[0] > 0))

    # K versus sensitivity regression
    es_reg = c['eps_star_primary']
    Yt = np.array([[lt(s, tid, 1.0, es_reg) for tid in tids] for s in seeds])
    zk = (logk - logk.mean()) / logk.std()
    zs = (sens - sens.mean()) / sens.std()
    Z = np.column_stack([np.ones(len(tids)), zk, zs])

    def ols(y, idx=None):
        Zi = Z if idx is None else Z[idx]
        yi = y if idx is None else y[idx]
        return np.linalg.lstsq(Zi, yi, rcond=None)[0][1:]
    bhat = ols(Yt.mean(0))
    bs = np.array([ols(boot.W[b] @ Yt / S) for b in range(boot.draws)])
    b2 = np.array([ols(W2[b] @ Yt / S, np.concatenate([lvl_idx[k][tw[b, k]] for k in range(len(levels))]))
                   for b in range(D2)])
    r_ks = np.corrcoef(zk, zs)[0, 1]
    A['K_vs_sensitivity'] = dict(eps_star=es_reg, coef_logK=float(bhat[0]), coef_logK_ci_seed=ci(bs[:, 0]),
                                 coef_logK_ci_two_stage=ci(b2[:, 0]), coef_sensitivity=float(bhat[1]),
                                 coef_sensitivity_ci_seed=ci(bs[:, 1]), coef_sensitivity_ci_two_stage=ci(b2[:, 1]),
                                 corr_logK_sensitivity=float(r_ks), vif=float(1 / (1 - r_ks ** 2)),
                                 alpha1_censored=cens)

    # plateau census and per-length loss
    Tp = c['plateau_reference_T']
    pc = {str(n): {str(a): 0 for a in c['alphas']} for n in levels}
    for s in seeds:
        for tid in tids:
            for a in c['alphas']:
                e16, e14 = eps[(s, tid, a, T_max)], eps[(s, tid, a, Tp)]
                if e16 >= c['plateau_ratio'] * e14 and e16 > c['plateau_min_bits']:
                    pc[str(tinfo[tid]['K'])][str(a)] += 1
    A['plateau_census'] = dict(counts=pc, runs_per_cell=S * 4)
    bylen = {r['target_id']: [] for r in rows}
    pl = {}
    for n in levels:
        v = [r['excess_bits_by_length'] for r in rows if r['K'] == n and r['alpha'] == 1.0 and r['T'] == T_max]
        pl[str(n)] = np.mean(v, axis=0).tolist()
    A['per_length_loss_alpha1_Tmax'] = dict(lengths=list(range(c['length_min'], c['length_max'] + 1)), by_level=pl)

    # descriptive tables
    A['tstar_summary'] = {tid: {str(es): {str(a): dict(gm=float(2 ** np.mean([lt(s, tid, a, es) for s in seeds])),
                                                       censored=sum(bool(tstar[(s, tid, a, es)][1]) for s in seeds))
                                          for a in c['alphas']} for es in c['eps_star_levels']} for tid in tids}
    A['curves_alpha1'] = {tid: [float(np.mean([eps[(s, tid, 1.0, T)] for s in seeds])) for T in Ts] for tid in tids}
    A['fitter_agreement'] = dict(all_agree=fits['all_fitters_agree'],
                                 max_rel_forecast_diff=max(a['max_rel_forecast_diff'] for a in fits['fitter_agreement']))
    A['g2_k_law'] = k_law
    A['forecasts_frozen_sha256'] = fits['forecasts_frozen_sha256']
    A['gates'] = {k: A[k]['verdict'] for k in ('G1', 'G2', 'G4', 'G5', 'G6', 'G7')}
    A['gates']['G3a'], A['gates']['G3b'] = A['G3']['G3a'], A['G3']['G3b']
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--results', default=str(HERE / 'results'))
    args = ap.parse_args()
    c = load_config()
    res = Path(args.results)
    tasks = json.loads((res / 'tasks.json').read_text())
    if args.stage == 1:
        h, out = stage1(c, res, tasks)
        print(json.dumps(dict(forecasts_frozen_sha256=h, g2_k_law=out['g2_k_law'], g2_fit_rmse=out['g2_fit_rmse'],
                              all_fitters_agree=out['all_fitters_agree'])))
    else:
        A = stage2(c, res, tasks)
        (res / 'analysis.json').write_text(json.dumps(A, indent=1, default=float))
        print(json.dumps(A['gates'], indent=1))


if __name__ == '__main__':
    main()
