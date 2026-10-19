"""Experiment 12 analysis (PROTOCOL Sections 3-5). Arithmetic mirrors Experiment 11's analyze.py (commit 9120e5ff).

  analyze.py --stage 1   reads fit rows of the main arm only; fits every law per seed (both K units, both fitters),
                         picks the G2 K law by fit-cell RMSE, writes fits.json and forecasts_frozen.json and prints
                         the forecasts' SHA-256
  analyze.py --stage 2   reads everything; held-out scores, T*, multipliers, elasticities, mechanism and confound
                         analyses, the secondary arms and gates G1-G9 into tstar.csv and analysis.json
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
    return (r.get('arm', 'main') == 'main' and int(r['K']) in c['fit_K'] and float(r['alpha']) in c['fit_alphas']
            and int(r['T']) <= c['fit_T_max'])


def read_rows(path, c, which, arm='main'):
    """Rows of metrics.csv for one arm; which = 'fit' drops held-out rows at read time, 'all' keeps every row."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['arm'] != arm:
                continue
            r = dict(arm=r['arm'], seed=int(r['seed']), target_id=r['target_id'], K=int(r['K']),
                     K_bits=int(r['K_bits']), alpha=float(r['alpha']), T=int(r['T']),
                     excess_bits=float(r['excess_bits']), accuracy=float(r['accuracy']),
                     train_bits=float(r['train_bits']),
                     excess_bits_by_output_bit=json.loads(r['excess_bits_by_output_bit']))
            if which == 'all' or is_fit(r, c):
                out.append(r)
    return out


def heldout_grid(c, tasks):
    """Coordinates of every held-out point (no observations): the K = 4, alpha = 0.15 and T = T_max slices."""
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
                res_fit = L.scored_residual(law, a['theta'], X)[0]
                fs[law] = dict(a, n_fit=len(rs), fit_sse_log2=float(np.sum(res_fit ** 2)),
                               independent=dict(theta=b['theta'], cost=b['cost']))
            fits[unit][str(s)] = fs
    # G2 K law: lower pooled fit-cell RMSE (primary unit), chosen before held-out rows are read
    rm = {law: float(np.sqrt(sum(fits['K_bits'][str(s)][law]['fit_sse_log2'] for s in seeds) /
                             sum(fits['K_bits'][str(s)][law]['n_fit'] for s in seeds)))
          for law in ('M_K_coef', 'M_K_exp')}
    k_law = min(rm, key=rm.get)
    for unit in UNITS:
        Xh = L.design(grid, unit, c)
        for s in seeds:
            rs = [r for r in rows if r['seed'] == s]
            X = L.design(rs, unit, c)
            fk = fits[unit][str(s)][k_law]
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
    x = x[~np.isnan(x)]
    if not len(x):
        return [float('nan'), float('nan')]
    x = np.where(np.isposinf(x), 1e300, np.where(np.isneginf(x), -1e300, x))
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


def gate_g9(exp11, this_g3, c):
    """G9 (PROTOCOL Section 5): same decided G3 verdict as Experiment 11, same sign of the carrying M_K_both
    parameter, ratio of point estimates (this / Experiment 11) in g9_ratio_range."""
    lo, hi = c['gates']['g9_ratio_range']
    if exp11 is None:
        return dict(verdict='NOT_ESTABLISHED', reason='Experiment 11 unavailable')
    g = exp11.get('G3', {})
    e11 = 'G3a' if g.get('G3a') == 'SUPPORTED' else 'G3b' if g.get('G3b') == 'SUPPORTED' else None
    mine = 'G3a' if this_g3.get('G3a') == 'SUPPORTED' else 'G3b' if this_g3.get('G3b') == 'SUPPORTED' else None
    out = dict(exp11_g3=e11 or 'unresolved', this_g3=mine or 'unresolved')
    if e11 is None:
        return dict(out, verdict='NOT_ESTABLISHED', reason='Experiment 11 G3 is unresolved')
    if mine != e11:
        return dict(out, verdict='NOT_ESTABLISHED', reason='G3 verdicts differ')
    key = 'kappa' if e11 == 'G3a' else 'gamma1'
    a, b = float(this_g3[key]), float(g[key])
    ratio = a / b if b != 0 else float('inf')
    ok = np.sign(a) == np.sign(b) and lo <= ratio <= hi
    return dict(out, parameter=key, this=a, exp11=b, ratio=ratio, verdict='SUPPORTED' if ok else 'NOT_ESTABLISHED',
                reason='' if ok else 'sign or ratio outside range')


def verdict(ok):
    return 'SUPPORTED' if ok else 'NOT_ESTABLISHED'


def sp(x, y):
    v = spearmanr(x, y).statistic
    return float(v) if np.isfinite(v) else float('nan')


def partial(x, y, z):
    rxy, rxz, ryz = sp(x, y), sp(x, z), sp(y, z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def load_exp11(c):
    for p in c['exp11_analysis_paths']:
        q = (HERE / p).resolve()
        if q.exists():
            return json.loads(q.read_text()), str(q), sha256(q)
    return None, None, None


# ---------------------------------------------------------------- stage 2

def stage2(c, res, tasks, two_stage_draws=None, exp11=None, use_exp11_file=True):
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
    levels = c['K_levels']
    by_level = {k: [i for i, t in enumerate(targets) if t['K'] == k] for k in levels}
    lvl_idx = [np.array(by_level[k]) for k in levels]
    i_hi, i_lo = by_level[levels[-1]], by_level[levels[0]]
    eps, trb, byb = {}, {}, {}
    for r in rows:
        key = (r['seed'], r['target_id'], r['alpha'], r['T'])
        eps[key], trb[key], byb[key] = r['excess_bits'], r['train_bits'], r['excess_bits_by_output_bit']
    A = {}
    g = c['gates']

    # held-out scoring
    obs = {(s, p['target_id'], p['alpha'], p['T']): eps[(s, p['target_id'], p['alpha'], p['T'])]
           for s in seeds for p in frozen['points']}
    slices = dict(all=lambda p: True, K4=lambda p: p['K'] in c['heldout_K'],
                  alpha015=lambda p: p['alpha'] in c['heldout_alphas'], Tmax=lambda p: p['T'] == T_max)
    sse = {u: {} for u in UNITS}
    nn = {u: {} for u in UNITS}
    invalid = {u: {} for u in UNITS}
    for unit in UNITS:
        for law in L.LAWS + ['M_K_floor']:
            for sl in slices:
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

    # T* table (examples and target passes)
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
                                      alpha=a, eps_star=es, T_star=v, T_star_passes=a * v / c['n_train'],
                                      censored=fl))
    with open(res / 'tstar.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(trows[0]))
        w.writeheader()
        w.writerows(trows)
    eps_p = c['eps_star_primary']

    def lt(s, tid, a, es):
        return np.log2(tstar[(s, tid, a, es)][0])

    # G1: complexity multiplier at alpha = 1
    LT = np.array([[lt(s, tid, 1.0, eps_p) for tid in tids] for s in seeds])  # seeds x targets
    mk_log = LT[:, i_hi].mean(1) - LT[:, i_lo].mean(1)
    MK = float(2 ** mk_log.mean())
    MK_seed = ci(2 ** boot.mean(mk_log))
    trng = np.random.default_rng(c['analysis_seed'] + 1)
    D2 = two_stage_draws or c['bootstrap_draws']
    tw = trng.integers(0, c['targets_per_level'], size=(D2, len(levels), c['targets_per_level']))
    W2 = boot.W[:D2]

    def two_stage_idx(b):
        return np.concatenate([lvl_idx[k][tw[b, k]] for k in range(len(levels))])
    mk2 = np.empty(D2)
    for b in range(D2):
        mk2[b] = 2 ** (W2[b] @ (LT[:, lvl_idx[-1][tw[b, -1]]].mean(1) - LT[:, lvl_idx[0][tw[b, 0]]].mean(1)) / S)
    MK_two = ci(mk2)
    cens = {fl: int(sum(tstar[(s, tid, 1.0, eps_p)][1] == fl for s in seeds for tid in tids))
            for fl in ('left', 'right')}
    A['G1'] = dict(statistic='M_K = GM T*(K=6) / GM T*(K=2) at alpha=1', eps_star=eps_p, estimate=MK,
                   ci_seed=MK_seed, ci_two_stage=MK_two, per_seed=(2 ** mk_log).tolist(), censored_alpha1=cens,
                   threshold=f">= {g['g1_min_multiplier']}, both lower bounds > {g['g1_lb_above']}",
                   verdict=verdict(MK >= g['g1_min_multiplier'] and MK_seed[0] > g['g1_lb_above']
                                   and MK_two[0] > g['g1_lb_above']))

    def mk_of(idx_hi, idx_lo, LTm):
        v = LTm[:, idx_hi].mean(1) - LTm[:, idx_lo].mean(1)
        return float(2 ** v.mean()), ci(2 ** boot.mean(v))
    aff = np.array([tinfo[t]['is_affine'] for t in tids])
    A['M_K_by_type'] = {
        'affine': dict(zip(['estimate', 'ci_seed'], mk_of([i for i in i_hi if aff[i]], [i for i in i_lo if aff[i]], LT))),
        'nonaffine': dict(zip(['estimate', 'ci_seed'], mk_of([i for i in i_hi if not aff[i]],
                                                             [i for i in i_lo if not aff[i]], LT)))}

    # G2: K law versus M_align_eff
    gk, gk_ci = gain('K_bits', 'M_align_eff', k_law)
    k4 = rmse['K_bits'][k_law]['K4'] < rmse['K_bits']['M_align_eff']['K4']
    A['G2'] = dict(statistic=f'held-out RMSE reduction of {k_law} over M_align_eff (log2 bits)', k_law=k_law,
                   estimate=gk, ci_seed=gk_ci, rmse_k_law=rmse['K_bits'][k_law]['all'],
                   rmse_align_eff=rmse['K_bits']['M_align_eff']['all'],
                   k4_rmse_k_law=rmse['K_bits'][k_law]['K4'], k4_rmse_align_eff=rmse['K_bits']['M_align_eff']['K4'],
                   invalid_k_law=invalid['K_bits'][k_law],
                   unit_program_length=dict(zip(['estimate', 'ci_seed'], gain('K', 'M_align_eff', k_law))),
                   threshold=f">= {g['g2_min_rel_gain']:.0%}, lower bound > 0, lower on K=4, 0 invalid",
                   verdict=verdict(gk >= g['g2_min_rel_gain'] and gk_ci[0] > 0 and k4
                                   and invalid['K_bits'][k_law] == 0))

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

    # G4: elasticities from M_K_both refit on all main points, per seed
    ref = {}
    for unit in UNITS:
        ref[unit] = []
        for s in seeds:
            X = L.design([r for r in rows if r['seed'] == s], unit, c)
            ref[unit].append(L.fit('M_K_both', X, c)['params'])
    Xb = L.design(rows[:1], 'K_bits', c)
    Xn = L.design(rows[:1], 'K', c)
    slope = float(np.polyfit(np.log(levels), np.log(c['K_bits']), 1)[0])
    assert abs(slope - c['len_to_bits_slope']) < 1e-3, slope
    slope = c['len_to_bits_slope']

    def E_at(es, unit='K_bits'):
        Xu, kc = (Xb, c['K_bits_center']) if unit == 'K_bits' else (Xn, c['fit']['K_centers']['K'])
        ea, ek = zip(*[L.elasticities('M_K_both', p, kc, es, Xu) for p in ref[unit]])
        ea, ek = np.array(ea), np.array(ek)
        est = ea.mean() / ek.mean() if ek.mean() > 0 else np.inf
        num, den = boot.mean(ea), boot.mean(ek)
        draws = np.where(den > 0, num / np.where(den > 0, den, 1), np.inf)
        return float(est), ci(draws), ea, ek
    E_levels = {}
    for es in c['eps_star_levels']:
        e, eci, ea_, ek_ = E_at(es)
        E_levels[str(es)] = dict(E_bits=e, ci_bits=eci, E_len=e / slope, ci_len=[x / slope for x in eci],
                                 e_alpha=float(ea_.mean()), e_K_bits=float(ek_.mean()))
    E, E_ci, ea, ek = E_at(eps_p)
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
            if len(Y) >= 3:
                beta = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), Xa, Xk]), np.array(Y), rcond=None)[0]
            else:
                beta = [np.nan, np.nan, np.nan]
            ba.append(beta[1])
            bk.append(beta[2])
        return np.array(ba, float), np.array(bk, float), cfr
    ba, bk, cfr = emf(eps_p)
    mf_eps = eps_p
    if np.median(cfr) > g['g4_mf_censor_fallback']:
        mf_eps = g['g4_mf_fallback_eps']
        ba, bk, cfr_fb = emf(mf_eps)
    else:
        cfr_fb = None
    bkm = np.nanmean(bk) if np.isfinite(bk).any() else np.nan
    bam = np.nanmean(ba) if np.isfinite(ba).any() else np.nan
    E_mf = float(bam / bkm) if np.isfinite(bkm) and bkm > 0 else (float('inf') if np.isfinite(bkm) else float('nan'))
    A['E'] = dict(levels=E_levels, curve=dict(eps_star=grid.tolist(), E_bits=curve), eps_flip=flip,
                  log_Kbits_on_log_len_slope=slope, E_refit_len=E_n, E_refit_len_ci=E_n_ci,
                  E_mf=E_mf, E_mf_eps_star=mf_eps, E_mf_censored_fraction_per_seed=cfr,
                  E_mf_censored_fraction_fallback=cfr_fb, E_mf_uncensored_pairs_per_seed=[
                      int(round((1 - x) * len(tids) * len(c['alphas']))) for x in (cfr_fb or cfr)],
                  E_mf_coef_alpha=float(bam), E_mf_coef_K=float(bkm), refit_params_bits=ref['K_bits'])
    ok4 = (E > g['g4_min_E'] and E_ci[0] > g['g4_min_E'] and E / slope > g['g4_min_E'] and
           E_ci[0] / slope > g['g4_min_E'] and E_mf > g['g4_min_E'])
    A['G4'] = dict(statistic='E = e_alpha / e_K (law-based, M_K_both on all points)', eps_star=eps_p,
                   estimate_bits=E, ci_bits=E_ci, estimate_len=E / slope, ci_len=[x / slope for x in E_ci],
                   E_mf=E_mf, E_mf_eps_star=mf_eps, threshold='E > 1 with lower bound > 1 in both units; E_mf > 1',
                   verdict=verdict(bool(ok4)))

    # range ratio R
    lr = np.array([np.mean([lt(s, tid, 0.05, eps_p) - lt(s, tid, 1.0, eps_p) for tid in tids]) for s in seeds])
    A['range_ratio_R'] = dict(estimate=float(2 ** (lr.mean() - mk_log.mean())),
                              ci_seed=ci(2 ** (boot.mean(lr) - boot.mean(mk_log))),
                              alignment_multiplier=float(2 ** lr.mean()), complexity_multiplier=MK)

    # alignment comparison and eta (Section 4.5)
    ag, ag_ci = gain('K_bits', 'M_align_add', 'M_align_eff')
    eta_eff = np.array([fits['fits']['K_bits'][str(s)]['M_align_eff']['params']['eta'] for s in seeds])
    eta_k = np.array([fits['fits']['K_bits'][str(s)][k_law]['params']['eta'] for s in seeds])
    dprims = set(p for t in tasks if t['role'] == 'distractor' for p in t['definition']['primitives'])
    eta_j, overlap = {}, {}
    for tid in tids:
        v = []
        for s in seeds:
            rs = [r for r in rows if r['seed'] == s and r['target_id'] == tid]
            v.append(L.fit('M_align_eff', L.design(rs, 'K_bits', c), c)['params']['eta'])
        eta_j[tid] = float(np.mean(v))
        overlap[tid] = len(set(tinfo[tid]['definition']['primitives']) & dprims)
    ej = np.array([eta_j[t] for t in tids])
    A['alignment'] = dict(rmse_align_eff=rmse['K_bits']['M_align_eff']['all'],
                          rmse_align_add=rmse['K_bits']['M_align_add']['all'],
                          eff_over_add_gain=ag, eff_over_add_gain_ci=ag_ci,
                          invalid_align_add=invalid['K_bits']['M_align_add'],
                          invalid_align_eff=invalid['K_bits']['M_align_eff'],
                          eta_align_eff=float(eta_eff.mean()), eta_align_eff_ci=ci(boot.mean(eta_eff)),
                          eta_k_law=float(eta_k.mean()), eta_k_law_ci=ci(boot.mean(eta_k)),
                          eta_refit_both=float(ea.mean()), eta_refit_both_ci=ci(boot.mean(ea)),
                          eta_j=eta_j, primitive_overlap_j=overlap,
                          spearman_eta_j_K=sp(ej, [tinfo[t]['K'] for t in tids]),
                          spearman_eta_j_overlap=sp(ej, [overlap[t] for t in tids]))

    # G5: complexity gap at alpha = 1
    Tg = c['gap_reference_T']

    def gap(T, logscale=False):
        f = (lambda v: np.log2(v)) if logscale else (lambda v: v)
        return np.array([np.mean([f(eps[(s, tids[i], 1.0, T)]) for i in i_hi]) -
                         np.mean([f(eps[(s, tids[i], 1.0, T)]) for i in i_lo]) for s in seeds])
    g12, g16 = gap(Tg), gap(T_max)
    rho = float(g16.mean() / g12.mean())
    rho_ci = ci(boot.ratio(g16, g12))
    l12, l16 = gap(Tg, True), gap(T_max, True)
    A['G5'] = dict(statistic=f'rho = g({T_max}) / g({Tg}), gap in bits (K=6 minus K=2) at alpha=1',
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

    # G7: per-target exponents versus local-rule entropy
    gam = np.zeros((S, len(tids)))
    for si, s in enumerate(seeds):
        for j, tid in enumerate(tids):
            rs = [dict(K=2, K_bits=11, alpha=1.0, T=T, excess_bits=eps[(s, tid, 1.0, T)]) for T in Ts]
            gam[si, j] = L.fit('M0_scale', L.design(rs, 'K_bits', c), c)['params']['gamma']
    H = np.array([tinfo[t]['local_rule_entropy'] for t in tids])
    logk = np.log([tinfo[t]['K_bits'] for t in tids])
    sens = np.array([tinfo[t]['sensitivity'] for t in tids])
    gbar = gam.mean(0)
    rs_est = sp(gbar, H)
    sd = np.array([sp(boot.W[b] @ gam / S, H) for b in range(boot.draws)])
    s2 = np.array([sp((W2[b] @ gam / S)[two_stage_idx(b)], H[two_stage_idx(b)]) for b in range(D2)])
    na = ~aff
    rs_na = sp(gbar[na], H[na])
    sd_ci = ci(sd)
    A['G7'] = dict(statistic='Spearman(gamma_j, H_rule_j) over 20 targets', estimate=rs_est, ci_seed=sd_ci,
                   ci_two_stage=ci(s2), nonaffine_estimate=rs_na, nonaffine_ci_seed=ci(
                       [sp((boot.W[b] @ gam / S)[na], H[na]) for b in range(min(boot.draws, 5000))]),
                   partial_given_logK=partial(gbar, H, logk),
                   gamma_j=dict(zip(tids, gbar.tolist())), H_rule_j=dict(zip(tids, H.tolist())),
                   threshold=f"<= {g['g7_max_spearman']} with seed upper bound < 0, and < 0 over the 10 non-affine",
                   verdict=verdict(np.isfinite(rs_est) and rs_est <= g['g7_max_spearman'] and sd_ci[1] < 0
                                   and np.isfinite(rs_na) and rs_na < 0))

    # G8: CTM/BDM cross-check
    allK = np.array([t['K'] for t in tasks], float)
    bdm_all = np.array([t['bdm_rowmajor'] for t in tasks], float)
    g8a_est = sp(bdm_all, allK)
    rng8 = np.random.default_rng(c['analysis_seed'] + 8)
    lv_all = [np.flatnonzero(allK == k) for k in levels]
    g8a_draws = np.empty(c['bootstrap_draws'])
    for b in range(len(g8a_draws)):
        idx = np.concatenate([q[rng8.integers(0, len(q), len(q))] for q in lv_all])
        g8a_draws[b] = sp(bdm_all[idx], allK[idx])
    g8a_ci = ci(g8a_draws)
    bdm_t = np.array([tinfo[t]['bdm_rowmajor'] for t in tids])
    lt_mean = LT.mean(0)
    g8b_est = sp(bdm_t, lt_mean)
    g8b_draws = np.array([sp(bdm_t, boot.W[b] @ LT / S) for b in range(boot.draws)])
    g8b_ci = ci(g8b_draws)
    other = {}
    for key in ('bdm_bitplane', 'zlib_bytes', 'lzma_bytes', 'zlib_u16_bytes'):
        v_all = np.array([t[key] for t in tasks], float)
        v_t = np.array([tinfo[t][key] for t in tids], float)
        other[key] = dict(spearman_with_K_200=sp(v_all, allK), spearman_with_Tstar_20=sp(v_t, lt_mean))
    gm8 = g['g8_min_spearman']
    A['G8a'] = dict(statistic='Spearman(bdm_rowmajor, K) over 200 tasks', estimate=g8a_est, ci_task=g8a_ci,
                    bdm_source=tasks[0].get('bdm_source'),
                    threshold=f'>= {gm8} with task-bootstrap lower bound > 0',
                    verdict=verdict(np.isfinite(g8a_est) and g8a_est >= gm8 and g8a_ci[0] > 0))
    A['G8b'] = dict(statistic=f'Spearman(bdm_rowmajor_j, T*_j) at alpha=1, eps*={eps_p}, over 20 targets',
                    estimate=g8b_est, ci_seed=g8b_ci, threshold=f'>= {gm8} with seed lower bound > 0',
                    verdict=verdict(np.isfinite(g8b_est) and g8b_est >= gm8 and g8b_ci[0] > 0))
    A['proxies_other'] = other

    # simplicity bias versus program length
    zl = np.array([tinfo[t]['zlib_bytes'] for t in tids], float)
    Kt = np.array([tinfo[t]['K'] for t in tids], float)
    A['simplicity_bias'] = dict(eps_star=eps_p, spearman_Tstar=dict(
        K=sp(lt_mean, Kt), bdm_rowmajor=sp(lt_mean, bdm_t),
        bdm_bitplane=sp(lt_mean, [tinfo[t]['bdm_bitplane'] for t in tids]), zlib=sp(lt_mean, zl),
        lzma=sp(lt_mean, [tinfo[t]['lzma_bytes'] for t in tids])),
        partial_Tstar_K_given_zlib=partial(lt_mean, Kt, zl), partial_Tstar_zlib_given_K=partial(lt_mean, zl, Kt))

    # K versus sensitivity versus type regression
    es_reg = c['regression_eps_star']
    Yt = np.array([[lt(s, tid, 1.0, es_reg) for tid in tids] for s in seeds])
    zk = (logk - logk.mean()) / logk.std()
    zs = (sens - sens.mean()) / sens.std()
    Z = np.column_stack([np.ones(len(tids)), zk, zs, aff.astype(float)])

    def ols(y, idx=None):
        Zi = Z if idx is None else Z[idx]
        yi = y if idx is None else y[idx]
        return np.linalg.lstsq(Zi, yi, rcond=None)[0][1:]
    bhat = ols(Yt.mean(0))
    bs = np.array([ols(boot.W[b] @ Yt / S) for b in range(boot.draws)])
    b2 = np.array([ols(W2[b] @ Yt / S, two_stage_idx(b)) for b in range(D2)])
    R = np.corrcoef(Z[:, 1:].T)
    vif = np.diag(np.linalg.pinv(R)).tolist()
    names = ['logK_bits', 'sensitivity', 'affine']
    A['K_vs_sensitivity_vs_type'] = dict(
        eps_star=es_reg, coefficients={n: dict(estimate=float(bhat[i]), ci_seed=ci(bs[:, i]), ci_two_stage=ci(b2[:, i]))
                                       for i, n in enumerate(names)},
        vif=dict(zip(names, vif)), censored_alpha1=int(sum(bool(tstar[(s, t, 1.0, es_reg)][1])
                                                            for s in seeds for t in tids)))

    # train-fit census
    Tp = c['plateau_reference_T']
    census = {}
    for tid in tids:
        key = f"K{tinfo[tid]['K']}_{'affine' if tinfo[tid]['is_affine'] else 'nonaffine'}"
        for a in c['alphas']:
            cc = census.setdefault(key, {}).setdefault(str(a), dict(runs=0, plateau=0, memorizing=0, not_fitting=0))
            for s in seeds:
                cc['runs'] += 1
                e16, e14 = eps[(s, tid, a, T_max)], eps[(s, tid, a, Tp)]
                if e16 >= c['plateau_ratio'] * e14 and e16 > c['plateau_min_bits']:
                    cc['plateau'] += 1
                    cc['memorizing'] += trb[(s, tid, a, T_max)] <= c['memorize_train_bits']
                    cc['not_fitting'] += trb[(s, tid, a, T_max)] > c['not_fit_train_bits']
    A['train_fit_census'] = census
    tot = dict(runs=0, plateau=0, memorizing=0, not_fitting=0)
    for v in census.values():
        for cc in v.values():
            for k in tot:
                tot[k] += int(cc[k])
    A['train_fit_census_total'] = tot
    A['per_output_bit_loss_alpha1_Tmax'] = {str(k): np.mean([byb[(s, tids[i], 1.0, T_max)] for s in seeds
                                                             for i in by_level[k]], axis=0).tolist() for k in levels}

    # secondary arms
    A['rep768'] = rep768_analysis(c, res, tasks, tinfo, tids, by_level, eps_p)
    A['mlp'] = mlp_analysis(c, res, tasks, tinfo, tids, by_level, eps_p)

    # G9: cross-language agreement with Experiment 11
    if exp11 is None and use_exp11_file:
        exp11, p11, h11 = load_exp11(c)
    else:
        p11 = h11 = None
    A['G9'] = dict(gate_g9(exp11, dict(G3a=g3a, G3b=g3b, kappa=A['G3']['kappa'], gamma1=A['G3']['gamma1']), c),
                   exp11_analysis_path=p11, exp11_analysis_sha256=h11)

    # descriptive tables
    A['tstar_summary'] = {tid: {str(es): {str(a): dict(
        gm=float(2 ** np.mean([lt(s, tid, a, es) for s in seeds])),
        gm_passes=float(a * 2 ** np.mean([lt(s, tid, a, es) for s in seeds]) / c['n_train']),
        censored=int(sum(bool(tstar[(s, tid, a, es)][1]) for s in seeds)))
        for a in c['alphas']} for es in c['eps_star_levels']} for tid in tids}
    A['curves_alpha1'] = {tid: [float(np.mean([eps[(s, tid, 1.0, T)] for s in seeds])) for T in Ts] for tid in tids}
    A['train_curves_alpha1'] = {tid: [float(np.mean([trb[(s, tid, 1.0, T)] for s in seeds])) for T in Ts]
                                for tid in tids}
    A['fitter_agreement'] = dict(all_agree=fits['all_fitters_agree'],
                                 max_rel_forecast_diff=max(a['max_rel_forecast_diff'] for a in fits['fitter_agreement']))
    A['g2_k_law'] = k_law
    A['forecasts_frozen_sha256'] = fits['forecasts_frozen_sha256']
    A['gates'] = {k: A[k]['verdict'] for k in ('G1', 'G2', 'G4', 'G5', 'G6', 'G7', 'G8a', 'G8b', 'G9')}
    A['gates']['G3a'], A['gates']['G3b'] = A['G3']['G3a'], A['G3']['G3b']
    return A


def _arm_tstar_mk(c, eps_arm, seeds, tids, by_level, eps_p, alpha=1.0):
    Ts = c['T_checkpoints']
    LT = np.array([[np.log2(t_star(Ts, [eps_arm[(s, t, alpha, T)] for T in Ts], eps_p)[0]) for t in tids]
                   for s in seeds])
    lv = c['K_levels']
    v = LT[:, by_level[lv[-1]]].mean(1) - LT[:, by_level[lv[0]]].mean(1)
    boot = Boot(len(seeds), c['bootstrap_draws'], c['analysis_seed'] + 2)
    return float(2 ** v.mean()), ci(2 ** boot.mean(v))


def _arm_census(c, eps_arm, trb_arm, seeds, tids, alphas):
    Tm, Tp = c['T_checkpoints'][-1], c['plateau_reference_T']
    out = dict(runs=0, plateau=0, memorizing=0, not_fitting=0)
    for s in seeds:
        for t in tids:
            for a in alphas:
                out['runs'] += 1
                e16, e14 = eps_arm[(s, t, a, Tm)], eps_arm[(s, t, a, Tp)]
                if e16 >= c['plateau_ratio'] * e14 and e16 > c['plateau_min_bits']:
                    out['plateau'] += 1
                    out['memorizing'] += trb_arm[(s, t, a, Tm)] <= c['memorize_train_bits']
                    out['not_fitting'] += trb_arm[(s, t, a, Tm)] > c['not_fit_train_bits']
    return out


def rep768_analysis(c, res, tasks, tinfo, tids, by_level, eps_p):
    rows = read_rows(res / 'metrics.csv', c, 'all', arm='rep768')
    if not rows:
        return dict(ran=False)
    cfg = c['rep768']
    seeds = sorted({r['seed'] for r in rows})
    e768 = {(r['seed'], r['target_id'], r['alpha'], r['T']): r['excess_bits'] for r in rows}
    t768 = {(r['seed'], r['target_id'], r['alpha'], r['T']): r['train_bits'] for r in rows}
    n_shared = c['n_test'] - (cfg['n_train'] - c['n_train'])  # perm[768:] = main test rows 256..511
    e512 = {}
    for s in seeds:
        for t in tids:
            for a in cfg['alphas']:
                z = np.load(res / 'predictions' / f'{s}_{t}_{a:.2f}.npz')
                v = z['test'][:, -n_shared:].astype(np.float64).mean(1)
                for i, T in enumerate(c['T_checkpoints']):
                    e512[(s, t, a, T)] = float(v[i])
    ratios = {str(a): {} for a in cfg['alphas']}
    for a in cfg['alphas']:
        for T in c['T_checkpoints']:
            ratios[str(a)][str(T)] = float(np.median([e768[(s, t, a, T)] / e512[(s, t, a, T)]
                                                     for s in seeds for t in tids]))
    Tm = str(c['T_checkpoints'][-1])
    med = ratios['1.0'][Tm] if '1.0' in ratios else None
    out = dict(ran=True, seeds=seeds, cells=len({(r['seed'], r['target_id'], r['alpha']) for r in rows}),
               n_shared_test_inputs=n_shared, median_ratio_768_over_512=ratios, median_ratio_alpha1_Tmax=med,
               note_partly_finite_input_generalization=bool(med is not None and med < cfg['note_ratio']))
    out['M_K_768'] = dict(zip(['estimate', 'ci_seed'], _arm_tstar_mk(c, e768, seeds, tids, by_level, eps_p)))
    out['M_K_512_shared'] = dict(zip(['estimate', 'ci_seed'], _arm_tstar_mk(c, e512, seeds, tids, by_level, eps_p)))
    etas = {}
    for name, E in (('768', e768), ('512_shared', e512)):
        v = []
        for s in seeds:
            rs = [dict(K=tinfo[t]['K'], K_bits=tinfo[t]['K_bits'], alpha=a, T=T, excess_bits=E[(s, t, a, T)])
                  for t in tids for a in cfg['alphas'] for T in c['T_checkpoints']]
            v.append(L.fit('M_align_eff', L.design(rs, 'K_bits', c), c)['params']['eta'])
        boot = Boot(len(seeds), c['bootstrap_draws'], c['analysis_seed'] + 3)
        etas[name] = dict(estimate=float(np.mean(v)), ci_seed=ci(boot.mean(v)), per_seed=v)
    out['eta_two_alpha'] = etas
    out['train_fit_census_768'] = _arm_census(c, e768, t768, seeds, tids, cfg['alphas'])
    return out


def mlp_analysis(c, res, tasks, tinfo, tids, by_level, eps_p):
    rows = read_rows(res / 'metrics.csv', c, 'all', arm='mlp')
    if not rows:
        return dict(ran=False)
    seeds = sorted({r['seed'] for r in rows})
    e = {(r['seed'], r['target_id'], r['alpha'], r['T']): r['excess_bits'] for r in rows}
    tb = {(r['seed'], r['target_id'], r['alpha'], r['T']): r['train_bits'] for r in rows}
    mk = _arm_tstar_mk(c, e, seeds, tids, by_level, eps_p)
    Tm = c['T_checkpoints'][-1]
    return dict(ran=True, seeds=seeds, M_K=dict(estimate=mk[0], ci_seed=mk[1]),
                train_fit_census=_arm_census(c, e, tb, seeds, tids, c['mlp']['alphas']),
                mean_test_bits_Tmax_by_level={str(k): float(np.mean([e[(s, tids[i], 1.0, Tm)] for s in seeds
                                                                     for i in by_level[k]])) for k in c['K_levels']},
                curves_alpha1={t: [float(np.mean([e[(s, t, 1.0, T)] for s in seeds])) for T in c['T_checkpoints']]
                               for t in tids})


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
