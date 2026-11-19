"""Experiment 13 analysis (arm 13-small): per-seed law fits on fit rows, frozen held-out forecasts, seed-block
bootstrap, gates G1-G5 (PROTOCOL Section 5 as frozen), position decomposition, multipliers, sensitivity and
mechanism analyses.

Usage (from expt_v1/):
  analyze.py --arm small --stage 1   fits on fit rows only; writes forecasts.csv and its SHA-256 (fits.json) before
                                     any held-out outcome is read
  analyze.py --arm small --stage 2   reads held-out outcomes; scores, bootstraps, gates; writes gates.json,
                                     bootstrap.npz, analysis.json
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import independent_fit as IF  # noqa: E402
import laws as L  # noqa: E402

VARIANTS = {'primary': ('K_bits', True), 'no_offset': ('K_bits', False), 'K_n': ('K', True),
            'K_canonical': ('K_bits_canonical', True)}


def load_config(arm):
    raw = json.loads((HERE / 'config.json').read_text())
    c = dict(raw['shared'])
    c.update(raw[arm])
    return c


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def pct(a, q):
    a = np.asarray(a, float)
    return float(np.percentile(a, q)) if np.all(np.isfinite(a)) else float(np.percentile(np.nan_to_num(a, posinf=1e300), q))


def ci(draws, lo=2.5, hi=97.5):
    return [pct(draws, lo), pct(draws, hi)]


# ---------------------------------------------------------------- data

def read_rows(out, tasks, outcome=True):
    canon = {t['task_id']: t['K_bits_canonical'] for t in tasks}
    rows = []
    with open(out / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            d = dict(seed=int(r['seed']), target_id=r['target_id'], K=int(r['K']), K_bits=int(r['K_bits']),
                     K_bits_canonical=canon[r['target_id']], alpha=float(r['alpha']), T=int(r['T']),
                     slice=r['slice'], ident_bits=float(r['ident_bits']), oracle_bits=float(r['oracle_bits']),
                     mixture_bits=float(r['mixture_bits']))
            if outcome:
                d.update(excess_bits=float(r['excess_bits']), learn_bits=float(r['learn_bits']),
                         model_bits=float(r['model_bits']))
            rows.append(d)
    return rows


def is_fit(c, r, Tmax):
    return r['K'] != c['heldout_level'] and abs(r['alpha'] - c['heldout_alpha']) > 1e-9 and r['T'] != Tmax


def unit_info(c, tasks, unit):
    tg = [t for t in tasks if t['role'] == 'target']
    grid = sorted({float(t[unit]) for t in tg})
    fitK = sorted({float(t[unit]) for t in tg if t['K'] != c['heldout_level']})
    return grid, gm(fitK)


def X_for(c, tasks, rows, variant, T0):
    unit, off = VARIANTS[variant]
    grid, kref = unit_info(c, tasks, unit)
    rr = [dict(r, excess_bits=r.get('excess_bits', 1.0)) for r in rows]
    return L.design(rr, unit, grid, kref, T0, c['eps_res'], off)


# ---------------------------------------------------------------- stage 1

def stage1(c, out, tasks):
    rows = [r for r in read_rows(out, tasks, outcome=False) if r['slice'] == 'all']
    Ts = sorted({r['T'] for r in rows})
    T0, Tmax = Ts[0], Ts[-1]
    seeds = sorted({r['seed'] for r in rows})
    full = {(r['seed'], r['target_id'], r['alpha'], r['T']): r for r in read_rows(out, tasks, True)
            if r['slice'] == 'all' and is_fit(c, r, Tmax)}  # outcomes of fit rows only
    fits, fc_rows = {}, []
    for v in VARIANTS:
        for s in seeds:
            fr = [full[(r['seed'], r['target_id'], r['alpha'], r['T'])] for r in rows
                  if r['seed'] == s and is_fit(c, r, Tmax)]
            hr = [r for r in rows if r['seed'] == s and not is_fit(c, r, Tmax)]
            Xf, Xh = X_for(c, tasks, fr, v, T0), X_for(c, tasks, hr, v, T0)
            for law in L.LAWS:
                f = L.fit(law, Xf, c)
                r_fit, _ = L.scored_residual(law, f['theta'], Xf)
                f['fit_rmse'] = float(np.sqrt(np.mean(r_fit ** 2)))
                f['n_fit'] = len(fr)
                fits[f'{v}|{s}|{law}'] = f
                p = L.predict_eps(law, f['theta'], Xh)
                bad = L.invalid_mask(law, p, Xh)
                for r, pv, b in zip(hr, p, bad):
                    fc_rows.append([v, s, law, r['target_id'], r['K'], r['alpha'], r['T'], repr(float(pv)), int(b)])
            print(f'stage1 {v} seed {s} done', flush=True)
    with open(out / 'forecasts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'seed', 'law', 'target_id', 'K', 'alpha', 'T', 'forecast', 'invalid'])
        w.writerows(fc_rows)
    sha = hashlib.sha256((out / 'forecasts.csv').read_bytes()).hexdigest()
    # G2's K law, chosen on fit rows only (pooled fit RMSE over seeds)
    def pooled_fit(law):
        return math.sqrt(np.mean([fits[f'primary|{s}|{law}']['fit_rmse'] ** 2 for s in seeds]))
    klaw = 'M_K_coef' if pooled_fit('M_K_coef') <= pooled_fit('M_K_exp') else 'M_K_exp'
    meta = dict(forecasts_sha256=sha, seeds=seeds, T0=T0, Tmax=Tmax, g2_K_law=klaw,
                g2_fit_rmse={l: pooled_fit(l) for l in ('M_K_coef', 'M_K_exp')}, eps_res=c['eps_res'])
    (out / 'fits.json').write_text(json.dumps(dict(meta=meta, fits=fits), indent=1))
    print(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- helpers for stage 2

def mf_tstar(Ts, vals, eps_star):
    """Model-free T*: first crossing of eps_star, log-log interpolated between checkpoints.
    Returns (T*, flag) with flag 'ok', 'left' (already below at T_min) or 'right' (never reached: T_max)."""
    v = np.maximum(np.asarray(vals, float), 1e-6)
    if v[0] <= eps_star:
        return float(Ts[0]), 'left'
    for i in range(1, len(Ts)):
        if v[i] <= eps_star:
            a, b = math.log(v[i - 1]), math.log(v[i])
            fr = (math.log(eps_star) - a) / (b - a)
            return float(math.exp(math.log(Ts[i - 1]) + fr * (math.log(Ts[i]) - math.log(Ts[i - 1])))), 'ok'
    return float(Ts[-1]), 'right'


def main_effect_share(y, a_idx, k_idx, t_idx):
    """Balanced 3-factor main-effects SS; returns SS_alpha / (SS_alpha + SS_K) plus the three SS."""
    g = y.mean()
    ss = []
    for idx in (a_idx, k_idx, t_idx):
        levels = np.unique(idx)
        ss.append(sum((idx == l).sum() * (y[idx == l].mean() - g) ** 2 for l in levels))
    return ss[0] / (ss[0] + ss[1]), ss


def boot_idx(c, S):
    rng = np.random.default_rng(c['analysis_seed'])
    return rng.integers(0, S, size=(c['bootstrap_draws'], S))


# ---------------------------------------------------------------- stage 2

def stage2(c, out, tasks):
    F = json.loads((out / 'fits.json').read_text())
    meta, fits = F['meta'], F['fits']
    assert hashlib.sha256((out / 'forecasts.csv').read_bytes()).hexdigest() == meta['forecasts_sha256']
    seeds, T0, Tmax = meta['seeds'], meta['T0'], meta['Tmax']
    S = len(seeds)
    rows_all = read_rows(out, tasks, True)
    rows = [r for r in rows_all if r['slice'] == 'all']
    Ts = sorted({r['T'] for r in rows})
    tg = sorted([t for t in tasks if t['role'] == 'target'], key=lambda t: (t['K'], t['task_id']))
    levels = sorted({t['K'] for t in tg})
    alphas = sorted({r['alpha'] for r in rows})
    eps_star = c['epsilon_star']
    B = boot_idx(c, S)
    res = dict(meta=meta, epsilon_star=eps_star)

    # held-out scoring per variant, seed, law
    sc = {}
    for v in VARIANTS:
        for s in seeds:
            hr = [r for r in rows if r['seed'] == s and not is_fit(c, r, Tmax)]
            Xh = X_for(c, tasks, hr, v, T0)
            lvl = np.array([r['K'] for r in hr])
            for law in L.LAWS:
                f = fits[f'{v}|{s}|{law}']
                rr, bad = L.scored_residual(law, f['theta'], Xh)
                sc[(v, s, law)] = dict(sse=float(np.sum(rr ** 2)), n=len(rr), invalid=int(bad.sum()),
                                       sse_level=float(np.sum(rr[lvl == c['heldout_level']] ** 2)),
                                       n_level=int((lvl == c['heldout_level']).sum()),
                                       by={k: dict(sse=float(np.sum(rr[m] ** 2)), n=int(m.sum())) for k, m in (
                                           ('level_only', np.array([is_only(c, r, Tmax, 'level') for r in hr])),
                                           ('alpha_only', np.array([is_only(c, r, Tmax, 'alpha') for r in hr])),
                                           ('T_only', np.array([is_only(c, r, Tmax, 'T') for r in hr])))})

    def rmse_arr(v, law, key='sse', nkey='n'):
        return np.array([sc[(v, s, law)][key] for s in seeds]), np.array([sc[(v, s, law)][nkey] for s in seeds])

    def pooled(v, law, idx=None, key='sse', nkey='n'):
        a, n = rmse_arr(v, law, key, nkey)
        if idx is None:
            return math.sqrt(a.sum() / n.sum())
        return np.sqrt(a[idx].sum(1) / n[idx].sum(1))

    def reduction(v, new, base, key='sse', nkey='n'):
        pt = 1 - pooled(v, new, key=key, nkey=nkey) / pooled(v, base, key=key, nkey=nkey)
        dr = 1 - pooled(v, new, B, key, nkey) / pooled(v, base, B, key, nkey)
        return pt, dr

    res['heldout_rmse'] = {v: {law: dict(rmse=pooled(v, law), invalid=sum(sc[(v, s, law)]['invalid'] for s in seeds),
                                         by_coordinate={k: math.sqrt(sum(sc[(v, s, law)]['by'][k]['sse'] for s in seeds)
                                                                     / max(1, sum(sc[(v, s, law)]['by'][k]['n'] for s in seeds)))
                                                        for k in ('level_only', 'alpha_only', 'T_only')})
                               for law in L.LAWS} for v in VARIANTS}

    # independent fitter agreement (held-out RMSE per seed and law, primary variant)
    disagree = []
    for s in seeds:
        fr = [r for r in rows if r['seed'] == s and is_fit(c, r, Tmax)]
        hr = [r for r in rows if r['seed'] == s and not is_fit(c, r, Tmax)]
        Xf, Xh = X_for(c, tasks, fr, 'primary', T0), X_for(c, tasks, hr, 'primary', T0)
        for law in L.LAWS:
            g = IF.fit(law, Xf, c)
            a = IF.heldout_rmse(law, fits[f'primary|{s}|{law}']['theta'], Xh)[0]
            b = IF.heldout_rmse(law, g['theta'], Xh)[0]
            if not IF.agree(a, b, c['independent_fit_tol']):
                disagree.append(dict(seed=s, law=law, laws_py=a, independent=b, cost_laws=fits[f'primary|{s}|{law}']['cost'],
                                     cost_independent=g['cost']))
    res['independent_fit'] = dict(agree=not disagree, disagreements=disagree)
    print('independent fit', 'AGREE' if not disagree else f'DISAGREE {disagree[0]}', flush=True)

    def per_seed_param(law, key, v='primary'):
        return np.array([fits[f'{v}|{s}|{law}']['params'].get(key, np.nan) for s in seeds])

    def summ(arr, f=np.mean):
        arr = np.asarray(arr, float)
        d = np.array([f(arr[i]) for i in B]) if f is not np.mean else arr[B].mean(1)
        return dict(estimate=float(f(arr)), ci95=ci(d), per_seed=arr.tolist()), d

    # ---------------- model-free T* (G1, G4 E_mf)
    def tstar_table(eps):
        tab = {}
        for s in seeds:
            for t in tg:
                for a in alphas:
                    vals = [next(r['learn_bits'] for r in rows if r['seed'] == s and r['target_id'] == t['task_id']
                                 and r['alpha'] == a and r['T'] == T) for T in Ts]
                    tab[(s, t['task_id'], a)] = mf_tstar(Ts, vals, eps)
        return tab
    tst = tstar_table(eps_star)
    kmin, kmax = min(levels), max(levels)
    ids = {k: [t['task_id'] for t in tg if t['K'] == k] for k in levels}
    Tj = {k: np.array([[tst[(s, j, 1.0)][0] for j in ids[k]] for s in seeds]) for k in levels}  # (S, 2)
    MK_seed = np.array([gm(Tj[kmax][i]) / gm(Tj[kmin][i]) for i in range(S)])
    MK = gm(MK_seed)
    d_seed = np.exp(np.log(MK_seed)[B].mean(1))
    rng2 = np.random.default_rng(c['analysis_seed'] + 1)
    d_two = np.empty(c['bootstrap_draws'])
    for i in range(c['bootstrap_draws']):
        ss = B[i]
        jj = rng2.integers(0, 2, size=(2, S, 2))
        vals = [math.log(gm(Tj[kmax][s][jj[0, n]]) / gm(Tj[kmin][s][jj[1, n]])) for n, s in enumerate(ss)]
        d_two[i] = math.exp(np.mean(vals))
    cens = {f: sum(1 for k, v in tst.items() if v[1] == f) for f in ('ok', 'left', 'right')}
    G1 = dict(M_K=MK, ci95_seed=ci(d_seed), ci95_two_stage=ci(d_two), per_seed=MK_seed.tolist(),
              censoring_alpha1={str(k): [[tst[(s, j, 1.0)][1] for j in ids[k]] for s in seeds] for k in (kmin, kmax)},
              censoring_all=cens, threshold=c['g1_min_MK'])
    G1['verdict'] = 'SUPPORTED' if (MK >= c['g1_min_MK'] and G1['ci95_seed'][0] > 1 and G1['ci95_two_stage'][0] > 1) \
        else 'NOT_ESTABLISHED'

    # ---------------- G2
    klaw = meta['g2_K_law']
    pt, dr = reduction('primary', klaw, 'M_align_eff')
    lvl_k = math.sqrt(sum(sc[('primary', s, klaw)]['sse_level'] for s in seeds) / sum(sc[('primary', s, klaw)]['n_level'] for s in seeds))
    lvl_a = math.sqrt(sum(sc[('primary', s, 'M_align_eff')]['sse_level'] for s in seeds) / sum(sc[('primary', s, 'M_align_eff')]['n_level'] for s in seeds))
    inv = sum(sc[('primary', s, klaw)]['invalid'] for s in seeds)
    G2 = dict(K_law=klaw, reduction=pt, ci95=ci(dr), rmse_K_law=pooled('primary', klaw),
              rmse_align_eff=pooled('primary', 'M_align_eff'), heldout_level_rmse_K_law=lvl_k,
              heldout_level_rmse_align_eff=lvl_a, invalid_K_law=inv)
    G2['verdict'] = 'SUPPORTED' if (pt >= c['rmse_reduction_threshold'] and G2['ci95'][0] > 0 and lvl_k < lvl_a
                                    and inv == 0) else 'NOT_ESTABLISHED'

    # ---------------- G3a / G3b
    kap, d_kap = summ(per_seed_param('M_K_both', 'kappa'))
    g1p, d_g1 = summ(per_seed_param('M_K_both', 'gamma1'))
    r_coef, r_exp = pooled('primary', 'M_K_coef'), pooled('primary', 'M_K_exp')
    pt3, dr3 = reduction('primary', 'M_K_exp', 'M_K_coef')
    G3a = dict(kappa=kap, gamma1=g1p, rmse_coef=r_coef, rmse_exp=r_exp)
    G3a['verdict'] = 'SUPPORTED' if (kap['ci95'][0] > 0 and g1p['ci95'][0] <= 0 <= g1p['ci95'][1]
                                     and r_coef <= r_exp) else 'NOT_ESTABLISHED'
    G3b = dict(gamma1=g1p, exp_vs_coef_reduction=pt3, ci95=ci(dr3), threshold=c['g3b_reduction'])
    G3b['verdict'] = 'SUPPORTED' if (g1p['ci95'][1] < 0 and pt3 >= c['g3b_reduction'] and G3b['ci95'][0] > 0) \
        else 'NOT_ESTABLISHED'

    # ---------------- G4: elasticity ratio E (descriptive M_K_both refit on all rows), both K units, and E_mf
    allfits = {}
    for v in ('primary', 'K_n'):
        for s in seeds:
            rs = [r for r in rows if r['seed'] == s]
            X = X_for(c, tasks, rs, v, T0)
            allfits[(v, s)] = (L.fit('M_K_both', X, c), X)
    lk = np.log([t['K_bits'] for t in tg])
    ln_ = np.log([t['K'] for t in tg])
    slope = float(np.polyfit(ln_, lk, 1)[0])

    def E_at(eps, v='primary'):
        ea, ek = [], []
        for s in seeds:
            f, X = allfits[(v, s)]
            a, k = L.elasticities('M_K_both', f['params'], X['K_ref'], eps, X)
            ea.append(a)
            ek.append(k)
        ea, ek = np.array(ea), np.array(ek)
        pt = ea.mean() / ek.mean() if ek.mean() > 0 else math.inf
        ma, mk = ea[B].mean(1), ek[B].mean(1)
        d = np.where(mk > 0, ma / np.where(mk > 0, mk, 1), np.inf)
        return pt, d, ea, ek
    E_b, dE_b, ea, ek = E_at(eps_star)
    E_s, dE_s = E_b / slope, dE_b / slope
    En, dEn, _, _ = E_at(eps_star, 'K_n')

    def E_mf(eps):
        tab = tstar_table(eps) if eps != eps_star else tst
        coefs, ncens = [], []
        for s in seeds:
            pts = [(math.log2(1 / a), math.log2(t['K_bits']), math.log2(tab[(s, t['task_id'], a)][0]))
                   for t in tg for a in alphas if tab[(s, t['task_id'], a)][1] == 'ok']
            ncens.append(len(tg) * len(alphas) - len(pts))
            if len(pts) >= 4:
                Z = np.array(pts)
                b = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z[:, 0], Z[:, 1]]), Z[:, 2], rcond=None)[0]
                coefs.append(b[1:])
        if not coefs:
            return None, ncens
        cm = np.mean(coefs, 0)
        return (float(cm[0] / cm[1]) if cm[1] > 0 else math.inf), ncens
    med_cens = float(np.median(E_mf(eps_star)[1]))
    eps_mf = eps_star if med_cens <= 0.5 * len(tg) * len(alphas) else 2 * eps_star
    Emf, ncens_mf = E_mf(eps_mf)
    G4 = dict(E_bits=E_b, ci95_bits=ci(dE_b), E_states=E_s, ci95_states=ci(dE_s), bits_per_state_slope=slope,
              E_states_refit_K_n=En, ci95_states_refit=ci(dEn), e_alpha=ea.tolist(), e_K_bits=ek.tolist(),
              E_mf=Emf, E_mf_eps=eps_mf, E_mf_censored_pairs_per_seed=ncens_mf)
    G4['verdict'] = 'SUPPORTED' if (E_b > 1 and G4['ci95_bits'][0] > 1 and E_s > 1 and G4['ci95_states'][0] > 1
                                    and Emf is not None and Emf > 1) else 'NOT_ESTABLISHED'

    # ---------------- G5: position decomposition
    pos = {}
    for sl in ('early', 'middle', 'late', 'all'):
        for q in ('excess_bits', 'learn_bits', 'ident_bits'):
            rs_seed = []
            for s in seeds:
                ys, ai, ki, ti = [], [], [], []
                for k in levels:
                    for a in alphas:
                        for T in Ts:
                            vals = [r[q] for r in rows_all if r['seed'] == s and r['slice'] == sl and r['K'] == k
                                    and r['alpha'] == a and r['T'] == T]
                            ys.append(float(np.mean(np.log2(np.maximum(vals, c['eps_res'])))))
                            ai.append(a)
                            ki.append(k)
                            ti.append(T)
                share, _ = main_effect_share(np.array(ys), np.array(ai), np.array(ki), np.array(ti))
                rs_seed.append(share)
            rs_seed = np.array(rs_seed)
            pos[f'{sl}|{q}'] = dict(r_alpha=float(rs_seed.mean()), ci95=ci(rs_seed[B].mean(1)),
                                    per_seed=rs_seed.tolist())
    e, l_ = pos['early|excess_bits'], pos['late|excess_bits']
    G5 = dict(r_alpha_early=e, r_alpha_late=l_)
    G5['verdict'] = 'SUPPORTED' if (e['ci95'][0] > c['g5_share'] and l_['ci95'][1] < c['g5_share']) else 'NOT_ESTABLISHED'
    G5['named_failure'] = ('alpha dominates everywhere' if l_['ci95'][0] > c['g5_share'] else
                           'K dominates everywhere' if e['ci95'][1] < c['g5_share'] else None)
    res['position_decomposition'] = pos

    # ---------------- draft-protocol versions (reported, not gated)
    conv_fail = sum(1 for s in seeds if not fits[f'primary|{s}|M_K_both']['converged']) / S
    mlaw = 'M_K_both' if conv_fail <= c['fallback_fail_share'] else 'M_K_coef'

    def law_MK(eps, law=mlaw, v='primary'):
        out_ = []
        for s in seeds:
            f = fits[f'{v}|{s}|{law}']
            Xf = X_for(c, tasks, [r for r in rows if r['seed'] == s and is_fit(c, r, Tmax)], v, T0)
            unit = VARIANTS[v][0]
            kv = sorted({float(t[unit]) for t in tg if t['K'] != c['heldout_level']})
            out_.append(L.t_star(law, f['params'], kv[-1], 1.0, eps, Xf) / L.t_star(law, f['params'], kv[0], 1.0, eps, Xf))
        return np.array(out_)
    mk_law = law_MK(eps_star)
    eta_s = per_seed_param(mlaw, 'eta')
    Malpha = 20.0 ** eta_s
    Rs = np.log(Malpha) / np.log(mk_law)
    dR = np.log(Malpha[B]).mean(1) / np.log(mk_law[B]).mean(1)
    gap = {}
    for T in Ts:
        gap[T] = np.array([np.mean([r['excess_bits'] for r in rows if r['seed'] == s and r['K'] == kmax
                                    and r['alpha'] == 1.0 and r['T'] == T])
                           - np.mean([r['excess_bits'] for r in rows if r['seed'] == s and r['K'] == kmin
                                      and r['alpha'] == 1.0 and r['T'] == T]) for s in seeds])
    T_last_fit = Ts[-2]
    gr = gap[T_last_fit].mean() / gap[Ts[0]].mean()
    dgr = gap[T_last_fit][B].mean(1) / gap[Ts[0]][B].mean(1)
    ptc, drc = reduction('primary', 'M_K_coef', 'M_align_eff')
    pte, dre = reduction('primary', 'M_K_exp', 'M_align_eff')
    rho = r_exp / r_coef
    drho = pooled('primary', 'M_K_exp', B) / pooled('primary', 'M_K_coef', B)
    res['draft_protocol_reported'] = dict(
        law_for_multipliers=mlaw, M_K_both_convergence_fail_share=conv_fail,
        G1_law_M_K=dict(estimate=gm(mk_law), ci95=ci(np.exp(np.log(mk_law)[B].mean(1))), per_seed=mk_law.tolist()),
        G2_bonferroni=dict(coef=dict(reduction=ptc, ci_bonf=[pct(drc, 1.25), pct(drc, 98.75)]),
                           exp=dict(reduction=pte, ci_bonf=[pct(dre, 1.25), pct(dre, 98.75)])),
        G3_rho=dict(rho=rho, ci95=ci(drho)),
        G4a_range_ratio=dict(R=float(np.mean(np.log(Malpha)) / np.mean(np.log(mk_law))), ci95=ci(dR),
                             M_alpha=gm(Malpha), M_K=gm(mk_law)),
        G4b_gap_ratio=dict(ratio=float(gr), ci95=ci(dgr), gap_first=float(gap[Ts[0]].mean()),
                           gap_last_fit=float(gap[T_last_fit].mean()), T_first=Ts[0], T_last_fit=T_last_fit),
        elasticity_ratio_e_alpha_over_e_K=E_b)

    # ---------------- multiplier curves
    curves = {}
    for eps in c['eps_curve']:
        Eb, dEb, _, _ = E_at(eps)
        mkl = law_MK(eps)
        tt = tstar_table(eps)
        mk_mf = gm([gm([tt[(s, j, 1.0)][0] for j in ids[kmax]]) / gm([tt[(s, j, 1.0)][0] for j in ids[kmin]])
                    for s in seeds])
        curves[str(eps)] = dict(E_bits=Eb, E_ci95=ci(dEb), M_K_law=gm(mkl), M_alpha_law=gm(Malpha), M_K_model_free=mk_mf,
                                right_censored_alpha1=sum(1 for s in seeds for k in levels for j in ids[k]
                                                          if tt[(s, j, 1.0)][1] == 'right'))
    res['multiplier_curves'] = curves

    # ---------------- parameters per seed and alignment comparison
    res['fitted_laws_per_seed'] = {law: {str(s): fits[f'primary|{s}|{law}']['params'] for s in seeds} for law in L.LAWS}
    eta_eff, _ = summ(per_seed_param('M_align_eff', 'eta'))
    eta_k, _ = summ(per_seed_param(klaw, 'eta'))
    pta, dra = reduction('primary', 'M_align_eff', 'M_align_add')
    res['alignment_comparison'] = dict(eta_align_eff=eta_eff, eta_K_law=eta_k, align_eff_vs_add_reduction=pta,
                                       ci95=ci(dra))

    # ---------------- mechanism analyses
    mech = {}
    gam = {}
    for t in tg:
        gs = []
        for s in seeds:
            rs = [r for r in rows if r['seed'] == s and r['target_id'] == t['task_id'] and r['alpha'] == 1.0]
            y = np.log2(np.maximum([r['excess_bits'] for r in rs], c['eps_res']))
            x = np.log2([r['T'] for r in rs])
            gs.append(float(-np.polyfit(x, y, 1)[0]))
        gam[t['task_id']] = float(np.mean(gs))
    tail = [t['visit_tail_exponent'] for t in tg]
    sp = spearmanr([gam[t['task_id']] for t in tg], tail)
    mech['gamma_alpha1_per_target'] = gam
    mech['spearman_gamma_vs_tail'] = dict(rho=float(sp.statistic), p=float(sp.pvalue), tail=tail)
    shape_fit = {}
    for t in tg:
        ei = {a: next(r['ident_bits'] for r in rows if r['target_id'] == t['task_id'] and r['alpha'] == a)
              for a in alphas}
        xa = np.array([a for a in alphas if a < 1])
        ya = np.array([ei[a] for a in xa])
        lin = 1 - xa
        lg = np.log2(1 / xa)
        rss_lin = float(np.sum((ya - lin * (lin @ ya) / (lin @ lin)) ** 2))
        rss_log = float(np.sum((ya - lg * (lg @ ya) / (lg @ lg)) ** 2))
        shape_fit[t['task_id']] = dict(eps_id={str(a): ei[a] for a in alphas}, rss_linear=rss_lin, rss_log=rss_log)
    mech['eps_id_shape'] = dict(per_target=shape_fit,
                                total_rss_linear=sum(v['rss_linear'] for v in shape_fit.values()),
                                total_rss_log=sum(v['rss_log'] for v in shape_fit.values()))
    early = [r for r in rows_all if r['slice'] == 'early' and r['T'] == Tmax and r['alpha'] < 1]
    mech['early_share_eps_id_at_Tmax'] = {str(a): float(np.mean([r['ident_bits'] / r['excess_bits'] for r in early
                                                               if r['alpha'] == a and r['excess_bits'] > 0]))
                                          for a in alphas if a < 1}
    dl = list(csv.DictReader(open(out / 'distractor_loss.csv')))
    mech['distractor_excess_at_Tmax_by_alpha'] = {str(a): float(np.mean([float(d['model_bits']) - float(d['oracle_bits'])
                                                                        for d in dl if float(d['alpha']) == a and int(d['T']) == Tmax]))
                                                  for a in alphas}
    post = json.loads((out / 'posterior16.json').read_text())
    mech['posterior_target_after_16'] = {k: v for k, v in post.items() if k.split('|')[0] in ids_all(tg)}
    res['mechanism'] = mech
    res['diversity'] = diversity_line()
    res['gates'] = dict(G1=G1, G2=G2, G3a=G3a, G3b=G3b, G4=G4, G5=G5)
    res['C3'] = ('coefficient' if G3a['verdict'] == 'SUPPORTED' else
                 'exponent' if G3b['verdict'] == 'SUPPORTED' else 'unresolved')
    res['cell_means'] = cell_means(rows_all, tg, alphas, Ts)
    gates = {k: dict(verdict=v['verdict']) for k, v in res['gates'].items()}
    (out / 'gates.json').write_text(json.dumps(dict(forecasts_sha256=meta['forecasts_sha256'], gates=gates,
                                                    detail=res['gates'], C3=res['C3']), indent=1, default=float))
    (out / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez(out / 'bootstrap.npz', seed_index=B, G1_seed=d_seed, G1_two_stage=d_two, G2=dr, G3b=dr3, G4_bits=dE_b,
             G4_states=dE_s, kappa=d_kap, gamma1=d_g1)
    for k, v in gates.items():
        print(k, v['verdict'])
    return res


def ids_all(tg):
    return {t['task_id'] for t in tg}


def is_only(c, r, Tmax, coord):
    held = dict(level=r['K'] == c['heldout_level'], alpha=abs(r['alpha'] - c['heldout_alpha']) < 1e-9,
                T=r['T'] == Tmax)
    return held[coord] and sum(held.values()) == 1


def cell_means(rows_all, tg, alphas, Ts):
    out = {}
    for sl in ('early', 'middle', 'late', 'all'):
        for k in sorted({t['K'] for t in tg}):
            for a in alphas:
                for T in Ts:
                    rs = [r for r in rows_all if r['slice'] == sl and r['K'] == k and r['alpha'] == a and r['T'] == T]
                    out[f'{sl}|{k}|{a}|{T}'] = dict(excess=float(np.mean([r['excess_bits'] for r in rs])),
                                                     ident=float(np.mean([r['ident_bits'] for r in rs])),
                                                     learn=float(np.mean([r['learn_bits'] for r in rs])))
    return out


def diversity_line():
    p = HERE.parent.parent / '14_task2vec_vs_exact_complexity' / 'expt_v1'
    q = HERE.parent.parent / 'archive' / '14_task2vec_vs_exact_complexity' / 'expt_v1'
    if any(x.exists() and any(x.glob('*.py')) for x in (p, q)):
        return 'Experiment 14 code exists but the diversity link was not computed in this run.'
    return 'follow-up: diversity coefficient of the Experiment 13 mixes (Experiment 14 code did not exist at run time)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='small')
    ap.add_argument('--stage', type=int, choices=[1, 2], required=True)
    ap.add_argument('--output', default=None)
    a = ap.parse_args()
    c = load_config(a.arm)
    out = Path(a.output) if a.output else HERE / 'results' / a.arm
    tasks = json.loads((out / 'tasks.json').read_text())
    if a.arm == 'gpu':  # 13-GPU: pooled full N laws, per-size laws and G6 (analyze_gpu.py)
        import analyze_gpu as AG
        (AG.stage1 if a.stage == 1 else AG.stage2)(c, out, tasks)
    elif a.stage == 1:
        stage1(c, out, tasks)
    else:
        stage2(c, out, tasks)


if __name__ == '__main__':
    main()
