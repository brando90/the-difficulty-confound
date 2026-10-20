"""Experiment 15 analysis: alpha-hat for every (arm, target, mix, measure, probe), the affine decomposition, the
controls, and Gates 1, 2, 3, 4a and 4b (PROTOCOL Sections 4-5).

Writes results/alpha_hat.csv, decomposition.csv, controls.csv, laws.json, gates.json and bootstrap.npz.
Usage: analyze.py [--results results]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import common as C
import independent_fit as IFX
import jobs as J
import laws as LW
import scores as S

HERE = Path(__file__).resolve().parent

NONPROBE_MAIN = ['a', 'aprime', 'a2', 'b2', 'oracle', 'random', 'a_leak', 'a_perm', 'a_inp']
PROBE_MAIN = ['b1', 'd', 'b1_leak', 'b1_perm', 'b1_inp', 'd_perm']
COMBOS = {'c': ('b1', 'a'), 'c2': ('b2', 'a'), 'c_leak': ('b1_leak', 'a_leak'), 'c_perm': ('b1_perm', 'a_perm'),
          'c_inp': ('b1_inp', 'a_inp')}
SETLEVEL = ['e', 'e2']
MEASURES = ['a', 'aprime', 'a2', 'b1', 'b2', 'c', 'c2', 'd', 'e', 'e2', 'oracle', 'random']
CONTROLS = ['a_leak', 'b1_leak', 'c_leak', 'a_perm', 'b1_perm', 'c_perm', 'd_perm', 'a_inp', 'b1_inp', 'c_inp']
CONTROL_OF = {'a_leak': 'a', 'b1_leak': 'b1', 'c_leak': 'c', 'a_perm': 'a', 'b1_perm': 'b1', 'c_perm': 'c',
              'd_perm': 'd', 'a_inp': 'a', 'b1_inp': 'b1', 'c_inp': 'c'}


def key_alpha(a):
    return round(float(a), 6)


# ---------------------------------------------------------------- alpha-hat

def arm_measures(arm):
    if arm == 'graded':
        return ['a', 'aprime', 'a2', 'b2', 'oracle', 'random'], ['b1', 'd'], {'c': ('b1', 'a'), 'c2': ('b2', 'a')}
    return NONPROBE_MAIN, PROBE_MAIN, COMBOS


def compute_alpha_hat(c, res, arm, probes=None):
    """dict measure -> array (targets, mixes, probes) of alpha-hat (probes axis 1 for non-probe measures), plus the
    decomposition rows and the token-weighted true share."""
    st = J.Setup(c, arm, res)
    st.constant_scores = set()
    probes = probes or c['probes']
    fam = st.fam
    sd = Path(res) / 'scores'
    nonp, prb, combos = arm_measures(arm)
    nT, nA = len(st.targets), len(st.alphas)
    out = {}
    decomp = []
    a_tok = np.zeros((nT, nA))
    raw = {m: np.load(sd / f'{arm}_{m}_all.npz')['raw'] for m in nonp}
    for m in prb:
        raw[m] = np.stack([np.load(sd / f'{arm}_{m}_{p}.npz')['raw'] for p in probes], -1)
    for ti, T in enumerate(st.targets):
        lay = st.layout(T)
        pool = st.pool(T)
        t = C.n_tokens(fam, pool)
        t512 = C.n_tokens(fam, st.pool(T, c['ncd_units_per_task']))
        nu = {al: st.n_u(T, al) for al in st.alphas}
        ind = np.repeat((np.arange(len(lay)) == 0).astype(float), st.nU)
        mts, mt = S.block_means(t, ind, [st.nU] * len(lay))
        for ai, al in enumerate(st.alphas):
            a_tok[ti, ai] = S.eq2_stratified(nu[al], mts, mt)

        def norm(m, pi):
            r = raw[m][ti] if m in nonp else raw[m][ti][:, pi]
            if np.ptp(r) == 0:
                st.constant_scores.add((m, T, probes[pi] if m in prb else 'none'))
            return S.minmax_or_zero(r)

        def ah(s, m):
            tt, size = (t512, c['ncd_units_per_task']) if m == 'a2' else (t, st.nU)
            mts, mt = S.block_means(tt, s, [size] * len(lay))
            ref_n = nu[st.alphas[0]] if arm != 'graded' else nu['m3']
            mD = float(np.dot(ref_n[1:], mts[1:]) / np.dot(ref_n[1:], mt[1:]))
            return [S.eq2_stratified(nu[al], mts, mt) for al in st.alphas], float(mts[0] / mt[0]), mD

        names = list(nonp) + list(prb) + list(combos)
        for m in names:
            npr = len(probes) if (m in prb or m in combos and combos[m][0] in prb) else 1
            arr = np.zeros((nA, npr))
            mT, mD, resol = [], [], []
            for pi in range(npr):
                if m in combos:
                    u, v = combos[m]
                    s = 0.5 * norm(u, pi) + 0.5 * norm(v, pi)
                    resol.append(len(np.unique(s)))
                else:
                    s = norm(m, pi)
                    resol.append(len(np.unique(raw[m][ti] if m in nonp else raw[m][ti][:, pi])))
                v_, mt_, md_ = ah(s, m)
                arr[:, pi] = v_
                mT.append(mt_)
                mD.append(md_)
            out.setdefault(m, np.zeros((nT, nA, npr)))[ti] = arr
            decomp.append(dict(arm=arm, target=T, K=st.by_id[T]['K'], measure=m, m_T=float(np.mean(mT)),
                               m_D=float(np.mean(mD)), separation=float(np.mean(mT) - np.mean(mD)),
                               resolution_distinct_values=float(np.mean(resol)), n_pool=int(len(t))))
    t2 = json.loads((sd / f'{arm}_task2vec.json').read_text())
    for est in SETLEVEL:
        arr = np.zeros((nT, nA, len(probes)))
        for ti, T in enumerate(st.targets):
            for pi, p in enumerate(probes):
                ahd = t2[f'{p}|{T}'][est]['alpha_hat']
                for ai, al in enumerate(st.alphas):
                    arr[ti, ai, pi] = ahd[al if arm == 'graded' else f'alpha{al:g}']
        out[est] = arr
    return st, out, decomp, a_tok


# ---------------------------------------------------------------- bootstrap draws

def level_draws(rng, levels, D):
    """(D, nT) target indices resampled with replacement within each K level (levels: list of index arrays)."""
    return np.concatenate([np.asarray(ix)[rng.integers(0, len(ix), (D, len(ix)))] for ix in levels], axis=1)


def probe_mean(A, P=None):
    if P is None:
        return A.mean(-1)
    if A.shape[-1] == 1:
        return np.repeat(A[None, ..., 0], len(P), 0)
    return A[..., P].mean(-1) if P.ndim == 1 else None


def resampled(A, Tidx, P):
    """(D, nT, nA) alpha-hat for target draws Tidx (D, nT) and probe draws P (D, nP)."""
    nA = A.shape[1]
    if A.shape[-1] == 1:
        return A[Tidx, :, 0]
    return A[Tidx[:, :, None, None], np.arange(nA)[None, None, :, None], P[:, None, None, :]].mean(-1)


def g1_stats(A, alphas, levels, Tdraws, P):
    """Per level: point Spearman of probe-mean alpha-hat vs alpha over (target, alpha) and its bootstrap draws."""
    y = A.mean(-1)
    out = []
    for li, ix in enumerate(levels):
        ix = np.asarray(ix)
        pt = LW.spearman(y[ix].ravel(), np.tile(alphas, len(ix)))
        Td = Tdraws[li]
        Ad = resampled(A, Td, P)  # (D, n_level, nA)
        D = Ad.shape[0]
        dr = LW.spearman_rows(Ad.reshape(D, -1), np.tile(alphas, (D, len(ix))))
        out.append((pt, dr))
    return out


def g2_point(A, logK, alphas):
    y = A.mean(-1)
    nT, nA = y.shape
    return LW.fe_slope(y.ravel(), np.repeat(logK, nA), np.tile(np.arange(nA), nT))


def g2_draws(A, logK, Tidx, P):
    Ad = resampled(A, Tidx, P)
    yb = Ad.mean(-1)
    x = logK[Tidx]
    xc = x - x.mean(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (xc * yb).sum(1) / (xc * xc).sum(1)


# ---------------------------------------------------------------- main analysis

def analyze(c, res):
    t_start = time.time()
    res = Path(res)
    D = c['bootstrap_draws']
    g = c['gates']
    probes = c['probes']
    nP = len(probes)
    AH, ST, DEC, ATOK = {}, {}, [], {}
    for arm in J.ARMS:
        ST[arm], AH[arm], dec, ATOK[arm] = compute_alpha_hat(c, res, arm)
        DEC += dec

    # alpha_hat.csv
    rows = []
    for arm in J.ARMS:
        st = ST[arm]
        G = getattr(st, 'G', None)
        for m, A in AH[arm].items():
            for ti, T in enumerate(st.targets):
                for ai, al in enumerate(st.alphas):
                    base = dict(arm=arm, family=st.fam, target=T, K=st.by_id[T]['K'], K_bits=st.by_id[T]['K_bits'],
                                mix=al if arm == 'graded' else '', alpha=(G['mixes'][f'{T}_{al}']['alpha_share']
                                                                            if arm == 'graded' else al),
                                alpha_tok=ATOK[arm][ti, ai],
                                alpha_soft=G['mixes'][f'{T}_{al}']['alpha_soft'] if arm == 'graded' else '',
                                alpha_agree=G['mixes'][f'{T}_{al}']['alpha_agree'] if arm == 'graded' else '',
                                measure=m)
                    for pi in range(A.shape[-1]):
                        rows.append(dict(base, probe=probes[pi] if A.shape[-1] > 1 else 'none',
                                         alpha_hat=A[ti, ai, pi]))
                    rows.append(dict(base, probe='mean', alpha_hat=A[ti, ai].mean()))
    write_csv(res / 'alpha_hat.csv', rows)
    write_csv(res / 'decomposition.csv', DEC)

    gates, boot = {}, {}
    rng_p = np.random.default_rng([c['analysis_seed'], 1])
    P = rng_p.integers(0, nP, (D, nP))
    boot['probe_draws'] = P

    # Gates 1 and 2, every measure and control, both families
    stats = {}
    for fam in C.FAMILIES:
        st = ST[fam]
        Ks = np.array([st.by_id[T]['K'] for T in st.targets])
        lv = sorted(set(Ks.tolist()))
        levels = [np.flatnonzero(Ks == k) for k in lv]
        rng_t = np.random.default_rng([c['analysis_seed'], 2, C.FAM_CODE[fam]])
        Tlev = [np.asarray(ix)[rng_t.integers(0, len(ix), (D, len(ix)))] for ix in levels]
        Tall = np.concatenate(Tlev, axis=1)
        boot[f'{fam}_target_draws'] = Tall
        alphas = np.array(st.alphas)
        logK = np.log2(Ks.astype(float))
        star = AH[fam]['oracle']
        for m, A in AH[fam].items():
            g1 = g1_stats(A, alphas, levels, Tlev, P)
            s_pt = g2_point(A, logK, alphas)
            s_dr = g2_draws(A, logK, Tall, P)
            diff = A - star if A.shape[-1] == 1 else A - star
            sd_pt = g2_point(diff, logK, alphas)
            sd_dr = g2_draws(diff, logK, Tall, P)
            stats[(fam, m)] = dict(
                g1=[dict(K=int(k), rho=pt, lb=percentile_ci(dr)[0], ci=percentile_ci(dr),
                         rho_independent=IFX.spearman(A.mean(-1)[levels[i]].ravel(),
                                                      np.tile(alphas, len(levels[i]))))
                    for i, (k, (pt, dr)) in enumerate(zip(lv, g1))],
                g2=dict(slope=s_pt, ci=percentile_ci(s_dr),
                        slope_independent=IFX.fe_slope(A.mean(-1).ravel(), np.repeat(logK, len(alphas)),
                                                       np.tile(np.arange(len(alphas)), len(Ks))),
                        slope_minus_oracle=sd_pt, slope_minus_oracle_ci=percentile_ci(sd_dr)),
                mean_alpha_hat_by_alpha=A.mean(-1).mean(0).tolist())
            if m == 'c':
                boot[f'{fam}_g1_c'] = np.stack([dr for _, dr in g1])
                boot[f'{fam}_g2_c'] = s_dr
            if m == 'oracle':
                boot[f'{fam}_g2_oracle'] = s_dr
    g1_ok = all(x['rho'] >= g['g1_min_rho'] and x['lb'] >= g['g1_lb_min']
                for fam in C.FAMILIES for x in stats[(fam, 'c')]['g1'])
    gates['G1'] = dict(verdict=LW_verdict(g1_ok), measure='c',
                       levels={fam: stats[(fam, 'c')]['g1'] for fam in C.FAMILIES},
                       threshold=f"rho >= {g['g1_min_rho']} and 95% LB >= {g['g1_lb_min']} in every K level")
    band = g['g2_band']
    g2 = {}
    for fam in C.FAMILIES:
        s = stats[(fam, 'c')]['g2']
        so = stats[(fam, 'oracle')]['g2']
        g2[fam] = dict(slope=s['slope'], ci=s['ci'], inside=bool(-band <= s['ci'][0] and s['ci'][1] <= band),
                       oracle_slope=so['slope'], oracle_ci=so['ci'],
                       slope_minus_oracle=s['slope_minus_oracle'], slope_minus_oracle_ci=s['slope_minus_oracle_ci'])
    gates['G2'] = dict(verdict=LW_verdict(all(v['inside'] for v in g2.values())), measure='c', families=g2,
                       threshold=f'95% CI of the slope inside [-{band}, +{band}] per doubling of K, each family')

    # Gate 3
    laws_out = {}
    g3 = {}
    for fam in C.FAMILIES:
        st = ST[fam]
        cu = LW.upstream(c, fam)[2]
        held_K = cu['heldout_n'] if fam == 'dfa' else cu['heldout_K']
        y = AH[fam]['c'].mean(-1)
        pairs = [(T, al) for T in st.targets for al in st.alphas]
        fit_idx = [i for i, (T, al) in enumerate(pairs)
                   if st.by_id[T]['K'] not in held_K and al not in cu['heldout_alphas']]
        ho_idx = [i for i in range(len(pairs)) if i not in fit_idx]
        xa = y.ravel()
        ya = np.array([al for _, al in pairs])
        iso = LW.Isotonic(xa[fit_idx], ya[fit_idx], rows=fit_idx, heldout_rows=ho_idx)
        iso_i = IFX.Isotonic(xa[fit_idx], ya[fit_idx])
        cal = iso(xa)
        amaps = dict(true={(T, key_alpha(al)): al for T, al in pairs},
                     calibrated={(T, key_alpha(al)): cal[i] for i, (T, al) in enumerate(pairs)},
                     raw={(T, key_alpha(al)): xa[i] for i, (T, al) in enumerate(pairs)})
        seeds, fits = gate3_fits(c, fam, amaps)
        _, fits_i = gate3_fits(c, fam, amaps, independent=True)
        W = LW.Boot(len(seeds), D, c['analysis_seed']).W
        d_cal = LW.rel_change_draws(fits['true'], fits['calibrated'], W)
        d_raw = LW.rel_change_draws(fits['true'], fits['raw'], W)
        r = {k: LW.rmse(v) for k, v in fits.items()}
        inv = {k: int(sum(p['invalid'] for p in v)) for k, v in fits.items()}
        delta = r['calibrated'] / r['true'] - 1
        g3[fam] = dict(rmse=r, invalid=inv, delta_calibrated=delta, delta_calibrated_ci=percentile_ci(d_cal),
                       delta_raw=r['raw'] / r['true'] - 1, delta_raw_ci=percentile_ci(d_raw),
                       passed=bool(delta <= g['g3_max_rel_change'] and percentile_ci(d_cal)[1] <= g['g3_ub_max']
                                   and inv['calibrated'] == 0),
                       calibrator=dict(fit_pairs=[list(pairs[i]) for i in fit_idx],
                                       x=xa.tolist(), calibrated=cal.tolist(),
                                       max_abs_diff_independent=float(np.max(np.abs(cal - iso_i(xa))))),
                       seeds=seeds)
        boot[f'{fam}_g3_delta_cal'] = d_cal
        boot[f'{fam}_g3_delta_raw'] = d_raw
        laws_out[fam] = dict(law=LW.LAW, unit=LW.UNIT, fits=fits, independent=fits_i,
                             agreement=law_agreement(fits, fits_i, c['law_rel_tol']))
        # calibrated alpha-hat in the law for every other measure (reported, laws.py fitter only)
        other = {}
        for m in MEASURES:
            if m == 'c':
                continue
            ym = AH[fam][m].mean(-1).ravel()
            if not np.all(np.diff(np.sort(ym[fit_idx])) >= 0) or np.ptp(ym[fit_idx]) == 0:
                pass
            cm = LW.Isotonic(ym[fit_idx], ya[fit_idx])(ym)
            _, fm = gate3_fits(c, fam, dict(calibrated={(T, key_alpha(al)): cm[i] for i, (T, al) in enumerate(pairs)}))
            other[m] = dict(rmse=LW.rmse(fm['calibrated']), delta=LW.rmse(fm['calibrated']) / r['true'] - 1,
                            invalid=int(sum(p['invalid'] for p in fm['calibrated'])))
        g3[fam]['other_measures_calibrated'] = other
    gates['G3'] = dict(verdict=LW_verdict(all(v['passed'] for v in g3.values())), measure='c', families=g3,
                       threshold=f"Delta <= {g['g3_max_rel_change']}, 95% UB <= {g['g3_ub_max']}, 0 invalid")

    # Gate 4a
    grows = LW.read_graded_metrics(res / 'graded' / 'metrics.csv')
    seeds4, per_share = LW.gate4a_fits(c, grows, 'alpha_share')
    _, per_soft = LW.gate4a_fits(c, grows, 'alpha_soft')
    _, per_agree = LW.gate4a_fits(c, grows, 'alpha_agree')
    _, per_share_i = LW.gate4a_fits(c, grows, 'alpha_share', independent=True)
    _, per_soft_i = LW.gate4a_fits(c, grows, 'alpha_soft', independent=True)
    W4 = LW.Boot(len(seeds4), D, c['analysis_seed']).W
    red_d = -LW.rel_change_draws(per_share, per_soft, W4)
    red = 1 - LW.rmse(per_soft) / LW.rmse(per_share)
    red_agree = 1 - LW.rmse(per_agree) / LW.rmse(per_share)
    inv4 = dict(share=sum(p['invalid'] for p in per_share), soft=sum(p['invalid'] for p in per_soft),
                agree=sum(p['invalid'] for p in per_agree))
    boot['g4a_reduction'] = red_d
    gates['G4a'] = dict(
        verdict=LW_verdict(red >= g['g4a_min_rel_reduction'] and percentile_ci(red_d)[0] > g['g4a_lb_above']
                           and inv4['soft'] == 0 and inv4['share'] == 0),
        rmse=dict(share=LW.rmse(per_share), soft=LW.rmse(per_soft), agree=LW.rmse(per_agree)),
        reduction=red, reduction_ci=percentile_ci(red_d), reduction_agree=red_agree,
        reduction_agree_ci=percentile_ci(-LW.rel_change_draws(per_share, per_agree, W4)), invalid=inv4, seeds=seeds4,
        threshold=f"reduction >= {g['g4a_min_rel_reduction']:.0%} with 95% LB > 0, 0 invalid")
    laws_out['graded'] = dict(share=per_share, soft=per_soft, agree=per_agree,
                              agreement=law_agreement(dict(share=per_share, soft=per_soft),
                                                      dict(share=per_share_i, soft=per_soft_i), c['law_rel_tol'],
                                                      folds=True))
    # loss by mix (for the figure and the report)
    by_mix = {}
    Tm = max(r['T'] for r in grows)
    for r in grows:
        if r['T'] == Tm:
            by_mix.setdefault(f"{r['target_id']}_{r['mix']}", []).append(r['excess_bits'])
    gates['G4a']['final_loss_by_mix'] = {k: float(np.mean(v)) for k, v in by_mix.items()}

    # Gate 4b
    st = ST['graded']
    Gm = st.G['mixes']
    soft = np.array([[Gm[f'{T}_{m}']['alpha_soft'] for m in st.alphas] for T in st.targets])
    share = np.array([[Gm[f'{T}_{m}']['alpha_share'] for m in st.alphas] for T in st.targets])
    rng4 = np.random.default_rng([c['analysis_seed'], 4])
    T4 = rng4.integers(0, len(st.targets), (D, len(st.targets)))
    g4b = {}
    for m, A in AH['graded'].items():
        y = A.mean(-1)
        r_soft, r_share = LW.spearman(y.ravel(), soft.ravel()), LW.spearman(y.ravel(), share.ravel())
        Ad = resampled(A, T4, P)
        Dn = Ad.shape[0]
        ds = LW.spearman_rows(Ad.reshape(Dn, -1), soft[T4].reshape(Dn, -1))
        dh = LW.spearman_rows(Ad.reshape(Dn, -1), share[T4].reshape(Dn, -1))
        g4b[m] = dict(rho_soft=r_soft, rho_share=r_share, diff=r_soft - r_share, diff_ci=percentile_ci(ds - dh),
                      rho_soft_independent=IFX.spearman(y.ravel(), soft.ravel()))
        if m == 'c':
            boot['g4b_diff_c'] = ds - dh
    x = g4b['c']
    gates['G4b'] = dict(verdict=LW_verdict(x['diff'] >= g['g4b_min_diff'] and x['diff_ci'][0] > g['g4b_lb_above']),
                        measure='c', **x, all_measures=g4b,
                        threshold=f"rho(soft) - rho(share) >= {g['g4b_min_diff']} with 95% LB > 0")

    # Eq. 2 check (stratified vs exact on full mixes, measure c, one seed)
    eq = json.loads((res / 'eq2_check.json').read_text())
    worst = 0.0
    for r in eq:
        st_ = ST[r['family']]
        ti, ai = st_.targets.index(r['target']), st_.alphas.index(r['alpha'])
        strat = AH[r['family']]['c'][ti, ai]
        r['stratified_c_per_probe'] = strat.tolist()
        r['abs_diff'] = float(np.max(np.abs(np.array(r['exact_c_per_probe']) - strat)))
        worst = max(worst, r['abs_diff'])
    gates['eq2_check'] = dict(max_abs_diff=worst, tol=c['eq2_check_tol'], passed=bool(worst <= c['eq2_check_tol']),
                              rows=eq)

    # per-measure table and controls
    per_measure = {}
    for fam in C.FAMILIES:
        for m in AH[fam]:
            s = stats[(fam, m)]
            per_measure[f'{fam}|{m}'] = dict(
                g1_rho=[x['rho'] for x in s['g1']], g1_lb=[x['lb'] for x in s['g1']],
                g1_min_rho=min(x['rho'] for x in s['g1']), g2_slope=s['g2']['slope'], g2_ci=s['g2']['ci'],
                g2_slope_minus_oracle=s['g2']['slope_minus_oracle'], mean_alpha_hat_by_alpha=s['mean_alpha_hat_by_alpha'],
                g1_rho_independent=[x['rho_independent'] for x in s['g1']],
                g2_slope_independent=s['g2']['slope_independent'])
    gates['per_measure'] = per_measure
    crow = []
    for fam in C.FAMILIES:
        for m in CONTROLS + ['random']:
            s, b = per_measure[f'{fam}|{m}'], per_measure[f'{fam}|{CONTROL_OF.get(m, m)}']
            sep = np.mean([d['separation'] for d in DEC if d['arm'] == fam and d['measure'] == m])
            crow.append(dict(family=fam, control=m, base_measure=CONTROL_OF.get(m, ''),
                             g1_min_rho=s['g1_min_rho'], g1_min_rho_base=b['g1_min_rho'],
                             g1_mean_rho=float(np.mean(s['g1_rho'])), g1_mean_rho_base=float(np.mean(b['g1_rho'])),
                             g2_slope=s['g2_slope'], mean_separation=float(sep),
                             alpha_hat_at_alpha_min=s['mean_alpha_hat_by_alpha'][0],
                             alpha_hat_at_alpha_max=s['mean_alpha_hat_by_alpha'][-1]))
    write_csv(res / 'controls.csv', crow)
    refs_log = json.loads((res / 'refs_log.json').read_text())
    gates['inputs'] = dict(overlap=refs_log['overlap'], tv=refs_log['tv'], references=refs_log['references'],
                           mix_check=refs_log['mix_check'], graded=refs_log['graded'])
    gates['constant_scores'] = {arm: sorted([list(map(str, x)) for x in ST[arm].constant_scores])
                                for arm in J.ARMS}
    gates['analysis_seconds'] = time.time() - t_start
    C.atomic_write_text(res / 'gates.json', json.dumps(gates, indent=1, default=float))
    C.atomic_write_text(res / 'laws.json', json.dumps(laws_out, indent=1, default=float))
    C.atomic_savez(res / 'bootstrap.npz', **boot)
    return gates


def gate3_fits(c, fam, amaps, independent=False):
    return LW.gate3_fits(c, fam, {k: {(t, key_alpha(a)): v for (t, a), v in m.items()} for k, m in amaps.items()},
                         independent=independent)


def law_agreement(fa, fb, tol, folds=False):
    worst = 0.0
    for k in fa:
        for pa, pb in zip(fa[k], fb[k]):
            if folds:
                for x, y in zip(pa['folds'], pb['folds']):
                    worst = max(worst, abs(x['cost'] - y['cost']) / max(abs(x['cost']), 1e-300))
            else:
                worst = max(worst, abs(pa['cost'] - pb['cost']) / max(abs(pa['cost']), 1e-300))
                f1, f2 = np.array(pa['forecast']), np.array(pb['forecast'])
                worst = max(worst, float(np.max(np.abs(f1 - f2) / np.abs(f1))))
    return dict(max_rel_diff=worst, tol=tol, agree=bool(worst <= tol))


def percentile_ci(d):
    return LW.percentile_ci(d)


def LW_verdict(ok):
    return 'SUPPORTED' if ok else 'NOT_ESTABLISHED'


def write_csv(path, rows):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    a = ap.parse_args()
    c = C.load_config()
    g = analyze(c, HERE / a.results)
    print(json.dumps({k: g[k]['verdict'] for k in ('G1', 'G2', 'G3', 'G4a', 'G4b')}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
