"""Experiment 13 analysis, arm 13-GPU (analyze.py --arm gpu dispatches here).

Scopes: 'pooled' (all three sizes, the full N laws of PROTOCOL Section 4.1) and each size S, M, L (the six laws).
Laws are fit per seed on fit rows only (held out: level 32, alpha 0.15, each run's largest checkpoint); every held-out
forecast is written to forecasts.csv and hashed before any held-out outcome is read. G1-G5 are computed in every scope
with the definitions gated in 13-small (PROTOCOL Section 5, alignment with Experiment 11) plus the 13-GPU rule that
each seed's own point estimate must also fall on the supported side; the verdict of record is the pooled one and the
per-size verdicts are reported with it. G6 compares M_K of the per-size M_K_both fits. The draft gate versions,
multiplier curves, sensitivity variants and mechanism analyses are reported, not gated.

Usage (from expt_v1/):
  analyze.py --arm gpu --stage 1   fits; writes forecasts.csv and fits.json (with the forecasts SHA-256)
  analyze.py --arm gpu --stage 2   scores, bootstraps, gates; writes gates.json, analysis.json, bootstrap.npz
"""
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

import analyze as AN  # noqa: E402
import independent_fit as IF  # noqa: E402
import laws as L  # noqa: E402

SIZES = ['S', 'M', 'L']
SCOPES = ['pooled'] + SIZES
VARIANTS = AN.VARIANTS
ROLE = {'pooled': dict(eff='N_align_eff', coef='N_K_coef', exp='N_K_exp', both='N_K_both')}
ROLE.update({s: dict(eff='M_align_eff', coef='M_K_coef', exp='M_K_exp', both='M_K_both') for s in SIZES})
gm, ci, pct = AN.gm, AN.ci, AN.pct


def laws_of(scope):
    return L.N_LAWS if scope == 'pooled' else L.LAWS


# ---------------------------------------------------------------- data

def read_rows(out, tasks, outcome=True):
    canon = {t['task_id']: t['K_bits_canonical'] for t in tasks}
    rows = []
    with open(out / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            d = dict(seed=int(r['seed']), target_id=r['target_id'], K=int(r['K']), K_bits=int(r['K_bits']),
                     K_bits_canonical=canon[r['target_id']], alpha=float(r['alpha']), size=r['size'], N=int(r['N']),
                     T=int(r['T']), slice=r['slice'], ident_bits=float(r['ident_bits']),
                     oracle_bits=float(r['oracle_bits']), mixture_bits=float(r['mixture_bits']))
            if outcome:
                d.update(excess_bits=float(r['excess_bits']), learn_bits=float(r['learn_bits']),
                         model_bits=float(r['model_bits']))
            rows.append(d)
    return rows


def tmax_of(rows):
    return {s: max(r['T'] for r in rows if r['size'] == s) for s in SIZES}


def is_fit(c, r, Tmax):
    return r['K'] != c['heldout_level'] and abs(r['alpha'] - c['heldout_alpha']) > 1e-9 and r['T'] != Tmax[r['size']]


def in_scope(r, scope):
    return scope == 'pooled' or r['size'] == scope


def X_for(c, tasks, rows, variant, T0):
    return AN.X_for(c, tasks, rows, variant, T0)


# ---------------------------------------------------------------- stage 1

def stage1(c, out, tasks):
    rows = [r for r in read_rows(out, tasks, outcome=False) if r['slice'] == 'all']
    Tmax = tmax_of(rows)
    T0 = min(r['T'] for r in rows)
    seeds = sorted({r['seed'] for r in rows})
    key = lambda r: (r['seed'], r['target_id'], r['alpha'], r['size'], r['T'])  # noqa: E731
    full = {key(r): r for r in read_rows(out, tasks, True) if r['slice'] == 'all' and is_fit(c, r, Tmax)}
    fits, fc_rows = {}, []
    for v in VARIANTS:
        for scope in SCOPES:
            for s in seeds:
                fr = [full[key(r)] for r in rows if r['seed'] == s and in_scope(r, scope) and is_fit(c, r, Tmax)]
                hr = [r for r in rows if r['seed'] == s and in_scope(r, scope) and not is_fit(c, r, Tmax)]
                Xf, Xh = X_for(c, tasks, fr, v, T0), X_for(c, tasks, hr, v, T0)
                for law in laws_of(scope):
                    f = L.fit(law, Xf, c)
                    r_fit, _ = L.scored_residual(law, f['theta'], Xf)
                    f['fit_rmse'] = float(np.sqrt(np.mean(r_fit ** 2)))
                    f['n_fit'] = len(fr)
                    fits[f'{v}|{scope}|{s}|{law}'] = f
                    p = L.predict_eps(law, f['theta'], Xh)
                    bad = L.invalid_mask(law, p, Xh)
                    for r, pv, b in zip(hr, p, bad):
                        fc_rows.append([v, scope, s, law, r['target_id'], r['K'], r['alpha'], r['size'], r['T'],
                                        repr(float(pv)), int(b)])
            print(f'stage1 {v} seeds done', flush=True)
    with open(out / 'forecasts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'scope', 'seed', 'law', 'target_id', 'K', 'alpha', 'size', 'T', 'forecast', 'invalid'])
        w.writerows(fc_rows)
    sha = hashlib.sha256((out / 'forecasts.csv').read_bytes()).hexdigest()

    def pooled_fit(scope, law):
        return math.sqrt(np.mean([fits[f'primary|{scope}|{s}|{law}']['fit_rmse'] ** 2 for s in seeds]))
    klaw, krmse = {}, {}
    for scope in SCOPES:
        rc, re = pooled_fit(scope, ROLE[scope]['coef']), pooled_fit(scope, ROLE[scope]['exp'])
        klaw[scope] = ROLE[scope]['coef'] if rc <= re else ROLE[scope]['exp']
        krmse[scope] = {ROLE[scope]['coef']: rc, ROLE[scope]['exp']: re}
    meta = dict(forecasts_sha256=sha, seeds=seeds, T0=T0, Tmax=Tmax, g2_K_law=klaw, g2_fit_rmse=krmse,
                eps_res=c['eps_res'])
    (out / 'fits.json').write_text(json.dumps(dict(meta=meta, fits=fits), indent=1))
    print(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- stage 2

def stage2(c, out, tasks):
    F = json.loads((out / 'fits.json').read_text())
    meta, fits = F['meta'], F['fits']
    assert hashlib.sha256((out / 'forecasts.csv').read_bytes()).hexdigest() == meta['forecasts_sha256']
    seeds, T0, Tmax = meta['seeds'], meta['T0'], meta['Tmax']
    S = len(seeds)
    rows_all = read_rows(out, tasks, True)
    rows = [r for r in rows_all if r['slice'] == 'all']
    Ts = {z: sorted({r['T'] for r in rows if r['size'] == z}) for z in SIZES}
    tg = sorted([t for t in tasks if t['role'] == 'target'], key=lambda t: (t['K'], t['task_id']))
    levels = sorted({t['K'] for t in tg})
    alphas = sorted({r['alpha'] for r in rows})
    eps_star = c['epsilon_star']
    B = AN.boot_idx(c, S)
    kmin, kmax = min(levels), max(levels)
    ids = {k: [t['task_id'] for t in tg if t['K'] == k] for k in levels}
    res = dict(meta=meta, epsilon_star=eps_star, scopes={})
    look = {}
    for r in rows:
        look[(r['seed'], r['size'], r['target_id'], r['alpha'], r['T'])] = r

    # ---------------- held-out scoring per variant, scope, seed, law
    sc = {}
    for v in VARIANTS:
        for scope in SCOPES:
            for s in seeds:
                hr = [r for r in rows if r['seed'] == s and in_scope(r, scope) and not is_fit(c, r, Tmax)]
                Xh = X_for(c, tasks, hr, v, T0)
                lvl = np.array([r['K'] for r in hr])
                held = [(r['K'] == c['heldout_level'], abs(r['alpha'] - c['heldout_alpha']) < 1e-9,
                         r['T'] == Tmax[r['size']]) for r in hr]
                only = {k: np.array([h[i] and sum(h) == 1 for h in held]) for i, k in
                        enumerate(('level_only', 'alpha_only', 'T_only'))}
                for law in laws_of(scope):
                    rr, bad = L.scored_residual(law, fits[f'{v}|{scope}|{s}|{law}']['theta'], Xh)
                    sc[(v, scope, s, law)] = dict(
                        sse=float(np.sum(rr ** 2)), n=len(rr), invalid=int(bad.sum()),
                        sse_level=float(np.sum(rr[lvl == c['heldout_level']] ** 2)),
                        n_level=int((lvl == c['heldout_level']).sum()),
                        by={k: dict(sse=float(np.sum(rr[m] ** 2)), n=int(m.sum())) for k, m in only.items()})

    def arr(v, scope, law, key='sse'):
        return np.array([sc[(v, scope, s, law)][key] for s in seeds])

    def pooled(v, scope, law, idx=None, key='sse', nkey='n'):
        a, n = arr(v, scope, law, key), arr(v, scope, law, nkey)
        if idx is None:
            return math.sqrt(a.sum() / n.sum())
        return np.sqrt(a[idx].sum(1) / n[idx].sum(1))

    def reduction(v, scope, new, base, key='sse', nkey='n'):
        pt = 1 - pooled(v, scope, new, key=key, nkey=nkey) / pooled(v, scope, base, key=key, nkey=nkey)
        dr = 1 - pooled(v, scope, new, B, key, nkey) / pooled(v, scope, base, B, key, nkey)
        per = 1 - np.sqrt(arr(v, scope, new, key) / arr(v, scope, new, nkey)) / \
            np.sqrt(arr(v, scope, base, key) / arr(v, scope, base, nkey))
        return pt, dr, per

    res['heldout_rmse'] = {v: {scope: {law: dict(
        rmse=pooled(v, scope, law), invalid=int(arr(v, scope, law, 'invalid').sum()),
        by_coordinate={k: math.sqrt(sum(sc[(v, scope, s, law)]['by'][k]['sse'] for s in seeds)
                                    / max(1, sum(sc[(v, scope, s, law)]['by'][k]['n'] for s in seeds)))
                       for k in ('level_only', 'alpha_only', 'T_only')})
        for law in laws_of(scope)} for scope in SCOPES} for v in VARIANTS}

    # ---------------- independent fitter (held-out RMSE per scope, seed and law, primary variant)
    disagree = []
    for scope in SCOPES:
        for s in seeds:
            fr = [r for r in rows if r['seed'] == s and in_scope(r, scope) and is_fit(c, r, Tmax)]
            hr = [r for r in rows if r['seed'] == s and in_scope(r, scope) and not is_fit(c, r, Tmax)]
            Xf, Xh = X_for(c, tasks, fr, 'primary', T0), X_for(c, tasks, hr, 'primary', T0)
            for law in laws_of(scope):
                g = IF.fit(law, Xf, c)
                f = fits[f'primary|{scope}|{s}|{law}']
                a, b = IF.heldout_rmse(law, f['theta'], Xh)[0], IF.heldout_rmse(law, g['theta'], Xh)[0]
                if not IF.agree(a, b, c['independent_fit_tol']):
                    disagree.append(dict(scope=scope, seed=s, law=law, laws_py=a, independent=b,
                                         cost_laws=f['cost'], cost_independent=g['cost']))
        print(f'independent fit {scope} done', flush=True)
    res['independent_fit'] = dict(agree=not disagree, disagreements=disagree)
    print('independent fit', 'AGREE' if not disagree else f'DISAGREE {disagree[0]}', flush=True)

    # ---------------- model-free T* tables per size
    def tstar_table(eps):
        tab = {}
        for z in SIZES:
            for s in seeds:
                for t in tg:
                    for a in alphas:
                        vals = [look[(s, z, t['task_id'], a, T)]['learn_bits'] for T in Ts[z]]
                        tab[(z, s, t['task_id'], a)] = AN.mf_tstar(Ts[z], vals, eps)
        return tab
    tst = tstar_table(eps_star)
    Tj = {z: {k: np.array([[tst[(z, s, j, 1.0)][0] for j in ids[k]] for s in seeds]) for k in levels} for z in SIZES}

    def law_fits_all(scope, v):
        """'both' law refit on all rows of the scope (descriptive, for E)."""
        out_ = {}
        for s in seeds:
            rs = [r for r in rows if r['seed'] == s and in_scope(r, scope)]
            X = X_for(c, tasks, rs, v, T0)
            out_[s] = (L.fit(ROLE[scope]['both'], X, c), X)
        return out_
    lk = np.log([t['K_bits'] for t in tg])
    ln_ = np.log([t['K'] for t in tg])
    slope = float(np.polyfit(ln_, lk, 1)[0])
    G6_MK = {}
    for scope in SCOPES:
        sizes = SIZES if scope == 'pooled' else [scope]
        role = ROLE[scope]
        out_scope = {}

        def per_seed_param(law, key, v='primary'):
            return np.array([fits[f'{v}|{scope}|{s}|{law}']['params'].get(key, np.nan) for s in seeds])

        def summ(a):
            a = np.asarray(a, float)
            d = a[B].mean(1)
            return dict(estimate=float(a.mean()), ci95=ci(d), per_seed=a.tolist()), d

        # G1: model-free M_K at alpha = 1 (geometric mean over the scope's sizes)
        MK_seed = np.array([gm([gm(Tj[z][kmax][i]) / gm(Tj[z][kmin][i]) for z in sizes]) for i in range(S)])
        MK = gm(MK_seed)
        d_seed = np.exp(np.log(MK_seed)[B].mean(1))
        rng2 = np.random.default_rng(c['analysis_seed'] + 1)
        d_two = np.empty(c['bootstrap_draws'])
        for i in range(c['bootstrap_draws']):
            jj = rng2.integers(0, 2, size=(2, S, 2))
            d_two[i] = math.exp(np.mean([np.mean([math.log(gm(Tj[z][kmax][s][jj[0, n]]) / gm(Tj[z][kmin][s][jj[1, n]]))
                                                  for z in sizes]) for n, s in enumerate(B[i])]))
        cens = {f: sum(1 for k, v in tst.items() if k[0] in sizes and v[1] == f) for f in ('ok', 'left', 'right')}
        G1 = dict(M_K=MK, ci95_seed=ci(d_seed), ci95_two_stage=ci(d_two), per_seed=MK_seed.tolist(),
                  censoring_all=cens, threshold=c['g1_min_MK'],
                  every_seed=bool(np.all(MK_seed >= c['g1_min_MK'])))
        G1['verdict'] = 'SUPPORTED' if (MK >= c['g1_min_MK'] and G1['ci95_seed'][0] > 1 and G1['ci95_two_stage'][0] > 1
                                        and G1['every_seed']) else 'NOT_ESTABLISHED'

        # G2
        klaw = meta['g2_K_law'][scope]
        pt, dr, per = reduction('primary', scope, klaw, role['eff'])
        lvl = {law: math.sqrt(arr('primary', scope, law, 'sse_level').sum() / arr('primary', scope, law, 'n_level').sum())
               for law in (klaw, role['eff'])}
        inv = int(arr('primary', scope, klaw, 'invalid').sum())
        G2 = dict(K_law=klaw, reduction=pt, ci95=ci(dr), per_seed=per.tolist(), rmse_K_law=pooled('primary', scope, klaw),
                  rmse_align_eff=pooled('primary', scope, role['eff']), heldout_level_rmse_K_law=lvl[klaw],
                  heldout_level_rmse_align_eff=lvl[role['eff']], invalid_K_law=inv,
                  every_seed=bool(np.all(per >= c['rmse_reduction_threshold'])))
        G2['verdict'] = 'SUPPORTED' if (pt >= c['rmse_reduction_threshold'] and G2['ci95'][0] > 0
                                        and lvl[klaw] < lvl[role['eff']] and inv == 0 and G2['every_seed']) \
            else 'NOT_ESTABLISHED'

        # G3a / G3b
        kap, d_kap = summ(per_seed_param(role['both'], 'kappa'))
        g1p, d_g1 = summ(per_seed_param(role['both'], 'gamma1'))
        r_coef, r_exp = pooled('primary', scope, role['coef']), pooled('primary', scope, role['exp'])
        rc_s = np.sqrt(arr('primary', scope, role['coef']) / arr('primary', scope, role['coef'], 'n'))
        re_s = np.sqrt(arr('primary', scope, role['exp']) / arr('primary', scope, role['exp'], 'n'))
        pt3, dr3, per3 = reduction('primary', scope, role['exp'], role['coef'])
        G3a = dict(kappa=kap, gamma1=g1p, rmse_coef=r_coef, rmse_exp=r_exp,
                   every_seed=bool(np.all(np.array(kap['per_seed']) > 0) and np.all(rc_s <= re_s)))
        G3a['verdict'] = 'SUPPORTED' if (kap['ci95'][0] > 0 and g1p['ci95'][0] <= 0 <= g1p['ci95'][1]
                                         and r_coef <= r_exp and G3a['every_seed']) else 'NOT_ESTABLISHED'
        G3b = dict(gamma1=g1p, exp_vs_coef_reduction=pt3, ci95=ci(dr3), per_seed=per3.tolist(),
                   threshold=c['g3b_reduction'],
                   every_seed=bool(np.all(np.array(g1p['per_seed']) < 0) and np.all(per3 >= c['g3b_reduction'])))
        G3b['verdict'] = 'SUPPORTED' if (g1p['ci95'][1] < 0 and pt3 >= c['g3b_reduction'] and G3b['ci95'][0] > 0
                                         and G3b['every_seed']) else 'NOT_ESTABLISHED'

        # G4: E = e_alpha / e_K from the 'both' law refit on all rows, both K units, and E_mf
        allfits = {v: law_fits_all(scope, v) for v in ('primary', 'K_n')}

        def E_at(eps, v='primary'):
            ea, ek = [], []
            for s in seeds:
                f, X = allfits[v][s]
                a, k = L.elasticities(role['both'], f['params'], X['K_ref'], eps, X)
                ea.append(a)
                ek.append(k)
            ea, ek = np.array(ea), np.array(ek)
            pt_ = ea.mean() / ek.mean() if ek.mean() > 0 else math.inf
            ma, mk = ea[B].mean(1), ek[B].mean(1)
            d = np.where(mk > 0, ma / np.where(mk > 0, mk, 1), np.inf)
            per_ = np.where(ek > 0, ea / np.where(ek > 0, ek, 1), np.inf)
            return pt_, d, ea, ek, per_
        E_b, dE_b, ea, ek, perE = E_at(eps_star)
        E_s, dE_s = E_b / slope, dE_b / slope
        En, dEn, _, _, perEn = E_at(eps_star, 'K_n')

        def E_mf(eps):
            tab = tstar_table(eps) if eps != eps_star else tst
            coefs, ncens = [], []
            for s in seeds:
                pts = [(sizes.index(z), math.log2(1 / a), math.log2(t['K_bits']), math.log2(tab[(z, s, t['task_id'], a)][0]))
                       for z in sizes for t in tg for a in alphas if tab[(z, s, t['task_id'], a)][1] == 'ok']
                ncens.append(len(sizes) * len(tg) * len(alphas) - len(pts))
                Z = np.array(pts) if pts else np.zeros((0, 4))
                dummies = np.eye(len(sizes))[Z[:, 0].astype(int)] if len(Z) else np.zeros((0, len(sizes)))
                if len(Z) >= len(sizes) + 3:
                    b = np.linalg.lstsq(np.column_stack([dummies, Z[:, 1], Z[:, 2]]), Z[:, 3], rcond=None)[0]
                    coefs.append(b[-2:])
                else:
                    coefs.append(np.array([np.nan, np.nan]))
            coefs = np.array(coefs)
            ok = np.all(np.isfinite(coefs), 1)
            if not ok.any():
                return None, ncens, []
            cm = coefs[ok].mean(0)
            per_ = [float(a / k) if (np.isfinite(k) and k > 0) else math.inf for a, k in coefs]
            return (float(cm[0] / cm[1]) if cm[1] > 0 else math.inf), ncens, per_
        med_cens = float(np.median(E_mf(eps_star)[1]))
        eps_mf = eps_star if med_cens <= 0.5 * len(sizes) * len(tg) * len(alphas) else 2 * eps_star
        Emf, ncens_mf, perEmf = E_mf(eps_mf)
        G4 = dict(E_bits=E_b, ci95_bits=ci(dE_b), E_states=E_s, ci95_states=ci(dE_s), bits_per_state_slope=slope,
                  E_states_refit_K_n=En, ci95_states_refit=ci(dEn), e_alpha=ea.tolist(), e_K_bits=ek.tolist(),
                  per_seed_E_bits=perE.tolist(), per_seed_E_states=(perE / slope).tolist(),
                  E_mf=Emf, E_mf_eps=eps_mf, E_mf_censored_pairs_per_seed=ncens_mf, per_seed_E_mf=perEmf,
                  every_seed=bool(np.all(perE > 1) and np.all(perE / slope > 1) and len(perEmf) == S
                                  and all(x > 1 for x in perEmf)))
        G4['verdict'] = 'SUPPORTED' if (E_b > 1 and G4['ci95_bits'][0] > 1 and E_s > 1 and G4['ci95_states'][0] > 1
                                        and Emf is not None and Emf > 1 and G4['every_seed']) else 'NOT_ESTABLISHED'

        # G5: position decomposition on the balanced grid (levels x alpha x checkpoint index), sizes averaged
        pos = {}
        for sl in ('early', 'middle', 'late', 'all'):
            for q in ('excess_bits', 'learn_bits', 'ident_bits'):
                rs_seed = []
                for s in seeds:
                    ys, ai, ki, ti = [], [], [], []
                    for k in levels:
                        for a in alphas:
                            for it in range(len(Ts[SIZES[0]])):
                                vals = [math.log2(max(r[q], c['eps_res'])) for r in rows_all
                                        if r['seed'] == s and r['slice'] == sl and r['K'] == k and r['alpha'] == a
                                        and r['size'] in sizes and r['T'] == Ts[r['size']][it]]
                                ys.append(float(np.mean(vals)))
                                ai.append(a)
                                ki.append(k)
                                ti.append(it)
                    share, _ = AN.main_effect_share(np.array(ys), np.array(ai), np.array(ki), np.array(ti))
                    rs_seed.append(share)
                rs_seed = np.array(rs_seed)
                pos[f'{sl}|{q}'] = dict(r_alpha=float(rs_seed.mean()), ci95=ci(rs_seed[B].mean(1)),
                                        per_seed=rs_seed.tolist())
        e, l_ = pos['early|excess_bits'], pos['late|excess_bits']
        G5 = dict(r_alpha_early=e, r_alpha_late=l_,
                  every_seed=bool(np.all(np.array(e['per_seed']) > c['g5_share'])
                                  and np.all(np.array(l_['per_seed']) < c['g5_share'])))
        G5['verdict'] = 'SUPPORTED' if (e['ci95'][0] > c['g5_share'] and l_['ci95'][1] < c['g5_share']
                                        and G5['every_seed']) else 'NOT_ESTABLISHED'
        G5['named_failure'] = ('alpha dominates everywhere' if l_['ci95'][0] > c['g5_share'] else
                               'K dominates everywhere' if e['ci95'][1] < c['g5_share'] else None)

        # draft versions (reported, not gated)
        conv_fail = sum(1 for s in seeds if not fits[f'primary|{scope}|{s}|{role["both"]}']['converged']) / S
        mlaw = role['both'] if conv_fail <= c['fallback_fail_share'] else role['coef']

        def law_MK(eps, law=mlaw, v='primary'):
            o = []
            for s in seeds:
                f = fits[f'{v}|{scope}|{s}|{law}']
                Xf = X_for(c, tasks, [r for r in rows if r['seed'] == s and in_scope(r, scope) and is_fit(c, r, Tmax)],
                           v, T0)
                unit = VARIANTS[v][0]
                kv = sorted({float(t[unit]) for t in tg if t['K'] != c['heldout_level']})
                o.append(L.t_star(law, f['params'], kv[-1], 1.0, eps, Xf) / L.t_star(law, f['params'], kv[0], 1.0, eps, Xf))
            return np.array(o)
        mk_law = law_MK(eps_star)
        G6_MK[scope] = dict(law=mlaw, per_seed=mk_law, convergence_fail_share=conv_fail)
        Malpha = 20.0 ** per_seed_param(mlaw, 'eta')
        dR = np.log(Malpha[B]).mean(1) / np.log(mk_law[B]).mean(1)
        gap = {}
        for it in range(len(Ts[SIZES[0]])):
            gap[it] = np.array([np.mean([r['excess_bits'] for r in rows if r['seed'] == s and r['K'] == kmax
                                         and r['alpha'] == 1.0 and r['size'] in sizes and r['T'] == Ts[r['size']][it]])
                                - np.mean([r['excess_bits'] for r in rows if r['seed'] == s and r['K'] == kmin
                                           and r['alpha'] == 1.0 and r['size'] in sizes and r['T'] == Ts[r['size']][it]])
                                for s in seeds])
        last_fit = len(Ts[SIZES[0]]) - 2
        ptc, drc, _ = reduction('primary', scope, role['coef'], role['eff'])
        pte, dre, _ = reduction('primary', scope, role['exp'], role['eff'])
        drho = pooled('primary', scope, role['exp'], B) / pooled('primary', scope, role['coef'], B)
        drafts = dict(
            law_for_multipliers=mlaw, convergence_fail_share=conv_fail,
            G1_law_M_K=dict(estimate=gm(mk_law), ci95=ci(np.exp(np.log(mk_law)[B].mean(1))), per_seed=mk_law.tolist()),
            G2_bonferroni=dict(coef=dict(reduction=ptc, ci_bonf=[pct(drc, 1.25), pct(drc, 98.75)]),
                               exp=dict(reduction=pte, ci_bonf=[pct(dre, 1.25), pct(dre, 98.75)])),
            G3_rho=dict(rho=r_exp / r_coef, ci95=ci(drho)),
            G4a_range_ratio=dict(R=float(np.mean(np.log(Malpha)) / np.mean(np.log(mk_law))), ci95=ci(dR),
                                 M_alpha=gm(Malpha), M_K=gm(mk_law)),
            G4b_gap_ratio=dict(ratio=float(gap[last_fit].mean() / gap[0].mean()),
                               ci95=ci(gap[last_fit][B].mean(1) / gap[0][B].mean(1)),
                               gap_first=float(gap[0].mean()), gap_last_fit=float(gap[last_fit].mean())),
            elasticity_ratio_e_alpha_over_e_K=E_b)
        curves = {}
        for eps in c['eps_curve']:
            Eb, dEb, _, _, _ = E_at(eps)
            tt = tstar_table(eps)
            mk_mf = gm([gm([gm([tt[(z, s, j, 1.0)][0] for j in ids[kmax]]) / gm([tt[(z, s, j, 1.0)][0] for j in ids[kmin]])
                            for z in sizes]) for s in seeds])
            curves[str(eps)] = dict(E_bits=Eb, E_ci95=ci(dEb), M_K_law=gm(law_MK(eps)), M_alpha_law=gm(Malpha),
                                    M_K_model_free=mk_mf,
                                    right_censored_alpha1=sum(1 for z in sizes for s in seeds for k in levels
                                                              for j in ids[k] if tt[(z, s, j, 1.0)][1] == 'right'))
        out_scope.update(gates=dict(G1=G1, G2=G2, G3a=G3a, G3b=G3b, G4=G4, G5=G5), position_decomposition=pos,
                         draft_protocol_reported=drafts, multiplier_curves=curves,
                         fitted_laws_per_seed={law: {str(s): fits[f'primary|{scope}|{s}|{law}']['params'] for s in seeds}
                                               for law in laws_of(scope)})
        if scope != 'pooled':
            pta, dra, _ = reduction('primary', scope, 'M_align_eff', 'M_align_add')
            out_scope['alignment_comparison'] = dict(eta_align_eff=summ(per_seed_param('M_align_eff', 'eta'))[0],
                                                     align_eff_vs_add_reduction=pta, ci95=ci(dra))
        else:
            pta, dra, _ = reduction('primary', scope, 'N_K_both', 'N_AS_both')
            out_scope['alignment_comparison'] = dict(effective_data_vs_alignment_scaling_reduction=pta, ci95=ci(dra))
        out_scope['C3'] = ('coefficient' if G3a['verdict'] == 'SUPPORTED' else
                           'exponent' if G3b['verdict'] == 'SUPPORTED' else 'unresolved')
        res['scopes'][scope] = out_scope
        print(scope, {k: v['verdict'] for k, v in out_scope['gates'].items()}, flush=True)

    # ---------------- G6: does a bigger model shrink the complexity multiplier? (per-size M_K_both fits)
    mk = {z: G6_MK[z]['per_seed'] for z in SIZES}
    ratio = mk['L'] / mk['S']
    d6 = np.exp(np.log(ratio)[B].mean(1))
    pts = {z: gm(mk[z]) for z in SIZES}
    G6 = dict(M_K={z: dict(estimate=pts[z], per_seed=mk[z].tolist(), law=G6_MK[z]['law']) for z in SIZES},
              ratio_L_over_S=gm(ratio), ci95=ci(d6), per_seed=ratio.tolist(), threshold=c['g6_ratio'],
              monotone=bool(pts['S'] >= pts['M'] >= pts['L']), every_seed=bool(np.all(ratio <= c['g6_ratio'])))
    G6['verdict'] = 'SUPPORTED' if (G6['ratio_L_over_S'] <= c['g6_ratio'] and G6['ci95'][1] < 1 and G6['monotone']
                                    and G6['every_seed']) else 'NOT_ESTABLISHED'
    G6['reverse'] = bool(G6['ratio_L_over_S'] >= c['g6_reverse_ratio'] and G6['ci95'][0] > 1)
    G6['named_outcome'] = 'bigger models pay relatively more for complexity' if G6['reverse'] else None
    res['G6'] = G6

    # ---------------- mechanism analyses (reported, not gated)
    mech = {}
    for z in SIZES:
        gam = {}
        for t in tg:
            gs = []
            for s in seeds:
                rs = [look[(s, z, t['task_id'], 1.0, T)] for T in Ts[z]]
                y = np.log2(np.maximum([r['excess_bits'] for r in rs], c['eps_res']))
                gs.append(float(-np.polyfit(np.log2(Ts[z]), y, 1)[0]))
            gam[t['task_id']] = float(np.mean(gs))
        tail = [t['visit_tail_exponent'] for t in tg]
        sp = spearmanr([gam[t['task_id']] for t in tg], tail)
        early = [r for r in rows_all if r['slice'] == 'early' and r['size'] == z and r['T'] == Tmax[z] and r['alpha'] < 1]
        mech[z] = dict(gamma_alpha1_per_target=gam, spearman_gamma_vs_tail=dict(rho=float(sp.statistic),
                                                                                p=float(sp.pvalue), tail=tail),
                       early_share_eps_id_at_Tmax={str(a): float(np.mean([r['ident_bits'] / r['excess_bits']
                                                                          for r in early if r['alpha'] == a
                                                                          and r['excess_bits'] > 0]))
                                                   for a in alphas if a < 1})
    shape_fit = {}
    for t in tg:
        ei = {a: next(r['ident_bits'] for r in rows if r['target_id'] == t['task_id'] and r['alpha'] == a) for a in alphas}
        xa = np.array([a for a in alphas if a < 1])
        ya = np.array([ei[a] for a in xa])
        lin, lg = 1 - xa, np.log2(1 / xa)
        shape_fit[t['task_id']] = dict(eps_id={str(a): ei[a] for a in alphas},
                                       rss_linear=float(np.sum((ya - lin * (lin @ ya) / (lin @ lin)) ** 2)),
                                       rss_log=float(np.sum((ya - lg * (lg @ ya) / (lg @ lg)) ** 2)))
    mech['eps_id_shape'] = dict(per_target=shape_fit, total_rss_linear=sum(v['rss_linear'] for v in shape_fit.values()),
                                total_rss_log=sum(v['rss_log'] for v in shape_fit.values()))
    dl = list(csv.DictReader(open(out / 'distractor_loss.csv')))
    mech['distractor_excess_at_Tmax_by_alpha'] = {
        z: {str(a): float(np.mean([float(d['model_bits']) - float(d['oracle_bits']) for d in dl
                                   if d['size'] == z and float(d['alpha']) == a and int(d['T']) == Tmax[z]]))
            for a in alphas} for z in SIZES}
    post = json.loads((out / 'posterior16.json').read_text())
    mech['posterior_target_after_16'] = {k: v for k, v in post.items() if k.split('|')[0] in {t['task_id'] for t in tg}}
    res['mechanism'] = mech
    res['diversity'] = AN.diversity_line()

    gates = {}
    for g in ('G1', 'G2', 'G3a', 'G3b', 'G4', 'G5'):
        gates[g] = dict(verdict=res['scopes']['pooled']['gates'][g]['verdict'],
                        per_size={z: res['scopes'][z]['gates'][g]['verdict'] for z in SIZES})
    gates['G6'] = dict(verdict=G6['verdict'])
    res['gates'] = gates
    res['C3'] = {scope: res['scopes'][scope]['C3'] for scope in SCOPES}
    (out / 'gates.json').write_text(json.dumps(dict(forecasts_sha256=meta['forecasts_sha256'], gates=gates,
                                                    detail={scope: res['scopes'][scope]['gates'] for scope in SCOPES},
                                                    G6=G6, C3=res['C3']), indent=1, default=float))
    (out / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez(out / 'bootstrap.npz', seed_index=B, G6_ratio=d6)
    for k, v in gates.items():
        print(k, v)
    return res
