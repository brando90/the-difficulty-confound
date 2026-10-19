"""Experiment 18 analysis (PROTOCOL Sections 4-5): laws, held-out scoring, model-free T*, elasticities, gates, the
mismatch sweep and the hard-to-easy arm. Writes <out>/analysis.json.

Usage (from expt_v1/): analyze.py [--results results] [--out results/analysis.json] [--datasets mnist ...]
A dataset whose main grid is incomplete is analysed on the runs present and flagged `complete: false`."""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import independent_fit as IF  # noqa: E402
import laws as L  # noqa: E402
import tasks as TK  # noqa: E402

CFG = json.load(open(os.path.join(HERE, 'config.json')))
LEVEL_OF = {t: lv for lv, ts in TK.MAIN_LEVELS for t in ts}
B_SEED = CFG['analysis_seed']
NB = CFG['bootstrap_draws']


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.mean(np.log(x))))


def ci(d, lo=2.5, hi=97.5):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return [float(np.percentile(d, lo)), float(np.percentile(d, hi))] if len(d) else [None, None]


def boot(S, n=NB, seed=B_SEED):
    return np.random.default_rng(seed).integers(0, S, size=(n, S))


def mf_tstar(Ts, vals, eps_star):
    """Model-free T*: first crossing of eps_star, log-log interpolated (Experiment 13's rule)."""
    v = np.maximum(np.asarray(vals, float), 1e-6)
    if v[0] <= eps_star:
        return float(Ts[0]), 'left'
    for i in range(1, len(Ts)):
        if v[i] <= eps_star:
            a, b = math.log(v[i - 1]), math.log(v[i])
            fr = (math.log(eps_star) - a) / (b - a)
            return float(math.exp(math.log(Ts[i - 1]) + fr * (math.log(Ts[i]) - math.log(Ts[i - 1])))), 'ok'
    return float(Ts[-1]), 'right'


def load_runs(res, arm, ds):
    out = []
    for p in sorted(glob.glob(os.path.join(res, arm, f'{arm}__{ds}__*.json'))):
        out.append(json.load(open(p)))
    return out


def main_rows(runs):
    rows = []
    for r in runs:
        s = r['spec']
        t = s['target']
        for row in r['rows']:
            rows.append(dict(seed=s['seed'], task_id=t, level=LEVEL_OF[t], K_bits=TK.k_bits(t),
                             K_classes=2.0 ** TK.k_bits(t), alpha=s['alpha'], T=row['T'], mismatch=s['mismatch'],
                             excess_bits=row[f'eps__{t}'], ident_bits=0.0, err=row[f'err__{t}']))
    return rows


def X_of(rows, unit, Kgrid, Kref, T0):
    return L.design(rows, unit, Kgrid, Kref, T0, CFG['eps_res'], offset=False)


def fit_rows(rows, Tmax):
    return [r for r in rows if r['level'] != CFG['level_heldout'] and r['alpha'] != CFG['alpha_heldout']
            and r['T'] != Tmax]


def regime(rows):
    return [r for r in rows if r['excess_bits'] <= CFG['learning_regime_max_eps']]


def analyse_main(runs, ds):
    rows = main_rows(runs)
    seeds = sorted({r['seed'] for r in rows})
    Ts = sorted({r['T'] for r in rows})
    Tmax, T0 = max(Ts), min(Ts)
    eps_star = CFG['eps_star'][ds]
    alphas = CFG['alphas']
    targets = [t for _, ts in TK.MAIN_LEVELS for t in ts]
    fit_levels = [lv for lv, _ in TK.MAIN_LEVELS if lv != CFG['level_heldout']]
    Kfit = [TK.k_bits(ts[0]) for lv, ts in TK.MAIN_LEVELS if lv in fit_levels]
    Kref = gm(Kfit)
    Kgrid = [TK.k_bits(ts[0]) for _, ts in TK.MAIN_LEVELS]
    A = dict(dataset=ds, n_runs=len(runs), n_runs_expected=len(seeds and CFG['seeds']) * len(targets) * len(alphas),
             seeds=seeds, T_checkpoints=Ts, eps_star=eps_star, K_ref_bits=Kref)
    A['complete'] = A['n_runs'] == A['n_runs_expected']
    S = len(seeds)
    Bi = boot(S)

    # ---- cell table (seed-mean eps per target, alpha, T) for the report
    cell = {}
    for r in rows:
        cell.setdefault((r['task_id'], r['alpha'], r['T']), []).append(r['excess_bits'])
    A['cells'] = [dict(task_id=k[0], alpha=k[1], T=k[2], eps_mean=float(np.mean(v)), n=len(v))
                  for k, v in sorted(cell.items())]

    # ---- model-free T*
    def tstar_table(eps):
        tab = {}
        for s in seeds:
            for t in targets:
                for a in alphas:
                    rr = sorted([r for r in rows if r['seed'] == s and r['task_id'] == t and r['alpha'] == a],
                                key=lambda r: r['T'])
                    if len(rr) == len(Ts):
                        tab[(s, t, a)] = mf_tstar([r['T'] for r in rr], [r['excess_bits'] for r in rr], eps)
        return tab
    tst = tstar_table(eps_star)
    A['tstar_mf'] = [dict(seed=k[0], task_id=k[1], alpha=k[2], T_star=v[0], flag=v[1]) for k, v in tst.items()]
    A['tstar_censored'] = {f: sum(v[1] == f for v in tst.values()) for f in ('ok', 'left', 'right')}

    # ---- G1: M_K at alpha = 1, L4 over L1
    L1, L4 = dict(TK.MAIN_LEVELS)['L1'], dict(TK.MAIN_LEVELS)['L4']
    full = [s for s in seeds if all((s, t, 1.0) in tst for t in L1 + L4)]
    mk = np.array([gm([tst[(s, t, 1.0)][0] for t in L4]) / gm([tst[(s, t, 1.0)][0] for t in L1]) for s in full])
    G1 = dict(M_K=gm(mk) if len(mk) else None, per_seed=mk.tolist())
    if len(mk) >= 2:
        bb = boot(len(mk))
        G1['ci95_seed'] = ci(np.exp(np.log(mk)[bb].mean(1)))
        rng = np.random.default_rng(B_SEED + 1)
        two = []
        for _ in range(NB // 4):
            ss = rng.choice(full, len(full))
            v = []
            for s in ss:
                t4, t1 = rng.choice(L4, 2), rng.choice(L1, 2)
                v.append(math.log(gm([tst[(s, t, 1.0)][0] for t in t4]) / gm([tst[(s, t, 1.0)][0] for t in t1])))
            two.append(math.exp(np.mean(v)))
        G1['ci95_two_stage'] = ci(two)
        G1['verdict'] = 'SUPPORTED' if (G1['M_K'] >= CFG['g1_min_MK'] and G1['ci95_seed'][0] > 1
                                        and G1['ci95_two_stage'][0] > 1) else 'NOT_ESTABLISHED'
    G1['censored_cells_alpha1'] = sum(tst[(s, t, 1.0)][1] != 'ok' for s in full for t in L1 + L4)
    A['G1'] = G1

    # ---- laws per seed on fit rows (learning-regime filter), held-out scoring
    laws_ = ['M0_scale', 'M_align_eff', 'M_K_coef', 'M_K_exp', 'M_K_both']
    fits, ho = {}, {}
    n_drop = 0
    for s in seeds:
        rs = [r for r in rows if r['seed'] == s]
        fr = fit_rows(rs, Tmax)
        frr = regime(fr)
        n_drop += len(fr) - len(frr)
        hr = [r for r in rs if r not in fr]
        Xf = X_of(frr, 'K_bits', Kgrid, Kref, T0)
        Xh = X_of(hr, 'K_bits', Kgrid, Kref, T0)
        Xho = X_of([r for r in hr if r['level'] == CFG['level_heldout']], 'K_bits', Kgrid, Kref, T0)
        for law in laws_:
            f = L.fit(law, Xf, CFG)
            r_, bad = L.scored_residual(law, np.array(f['theta']), Xh)
            r2, _ = L.scored_residual(law, np.array(f['theta']), Xho)
            rf, _ = L.scored_residual(law, np.array(f['theta']), Xf)
            fits[(law, s)] = f
            ho[(law, s)] = dict(sq=(r_ ** 2), sq_level=(r2 ** 2), n_invalid=int(bad.sum()), fit_sq=(rf ** 2))
    A['fit_rows_dropped_learning_regime'] = n_drop
    A['laws'] = {law: dict(params_per_seed=[fits[(law, s)]['params'] for s in seeds],
                           heldout_rmse=float(np.sqrt(np.mean([ho[(law, s)]['sq'].mean() for s in seeds]))),
                           fit_rmse=float(np.sqrt(np.mean([ho[(law, s)]['fit_sq'].mean() for s in seeds]))),
                           n_invalid=sum(ho[(law, s)]['n_invalid'] for s in seeds)) for law in laws_}

    # ---- G2
    klaw = min(['M_K_coef', 'M_K_exp'], key=lambda l: A['laws'][l]['fit_rmse'])
    msq = np.array([ho[(klaw, s)]['sq'].mean() for s in seeds])
    bsq = np.array([ho[('M_align_eff', s)]['sq'].mean() for s in seeds])
    red = 1 - math.sqrt(msq.mean() / bsq.mean())
    dred = 1 - np.sqrt(msq[Bi].mean(1) / bsq[Bi].mean(1))
    lvl_k = np.mean([ho[(klaw, s)]['sq_level'].mean() for s in seeds])
    lvl_b = np.mean([ho[('M_align_eff', s)]['sq_level'].mean() for s in seeds])
    G2 = dict(K_law=klaw, reduction=red, ci95=ci(dred), level_rmse_K=math.sqrt(lvl_k), level_rmse_align=math.sqrt(lvl_b),
              n_invalid=A['laws'][klaw]['n_invalid'])
    G2['verdict'] = 'SUPPORTED' if (red >= CFG['rmse_reduction_threshold'] and G2['ci95'][0] > 0 and lvl_k < lvl_b
                                    and G2['n_invalid'] == 0) else 'NOT_ESTABLISHED'
    A['G2'] = G2

    # ---- G3 (reported)
    rho = math.sqrt(np.mean([ho[('M_K_exp', s)]['sq'].mean() for s in seeds]) /
                    np.mean([ho[('M_K_coef', s)]['sq'].mean() for s in seeds]))
    g1s = [fits[('M_K_both', s)]['params']['gamma1'] for s in seeds]
    kap = [fits[('M_K_both', s)]['params']['kappa'] for s in seeds]
    A['G3_reported'] = dict(rho_exp_over_coef=rho, gamma1_per_seed=g1s, kappa_per_seed=kap)

    # ---- G4: E from M_K_both refit on all rows (regime filter), both K units; E_mf
    allfit = {}
    for unit in ('K_bits', 'K_classes'):
        for s in seeds:
            rs = regime([r for r in rows if r['seed'] == s])
            kg = [2.0 ** k for k in Kgrid] if unit == 'K_classes' else Kgrid
            kr = 2.0 ** Kref if unit == 'K_classes' else Kref
            X = X_of(rs, unit, kg, kr, T0)
            allfit[(unit, s)] = (L.fit('M_K_both', X, CFG), X)

    def E_at(eps, unit):
        ea, ek = [], []
        for s in seeds:
            f, X = allfit[(unit, s)]
            a, k = L.elasticities('M_K_both', f['params'], X['K_ref'], eps, X)
            ea.append(a)
            ek.append(k)
        ea, ek = np.array(ea), np.array(ek)
        pt = ea.mean() / ek.mean() if ek.mean() > 0 else math.inf
        ma, mk_ = ea[Bi].mean(1), ek[Bi].mean(1)
        d = np.where(mk_ > 0, ma / np.where(mk_ > 0, mk_, 1), np.inf)
        return float(pt), d, ea, ek
    Eb, dEb, ea, ek = E_at(eps_star, 'K_bits')
    Ec, dEc, _, _ = E_at(eps_star, 'K_classes')

    def E_mf(eps):
        tab = tst if eps == eps_star else tstar_table(eps)
        coefs, ncens = [], []
        for s in seeds:
            pts = [(math.log2(1 / a), math.log2(TK.k_bits(t)), math.log2(tab[(s, t, a)][0]))
                   for t in targets for a in alphas if (s, t, a) in tab and tab[(s, t, a)][1] == 'ok']
            ncens.append(len(targets) * len(alphas) - len(pts))
            if len(pts) >= 4:
                Z = np.array(pts)
                b = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z[:, 0], Z[:, 1]]), Z[:, 2], rcond=None)[0]
                coefs.append(b[1:])
        if not coefs:
            return None, ncens, None
        cm = np.mean(coefs, 0)
        return (float(cm[0] / cm[1]) if cm[1] > 0 else math.inf), ncens, cm.tolist()
    med = float(np.median(E_mf(eps_star)[1]))
    eps_mf = eps_star if med <= 0.5 * len(targets) * len(alphas) else 2 * eps_star
    Emf, ncm, cmf = E_mf(eps_mf)
    G4 = dict(E_bits=Eb, ci95_bits=ci(dEb), E_classes=Ec, ci95_classes=ci(dEc), e_alpha=ea.tolist(),
              e_K_bits=ek.tolist(), E_mf=Emf, E_mf_eps=eps_mf, E_mf_coefs=cmf, E_mf_censored_per_seed=ncm)
    G4['verdict'] = 'SUPPORTED' if (Eb > 1 and G4['ci95_bits'][0] > 1 and Ec > 1 and Emf is not None and Emf > 1) \
        else 'NOT_ESTABLISHED'
    if Eb < 1 and G4['ci95_bits'][1] < 1 and Emf is not None and Emf < 1:
        G4['replication'] = 'SAME_DIRECTION'
    elif G4['verdict'] == 'SUPPORTED':
        G4['replication'] = 'OPPOSITE'
    else:
        G4['replication'] = 'UNRESOLVED'
    G4['M_alpha_law'] = gm([20 ** e for e in ea])
    # E at other targets (law), descriptive
    G4['E_curve'] = {str(e): E_at(e, 'K_bits')[0] for e in (0.05, 0.1, 0.2, 0.4)}
    A['G4'] = G4

    # ---- independent fitter on M_K_both (fit rows): held-out RMSE agreement
    agree = []
    for s in seeds:
        rs = [r for r in rows if r['seed'] == s]
        fr = regime(fit_rows(rs, Tmax))
        hr = [r for r in rs if r not in fit_rows(rs, Tmax)]
        Xf, Xh = X_of(fr, 'K_bits', Kgrid, Kref, T0), X_of(hr, 'K_bits', Kgrid, Kref, T0)
        f2 = IF.fit('M_K_both', Xf, CFG)
        a1 = IF.heldout_rmse('M_K_both', np.array(fits[('M_K_both', s)]['theta']), Xh)[0]
        a2 = IF.heldout_rmse('M_K_both', np.array(f2['theta']), Xh)[0]
        agree.append([a1, a2])
    A['independent_fit'] = dict(pairs=agree, agree=all(IF.agree(a, b, CFG['independent_fit_tol']) for a, b in agree))
    return A


def analyse_mis(res):
    m = CFG['mismatch_sweep']
    ds = m['dataset']
    main = [r for r in load_runs(res, 'main', ds) if r['spec']['alpha'] == 1.0 and r['spec']['seed'] in m['seeds']]
    mis = load_runs(res, 'mis', ds)
    out = {}
    eps_star = CFG['eps_star'][ds]
    Kgrid = [TK.k_bits(ts[0]) for _, ts in TK.MAIN_LEVELS]
    targets = [t for _, ts in TK.MAIN_LEVELS for t in ts]
    types = [CFG['primary_mismatch']] + m['types']
    for ty in types:
        if ty == CFG['primary_mismatch']:
            runs = [r for r in load_runs(res, 'main', ds) if r['spec']['seed'] in m['seeds']
                    and r['spec']['alpha'] in m['alphas'] + [1.0]]
        else:
            runs = [r for r in mis if r['spec']['mismatch'] == ty] + main
        rows = main_rows(runs)
        if not rows:
            continue
        T0 = min(r['T'] for r in rows)
        Kref = gm([TK.k_bits(ts[0]) for lv, ts in TK.MAIN_LEVELS if lv != CFG['level_heldout']])
        ea, ek, per = [], [], {}
        for s in m['seeds']:
            rs = regime([r for r in rows if r['seed'] == s])
            if len(rs) < 10:
                continue
            X = X_of(rs, 'K_bits', Kgrid, Kref, T0)
            f = L.fit('M_K_both', X, CFG)
            a, k = L.elasticities('M_K_both', f['params'], Kref, eps_star, X)
            ea.append(a)
            ek.append(k)
        Ts = sorted({r['T'] for r in rows})
        malpha = {}
        for lv, ts in TK.MAIN_LEVELS:
            v = []
            for s in m['seeds']:
                for t in ts:
                    tt = {}
                    for a in (0.05, 1.0):
                        rr = sorted([r for r in rows if r['seed'] == s and r['task_id'] == t and r['alpha'] == a],
                                    key=lambda r: r['T'])
                        if len(rr) == len(Ts):
                            tt[a] = mf_tstar(Ts, [r['excess_bits'] for r in rr], eps_star)
                    if len(tt) == 2:
                        v.append((tt[0.05][0] / tt[1.0][0], tt[0.05][1], tt[1.0][1]))
            if v:
                malpha[lv] = dict(M_alpha_mf=gm([x[0] for x in v]), censored=sum(x[1] != 'ok' or x[2] != 'ok' for x in v))
        # mean eps at the largest checkpoint, alpha = 0.05, per level
        last = {}
        for lv, ts in TK.MAIN_LEVELS:
            vv = [r['excess_bits'] for r in rows if r['level'] == lv and r['alpha'] == 0.05 and r['T'] == max(Ts)]
            if vv:
                last[lv] = float(np.mean(vv))
        out[ty] = dict(n_runs=len(runs), eta_per_seed=ea, e_K_per_seed=ek,
                       E=(float(np.mean(ea) / np.mean(ek)) if ek and np.mean(ek) > 0 else None),
                       M_alpha_mf_by_level=malpha, eps_at_Tmax_alpha005=last)
    return out


def analyse_H(res, ds):
    runs = load_runs(res, 'H', ds)
    if not runs:
        return None
    er = CFG['eps_res']
    easy, hard = TK.H_TARGETS['easy'], TK.H_TARGETS['hard']
    seeds = sorted({r['spec']['seed'] for r in runs})
    joint, probe, anchor = {}, {}, {}
    for r in runs:
        s = r['spec']
        rows = [x for x in r['rows'] if x['T'] > 0]
        if s['mode'] == 'joint':
            t = s['target']
            joint[(s['pool'], t, s['seed'])] = [x[f'eps__{t}'] for x in rows]
        elif s['mode'] == 'pool_only':
            for t in easy + hard:
                probe[(s['pool'], t, s['seed'])] = [x[f'probe_eps__{t}'] for x in rows]
            probe[('random_init', 'all', s['seed'])] = {t: r['rows'][0][f'probe_eps__{t}'] for t in easy + hard}
        else:
            t = s['target']
            anchor[(t, s['seed'])] = dict(T=[x['T'] for x in rows], eps=[x[f'eps__{t}'] for x in rows])

    def S(v):
        return float(np.mean(np.log2(np.maximum(v, er))))

    def last(v):
        return float(math.log2(max(v[-1], er)))

    out = dict(dataset=ds, n_runs=len(runs), n_runs_expected=19 * len(CFG['seeds']), seeds=seeds)
    for mode, tab in (('joint', joint), ('probe', probe)):
        M = {}
        for score_name, fn in (('S', S), ('last', last)):
            sc = {}
            for pool in TK.POOLS:
                for t in easy + hard:
                    for s in seeds:
                        if (pool, t, s) in tab:
                            sc[(pool, t, s)] = fn(tab[(pool, t, s)])
            ok = [s for s in seeds if all((p, t, s) in sc for p in TK.POOLS for t in easy + hard)]

            def ratio(num, den, tg):
                per = np.array([np.mean([sc[(num, t, s)] - sc[(den, t, s)] for t in tg]) for s in ok])
                if not len(per):
                    return dict(ratio=None)
                bb = boot(len(per))
                return dict(ratio=float(2 ** per.mean()), ci95=ci(2 ** per[bb].mean(1)), per_seed=(2 ** per).tolist())
            M[score_name] = dict(
                seeds_complete=ok,
                rho_HE=ratio('hard', 'easy', easy), rho_EH=ratio('easy', 'hard', hard),
                mixed_vs_easy_on_easy=ratio('mixed', 'easy', easy), mixed_vs_hard_on_hard=ratio('mixed', 'hard', hard),
                eps_mean={f'{p}|{t}': float(np.mean([2 ** sc[(p, t, s)] for s in ok])) for p in TK.POOLS
                          for t in easy + hard} if ok else {})
        mg = margin = CFG['H']['margin']
        he, eh = M['S']['rho_HE'], M['S']['rho_EH']
        if he.get('ratio') is not None:
            M['GH1'] = 'CARRIES' if he['ci95'][1] < margin else ('DOES_NOT_CARRY' if he['ci95'][0] > margin else 'UNRESOLVED')
            M['GH2'] = 'SUPPORTED' if (eh['ratio'] > mg and eh['ci95'][0] > mg) else 'NOT_ESTABLISHED'
        out[mode] = M
    out['probe_random_init'] = {t: float(np.mean([probe[('random_init', 'all', s)][t] for s in seeds
                                                   if ('random_init', 'all', s) in probe])) for t in easy + hard} \
        if any(k[0] == 'random_init' for k in probe) else None
    # anchor at matched target examples: joint alpha=0.05 at T_max vs anchor at 0.05 T_max (log-log interpolation)
    comp = {}
    for t in easy + hard:
        v = []
        for s in seeds:
            if (t, s) not in anchor:
                continue
            a = anchor[(t, s)]
            Tm = a['T'][-1] * CFG['H']['alpha_joint']
            le = np.interp(math.log(Tm), np.log(a['T']), np.log(np.maximum(a['eps'], er)))
            row = dict(seed=s, anchor_eps_at_matched_target_examples=float(math.exp(le)),
                       anchor_eps_at_Tmax=float(a['eps'][-1]))
            for p in TK.POOLS:
                if (p, t, s) in joint:
                    row[f'joint_{p}_eps_at_Tmax'] = float(joint[(p, t, s)][-1])
            v.append(row)
        comp[t] = v
    out['anchor_comparison'] = comp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out')
    ap.add_argument('--datasets', nargs='*', default=CFG['datasets'])
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, 'analysis.json')
    A = dict(main={}, H={}, mismatch_sweep=None)
    for ds in a.datasets:
        runs = load_runs(a.results, 'main', ds)
        if runs:
            A['main'][ds] = analyse_main(runs, ds)
            print(ds, 'main', len(runs), 'runs', A['main'][ds]['G4']['replication'], round(A['main'][ds]['G4']['E_bits'], 3),
                  flush=True)
        h = analyse_H(a.results, ds)
        if h:
            A['H'][ds] = h
            print(ds, 'H', h['n_runs'], 'runs', h.get('joint', {}).get('GH1'), flush=True)
    if glob.glob(os.path.join(a.results, 'mis', '*.json')):
        A['mismatch_sweep'] = analyse_mis(a.results)
    A['config'] = CFG
    with open(out + '.tmp', 'w') as f:
        json.dump(A, f, indent=1, default=float)
    os.replace(out + '.tmp', out)
    print('wrote', out)


if __name__ == '__main__':
    main()
