"""Experiment 16 analysis (PROTOCOL Sections 4-5, with Section 5's alignment to Experiment 11's frozen G1-G4).

Stage 1 fits every law per seed on fit rows only and writes forecasts.csv (and its SHA-256) for every held-out row
before any held-out outcome is read. Stage 2 scores the forecasts, computes data multipliers, and decides the gates.

    python analyze.py --arm mlp --output results/mlp
    python analyze.py --arm transformer --output results/transformer
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import independent_fit as IF
import laws as L

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / 'experiments').is_dir())
EXP11 = REPO / 'experiments/11_dfa_exact_complexity_vs_alignment/expt_v1/results/analysis.json'
EXP12 = REPO / 'experiments/12_dsl_shortest_program_complexity_vs_alignment/expt_v1/results/analysis.json'


def load_config():
    return json.loads((ROOT / 'config.json').read_text())


def gm(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def interval(point, draws, lo=2.5, hi=97.5):
    d = np.asarray(draws, float)
    return dict(estimate=float(point), lo=float(np.percentile(d, lo)), hi=float(np.percentile(d, hi)))


# --- arm specification --------------------------------------------------------------------------------------

def spec(arm, c):
    """Fit/held-out coordinates, levels and the largest model for each arm (PROTOCOL 2.1 and 2.6)."""
    if arm == 'mlp':
        m = c['mlp']
        bs = m['batch_size']
        fit_T = [u * bs for u in m['checkpoints_updates'] if u != m['heldout_updates']]
        return dict(
            arm=arm, seeds=m['seeds'], eps_star=m['epsilon_star'], levels=[1, 2, 3], fit_s=[1, 3], held_s=2,
            s_lo=1, s_hi=3, fit_alpha=[a for a in m['alpha_grid'] if a != m['heldout_alpha']], held_alpha=m['heldout_alpha'],
            fit_T=fit_T, held_T=m['heldout_updates'] * bs, T_all=[u * bs for u in m['checkpoints_updates']],
            alpha_top=max(m['alpha_grid']), alpha_bottom=min(m['alpha_grid']), size_key='width',
            fit_sizes=[w for w in m['widths'] if w != m['heldout_width']], held_size=m['heldout_width'],
            largest=max(m['widths']), every_seed=False, formats=['mlp'], primary_format='mlp',
            gap_T=(100 * bs, 1000 * bs))
    t = c['transformer']
    fmt = t['primary_format_final'] or t['primary_format']
    s_levels = [lv + 1 for lv in t['train_levels']]
    T_all = [2 ** k for k in t['T_checkpoints_log2']]
    return dict(
        arm=arm, seeds=t['seeds'], eps_star=t['epsilon_star'], levels=s_levels,
        fit_s=[s for s in s_levels if s != t['heldout_c'] + 1], held_s=t['heldout_c'] + 1, s_lo=min(s_levels),
        s_hi=max(s_levels), fit_alpha=[a for a in t['alpha_grid'] if a != t['heldout_alpha']],
        held_alpha=t['heldout_alpha'], fit_T=[T for T in T_all if T != 2 ** t['heldout_T_log2']],
        held_T=2 ** t['heldout_T_log2'], T_all=T_all, alpha_top=max(t['alpha_grid']), alpha_bottom=min(t['alpha_grid']),
        size_key='size', fit_sizes=t['size_order'], held_size=None, largest=t['size_order'][-1], every_seed=True,
        formats=[t['primary_format'], t['control_format']], primary_format=fmt, gap_T=(2 ** 16, 2 ** 22))


def read_rows(out):
    with (Path(out) / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('seed', 'K', 'alpha', 'T', 'step', 'excess_bits', 'accuracy', 'n_test', 'N', 'c', 's'):
            r[k] = float(r[k])
        r['seed'] = int(r['seed'])
    return rows


def target_rows(rows, sp, fmt=None):
    fmt = fmt or sp['primary_format']
    return [r for r in rows if r['eval_stratum'] == r['target_id'] and r['format'] == fmt]


def size_of(r, sp):
    return r[sp['size_key']] if sp['arm'] == 'transformer' else int(float(r['width']))


def is_fit(r, sp):
    return (r['s'] in sp['fit_s'] and r['alpha'] in sp['fit_alpha'] and r['T'] in sp['fit_T']
            and (sp['held_size'] is None or size_of(r, sp) != sp['held_size']))


def X_for(rows, sp, c, s_grid=None):
    s_grid = s_grid or sp['fit_s']
    s_ref = gm(s_grid)
    N0 = min(r['N'] for r in rows)
    return L.design(rows, s_grid, s_ref, min(sp['T_all']), N0, c['shared']['eps_res'])


# --- stage 1: frozen forecasts -------------------------------------------------------------------------------

def stage1(rows, sp, c, out):
    """Fit on fit rows, forecast every held-out row, write forecasts.csv and its hash before reading outcomes."""
    tr = target_rows(rows, sp)
    fits, forecasts = {}, []
    N0 = min(r['N'] for r in tr)
    for seed in sp['seeds']:
        fit_rows = [r for r in tr if r['seed'] == seed and is_fit(r, sp)]
        held = [{k: v for k, v in r.items() if k not in ('excess_bits', 'accuracy')} | {'excess_bits': 1.0}
                for r in tr if r['seed'] == seed and not is_fit(r, sp)]
        Xf = L.design(fit_rows, sp['fit_s'], gm(sp['fit_s']), min(sp['T_all']), N0, c['shared']['eps_res'])
        Xh = L.design(held, sp['fit_s'], gm(sp['fit_s']), min(sp['T_all']), N0, c['shared']['eps_res'])
        fits[seed] = {}
        for law in L.LAWS:
            f = L.fit(law, Xf, c)
            fit_rmse = float(np.sqrt(np.mean(L.scored_residual(law, f['theta'], Xf)[0] ** 2)))
            fits[seed][law] = dict(f, fit_rmse=fit_rmse, n_fit=len(fit_rows))
            pred = L.predict_eps(law, f['theta'], Xh)
            for r, p in zip(held, pred):
                forecasts.append(dict(seed=seed, law=law, s=r['s'], alpha=r['alpha'], T=r['T'], N=r['N'],
                                      size=size_of(r, sp), forecast=float(p)))
    # The G2 K law is chosen on pooled fit-row RMSE, before any held-out outcome is read (Experiment 11 G2).
    pooled = {law: float(np.sqrt(np.mean([fits[s][law]['fit_rmse'] ** 2 for s in sp['seeds']])))
              for law in ('M_K_coef', 'M_K_exp')}
    selection = dict(g2_k_law=min(pooled, key=pooled.get), pooled_fit_rmse=pooled)
    with (out / 'forecasts.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(forecasts[0]))
        w.writeheader()
        w.writerows(forecasts)
    digest = hashlib.sha256((out / 'forecasts.csv').read_bytes()).hexdigest()
    (out / 'forecasts.sha256').write_text(digest + '\n')
    (out / 'fits.json').write_text(json.dumps(dict(fits={str(k): v for k, v in fits.items()}, selection=selection,
                                                   forecasts_sha256=digest), indent=1) + '\n')
    return fits, selection


# --- stage 2 helpers -----------------------------------------------------------------------------------------

def mf_tstar(Ts, vals, eps_star):
    """Model-free T*: log-log interpolation of one run's curve; censored at the first or last checkpoint."""
    Ts, vals = np.asarray(Ts, float), np.maximum(np.asarray(vals, float), 1e-300)
    if vals[0] <= eps_star:
        return Ts[0], 'left'
    for i in range(1, len(Ts)):
        if vals[i] <= eps_star:
            x0, x1, y0, y1 = map(math.log2, (Ts[i - 1], Ts[i], vals[i - 1], vals[i]))
            return 2.0 ** (x0 + (math.log2(eps_star) - y0) * (x1 - x0) / (y1 - y0)), None
    return Ts[-1], 'right'


def curve(tr, sp, seed, s, alpha, size):
    pts = sorted((r['T'], r['excess_bits']) for r in tr if r['seed'] == seed and r['s'] == s and r['alpha'] == alpha
                 and size_of(r, sp) == size)
    return [p[0] for p in pts], [p[1] for p in pts]


def boot(c, n_seeds):
    sh = c['shared']
    return np.random.default_rng(sh['analysis_seed']).integers(0, n_seeds, (sh['bootstrap_draws'], n_seeds))


def heldout_scores(rows, sp, c, fits, forecasts):
    """Per seed and law: sum of squared censored residuals and counts on all held-out rows, the held-out level only,
    and invalid-forecast counts."""
    tr = target_rows(rows, sp)
    obs = {(r['seed'], r['s'], r['alpha'], r['T'], size_of(r, sp)): r['excess_bits'] for r in tr if not is_fit(r, sp)}
    eps_res, out = c['shared']['eps_res'], {}
    for f in forecasts:
        key = (f['seed'], f['s'], f['alpha'], f['T'], f['size'])
        y = obs[key]
        p = f['forecast']
        bad = (not np.isfinite(p)) or p <= 0
        p = eps_res if bad else p
        r = math.log2(p) - math.log2(max(y, 1e-300)) if y >= eps_res else max(0.0, math.log2(p) - math.log2(eps_res))
        d = out.setdefault((f['seed'], f['law']), dict(sse=0.0, n=0, sse_level=0.0, n_level=0, invalid=0))
        d['sse'] += r * r
        d['n'] += 1
        d['invalid'] += int(bad)
        if f['s'] == sp['held_s']:
            d['sse_level'] += r * r
            d['n_level'] += 1
    return out


def pooled_rmse(scores, law, seeds, idx=None, level=False):
    k = ('sse_level', 'n_level') if level else ('sse', 'n')
    idx = np.arange(len(seeds)) if idx is None else idx
    sse = np.array([scores[(s, law)][k[0]] for s in seeds])
    n = np.array([scores[(s, law)][k[1]] for s in seeds])
    if idx.ndim == 1:
        return float(np.sqrt(sse[idx].sum() / n[idx].sum()))
    return np.sqrt(sse[idx].sum(1) / n[idx].sum(1))


def reduction(scores, base, new, seeds, draws, level=False):
    point = 1 - pooled_rmse(scores, new, seeds, level=level) / pooled_rmse(scores, base, seeds, level=level)
    b = 1 - pooled_rmse(scores, new, seeds, draws, level) / pooled_rmse(scores, base, seeds, draws, level)
    per_seed = [1 - math.sqrt(scores[(s, new)]['sse' + ('_level' if level else '')] / scores[(s, base)]['sse' + ('_level' if level else '')])
                for s in seeds]
    return dict(interval(point, b), per_seed=per_seed)


def law_multiplier(p, sp, X, eps_star, alpha=None):
    """Law-based chain multiplier M_s = T*(s_hi)/T*(s_lo) at the top alpha (PROTOCOL 4.3)."""
    a = sp['alpha_top'] if alpha is None else alpha
    return L.t_star(p, sp['s_hi'], a, eps_star, X) / L.t_star(p, sp['s_lo'], a, eps_star, X)


def descriptive_fits(tr, sp, c, law='M_K_both', size=None):
    """Per-seed refit on all rows of the primary format (optionally one size): a description, not a forecast."""
    out = {}
    for seed in sp['seeds']:
        rr = [r for r in tr if r['seed'] == seed and (size is None or size_of(r, sp) == size)]
        X = X_for(rr, sp, c, s_grid=sp['levels'])
        out[seed] = (L.fit(law, X, c), X)
    return out


# --- stage 2 -------------------------------------------------------------------------------------------------

def stage2(rows, sp, c, out, fits, selection):
    sh, seeds = c['shared'], sp['seeds']
    draws = boot(c, len(seeds))
    tr = target_rows(rows, sp)
    with (out / 'forecasts.csv').open() as f:
        forecasts = [dict(r, seed=int(r['seed']), s=float(r['s']), alpha=float(r['alpha']), T=float(r['T']),
                          size=(r['size'] if sp['arm'] == 'transformer' else int(float(r['size']))), forecast=float(r['forecast']))
                     for r in csv.DictReader(f)]
    assert hashlib.sha256((out / 'forecasts.csv').read_bytes()).hexdigest() == (out / 'forecasts.sha256').read_text().strip()
    scores = heldout_scores(rows, sp, c, fits, forecasts)
    eps_star = sp['eps_star']
    res = dict(arm=sp['arm'], eps_star=eps_star, n_seeds=len(seeds), bootstrap_draws=sh['bootstrap_draws'],
               distinct_resamples=math.comb(2 * len(seeds) - 1, len(seeds)), selection=selection)
    res['heldout_rmse'] = {law: interval(pooled_rmse(scores, law, seeds), pooled_rmse(scores, law, seeds, draws))
                           for law in L.LAWS}
    res['heldout_invalid'] = {law: int(sum(scores[(s, law)]['invalid'] for s in seeds)) for law in L.LAWS}
    every = sp['every_seed']
    lv = 'level' if sp['arm'] == 'mlp' else 'seed'

    # G1: model-free M_s at the top alpha, largest model (Experiment 11 G1; one target per level, so seed-block only).
    t_hi, t_lo, cens = [], [], []
    for seed in seeds:
        for s, store in ((sp['s_hi'], t_hi), (sp['s_lo'], t_lo)):
            T, v, = curve(tr, sp, seed, s, sp['alpha_top'], sp['largest'])
            t, flag = mf_tstar(T, v, eps_star)
            store.append(t)
            cens.append(dict(seed=seed, s=s, censored=flag))
    m_seed = np.array(t_hi) / np.array(t_lo)
    g1 = interval(gm(m_seed), np.exp(np.log(m_seed)[draws].mean(1)))
    g1.update(per_seed=m_seed.tolist(), censoring=cens, alpha=sp['alpha_top'], model=sp['largest'])
    g1['verdict'] = ok(g1['estimate'] >= sh['g1_min_multiplier'] and g1['lo'] > 1 and (not every or all(m_seed > 1)))
    res['G1'] = g1

    # G2: the K law (chosen on fit RMSE in stage 1) against M_align_eff on held-out rows.
    kl = selection['g2_k_law']
    red = reduction(scores, 'M_align_eff', kl, seeds, draws)
    red_level = 1 - pooled_rmse(scores, kl, seeds, level=True) / pooled_rmse(scores, 'M_align_eff', seeds, level=True)
    inv = res['heldout_invalid'][kl]
    res['G2'] = dict(k_law=kl, reduction=red, reduction_heldout_level=red_level, invalid=inv,
                     verdict=ok(red['estimate'] >= sh['rmse_reduction_threshold'] and red['lo'] > 0 and red_level > 0
                                and inv == 0 and (not every or all(x > 0 for x in red['per_seed']))))

    # G3a / G3b: M_K_both parameters from the stage-1 fits (fit rows), plus coef-versus-exp held-out RMSE.
    kap = np.array([fits[s]['M_K_both']['params']['kappa'] for s in seeds])
    g1s = np.array([fits[s]['M_K_both']['params']['gamma1'] for s in seeds])
    kap_i, g1_i = interval(kap.mean(), kap[draws].mean(1)), interval(g1s.mean(), g1s[draws].mean(1))
    exp_vs_coef = reduction(scores, 'M_K_coef', 'M_K_exp', seeds, draws)
    coef_le_exp = pooled_rmse(scores, 'M_K_coef', seeds) <= pooled_rmse(scores, 'M_K_exp', seeds)
    g3a = kap_i['lo'] > 0 and g1_i['lo'] <= 0 <= g1_i['hi'] and coef_le_exp and (not every or all(kap > 0))
    g3b = (g1_i['hi'] < 0 and exp_vs_coef['estimate'] >= sh['g3b_min_reduction'] and exp_vs_coef['lo'] > 0
           and (not every or (all(g1s < 0) and all(x > 0 for x in exp_vs_coef['per_seed']))))
    res['G3'] = dict(kappa=kap_i, gamma1=g1_i, exp_vs_coef=exp_vs_coef, coef_rmse_le_exp=bool(coef_le_exp),
                     G3a=ok(g3a), G3b=ok(g3b), picture='coefficient' if g3a else 'exponent' if g3b else 'unresolved')

    # G4: elasticity ratio E from the descriptive M_K_both refit, and the model-free E_mf.
    desc = descriptive_fits(tr, sp, c)
    s_center = gm(sp['levels'])
    e_a = np.array([L.elasticities(desc[s][0]['params'], s_center, eps_star, desc[s][1])[0] for s in seeds])
    e_s = np.array([L.elasticities(desc[s][0]['params'], s_center, eps_star, desc[s][1])[1] for s in seeds])
    ea_b, es_b = e_a[draws].mean(1), e_s[draws].mean(1)
    E_point = e_a.mean() / e_s.mean() if e_s.mean() > 0 else math.inf
    E_draws = np.where(es_b > 0, ea_b / np.where(es_b > 0, es_b, 1), math.inf)
    E = interval(E_point, E_draws)
    E.update(per_seed=(e_a / np.where(e_s > 0, e_s, np.nan)).tolist(), e_alpha=e_a.tolist(), e_s=e_s.tolist(), s_center=s_center)
    emf, emf_eps = model_free_E(tr, sp, eps_star)
    if emf['censored_share_median'] > sh['censored_share_for_emf_fallback']:
        emf, emf_eps = model_free_E(tr, sp, 2 * eps_star)
    res['G4'] = dict(E=E, E_mf=emf, E_mf_eps_star=emf_eps, units='one unit (s = c + 1); Experiment 11 also used states',
                     verdict=ok(E['estimate'] > 1 and E['lo'] > 1 and emf['estimate'] > 1
                                and (not every or all(np.nan_to_num(np.array(E['per_seed']), nan=0) > 1))))

    # Draft gates of this protocol, reported but not gated (Section 5 alignment note).
    res['draft'] = draft_versions(rows, tr, sp, c, scores, fits, desc, draws)

    # G5: same C3 verdict as exact complexity (Experiments 11 and 12).
    res['G5'] = g5(res['G3']['picture'], g1_i, sh)

    if sp['arm'] == 'transformer':
        res['G6'] = g6(tr, sp, c, draws)
        res['G7'] = g7(rows, sp, c, draws)
    res['independent_fit'] = independent_agreement(rows, sp, c, fits)
    res['sensitivity'] = sensitivity(rows, sp, c)
    res['mechanism'] = mechanism(rows, sp)
    res['multiplier_curves'] = multiplier_curves(desc, sp)
    return res


def ok(b):
    return 'SUPPORTED' if bool(b) else 'NOT_ESTABLISHED'


def model_free_E(tr, sp, eps_star):
    """Per seed, regress log2 T* on log2(1/alpha) and log2 s over uncensored (level, alpha) pairs at the largest model."""
    alphas = sorted({r['alpha'] for r in tr})
    ratios, shares = [], []
    for seed in sp['seeds']:
        ys, xa, xs, n_cens = [], [], [], 0
        for s in sp['levels']:
            for a in alphas:
                T, v = curve(tr, sp, seed, s, a, sp['largest'])
                if not T:
                    continue
                t, flag = mf_tstar(T, v, eps_star)
                if flag:
                    n_cens += 1
                    continue
                ys.append(math.log2(t)), xa.append(math.log2(1 / a)), xs.append(math.log2(s))
        shares.append(n_cens / (len(sp['levels']) * len(alphas)))
        if len(ys) >= 4 and np.ptp(xa) > 0 and np.ptp(xs) > 0:
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(ys)), xa, xs]), ys, rcond=None)[0]
            ratios.append(coef[1] / coef[2] if coef[2] > 0 else math.inf)
        else:
            ratios.append(float('nan'))
    finite = [x for x in ratios if np.isfinite(x)]
    est = float(np.mean(finite)) if finite else float('nan')
    return dict(estimate=est, per_seed=ratios, censored_share_median=float(np.median(shares))), eps_star


def draft_versions(rows, tr, sp, c, scores, fits, desc, draws):
    """The draft G1-G4 of this protocol: law-based M_s, Bonferroni G2 pair, ratio rho, range ratio R and gap Delta."""
    seeds, eps_star = sp['seeds'], sp['eps_star']
    ms = np.array([law_multiplier(desc[s][0]['params'], sp, desc[s][1], eps_star) for s in seeds])
    eta = np.array([desc[s][0]['params']['eta'] for s in seeds])
    m_alpha = (sp['alpha_top'] / sp['alpha_bottom']) ** eta
    R = np.log(m_alpha) / np.log(ms)
    g2 = {law: reduction(scores, 'M_align_eff', law, seeds, draws) for law in ('M_K_coef', 'M_K_exp')}
    for law, v in g2.items():
        b = 1 - pooled_rmse(scores, law, seeds, draws) / pooled_rmse(scores, 'M_align_eff', seeds, draws)
        v['bonferroni_lo'] = float(np.percentile(b, 100 * c['shared']['bonferroni_alpha'] / 2))
    rho_point = pooled_rmse(scores, 'M_K_exp', seeds) / pooled_rmse(scores, 'M_K_coef', seeds)
    rho = interval(rho_point, pooled_rmse(scores, 'M_K_exp', seeds, draws) / pooled_rmse(scores, 'M_K_coef', seeds, draws))
    t0, t1 = sp['gap_T']
    gaps = []
    for seed in seeds:
        gap = {}
        for T in (t0, t1):
            vals = {s: np.mean([r['excess_bits'] for r in tr if r['seed'] == seed and r['s'] == s
                                and r['alpha'] == sp['alpha_top'] and r['T'] == T]) for s in (sp['s_hi'], sp['s_lo'])}
            gap[T] = vals[sp['s_hi']] - vals[sp['s_lo']]
        gaps.append(gap[t1] / gap[t0] if gap[t0] != 0 else float('nan'))
    gaps = np.array(gaps)
    return dict(law_M_s=interval(gm(ms), np.exp(np.log(ms)[draws].mean(1))), M_alpha=interval(gm(m_alpha), np.exp(np.log(m_alpha)[draws].mean(1))),
                range_ratio_R=interval(np.mean(R), R[draws].mean(1)), g2_bonferroni_pair=g2, rho=rho,
                gap_ratio_Delta=interval(np.nanmean(gaps), np.nanmean(gaps[draws], 1)), gap_T=list(sp['gap_T']))


def g5(picture, g1_i, sh):
    """Compare this G3 verdict with Experiments 11 and 12 (their decided G3a/G3b), naming the reason otherwise."""
    other = {}
    for name, path in (('Experiment 11', EXP11), ('Experiment 12', EXP12)):
        if not path.exists():
            other[name] = 'not complete'
            continue
        a = json.loads(path.read_text())
        g3a = a.get('G3', {}).get('G3a', {})
        g3b = a.get('G3', {}).get('G3b', {})
        v3a = g3a.get('verdict') if isinstance(g3a, dict) else g3a
        v3b = g3b.get('verdict') if isinstance(g3b, dict) else g3b
        other[name] = 'coefficient' if v3a == 'SUPPORTED' else 'exponent' if v3b == 'SUPPORTED' else 'undecided'
    if picture == 'unresolved':
        reason = 'this experiment undecided'
    elif other['Experiment 11'] in ('undecided', 'not complete'):
        reason = f"Experiment 11 {other['Experiment 11']}"
    elif other['Experiment 11'] != picture or other['Experiment 12'] not in ('undecided', 'not complete', picture):
        reason = 'verdicts differ'
    else:
        reason = None
    return dict(this=picture, others=other, verdict=ok(reason is None), reason=reason)


def g6(tr, sp, c, draws):
    """Per-size M_K_both fits; the chain multiplier should shrink with model size."""
    ms = {}
    for size in sp['fit_sizes']:
        d = descriptive_fits(tr, sp, c, size=size)
        ms[size] = np.array([law_multiplier(d[s][0]['params'], sp, d[s][1], sp['eps_star']) for s in sp['seeds']])
    lo_size, hi_size = sp['fit_sizes'][0], sp['fit_sizes'][-1]
    ratio = ms[hi_size] / ms[lo_size]
    est = interval(gm(ratio), np.exp(np.log(ratio)[draws].mean(1)))
    pts = [gm(ms[s]) for s in sp['fit_sizes']]
    mono = all(pts[i] > pts[i + 1] for i in range(len(pts) - 1))
    return dict(ratio=est, M_s_by_size={k: v.tolist() for k, v in ms.items()}, monotone=mono,
                verdict=ok(est['estimate'] <= c['shared']['g6_ratio'] and est['hi'] < 1 and mono and all(ratio < 1)))


def g7(rows, sp, c, draws):
    """At the control size, log M_s(plain) / log M_s(reversed)."""
    t = c['transformer']
    logs = {}
    for fmt in ('plain', 'reversed'):
        tr = target_rows(rows, sp, fmt=fmt)
        d = descriptive_fits(tr, sp, c, size=t['control_size'])
        logs[fmt] = np.log([law_multiplier(d[s][0]['params'], sp, d[s][1], sp['eps_star']) for s in sp['seeds']])
    ratio = logs['plain'] / logs['reversed']
    est = interval(ratio.mean(), ratio[draws].mean(1))
    return dict(log_ratio=est, per_seed=ratio.tolist(),
                verdict=ok(est['estimate'] >= c['shared']['g7_log_ratio'] and est['lo'] > 1 and all(ratio > 1)))


def independent_agreement(rows, sp, c, fits):
    """Refit every law with independent_fit.py and compare held-out RMSE per seed to independent_fit_tol."""
    tr = target_rows(rows, sp)
    N0 = min(r['N'] for r in tr)
    worst, first_bad = 0.0, None
    for seed in sp['seeds']:
        fr = [r for r in tr if r['seed'] == seed and is_fit(r, sp)]
        hr = [r for r in tr if r['seed'] == seed and not is_fit(r, sp)]
        Xf = L.design(fr, sp['fit_s'], gm(sp['fit_s']), min(sp['T_all']), N0, c['shared']['eps_res'])
        Xh = L.design(hr, sp['fit_s'], gm(sp['fit_s']), min(sp['T_all']), N0, c['shared']['eps_res'])
        for law in L.LAWS:
            a = IF.heldout_rmse(law, fits[seed][law]['theta'], Xh)[0]
            b = IF.heldout_rmse(law, IF.fit(law, Xf, c)['theta'], Xh)[0]
            rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
            worst = max(worst, rel)
            if rel > c['shared']['independent_fit_tol'] and first_bad is None:
                first_bad = dict(seed=seed, law=law, laws_rmse=a, independent_rmse=b)
    return dict(status='AGREE' if first_bad is None else 'DISAGREE', max_relative_difference=worst, first=first_bad)


def sensitivity(rows, sp, c):
    """Reported, not gated: s = c on c >= 1 only, and (16b) the exact-match error in place of excess bits."""
    tr = target_rows(rows, sp)
    outr = {}
    sub = [dict(r, s=r['c']) for r in tr if r['c'] >= 1]
    levels = sorted({r['s'] for r in sub})
    if len(levels) >= 2:
        sp2 = dict(sp, levels=levels, s_lo=min(levels), s_hi=max(levels))
        d = descriptive_fits(sub, sp2, c)
        outr['s_equals_c'] = dict(M_s=gm([law_multiplier(d[s][0]['params'], sp2, d[s][1], sp['eps_star']) for s in sp['seeds']]),
                                  eta=float(np.mean([d[s][0]['params']['eta'] for s in sp['seeds']])))
    em = [dict(r, excess_bits=max(1 - r['accuracy'], c['shared']['eps_res'])) for r in tr]
    d = descriptive_fits(em, sp, c)
    outr['exact_match_error'] = dict(gamma1=float(np.mean([d[s][0]['params']['gamma1'] for s in sp['seeds']])),
                                     kappa=float(np.mean([d[s][0]['params']['kappa'] for s in sp['seeds']])))
    return outr


def mechanism(rows, sp):
    """Per-digit excess at the last checkpoint and off-target (probe) strata, averaged over seeds."""
    last = max(sp['T_all'])
    out = dict(per_digit={}, off_target={})
    for r in rows:
        if r['T'] != last or r['alpha'] != sp['alpha_top']:
            continue
        if r['eval_stratum'] == r['target_id']:
            key = f"{r['format']}|{r['target_id']}"
            acc = out['per_digit'].setdefault(key, [])
            acc.append([float(r[f'digit_bits_{j}']) for j in range(9) if r[f'digit_bits_{j}'] not in ('', None)])
        else:
            key = f"{r['format']}|target={r['target_id']}|eval={r['eval_stratum']}"
            out['off_target'].setdefault(key, []).append(r['excess_bits'])
    out['per_digit'] = {k: np.mean(v, 0).tolist() for k, v in out['per_digit'].items()}
    out['off_target'] = {k: float(np.mean(v)) for k, v in out['off_target'].items()}
    return out


def multiplier_curves(desc, sp):
    """Law-based M_s, M_alpha and E over eps in {0.03, 0.1, 0.3, 1} bits (PROTOCOL 4.4)."""
    out = {}
    for eps in (0.03, 0.1, 0.3, 1.0):
        ms = [law_multiplier(f['params'], sp, X, eps) for f, X in desc.values()]
        el = [L.elasticities(p, gm(sp['levels']), eps, X) for p, X in ((v[0]['params'], v[1]) for v in desc.values())]
        ea, es = np.mean([e[0] for e in el]), np.mean([e[1] for e in el])
        out[str(eps)] = dict(M_s=gm(ms), E=float(ea / es) if es > 0 else math.inf)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) and not np.isinf(o) else (str(o) if np.isinf(o) else float(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['mlp', 'transformer'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    c = load_config()
    sp = spec(args.arm, c)
    rows = read_rows(args.output)
    fits, selection = stage1(rows, sp, c, args.output)
    fits = {int(k): v for k, v in json.loads((args.output / 'fits.json').read_text())['fits'].items()}
    res = stage2(rows, sp, c, args.output, fits, selection)
    (args.output / 'gates.json').write_text(json.dumps(_jsonable(res), indent=1) + '\n')
    summary = {k: res[k]['verdict'] for k in ('G1', 'G2', 'G4', 'G5', 'G6', 'G7') if k in res}
    summary.update(G3a=res['G3']['G3a'], G3b=res['G3']['G3b'], independent_fit=res['independent_fit']['status'])
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
