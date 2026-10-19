"""Statistics and law fits for the gates (PROTOCOL Sections 2.6, 4 and 5).

The law forms and their fitter are Experiment 11's and 12's own laws.py (imported, hash-checked through common.mods);
this file adds the isotonic calibrator, the Gate 2 fixed-effects slope, Spearman with average ranks, the
leave-one-mix-out of Gate 4a and the bootstraps. independent_fit.py reimplements every piece in plain numpy.
"""
import csv
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

import common as C

LAW = 'M_K_both'
UNIT = 'K_bits'


# ---------------------------------------------------------------- statistics

def spearman(x, y):
    """Spearman rho with average ranks for ties (Pearson correlation of the ranks)."""
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(np.dot(rx, ry) / np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))


def spearman_rows(X, Y):
    """Row-wise Spearman of two (draws, n) arrays."""
    rx, ry = rankdata(X, axis=1), rankdata(Y, axis=1)
    rx = rx - rx.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rx * ry).sum(1) / np.sqrt((rx * rx).sum(1) * (ry * ry).sum(1))


def fe_slope(y, x, groups, w=None):
    """OLS slope of y on x with one intercept per group (least squares with group dummies)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    g = np.unique(groups)
    Z = np.column_stack([x] + [(np.asarray(groups) == k).astype(float) for k in g])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(Z * sw[:, None], y * sw, rcond=None)[0]
    return float(beta[0])


class Isotonic:
    """Increasing isotonic regression of true alpha on alpha-hat; linear interpolation, clipped at the ends."""

    def __init__(self, x, y, rows=None, heldout_rows=None):
        if rows is not None and heldout_rows is not None:
            assert not set(rows) & set(heldout_rows), 'calibrator would read a held-out row'
        self.m = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(np.asarray(x, float),
                                                                               np.asarray(y, float))

    def __call__(self, x):
        return self.m.predict(np.asarray(x, float))


def percentile_ci(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [float(lo), float(hi)]


class Boot:
    """Seed-block bootstrap weights, as Experiment 11's analyze.Boot: W[b, s] = multiplicity of seed s."""

    def __init__(self, n_seeds, draws, seed):
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=draws).astype(float)


# ---------------------------------------------------------------- upstream rows and fits

def upstream(c, fam):
    m = C.mods(c)[fam]
    return m['laws'], m['IF'], m['cfg']


def is_fit(r, cu, fam):
    Ks = cu['fit_n'] if fam == 'dfa' else cu['fit_K']
    return int(r['K']) in Ks and float(r['alpha']) in cu['fit_alphas'] and int(r['T']) <= cu['fit_T_max']


def read_metrics(c, fam):
    """Experiment 11/12 metrics rows (DSL: main arm only)."""
    out = []
    with open(C.results_dir(c, fam) / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            if fam == 'dsl' and r['arm'] != 'main':
                continue
            out.append(dict(seed=int(r['seed']), target_id=r['target_id'], K=int(r['K']), K_bits=int(r['K_bits']),
                            alpha=float(r['alpha']), T=int(r['T']), excess_bits=float(r['excess_bits'])))
    return out


def with_alpha(rows, amap, key=('target_id', 'alpha')):
    """Rows with alpha replaced by amap[(target, true alpha)]."""
    return [dict(r, alpha=float(amap[tuple(r[k] for k in key)])) for r in rows]


def fit_and_score(L, cu, fit_rows, ho_rows, fitter):
    """Fit the law on fit_rows with `fitter`, score held-out rows: (sse, n, n_invalid, theta, cost)."""
    X = L.design(fit_rows, UNIT, cu)
    f = fitter(LAW, X, cu)
    Xh = L.design(ho_rows, UNIT, cu)
    pred = L.predict_eps(LAW, np.asarray(f['theta']), Xh)
    bad = ~np.isfinite(pred) | (pred <= 0)
    pred = np.where(bad, cu['eps_res'], pred)
    r = L.censored_residual(np.log2(pred), Xh)
    return dict(sse=float(np.sum(r ** 2)), n=int(len(r)), invalid=int(bad.sum()), theta=list(map(float, f['theta'])),
                cost=float(f['cost']), forecast=pred.tolist())


def gate3_fits(c, fam, amaps, independent=False):
    """Per seed: fit on Experiment 11/12 fit rows and score their held-out rows, for each alpha version."""
    L, IF, cu = upstream(c, fam)
    rows = read_metrics(c, fam)
    seeds = sorted({r['seed'] for r in rows})
    out = {}
    for name, amap in amaps.items():
        per = []
        for s in seeds:
            rs = with_alpha([r for r in rows if r['seed'] == s], amap)
            fr = [r for r, r0 in zip(rs, [r for r in rows if r['seed'] == s]) if is_fit(r0, cu, fam)]
            hr = [r for r, r0 in zip(rs, [r for r in rows if r['seed'] == s]) if not is_fit(r0, cu, fam)]
            per.append(fit_and_score(L, cu, fr, hr, IF.fit if independent else L.fit))
        out[name] = per
    return seeds, out


def rmse(per):
    return float(np.sqrt(sum(p['sse'] for p in per) / sum(p['n'] for p in per)))


def rel_change_draws(per_a, per_b, W):
    """RMSE_b / RMSE_a - 1 over seed-block draws."""
    sa, na = np.array([p['sse'] for p in per_a]), np.array([p['n'] for p in per_a], float)
    sb, nb = np.array([p['sse'] for p in per_b]), np.array([p['n'] for p in per_b], float)
    return np.sqrt((W @ sb) / (W @ nb)) / np.sqrt((W @ sa) / (W @ na)) - 1


# ---------------------------------------------------------------- Gate 4a: leave-one-mix-out

def lomo_split(rows, left_out, T_max):
    """Fit rows: the other mixes, T < T_max. Held-out: the left-out mix at every T, every other mix at T_max."""
    fit = [r for r in rows if r['mix'] != left_out and r['T'] < T_max]
    ho = [r for r in rows if r['mix'] == left_out or r['T'] == T_max]
    assert not any(r['mix'] == left_out for r in fit), 'fold trains on its left-out mix'
    return fit, ho


def gate4a_fits(c, rows, alpha_key, independent=False):
    """Per seed: pooled held-out SSE over the 5 folds, alpha = rows[alpha_key]."""
    L, IF, cu = upstream(c, 'dfa')
    seeds = sorted({r['seed'] for r in rows})
    T_max = max(r['T'] for r in rows)
    mixes = sorted({r['mix'] for r in rows})
    per = []
    for s in seeds:
        rs = [dict(r, alpha=float(r[alpha_key])) for r in rows if r['seed'] == s]
        acc = dict(sse=0.0, n=0, invalid=0, folds=[])
        for m in mixes:
            fr, hr = lomo_split(rs, m, T_max)
            f = fit_and_score(L, cu, fr, hr, IF.fit if independent else L.fit)
            acc['sse'] += f['sse']
            acc['n'] += f['n']
            acc['invalid'] += f['invalid']
            acc['folds'].append(dict(mix=m, theta=f['theta'], cost=f['cost'], sse=f['sse'], n=f['n']))
        per.append(acc)
    return seeds, per


def read_graded_metrics(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append(dict(seed=int(r['seed']), target_id=r['target_id'], mix=r['mix'], K=int(r['K']),
                            K_bits=int(r['K_bits']), alpha_share=float(r['alpha_share']),
                            alpha_soft=float(r['alpha_soft']), alpha_agree=float(r['alpha_agree']), T=int(r['T']),
                            excess_bits=float(r['excess_bits'])))
    return out


def load_json(p):
    return json.loads(open(p).read())
