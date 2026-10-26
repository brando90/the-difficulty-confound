"""Scaling laws of PROTOCOL Section 4.1, fit by multi-start nonlinear least squares on log2 excess loss.

Internally an exponent that depends on K, gamma(K) = gamma0 + gamma1 * log2 K, is parametrized by its values at the
two ends of the K grid (g_lo at K_min, g_hi at K_max), each bounded to [1e-3, 3]. Because gamma(K) is linear in
log2 K, this enforces the protocol's bound on the whole range [K_min, K_max] with box constraints only.
"""
import itertools

import numpy as np
from scipy.optimize import least_squares

LAWS = ['M0_scale', 'M_align_eff', 'M_align_add', 'M_K_coef', 'M_K_exp', 'M_K_both']
LOG_LAWS = ['M0_scale', 'M_align_eff', 'M_K_coef', 'M_K_exp', 'M_K_both']
K_RANGES = {'K_bits': (20.0, 352.0), 'K': (4.0, 32.0)}

# parameter layout: index 0 is log2 A for log-space laws
NAMES = {
    'M0_scale': ['logA', 'gamma'],
    'M_align_eff': ['logA', 'gamma', 'eta'],
    'M_K_coef': ['logA', 'gamma', 'eta', 'kappa'],
    'M_K_exp': ['logA', 'g_lo', 'g_hi', 'eta'],
    'M_K_both': ['logA', 'g_lo', 'g_hi', 'eta', 'kappa'],
    'M_align_add': ['a', 'c', 'd', 'gamma'],
}


def design(rows, k_unit, c):
    """Arrays for law fitting from metrics rows (dicts with K, K_bits, alpha, T, excess_bits)."""
    K = np.array([float(r[k_unit]) for r in rows])
    lo, hi = K_RANGES[k_unit]
    eps = np.array([float(r['excess_bits']) for r in rows])
    alpha = np.array([float(r['alpha']) for r in rows])
    T = np.array([float(r['T']) for r in rows])
    return dict(lK=np.log2(K), la=np.log2(alpha), lT=np.log2(T), alpha=alpha, T=T, eps=eps,
                y=np.log2(np.maximum(eps, 1e-300)), cens=eps < c['eps_res'], yres=np.log2(c['eps_res']),
                lKmin=np.log2(lo), lKmax=np.log2(hi), k_unit=k_unit)


def gamma_of_K(g_lo, g_hi, lK, X):
    return g_lo + (g_hi - g_lo) * (lK - X['lKmin']) / (X['lKmax'] - X['lKmin'])


def shape(law, th, X):
    """log2 prediction without the log2 A term (log-space laws)."""
    if law == 'M0_scale':
        return -th[1] * X['lT']
    if law == 'M_align_eff':
        return -th[1] * (th[2] * X['la'] + X['lT'])
    if law == 'M_K_coef':
        return th[3] * X['lK'] - th[1] * (th[2] * X['la'] + X['lT'])
    if law == 'M_K_exp':
        return -gamma_of_K(th[1], th[2], X['lK'], X) * (th[3] * X['la'] + X['lT'])
    if law == 'M_K_both':
        return th[4] * X['lK'] - gamma_of_K(th[1], th[2], X['lK'], X) * (th[3] * X['la'] + X['lT'])
    raise KeyError(law)


def base_law(base):
    return base[0] if isinstance(base, (tuple, list)) else base


def predict_eps(law, th, X, base=None):
    """Forecast of excess bits (linear scale). M_K_floor's theta is [a1, *K-law theta]; base names the K law."""
    if law == 'M_align_add':
        return th[0] + th[1] * X['T'] ** (-th[3]) - th[2] * X['alpha']
    if law == 'M_K_floor':
        bl = base_law(base)
        return 2.0 ** (th[1] + shape(bl, th[1:], X)) + th[0] * (1 - X['alpha'])
    return 2.0 ** (th[0] + shape(law, th, X))


def censored_residual(pred_log2, X):
    r = pred_log2 - X['y']
    rc = np.maximum(0.0, pred_log2 - X['yres'])
    return np.where(X['cens'], rc, r)


def scored_residual(law, th, X, base=None):
    """Censored residual on log2 scale, with invalid forecasts (nonfinite or <= 0) clipped to eps_res."""
    p = predict_eps(law, th, X, base)
    invalid = ~np.isfinite(p) | (p <= 0)
    p = np.where(invalid, 2.0 ** X['yres'], p)
    return censored_residual(np.log2(p), X), invalid


def bounds_for(law, c):
    f = c['fit']
    g, e, k = f['gamma_bounds'], f['eta_bounds'], f['kappa_bounds']
    inf = np.inf
    b = {
        'M0_scale': ([-inf, g[0]], [inf, g[1]]),
        'M_align_eff': ([-inf, g[0], e[0]], [inf, g[1], e[1]]),
        'M_K_coef': ([-inf, g[0], e[0], k[0]], [inf, g[1], e[1], k[1]]),
        'M_K_exp': ([-inf, g[0], g[0], e[0]], [inf, g[1], g[1], e[1]]),
        'M_K_both': ([-inf, g[0], g[0], e[0], k[0]], [inf, g[1], g[1], e[1], k[1]]),
        'M_align_add': ([-inf, -inf, -inf, g[0]], [inf, inf, inf, g[1]]),
    }[law]
    return np.array(b[0], float), np.array(b[1], float)


def exponent_starts(law, X, c):
    """Exponent parts of the 64-start grid (PROTOCOL Section 4.1), projected into the bounds."""
    f = c['fit']
    lo, hi = bounds_for(law, c)
    G, E = f['gamma_starts'], f['eta_starts']

    def ends(g0, g1):
        return g0 + g1 * X['lKmin'], g0 + g1 * X['lKmax']

    if law == 'M0_scale':
        S = [[g] for g in G]
    elif law == 'M_align_eff':
        S = [[g, e] for g, e in itertools.product(G, E)]
    elif law == 'M_K_coef':
        S = [[g, e, k] for g, e, k in itertools.product(G, E, f['kappa_starts'])]
    elif law == 'M_K_exp':
        S = [[*ends(g, g1), e] for g, e, g1 in itertools.product(G, E, f['gamma1_starts'])]
    elif law == 'M_K_both':
        S = [[*ends(g, kg[1]), e, kg[0]] for g, e, kg in itertools.product(G, E, f['both_starts'])]
    elif law == 'M_align_add':
        S = [[g] for g in G]
    else:
        raise KeyError(law)
    S = np.array(S, float)
    return np.clip(S, lo[1:] if law != 'M_align_add' else lo[3:], hi[1:] if law != 'M_align_add' else hi[3:])


def _ls_logA(law, ex, X):
    th = np.r_[0.0, ex]
    m = ~X['cens']
    return float(np.mean((X['y'] - shape(law, th, X))[m])) if m.any() else X['yres']


def _linear_add(gamma, X):
    Z = np.column_stack([np.ones_like(X['T']), X['T'] ** (-gamma), -X['alpha']])
    return np.linalg.lstsq(Z, X['eps'], rcond=None)[0]


TOL = dict(xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=5000)


def fit(law, X, c, base=None):
    """Best of the multi-start trust-region-reflective fits. Returns dict(theta, cost, params)."""
    best = None
    if law == 'M_K_floor':
        bl, bth = base
        lo0, hi0 = bounds_for(bl, c)
        lo, hi = np.r_[0.0, lo0], np.r_[np.inf, hi0]
        starts = [np.r_[a1, bth] for a1 in c['fit']['floor_a1_starts']]

        def res(z):
            p = 2.0 ** (z[1] + shape(bl, z[1:], X)) + z[0] * (1 - X['alpha'])
            return censored_residual(np.log2(np.maximum(p, 1e-300)), X)
    elif law == 'M_align_add':
        lo, hi = bounds_for(law, c)
        starts = [np.r_[_linear_add(s[0], X), s[0]] for s in exponent_starts(law, X, c)]

        def res(z):
            return predict_eps(law, z, X) - X['eps']
    else:
        lo, hi = bounds_for(law, c)
        starts = [np.r_[_ls_logA(law, s, X), s] for s in exponent_starts(law, X, c)]

        def res(z):
            return censored_residual(z[0] + shape(law, z, X), X)
    for s in starts:
        s = np.clip(s, lo, hi)
        r = least_squares(res, s, bounds=(lo, hi), method='trf', **TOL)
        cost = float(np.sum(r.fun ** 2))
        if best is None or cost < best['cost']:
            best = dict(theta=r.x.copy(), cost=cost)
    best['params'] = public_params(law, best['theta'], X, base)
    best['theta'] = best['theta'].tolist()
    return best


def public_params(law, th, X, base=None):
    th = np.asarray(th, float)
    if law == 'M_K_floor':
        p = public_params(base_law(base), th[1:], X)
        p['a1'] = float(th[0])
        return p
    p = dict(zip(NAMES[law], map(float, th)))
    if 'g_lo' in p:
        g1 = (p['g_hi'] - p['g_lo']) / (X['lKmax'] - X['lKmin'])
        p['gamma1'] = float(g1)
        p['gamma0'] = float(p['g_lo'] - g1 * X['lKmin'])
    return p


def objective(law, th, X, base=None):
    """Sum of squared fitting residuals (the quantity both fitters minimize)."""
    th = np.asarray(th, float)
    if law == 'M_align_add':
        return float(np.sum((predict_eps(law, th, X) - X['eps']) ** 2))
    if law == 'M_K_floor':
        bl = base_law(base)
        p = 2.0 ** (th[1] + shape(bl, th[1:], X)) + th[0] * (1 - X['alpha'])
        return float(np.sum(censored_residual(np.log2(np.maximum(p, 1e-300)), X) ** 2))
    return float(np.sum(censored_residual(th[0] + shape(law, th, X), X) ** 2))


def gamma_K_and_slope(law, p, lK, X):
    if law in ('M_K_exp', 'M_K_both'):
        return p['gamma0'] + p['gamma1'] * lK, p['gamma1']
    return p['gamma'], 0.0


def t_star(law, p, K, alpha, eps_star, X):
    """Analytic T* (examples) solving law(K, alpha, T) = eps_star for log-space laws."""
    lK = np.log2(K)
    g, _ = gamma_K_and_slope(law, p, lK, X)
    num = p['logA'] + p.get('kappa', 0.0) * lK - np.log2(eps_star)
    return 2.0 ** (num / g - p.get('eta', 0.0) * np.log2(alpha))


def elasticities(law, p, K_center, eps_star, X):
    """e_alpha = eta and e_K = d log T* / d log K at K_center (PROTOCOL Section 4.4)."""
    lK = np.log2(K_center)
    g, g1 = gamma_K_and_slope(law, p, lK, X)
    num = p['logA'] + p.get('kappa', 0.0) * lK - np.log2(eps_star)
    e_K = p.get('kappa', 0.0) / g - num * g1 / g ** 2
    return float(p.get('eta', 0.0)), float(e_K)
