"""The six laws of PROTOCOL Section 4.1 with the model-size term, fit by multi-start trust-region-reflective least
squares on log2 excess bits (adapted from Experiment 13's laws.py).

Forecast eps_hat = B * Nt**-beta + A * (learning shape), with Nt = N/N0, Tt = T/T0, st = s/s_ref (geometric mean of
the fit levels). gamma(s) = gamma0 + gamma1 * log2 st is parametrized by its values at the ends of the fit grid
(g_lo, g_hi), each bounded to the gamma bounds, so gamma(s) > 0 on the whole grid with box constraints only.
M_align_add = a + b*Nt**-beta + c*Tt**-gamma - d*alpha is fit in linear space. Observed points below eps_res enter
through a censored residual, as in Experiments 11 and 13. Floors are 0: addition is deterministic.
"""
import itertools
import math

import numpy as np
from scipy.optimize import least_squares

LAWS = ['M0_scale', 'M_align_eff', 'M_align_add', 'M_K_coef', 'M_K_exp', 'M_K_both']
NAMES = {
    'M0_scale': ['logA', 'logB', 'beta', 'gamma'],
    'M_align_eff': ['logA', 'logB', 'beta', 'gamma', 'eta'],
    'M_K_coef': ['logA', 'logB', 'beta', 'gamma', 'eta', 'kappa'],
    'M_K_exp': ['logA', 'logB', 'beta', 'g_lo', 'g_hi', 'eta'],
    'M_K_both': ['logA', 'logB', 'beta', 'g_lo', 'g_hi', 'eta', 'kappa'],
    'M_align_add': ['a', 'b', 'c', 'd', 'beta', 'gamma'],
}
INVALID_PAD = 1e-6
TOL = dict(xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
LOG_BOUND = 60.0


def design(rows, s_grid, s_ref, T0, N0, eps_res):
    """Fit arrays from metric rows (s, alpha, T, N, excess_bits). s_grid: the fit levels (sets gamma(s) ends)."""
    get = lambda k: np.array([float(r[k]) for r in rows])
    s, alpha, T, N, eps = get('s'), get('alpha'), get('T'), get('N'), get('excess_bits')
    return dict(lK=np.log2(s / s_ref), la=np.log2(alpha), lT=np.log2(T / T0), lN=np.log2(N / N0), alpha=alpha,
                Tt=T / T0, Nt=N / N0, eps=eps, y=np.log2(np.maximum(eps, 1e-300)), cens=eps < eps_res,
                yres=math.log2(eps_res), lKmin=math.log2(min(s_grid) / s_ref), lKmax=math.log2(max(s_grid) / s_ref),
                s_ref=s_ref, T0=T0, N0=N0, eps_res=eps_res)


def subset(X, m):
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in X.items()}


def gamma_of(g_lo, g_hi, lK, X):
    return g_lo + (g_hi - g_lo) * (lK - X['lKmin']) / (X['lKmax'] - X['lKmin'])


def shape(law, th, X):
    """log2 of the learning term without log2 A (th excludes logA, logB, beta)."""
    if law == 'M0_scale':
        return -th[0] * X['lT']
    if law == 'M_align_eff':
        return -th[0] * (th[1] * X['la'] + X['lT'])
    if law == 'M_K_coef':
        return th[2] * X['lK'] - th[0] * (th[1] * X['la'] + X['lT'])
    if law == 'M_K_exp':
        return -gamma_of(th[0], th[1], X['lK'], X) * (th[2] * X['la'] + X['lT'])
    if law == 'M_K_both':
        return th[3] * X['lK'] - gamma_of(th[0], th[1], X['lK'], X) * (th[2] * X['la'] + X['lT'])
    raise KeyError(law)


def predict_eps(law, th, X):
    th = np.asarray(th, float)
    if law == 'M_align_add':
        a, b, c, d, beta, gamma = th
        return a + b * X['Nt'] ** (-beta) + c * X['Tt'] ** (-gamma) - d * X['alpha']
    capacity = 2.0 ** np.clip(th[1] - th[2] * X['lN'], -1000, 1000)
    return capacity + 2.0 ** np.clip(th[0] + shape(law, th[3:], X), -1000, 1000)


def invalid_mask(p):
    return ~np.isfinite(p) | (p <= 0)


def censored_residual(pred_log2, X):
    r = pred_log2 - X['y']
    rc = np.maximum(0.0, pred_log2 - X['yres'])
    return np.where(X['cens'], rc, r)


def scored_residual(law, th, X):
    """Censored log2 residual; invalid forecasts are scored at log2(eps_res) (PROTOCOL 4.2 / Experiment 11)."""
    p = predict_eps(law, th, X)
    bad = invalid_mask(p)
    p = np.where(bad, X['eps_res'], p)
    return censored_residual(np.log2(p), X), bad


def bounds_for(law, c):
    b = c['shared']['bounds']
    g, e, k, be = b['gamma'], b['eta'], b['kappa'], b['beta']
    inf, L = np.inf, LOG_BOUND
    head_lo, head_hi = [-L, -L, be[0]], [L, L, be[1]]
    tail = {
        'M0_scale': ([g[0]], [g[1]]),
        'M_align_eff': ([g[0], e[0]], [g[1], e[1]]),
        'M_K_coef': ([g[0], e[0], k[0]], [g[1], e[1], k[1]]),
        'M_K_exp': ([g[0], g[0], e[0]], [g[1], g[1], e[1]]),
        'M_K_both': ([g[0], g[0], e[0], k[0]], [g[1], g[1], e[1], k[1]]),
    }
    if law == 'M_align_add':
        return np.array([-inf, -inf, -inf, -inf, be[0], g[0]]), np.array([inf, inf, inf, inf, be[1], g[1]])
    lo, hi = tail[law]
    return np.array(head_lo + lo, float), np.array(head_hi + hi, float)


def exponent_starts(law, X, c, jitter=None):
    """The 64-start exponent grid (Experiment 11), crossed with the beta starts and projected into the bounds.
    jitter (an RNG) gives the independent fitter its own, differently placed grid."""
    st = c['shared']['starts']
    G, E, KS, G1 = list(st['gamma']), list(st['eta']), list(st['kappa']), list(st['gamma1'])
    BS, BETA = [list(x) for x in st['both']], list(st['beta'])
    if jitter is not None:
        G = [g * float(jitter.uniform(0.7, 1.4)) for g in G]
        E = [e * float(jitter.uniform(0.7, 1.4)) for e in E]
        KS = [k + float(jitter.uniform(-0.3, 0.3)) for k in KS]
        G1 = [g + float(jitter.uniform(-0.03, 0.03)) for g in G1]
        BS = [[k + float(jitter.uniform(-0.3, 0.3)), g + float(jitter.uniform(-0.03, 0.03))] for k, g in BS]
        BETA = [b * float(jitter.uniform(0.7, 1.4)) for b in BETA]

    def ends(g0, g1):
        return g0 + g1 * X['lKmin'], g0 + g1 * X['lKmax']
    if law in ('M0_scale', 'M_align_add'):
        S = [[g] for g in G]
    elif law == 'M_align_eff':
        S = [[g, e] for g, e in itertools.product(G, E)]
    elif law == 'M_K_coef':
        S = [[g, e, k] for g, e, k in itertools.product(G, E, KS)]
    elif law == 'M_K_exp':
        S = [[*ends(g, g1), e] for g, e, g1 in itertools.product(G, E, G1)]
    elif law == 'M_K_both':
        S = [[*ends(g, kg[1]), e, kg[0]] for g, e, kg in itertools.product(G, E, BS)]
    else:
        raise KeyError(law)
    lo, hi = bounds_for(law, c)
    if law == 'M_align_add':
        S = [[b, s[0]] for b in BETA for s in S]
        return np.clip(np.array(S, float), lo[4:] + 1e-9, hi[4:])
    S = [[b, *s] for b in BETA for s in S]
    return np.clip(np.array(S, float), lo[2:] + 1e-9, hi[2:])


def start_head(law, ex, X):
    """logB at half the smallest observed excess; logA by least squares on the remaining learning term."""
    floor = max(float(np.min(X['eps'])) * 0.5, X['eps_res'] * 0.1)
    logB = math.log2(floor) + ex[0] * float(np.mean(X['lN']))
    learn = np.maximum(X['eps'] - floor, X['eps_res'] * 0.1)
    logA = float(np.mean(np.log2(learn) - shape(law, ex[1:], X)))
    return np.clip([logA, logB], -LOG_BOUND, LOG_BOUND)


def linear_add(beta, gamma, X):
    Z = np.column_stack([np.ones_like(X['Tt']), X['Nt'] ** (-beta), X['Tt'] ** (-gamma), -X['alpha']])
    return np.linalg.lstsq(Z, X['eps'], rcond=None)[0]


def residual_fn(law, X):
    if law == 'M_align_add':
        return lambda z: predict_eps(law, z, X) - X['eps']
    return lambda z: censored_residual(np.log2(np.maximum(predict_eps(law, z, X), 1e-300)), X)


def objective(law, th, X):
    return float(np.sum(residual_fn(law, X)(np.asarray(th, float)) ** 2))


def starts_for(law, X, c, jitter=None):
    if law == 'M_align_add':
        return [np.r_[linear_add(s[0], s[1], X), s] for s in exponent_starts(law, X, c, jitter)]
    return [np.r_[start_head(law, s, X), s] for s in exponent_starts(law, X, c, jitter)]


def fit(law, X, c):
    """Best of the multi-start fits: dict(theta, cost, params, converged)."""
    lo, hi = bounds_for(law, c)
    res = residual_fn(law, X)
    best = None
    for s in starts_for(law, X, c):
        r = least_squares(res, np.clip(s, lo, hi), bounds=(lo, hi), method='trf', **TOL)
        cost = float(np.sum(r.fun ** 2))
        if np.isfinite(cost) and (best is None or cost < best['cost']):
            best = dict(theta=r.x.copy(), cost=cost, converged=bool(r.status > 0))
    best['params'] = public_params(law, best['theta'], X)
    best['theta'] = best['theta'].tolist()
    return best


def public_params(law, th, X):
    p = dict(zip(NAMES[law], map(float, th)))
    if 'g_lo' in p:
        g1 = (p['g_hi'] - p['g_lo']) / (X['lKmax'] - X['lKmin'])
        p['gamma1'] = float(g1)
        p['gamma0'] = float(p['g_lo'] - g1 * X['lKmin'])
    return p


def gamma_at(p, lK):
    if 'gamma1' in p:
        return p['gamma0'] + p['gamma1'] * lK, p['gamma1']
    return p['gamma'], 0.0


def t_star(p, s, alpha, eps_star, X):
    """Examples needed for the fitted learning term to reach eps_star (capacity term excluded, PROTOCOL 4.3)."""
    lK = math.log2(s / X['s_ref'])
    g, _ = gamma_at(p, lK)
    num = p['logA'] + p.get('kappa', 0.0) * lK - math.log2(eps_star)
    return X['T0'] * 2.0 ** (num / g - p.get('eta', 0.0) * math.log2(alpha))


def elasticities(p, s, eps_star, X):
    """e_alpha = eta and e_s = d ln T* / d ln s at s."""
    lK = math.log2(s / X['s_ref'])
    g, g1 = gamma_at(p, lK)
    num = p['logA'] + p.get('kappa', 0.0) * lK - math.log2(eps_star)
    return float(p.get('eta', 0.0)), float(p.get('kappa', 0.0) / g - num * g1 / g ** 2)
