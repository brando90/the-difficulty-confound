"""Laws of PROTOCOL Section 4.1 with the exact, non-fitted identification offset eps_id, fit by multi-start
trust-region-reflective least squares on log2 excess loss.

Forecast: eps_hat = eps_id + 2 ** (logA + shape), with Kt = K / K_ref and Tt = T / T0. An exponent that depends on K,
gamma(K) = gamma0 + gamma1 * log2 Kt, is parametrized by its values at the two ends of the K grid (g_lo at K_min,
g_hi at K_max), each bounded to [1e-3, 3]; because gamma(K) is linear in log2 K this enforces gamma(K) > 0 at every
grid K with box constraints only. M_align_add = a + c Tt^-gamma - d alpha has no offset and is fit in linear space.
Observed points below the resolution floor eps_res enter through a censored residual (as in Experiment 11).

13-GPU full N laws (PROTOCOL Section 4.1, Nt = N / 3e6, N non-embedding): the effective-data forms
eps_hat = eps_id + B Nt^-beta + A (learning shape of the matching base law), with theta = [logA, logB, beta, base
exponents] (N_K_both is the full form, N_K_exp its kappa = 0 variant, N_K_coef its gamma1 = 0 variant and N_align_eff
the base with neither), and the alignment-scaling forms a + b Nt^-beta + c0 Kt^kappa Tt^-gamma(K) - d alpha
(N_AS_both, N_AS_exp with kappa = 0, N_AS_coef with gamma1 = 0), fit in linear space like M_align_add. The beta starts
are crossed with the 64-start exponent grid.
"""
import itertools
import math

import numpy as np
from scipy.optimize import least_squares

LAWS = ['M0_scale', 'M_align_eff', 'M_align_add', 'M_K_coef', 'M_K_exp', 'M_K_both']
LOG_LAWS = ['M0_scale', 'M_align_eff', 'M_K_coef', 'M_K_exp', 'M_K_both']
NAMES = {
    'M0_scale': ['logA', 'gamma'],
    'M_align_eff': ['logA', 'gamma', 'eta'],
    'M_K_coef': ['logA', 'gamma', 'eta', 'kappa'],
    'M_K_exp': ['logA', 'g_lo', 'g_hi', 'eta'],
    'M_K_both': ['logA', 'g_lo', 'g_hi', 'eta', 'kappa'],
    'M_align_add': ['a', 'c', 'd', 'gamma'],
}
INVALID_PAD = 1e-6
N0 = 3e6
LOG_BOUND = 60.0
N_BASE = {'N_align_eff': 'M_align_eff', 'N_K_coef': 'M_K_coef', 'N_K_exp': 'M_K_exp', 'N_K_both': 'M_K_both'}
AS_LAWS = ['N_AS_coef', 'N_AS_exp', 'N_AS_both']
N_LAWS = list(N_BASE) + AS_LAWS
LINEAR_LAWS = ['M_align_add'] + AS_LAWS
NAMES.update({law: ['logA', 'logB', 'beta'] + NAMES[base][1:] for law, base in N_BASE.items()})
NAMES.update({'N_AS_coef': ['a', 'b', 'c0', 'd', 'beta', 'gamma', 'kappa'],
              'N_AS_exp': ['a', 'b', 'c0', 'd', 'beta', 'g_lo', 'g_hi'],
              'N_AS_both': ['a', 'b', 'c0', 'd', 'beta', 'g_lo', 'g_hi', 'kappa']})


def design(rows, k_unit, K_grid, K_ref, T0, eps_res, offset=True):
    """Arrays for fitting from metric rows (dicts with the K unit, alpha, T, excess_bits, ident_bits).

    K_grid: the arm's K values in this unit (sets the gamma(K) end points); K_ref: geometric mean of the fit levels."""
    K = np.array([float(r[k_unit]) for r in rows])
    eps = np.array([float(r['excess_bits']) for r in rows])
    eid = np.array([float(r['ident_bits']) for r in rows]) if offset else np.zeros(len(rows))
    alpha = np.array([float(r['alpha']) for r in rows])
    T = np.array([float(r['T']) for r in rows])
    X = dict(lK=np.log2(K / K_ref), la=np.log2(alpha), lT=np.log2(T / T0), alpha=alpha, Tt=T / T0, eps=eps,
             eid=eid, y=np.log2(np.maximum(eps, 1e-300)), cens=eps < eps_res, yres=math.log2(eps_res),
             lKmin=math.log2(min(K_grid) / K_ref), lKmax=math.log2(max(K_grid) / K_ref), k_unit=k_unit,
             K_ref=K_ref, T0=T0, eps_res=eps_res)
    if rows and 'N' in rows[0]:  # 13-GPU: model size (non-embedding parameters)
        X['lN'] = np.log2(np.array([float(r['N']) for r in rows]) / N0)
    return X


def subset(X, m):
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in X.items()}


def gamma_of_K(g_lo, g_hi, lK, X):
    return g_lo + (g_hi - g_lo) * (lK - X['lKmin']) / (X['lKmax'] - X['lKmin'])


def shape(law, th, X):
    """log2 of the learning term without the log2 A part."""
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


def as_design(law, ex, X):
    """Columns [1, Nt^-beta, Kt^kappa Tt^-gamma(K), -alpha] of an alignment-scaling form, ex = [beta, exponents]."""
    if law == 'N_AS_coef':
        sh = ex[2] * X['lK'] - ex[1] * X['lT']
    elif law == 'N_AS_exp':
        sh = -gamma_of_K(ex[1], ex[2], X['lK'], X) * X['lT']
    else:
        sh = ex[3] * X['lK'] - gamma_of_K(ex[1], ex[2], X['lK'], X) * X['lT']
    one = np.ones_like(X['lT'])
    return np.column_stack([one, 2.0 ** np.clip(-ex[0] * X['lN'], -1000, 1000), 2.0 ** np.clip(sh, -1000, 1000),
                            -X['alpha']])


def predict_eps(law, th, X):
    th = np.asarray(th, float)
    if law in AS_LAWS:
        return as_design(law, th[4:], X) @ th[:4]
    if law in N_BASE:
        cap = 2.0 ** np.clip(th[1] - th[2] * X['lN'], -1000, 1000)
        return X['eid'] + cap + 2.0 ** np.clip(th[0] + shape(N_BASE[law], np.r_[0.0, th[3:]], X), -1000, 1000)
    if law == 'M_align_add':
        return th[0] + th[1] * X['Tt'] ** (-th[3]) - th[2] * X['alpha']
    return X['eid'] + 2.0 ** (th[0] + shape(law, th, X))


def floor_of(law, X):
    return np.maximum(X['eid'], 0.0) if law in LINEAR_LAWS else np.zeros_like(X['eps'])


def invalid_mask(law, p, X):
    """Nonfinite, <= 0, or (forms without the offset) below eps_id."""
    bad = ~np.isfinite(p) | (p <= 0)
    if law in LINEAR_LAWS:
        bad |= p < X['eid']
    return bad


def censored_residual(pred_log2, X):
    r = pred_log2 - X['y']
    rc = np.maximum(0.0, pred_log2 - X['yres'])
    return np.where(X['cens'], rc, r)


def scored_residual(law, th, X):
    """Censored residual on the log2 scale; invalid forecasts are scored at log2(floor + 1e-6)."""
    p = predict_eps(law, th, X)
    bad = invalid_mask(law, p, X)
    p = np.where(bad, floor_of(law, X) + INVALID_PAD, p)
    return censored_residual(np.log2(p), X), bad


def bounds_for(law, c):
    f = c['fit']
    g, e, k = f['gamma_bounds'], f['eta_bounds'], f['kappa_bounds']
    inf = np.inf
    if law in N_BASE:
        lo, hi = bounds_for(N_BASE[law], c)
        be = f['beta_bounds']
        return np.r_[-LOG_BOUND, -LOG_BOUND, be[0], lo[1:]], np.r_[LOG_BOUND, LOG_BOUND, be[1], hi[1:]]
    if law in AS_LAWS:
        be = f['beta_bounds']
        ex = {'N_AS_coef': ([g[0], k[0]], [g[1], k[1]]), 'N_AS_exp': ([g[0], g[0]], [g[1], g[1]]),
              'N_AS_both': ([g[0], g[0], k[0]], [g[1], g[1], k[1]])}[law]
        return np.array([-inf] * 4 + [be[0]] + ex[0], float), np.array([inf] * 4 + [be[1]] + ex[1], float)
    b = {
        'M0_scale': ([-inf, g[0]], [inf, g[1]]),
        'M_align_eff': ([-inf, g[0], e[0]], [inf, g[1], e[1]]),
        'M_K_coef': ([-inf, g[0], e[0], k[0]], [inf, g[1], e[1], k[1]]),
        'M_K_exp': ([-inf, g[0], g[0], e[0]], [inf, g[1], g[1], e[1]]),
        'M_K_both': ([-inf, g[0], g[0], e[0], k[0]], [inf, g[1], g[1], e[1], k[1]]),
        'M_align_add': ([-inf, -inf, -inf, g[0]], [inf, inf, inf, g[1]]),
    }[law]
    return np.array(b[0], float), np.array(b[1], float)


def exponent_starts(law, X, c, jitter=None):
    """The 64-start exponent grid, projected into the bounds. jitter (an RNG) gives the independent fitter its own
    differently placed grid."""
    f = c['fit']
    if law in N_LAWS:
        return n_exponent_starts(law, X, c, jitter)
    lo, hi = bounds_for(law, c)
    G, E = list(f['gamma_starts']), list(f['eta_starts'])
    KS, G1, BS = list(f['kappa_starts']), list(f['gamma1_starts']), [list(x) for x in f['both_starts']]
    if jitter is not None:
        G = [g * float(jitter.uniform(0.7, 1.4)) for g in G]
        E = [e * float(jitter.uniform(0.7, 1.4)) for e in E]
        KS = [k + float(jitter.uniform(-0.3, 0.3)) for k in KS]
        G1 = [g + float(jitter.uniform(-0.03, 0.03)) for g in G1]
        BS = [[k + float(jitter.uniform(-0.3, 0.3)), g + float(jitter.uniform(-0.03, 0.03))] for k, g in BS]

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
    S = np.array(S, float)
    j = 3 if law == 'M_align_add' else 1
    return np.clip(S, lo[j:] + 1e-9, hi[j:])


def n_exponent_starts(law, X, c, jitter=None):
    """[beta, exponents]: the base law's 64-start grid (alignment-scaling forms: gamma x gamma1 x kappa, 64 for the
    full form) crossed with c['beta_starts']."""
    f = c['fit']
    BETA = list(c['beta_starts'])
    if jitter is not None:
        BETA = [b * float(jitter.uniform(0.7, 1.4)) for b in BETA]
    if law in N_BASE:
        S = exponent_starts(N_BASE[law], X, c, jitter)
    else:
        G, KS, G1 = list(f['gamma_starts']), list(f['kappa_starts']), list(f['gamma1_starts'])
        if jitter is not None:
            G = [g * float(jitter.uniform(0.7, 1.4)) for g in G]
            KS = [k + float(jitter.uniform(-0.3, 0.3)) for k in KS]
            G1 = [g + float(jitter.uniform(-0.03, 0.03)) for g in G1]

        def ends(g0, g1):
            return [g0 + g1 * X['lKmin'], g0 + g1 * X['lKmax']]
        if law == 'N_AS_coef':
            S = [[g, k] for g, k in itertools.product(G, KS)]
        elif law == 'N_AS_exp':
            S = [ends(g, g1) for g, g1 in itertools.product(G, G1)]
        else:
            S = [ends(g, g1) + [k] for g, g1, k in itertools.product(G, G1, KS)]
    S = np.array([[b, *s] for b in BETA for s in S], float)
    lo, hi = bounds_for(law, c)
    j = 4 if law in AS_LAWS else 2
    return np.clip(S, lo[j:] + 1e-9, hi[j:])


def start_head(law, ex, X):
    """N log laws: logB at half the smallest excess over eps_id; logA by least squares on what remains."""
    rem = X['eps'] - X['eid']
    floor = max(float(np.min(rem)) * 0.5, X['eps_res'] * 0.1)
    logB = math.log2(floor) + ex[0] * float(np.mean(X['lN']))
    learn = np.maximum(rem - floor, X['eps_res'] * 0.1)
    logA = float(np.mean(np.log2(learn) - shape(N_BASE[law], np.r_[0.0, ex[1:]], X)))
    return np.clip([logA, logB], -LOG_BOUND, LOG_BOUND)


def linear_as(law, ex, X):
    return np.linalg.lstsq(as_design(law, ex, X), X['eps'], rcond=None)[0]


def starts_for(law, X, c, jitter=None):
    """Full starting vectors for the 13-GPU N laws."""
    if law in AS_LAWS:
        return [np.r_[linear_as(law, s, X), s] for s in exponent_starts(law, X, c, jitter)]
    return [np.r_[start_head(law, s, X), s] for s in exponent_starts(law, X, c, jitter)]


def start_logA(law, ex, X):
    th = np.r_[0.0, ex]
    learn = np.maximum(X['eps'] - X['eid'], X['eps_res'] * 0.1)
    return float(np.mean(np.log2(learn) - shape(law, th, X)))


def linear_add(gamma, X):
    Z = np.column_stack([np.ones_like(X['Tt']), X['Tt'] ** (-gamma), -X['alpha']])
    return np.linalg.lstsq(Z, X['eps'], rcond=None)[0]


def residual_fn(law, X):
    if law in LINEAR_LAWS:
        return lambda z: predict_eps(law, z, X) - X['eps']
    return lambda z: censored_residual(np.log2(np.maximum(predict_eps(law, z, X), 1e-300)), X)


def objective(law, th, X):
    return float(np.sum(residual_fn(law, X)(np.asarray(th, float)) ** 2))


TOL = dict(xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)


def fit(law, X, c):
    """Best of the multi-start fits. Returns dict(theta, cost, params, converged)."""
    lo, hi = bounds_for(law, c)
    if law in N_LAWS:
        starts = starts_for(law, X, c)
    elif law == 'M_align_add':
        starts = [np.r_[linear_add(s[0], X), s[0]] for s in exponent_starts(law, X, c)]
    else:
        starts = [np.r_[start_logA(law, s, X), s] for s in exponent_starts(law, X, c)]
    res = residual_fn(law, X)
    best = None
    for s in starts:
        s = np.clip(s, lo, hi)
        r = least_squares(res, s, bounds=(lo, hi), method='trf', **TOL)
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


def gamma_at(law, p, lK):
    if 'gamma1' in p:
        return p['gamma0'] + p['gamma1'] * lK, p['gamma1']
    return p['gamma'], 0.0


def t_star(law, p, K, alpha, eps_star, X):
    """T* in tokens where the fitted learning term equals eps_star (PROTOCOL Section 4.4)."""
    lK = math.log2(K / X['K_ref'])
    g, _ = gamma_at(law, p, lK)
    num = p['logA'] + p.get('kappa', 0.0) * lK - math.log2(eps_star)
    return X['T0'] * 2.0 ** (num / g - p.get('eta', 0.0) * math.log2(alpha))


def elasticities(law, p, K, eps_star, X):
    """e_alpha = eta and e_K = d ln T* / d ln K at K."""
    lK = math.log2(K / X['K_ref'])
    g, g1 = gamma_at(law, p, lK)
    num = p['logA'] + p.get('kappa', 0.0) * lK - math.log2(eps_star)  # log2 units
    e_K = p.get('kappa', 0.0) / g - num * g1 / g ** 2
    return float(p.get('eta', 0.0)), float(e_K)
