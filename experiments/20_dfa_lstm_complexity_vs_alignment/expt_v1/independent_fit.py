"""Independent fitter: profile the intercept out exactly, then L-BFGS-B over the exponents from the same starts.

Log-space laws: for fixed exponents the censored objective is a convex piecewise-quadratic function of log2 A, whose
minimizer is found exactly by scanning the censoring breakpoints. M_align_add: for fixed gamma, (a, c, d) are a
linear least-squares solution. M_K_floor has no closed-form profile, so L-BFGS-B runs over all its parameters.
Objective values and held-out forecasts must match laws.py to `agreement_rel_tol` relative (PROTOCOL Section 4.1).
"""
import numpy as np
from scipy.optimize import minimize

import laws as L


def profile_logA(g, X):
    """Exact argmin over a of sum_unc (a + g - y)^2 + sum_cens max(0, a + g - yres)^2."""
    unc, cen = ~X['cens'], X['cens']
    s_unc, n_unc = float(np.sum((X['y'] - g)[unc])), int(unc.sum())
    b = np.sort(X['yres'] - g[cen])  # censored point j becomes active when a > b_j
    s_c = 0.0
    for k in range(len(b) + 1):
        if k > 0:
            s_c += b[k - 1]
        if n_unc + k == 0:
            continue
        a = (s_unc + s_c) / (n_unc + k)
        lo = b[k - 1] if k > 0 else -np.inf
        hi = b[k] if k < len(b) else np.inf
        if lo <= a <= hi:
            return a
    return float(b[0]) if len(b) else 0.0


def fit(law, X, c, base=None):
    best = None
    if law == 'M_K_floor':
        bl, bth = base
        lo0, hi0 = L.bounds_for(bl, c)
        bounds = list(zip(np.r_[0.0, lo0], np.r_[np.inf, hi0]))
        bounds = [(lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None) for lo, hi in bounds]
        for a1 in c['fit']['floor_a1_starts']:
            r = minimize(lambda z: L.objective(law, z, X, bl), np.r_[a1, bth], method='L-BFGS-B', bounds=bounds,
                         options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000, maxfun=200000))
            if best is None or r.fun < best[1]:
                best = (r.x, float(r.fun))
        th = best[0]
    elif law == 'M_align_add':
        lo, hi = L.bounds_for(law, c)

        def obj(z):
            abc = L._linear_add(z[0], X)
            return L.objective(law, np.r_[abc, z[0]], X)
        for s in L.exponent_starts(law, X, c):
            r = minimize(obj, s, method='L-BFGS-B', bounds=[(lo[3], hi[3])],
                         options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000))
            if best is None or r.fun < best[1]:
                best = (r.x, float(r.fun))
        th = np.r_[L._linear_add(best[0][0], X), best[0][0]]
    else:
        lo, hi = L.bounds_for(law, c)

        def full(ex):
            g = L.shape(law, np.r_[0.0, ex], X)
            return np.r_[profile_logA(g, X), ex]

        def obj(ex):
            return L.objective(law, full(ex), X)
        for s in L.exponent_starts(law, X, c):
            r = minimize(obj, s, method='L-BFGS-B', bounds=list(zip(lo[1:], hi[1:])),
                         options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000, maxfun=200000))
            if best is None or r.fun < best[1]:
                best = (r.x, float(r.fun))
        th = full(best[0])
    b = bl if law == 'M_K_floor' else None
    return dict(theta=np.asarray(th).tolist(), cost=L.objective(law, th, X, b),
                params=L.public_params(law, th, X, b))


def agree(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(np.abs(a), np.abs(b)) + 1e-12))
