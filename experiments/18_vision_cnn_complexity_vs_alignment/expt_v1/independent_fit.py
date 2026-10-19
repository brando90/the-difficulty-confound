"""Independent fitter (PROTOCOL Section 4.1): scipy.optimize.minimize (L-BFGS-B) on the same objective as laws.py,
from a different, seeded starting grid. Held-out RMSEs must agree with laws.py to independent_fit_tol relative.

Log-space laws: for each start, log2 A is first set by a 1-D bounded search given the exponents, then L-BFGS-B runs
over all parameters. M_align_add: for fixed gamma, (a, c, d) are a linear least-squares solution; L-BFGS-B over gamma.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import laws as L

OPTS = dict(ftol=1e-15, gtol=1e-11, maxiter=20000, maxfun=200000)


def fit(law, X, c):
    rng = np.random.default_rng(c['fit']['independent_seed'])
    lo, hi = L.bounds_for(law, c)
    best = None
    if law == 'M_align_add':
        def obj(z):
            return L.objective(law, np.r_[L.linear_add(z[0], X), z[0]], X)
        for s in L.exponent_starts(law, X, c, jitter=rng):
            r = minimize(obj, s, method='L-BFGS-B', bounds=[(lo[3], hi[3])], options=OPTS)
            if best is None or r.fun < best[1]:
                best = (np.r_[L.linear_add(r.x[0], X), r.x[0]], float(r.fun))
    else:
        bnds = [(None, None)] + list(zip(lo[1:], hi[1:]))
        for s in L.exponent_starts(law, X, c, jitter=rng):
            a = minimize_scalar(lambda la: L.objective(law, np.r_[la, s], X), bounds=(-60, 20), method='bounded',
                                options=dict(xatol=1e-10)).x
            r = minimize(lambda z: L.objective(law, z, X), np.r_[a, s], method='L-BFGS-B', bounds=bnds, options=OPTS)
            if best is None or r.fun < best[1]:
                best = (r.x, float(r.fun))
    th = np.asarray(best[0], float)
    return dict(theta=th.tolist(), cost=L.objective(law, th, X), params=L.public_params(law, th, X))


def heldout_rmse(law, th, X):
    r, bad = L.scored_residual(law, th, X)
    return float(np.sqrt(np.mean(r ** 2))), int(bad.sum())


def agree(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(np.abs(a), np.abs(b)) + 1e-12))
