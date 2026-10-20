"""Independent fitter (PROTOCOL Section 4.1): L-BFGS-B on the same objective as laws.py from a different, seeded
start grid. Held-out RMSEs must agree with laws.py to independent_fit_tol relative; prints AGREE or the first
disagreement when run as a script on a results directory.

Log-space laws: for each start, log2 A is set by a bounded 1-D search given the other parameters, then L-BFGS-B runs
over all parameters. M_align_add: for fixed (beta, gamma), (a, b, c, d) solve a linear least-squares problem, and
L-BFGS-B runs over (beta, gamma).
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import laws as L

OPTS = dict(ftol=1e-15, gtol=1e-11, maxiter=20000, maxfun=200000)


def fit(law, X, c):
    rng = np.random.default_rng(c['shared']['starts']['independent_seed'])
    lo, hi = L.bounds_for(law, c)
    best = None
    if law == 'M_align_add':
        def full(z):
            return np.r_[L.linear_add(z[0], z[1], X), z]
        for s in L.exponent_starts(law, X, c, jitter=rng):
            r = minimize(lambda z: L.objective(law, full(z), X), s, method='L-BFGS-B',
                         bounds=list(zip(lo[4:], hi[4:])), options=OPTS)
            if best is None or r.fun < best[1]:
                best = (full(r.x), float(r.fun))
    else:
        bnds = list(zip(lo, hi))
        for s in L.starts_for(law, X, c, jitter=rng):
            s = np.clip(s, lo, hi)
            la = minimize_scalar(lambda v: L.objective(law, np.r_[v, s[1:]], X), bounds=(lo[0], hi[0]),
                                 method='bounded', options=dict(xatol=1e-10)).x
            r = minimize(lambda z: L.objective(law, z, X), np.r_[la, s[1:]], method='L-BFGS-B', bounds=bnds,
                         options=OPTS)
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
