"""Independent re-implementation of the bounded-law objective, minimized with a different optimizer.

Used by test_laws.py and verify.py to check that laws.fit_bounded_law reached (at least) the same optimum.
It shares no forecasting code with laws.py.
"""
import numpy as np
from scipy.optimize import minimize


def objective(theta, form, z, s, acc0, n, y, lam):
    eta = theta[0] + z @ theta[1:]
    if form == 'gain':
        frac = -np.expm1(-s * np.exp(np.minimum(np.maximum(eta, -30.0), 30.0)))
        pred = (1.0 - acc0) * frac
    else:
        p = np.minimum(np.maximum(acc0, 1e-6), 1.0 - 1e-6)
        pred = 0.5 * (1.0 + np.tanh(0.5 * (np.log(p) - np.log1p(-p) + s * eta))) - acc0
    return float(np.mean((pred - y) ** 2) + lam * np.sum(theta[1:] ** 2))


def refit(form, z, s, acc0, n, y, lam, starts):
    best = None
    for start in starts:
        theta0 = np.zeros(z.shape[1] + 1)
        theta0[0] = start
        res = minimize(objective, theta0, args=(form, z, s, acc0, n, y, lam), method='L-BFGS-B', options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)
