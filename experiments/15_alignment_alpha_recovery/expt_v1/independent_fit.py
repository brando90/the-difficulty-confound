"""Second implementation, plain numpy only (no scipy, no scikit-learn): Spearman with average ranks, the isotonic
calibrator (pool-adjacent-violators), the Gate 2 slope (within-group demeaning) and every law fit (Experiment 11/12's
own independent fitters, imported). verify.py compares each against laws.py.
"""
import numpy as np


def avg_ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def fe_slope(y, x, groups, w=None):
    """Slope with group intercepts: demean x and y within groups (weighted), then sum(w x y) / sum(w x x)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    groups = np.asarray(groups)
    xd, yd = x.copy(), y.copy()
    for g in np.unique(groups):
        m = groups == g
        xd[m] -= np.sum(w[m] * x[m]) / np.sum(w[m])
        yd[m] -= np.sum(w[m] * y[m]) / np.sum(w[m])
    return float(np.sum(w * xd * yd) / np.sum(w * xd * xd))


class Isotonic:
    """Pool-adjacent-violators on x-sorted data (ties in x pooled first), then linear interpolation between the
    block points, clipped beyond the ends."""

    def __init__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ux = np.unique(x)
        uy = np.array([y[x == v].mean() for v in ux])
        uw = np.array([float((x == v).sum()) for v in ux])
        vals, wts, cnt = [], [], []
        for v, w in zip(uy, uw):
            vals.append(v)
            wts.append(w)
            cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
                vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
                wts.append(w1 + w2)
                cnt.append(c1 + c2)
        self.x = ux
        self.y = np.repeat(vals, cnt)

    def __call__(self, x):
        return np.interp(np.asarray(x, float), self.x, self.y)
