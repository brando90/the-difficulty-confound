"""Second implementation of every Experiment 14 statistic in plain numpy (own rank code, no scipy), used by verify.py
to check stats.py to 1e-10 (takes the place of independent_fit.py).

Ranks: average rank of x_i = #{x_j < x_i} + (#{x_j == x_i} + 1) / 2 by direct comparison (stats.py sorts via
scipy). Partial correlation: per-sample least squares with numpy.linalg.lstsq (stats.py solves the normal
equations). Bootstrap indices follow the recipe in stats.py's docstring, re-implemented here.
"""
import numpy as np


def ranks(x):
    x = np.asarray(x, np.float64)
    less = (x[..., None, :] < x[..., :, None]).sum(-1)
    eq = (x[..., None, :] == x[..., :, None]).sum(-1)
    return less + (eq + 1) / 2.0


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    out = np.full(a.shape[:-1], np.nan)
    flat_a = a.reshape(-1, a.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    res = np.full(len(flat_a), np.nan)
    for i in range(len(flat_a)):
        u = flat_a[i] - flat_a[i].sum() / len(flat_a[i])
        v = flat_b[i] - flat_b[i].sum() / len(flat_b[i])
        den = np.sqrt(np.dot(u, u) * np.dot(v, v))
        if den > 1e-9:
            res[i] = np.dot(u, v) / den
    return res.reshape(out.shape) if out.shape else res[0]


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def partial_spearman(x, y, Z):
    rx, ry = ranks(x), ranks(y)
    Z = np.asarray(Z, np.float64)
    rZ = np.stack([ranks(Z[..., j]) for j in range(Z.shape[-1])], -1)
    shp = rx.shape[:-1]
    rx2, ry2 = rx.reshape(-1, rx.shape[-1]), ry.reshape(-1, ry.shape[-1])
    rZ2 = rZ.reshape(-1, rZ.shape[-2], rZ.shape[-1])
    ex, ey = np.empty_like(rx2), np.empty_like(ry2)
    for i in range(len(rx2)):
        A = np.column_stack([np.ones(rZ2.shape[1]), rZ2[i]])
        ex[i] = rx2[i] - A @ np.linalg.lstsq(A, rx2[i], rcond=None)[0]
        ey[i] = ry2[i] - A @ np.linalg.lstsq(A, ry2[i], rcond=None)[0]
    r = pearson(ex, ey)
    return r.reshape(shp) if shp else float(r.reshape(-1)[0])


def kendall_w(M):
    M = np.asarray(M, np.float64)
    m, n = M.shape
    R = np.stack([ranks(row) for row in M])
    col = R.sum(0)
    S = float(((col - col.mean()) ** 2).sum())
    ties = 0.0
    for row in M:
        vals = np.sort(row)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[j + 1] == vals[i]:
                j += 1
            t = j - i + 1
            ties += t ** 3 - t
            i = j + 1
    return 12.0 * S / (m ** 2 * (n ** 3 - n) - m * ties)


def t_star(Ts, eps, eps_star):
    for i, (T, e) in enumerate(zip(Ts, eps)):
        if e <= eps_star:
            if i == 0:
                return float(T)
            lt0, lt1 = np.log(Ts[i - 1]), np.log(T)
            le0, le1 = np.log(eps[i - 1]), np.log(e)
            frac = (np.log(eps_star) - le0) / (le1 - le0)
            return float(np.exp(lt0 + frac * (lt1 - lt0)))
    return float('inf')


def median(a, axis):
    s = np.sort(a, axis=axis)
    n = s.shape[axis]
    lo = np.take(s, (n - 1) // 2, axis=axis)
    hi = np.take(s, n // 2, axis=axis)
    with np.errstate(invalid='ignore'):
        return np.where(lo == hi, lo, (lo + hi) / 2.0)


def percentile_ci(draws):
    d = np.nan_to_num(np.asarray(draws, np.float64), nan=0.0)
    s = np.sort(d)
    out = []
    for q in (2.5, 97.5):     # numpy 'linear' percentile definition
        pos = q / 100.0 * (len(s) - 1)
        i = int(np.floor(pos))
        j = min(i + 1, len(s) - 1)
        out.append(float(s[i] + (pos - i) * (s[j] - s[i])))
    return out[0], out[1], int(np.isnan(draws).sum())


def _zscores(logn):
    mu = logn.sum(1) / logn.shape[1]
    sd = np.sqrt(((logn - mu[:, None]) ** 2).sum(1) / logn.shape[1])
    return (logn - mu[:, None]) / sd[:, None], mu, sd


def _chunks(c, gate, fam_index):
    rng = np.random.default_rng([c['analysis_seed'], gate, fam_index])
    left = c['bootstrap_draws']
    while left > 0:
        n = min(c['bootstrap_chunk'], left)
        left -= n
        yield rng, n


def _level_layout(K):
    K = list(K)
    sizes, offs, start = [], [], 0
    for v in sorted(set(K)):
        cnt = K.count(v)
        sizes.append(cnt)
        offs.append(start)
        start += cnt
    return sizes, offs


def gate12(c, fam_index, logn, K, cov, z_logn_all=None):
    """logn (4, n) of the embedded tasks (sorted by K); z-scores from these same tasks."""
    z, _, _ = _zscores(logn)
    kh = z.mean(0)
    p1 = spearman(kh, K)
    p2 = partial_spearman(kh, K, cov)
    sizes, offs = _level_layout(K)
    d1, d2 = [], []
    for rng, n in _chunks(c, 1, fam_index):
        lv = [rng.integers(0, k, (n, k)) + o for k, o in zip(sizes, offs)]
        pr = rng.integers(0, 4, (n, 4))
        ti = np.concatenate(lv, 1)
        for i in range(n):
            kb = z[pr[i]].mean(0)[ti[i]]
            d1.append(spearman(kb, K[ti[i]]))
            d2.append(partial_spearman(kb, K[ti[i]], cov[ti[i]]))
    return dict(rho=float(p1), partial=float(p2), rho_draws=np.array(d1), partial_draws=np.array(d2))


def gate3(c, fam_index, logn_t, K_t, logn_all, tstar_seeds):
    _, mu, sd = _zscores(logn_all)
    return gate3_z(c, fam_index, (logn_t - mu[:, None]) / sd[:, None], K_t, tstar_seeds)


def gate3_z(c, fam_index, z, K_t, tstar_seeds):
    """Gate 3 from the targets' z-scored log norms z (4, n_targets)."""
    ts = median(tstar_seeds, 1)
    p = spearman(z.mean(0), ts)
    sizes, offs = _level_layout(K_t)
    d = []
    for rng, n in _chunks(c, 3, fam_index):
        lv = [rng.integers(0, k, (n, k)) + o for k, o in zip(sizes, offs)]
        pr = rng.integers(0, 4, (n, 4))
        sd_ = rng.integers(0, tstar_seeds.shape[1], (n, tstar_seeds.shape[1]))
        ti = np.concatenate(lv, 1)
        for i in range(n):
            kb = z[pr[i]].mean(0)[ti[i]]
            tb = median(tstar_seeds[ti[i]][:, sd_[i]], 1)
            d.append(spearman(kb, tb))
    return dict(rho=float(p), rho_draws=np.array(d), tstar=ts)


def gate4(c, fam_index, dist, H, K, cell, rep):
    n_mix, _, n_pairs = dist.shape
    n_cells = int(cell.max()) + 1
    where = {(int(cell[i]), int(rep[i])): i for i in range(n_mix)}
    div = np.array([np.mean([dist[i, p].mean() for p in range(dist.shape[1])]) for i in range(n_mix)])
    rH, rK = spearman(div, H), spearman(div, K)
    dH, dG = [], []
    for rng, n in _chunks(c, 4, fam_index):
        pr = rng.integers(0, 4, (n, 4))
        rp = rng.integers(0, 2, (n, n_cells, 2))
        pa = rng.integers(0, n_pairs, (n, n_mix, 4, n_pairs))
        for i in range(n):
            mixes = [where[(cc, int(rp[i, cc, s]))] for cc in range(n_cells) for s in range(2)]
            vals = []
            for slot, mi in enumerate(mixes):
                vals.append(np.mean([dist[mi, pr[i, q]][pa[i, slot, q]].mean() for q in range(4)]))
            vals = np.array(vals)
            a = spearman(vals, H[mixes])
            b = spearman(vals, K[mixes])
            dH.append(a)
            dG.append(a - abs(b))
    return dict(div=div, rho_H=float(rH), rho_K=float(rK), gap=float(rH - abs(rK)), rho_H_draws=np.array(dH),
                gap_draws=np.array(dG))


def gate5(c, fam_index, logratio, K_n):
    p = float(np.exp(median(logratio.mean(0), 0)))
    sizes, offs = _level_layout(K_n)
    d = []
    for rng, n in _chunks(c, 5, fam_index):
        lv = [rng.integers(0, k, (n, k)) + o for k, o in zip(sizes, offs)]
        pr = rng.integers(0, 4, (n, 4))
        ti = np.concatenate(lv, 1)
        for i in range(n):
            d.append(np.exp(median(logratio[pr[i]].mean(0)[ti[i]], 0)))
    return dict(ratio=p, ratio_draws=np.array(d))
