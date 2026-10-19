"""Statistics for Experiment 14 (takes the place of laws.py): Spearman, partial Spearman, Kendall's W, T*, the
task-score K_hat, the stratified bootstraps and Gates 1-5 (PROTOCOL Sections 4-5).

Public API: rank_avg, spearman, partial_spearman, kendall_w, t_star, tstar_table, khat, gate12, gate3, gate4,
gate5, ci, boot_index_chunks.

Bootstrap recipe (independent_stats.py follows the same recipe with its own code; PROTOCOL Section 4):
  draws B = bootstrap_draws, generated in chunks of bootstrap_chunk draws, rng = numpy default_rng([analysis_seed,
  gate, family index]) with gate 1 (Gates 1 and 2 share draws), 3, 4 or 5 and family index 0 (DFA) or 1 (DSL).
  Per chunk, in this order: for each K level in ascending order, rng.integers(0, n_level, (chunk, n_level)) picks
  tasks of that level with replacement; then rng.integers(0, 4, (chunk, 4)) picks probe seeds with replacement;
  Gate 3 then draws rng.integers(0, n_seeds, (chunk, n_seeds)) training-seed blocks. Gate 4 instead draws, per
  chunk: probes rng.integers(0, 4, (chunk, 4)); replicates rng.integers(0, 2, (chunk, n_cells, 2)) within each
  (K, m) cell; batch pairs rng.integers(0, n_pairs, (chunk, n_mix, 4, n_pairs)) within each mix and probe slot.
  A draw whose statistic is undefined (zero rank variance) counts as 0 for the percentile interval; the count of
  such draws is reported.
K_hat: each probe's log L1 norm is z-scored with that probe's mean and standard deviation (ddof 0) over all
  embedded tasks of the family; a task's K_hat is the mean of its z-scores over the (resampled) probes.
Gate 5 ratio: per task, the geometric mean over (resampled) probes of L1(25% noise) / L1(0%); the statistic is
  the median over (resampled) tasks.
"""
import numpy as np
from scipy.stats import rankdata


# ---------------------------------------------------------------- correlation

def rank_avg(x, axis=-1):
    return rankdata(np.asarray(x, np.float64), method='average', axis=axis)


def _pearson_rows(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (a * b).sum(-1) / den
    return np.where(den > 1e-9, r, np.nan)


def spearman(x, y):
    """Spearman rho with average ranks (rows of 2-D inputs are separate samples). NaN if a rank vector is constant."""
    return _pearson_rows(rank_avg(x), rank_avg(y))


def _residualize(v, Z):
    """Residuals of v on [1, Z] by ordinary least squares, row by row (v: (..., n); Z: (..., n, k))."""
    ones = np.ones(Z.shape[:-1] + (1,))
    A = np.concatenate([ones, Z], -1)
    At = np.swapaxes(A, -1, -2)
    coef = np.linalg.solve(At @ A, At @ v[..., None])
    return v - (A @ coef)[..., 0]


def partial_spearman(x, y, Z):
    """Rank every variable, regress rank(x) and rank(y) on the covariate ranks with an intercept, Pearson of the
    residuals. Z: (..., n, k) covariates."""
    rx, ry = rank_avg(x), rank_avg(y)
    rZ = np.stack([rank_avg(Z[..., j]) for j in range(Z.shape[-1])], -1)
    return _pearson_rows(_residualize(rx, rZ), _residualize(ry, rZ))


def kendall_w(M):
    """Kendall's W for m raters (rows) by n items (columns), average ranks, with the tie correction."""
    M = np.asarray(M, np.float64)
    m, n = M.shape
    R = rank_avg(M, axis=1)
    S = ((R.sum(0) - m * (n + 1) / 2.0) ** 2).sum()
    Tc = 0.0
    for row in M:
        _, cnt = np.unique(row, return_counts=True)
        Tc += (cnt ** 3 - cnt).sum()
    return float(12.0 * S / (m * m * (n ** 3 - n) - m * Tc))


# ---------------------------------------------------------------- T*

def t_star(Ts, eps, eps_star):
    """First T with excess loss <= eps_star, log-log interpolation between adjacent checkpoints. Right-censored
    (never reached) = +inf; reached at the first checkpoint = that checkpoint."""
    Ts, eps = np.asarray(Ts, float), np.asarray(eps, float)
    below = np.flatnonzero(eps <= eps_star)
    if len(below) == 0:
        return np.inf
    i = int(below[0])
    if i == 0:
        return float(Ts[0])
    x0, x1 = np.log2(Ts[i - 1]), np.log2(Ts[i])
    y0, y1 = np.log2(eps[i - 1]), np.log2(eps[i])
    return float(2 ** (x0 + (np.log2(eps_star) - y0) * (x1 - x0) / (y1 - y0)))


def tstar_table(rows, target_ids, seeds, eps_star, alpha=1.0):
    """(n_targets, n_seeds) T* at alpha from metrics rows (dicts with seed, target_id, alpha, T, excess_bits)."""
    curves = {}
    for r in rows:
        if abs(float(r['alpha']) - alpha) > 1e-9:
            continue
        curves.setdefault((r['target_id'], int(r['seed'])), []).append((int(r['T']), float(r['excess_bits'])))
    out = np.zeros((len(target_ids), len(seeds)))
    for i, t in enumerate(target_ids):
        for j, s in enumerate(seeds):
            cv = sorted(curves[(t, int(s))])
            out[i, j] = t_star([a for a, _ in cv], [b for _, b in cv], eps_star)
    return out


# ---------------------------------------------------------------- K_hat and bootstrap machinery

def z_constants(logn):
    """logn: (n_probes, n_tasks) over all embedded tasks of a family -> (mean, sd) per probe (ddof 0)."""
    return logn.mean(1), logn.std(1)


def khat(logn, mu, sd, probe_idx=None):
    """K_hat per task: mean over probes (optionally resampled, shape (..., 4)) of z-scored log norms."""
    z = (logn - mu[:, None]) / sd[:, None]
    if probe_idx is None:
        return z.mean(0)
    return z[probe_idx].mean(-2)


def boot_index_chunks(c, gate, fam_index, level_sizes=None, n_seeds=None, g4=None):
    """Yield per-chunk index arrays following the recipe in the module docstring."""
    rng = np.random.default_rng([c['analysis_seed'], gate, fam_index])
    B, ch = c['bootstrap_draws'], c['bootstrap_chunk']
    for s in range(0, B, ch):
        n = min(ch, B - s)
        if gate == 4:
            n_cells, n_mix, n_pairs = g4
            pr = rng.integers(0, 4, (n, 4))
            rep = rng.integers(0, 2, (n, n_cells, 2))
            pairs = rng.integers(0, n_pairs, (n, n_mix, 4, n_pairs))
            yield dict(probes=pr, reps=rep, pairs=pairs)
            continue
        lv = [rng.integers(0, k, (n, k)) for k in level_sizes]
        pr = rng.integers(0, 4, (n, 4))
        out = dict(levels=lv, probes=pr)
        if gate == 3:
            out['seeds'] = rng.integers(0, n_seeds, (n, n_seeds))
        yield out


def _task_index(levels, offsets):
    return np.concatenate([lv + off for lv, off in zip(levels, offsets)], axis=1)


def ci(draws, level=95.0):
    d = np.where(np.isnan(draws), 0.0, draws)
    lo, hi = np.percentile(d, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi), int(np.isnan(draws).sum())


def _levels(level_of):
    """Tasks must be ordered by level: returns (sizes, offsets)."""
    lv = np.asarray(level_of)
    assert np.all(np.diff(lv) >= 0), 'tasks must be sorted by K level'
    vals = np.unique(lv)
    sizes = [int((lv == v).sum()) for v in vals]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    return sizes, offsets


# ---------------------------------------------------------------- gates

def gate12(c, fam_index, logn, K, cov, mu, sd):
    """Gate 1 (Spearman of K_hat with K) and Gate 2 (partial, covariates `cov` (n, k)) on embedded tasks (sorted by K).
    Returns dict with point estimates and bootstrap draws."""
    sizes, offsets = _levels(K)
    kh = khat(logn, mu, sd)
    p1 = float(spearman(kh, K))
    p2 = float(partial_spearman(kh, K, cov))
    d1, d2 = [], []
    for ch in boot_index_chunks(c, 1, fam_index, level_sizes=sizes):
        ti = _task_index(ch['levels'], offsets)
        z = (logn - mu[:, None]) / sd[:, None]
        khb = z[ch['probes']].mean(1)                 # (n, n_tasks) after probe resampling
        khb = np.take_along_axis(khb, ti, 1)          # resampled tasks
        Kb = K[ti]
        d1.append(spearman(khb, Kb))
        d2.append(partial_spearman(khb, Kb, cov[ti]))
    return dict(rho=p1, rho_draws=np.concatenate(d1), partial=p2, partial_draws=np.concatenate(d2))


def gate3(c, fam_index, logn_t, K_t, mu, sd, tstar_seeds):
    """Gate 3: Spearman of K_hat with median-over-seeds T* on the targets (sorted by K); +inf ranks highest."""
    sizes, offsets = _levels(K_t)
    kh = khat(logn_t, mu, sd)
    ts = np.median(tstar_seeds, axis=1)
    p = float(spearman(kh, ts))
    d = []
    z = (logn_t - mu[:, None]) / sd[:, None]
    for ch in boot_index_chunks(c, 3, fam_index, level_sizes=sizes, n_seeds=tstar_seeds.shape[1]):
        ti = _task_index(ch['levels'], offsets)
        khb = np.take_along_axis(z[ch['probes']].mean(1), ti, 1)
        tsb = np.stack([np.median(tstar_seeds[ti[i]][:, ch['seeds'][i]], axis=1) for i in range(len(ti))])
        d.append(spearman(khb, tsb))
    return dict(rho=p, rho_draws=np.concatenate(d), tstar=ts)


def gate4(c, fam_index, dist, H, K, cell, rep):
    """Gate 4 on coverage mixes: dist (n_mix, 4 probes, n_pairs) batch-pair distances; cell (n_mix,) cell index with
    replicate rep (n_mix,) in {0, 1}. div = mean over probes of the mean pair distance."""
    n_mix, n_pr, n_pairs = dist.shape
    cells = np.unique(cell)
    lookup = np.zeros((len(cells), 2), int)
    for i in range(n_mix):
        lookup[cell[i], rep[i]] = i
    div = dist.mean(2).mean(1)
    rH, rK = float(spearman(div, H)), float(spearman(div, K))
    dH, dG = [], []
    for ch in boot_index_chunks(c, 4, fam_index, g4=(len(cells), n_mix, n_pairs)):
        n = len(ch['probes'])
        mix_idx = lookup[np.arange(len(cells))[None, :, None], ch['reps']].reshape(n, -1)   # (n, n_cells*2)
        # per draw: the chosen mixes, probes and pairs (pair slots follow the mix position in the draw)
        pr = ch['probes']
        sel = dist[mix_idx]                                            # (n, M, 4, P)
        sel = np.take_along_axis(sel, pr[:, None, :, None].repeat(sel.shape[1], 1).repeat(n_pairs, 3), 2)
        pairs = ch['pairs'][:, : sel.shape[1]]                         # (n, M, 4, P)
        sel = np.take_along_axis(sel, pairs, 3)
        divb = sel.mean(3).mean(2)
        Hb, Kb = H[mix_idx], K[mix_idx]
        a, b = spearman(divb, Hb), spearman(divb, Kb)
        dH.append(a)
        dG.append(a - np.abs(b))
    return dict(div=div, rho_H=rH, rho_K=rK, gap=rH - abs(rK), rho_H_draws=np.concatenate(dH),
                gap_draws=np.concatenate(dG))


def gate5(c, fam_index, logratio, K_n):
    """Gate 5: logratio (4 probes, n_noise tasks sorted by K) = log L1(25%) - log L1(0%)."""
    sizes, offsets = _levels(K_n)
    p = float(np.exp(np.median(logratio.mean(0))))
    d = []
    for ch in boot_index_chunks(c, 5, fam_index, level_sizes=sizes):
        ti = _task_index(ch['levels'], offsets)
        lr = np.take_along_axis(logratio[ch['probes']].mean(1), ti, 1)
        d.append(np.exp(np.median(lr, axis=1)))
    return dict(ratio=p, ratio_draws=np.concatenate(d))
