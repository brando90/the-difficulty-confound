"""Unit test 8 of PROMPT.md: statistics against scipy and independent_stats.py, bootstrap reproducibility, T*.
Run: python -m pytest -q test_stats.py"""
import copy

import numpy as np
import pytest
from scipy import linalg, stats as ss

import independent_stats as I
import stats as S
import tasks14 as T


@pytest.fixture(scope='module')
def c():
    c = copy.deepcopy(T.load_config())
    c['bootstrap_draws'] = 600
    c['bootstrap_chunk'] = 250
    return c


def test_spearman_matches_scipy_and_independent():
    g = np.random.default_rng(0)
    for _ in range(50):
        x = g.integers(0, 6, 40).astype(float)          # ties on purpose
        y = x + g.normal(size=40)
        r = float(S.spearman(x, y))
        assert abs(r - ss.spearmanr(x, y).statistic) < 1e-12
        assert abs(r - float(I.spearman(x, y))) < 1e-12


def test_partial_spearman_matches_scipy_and_independent():
    g = np.random.default_rng(1)
    for _ in range(50):
        Z = np.column_stack([g.normal(size=60), g.integers(0, 4, 60)])
        x = Z[:, 0] + g.normal(size=60)
        y = Z[:, 1] + 0.5 * x + g.normal(size=60)
        r = float(S.partial_spearman(x, y, Z))
        rx, ry = ss.rankdata(x), ss.rankdata(y)
        A = np.column_stack([np.ones(60), ss.rankdata(Z[:, 0]), ss.rankdata(Z[:, 1])])
        ex = rx - A @ linalg.lstsq(A, rx)[0]
        ey = ry - A @ linalg.lstsq(A, ry)[0]
        assert abs(r - ss.pearsonr(ex, ey).statistic) < 1e-12
        assert abs(r - float(I.partial_spearman(x, y, Z))) < 1e-12


def test_kendall_w():
    g = np.random.default_rng(2)
    M = g.integers(0, 5, (4, 30)).astype(float)
    assert abs(S.kendall_w(M) - I.kendall_w(M)) < 1e-12
    same = np.tile(np.arange(10.0), (4, 1))
    assert abs(S.kendall_w(same) - 1.0) < 1e-12


def test_tstar_power_law_exact_and_censored():
    Ts = 2.0 ** np.arange(8, 17)
    eps = 37.0 * Ts ** -0.6
    e_star = 0.05
    exact = (37.0 / e_star) ** (1 / 0.6)
    assert Ts[0] < exact < Ts[-1]
    assert abs(S.t_star(Ts, eps, e_star) / exact - 1) < 1e-12
    assert abs(I.t_star(Ts, eps, e_star) / exact - 1) < 1e-12
    assert S.t_star(Ts, eps, 1e-9) == np.inf and I.t_star(Ts, eps, 1e-9) == np.inf
    assert S.t_star(Ts, eps, 100.0) == Ts[0]
    # +inf ranks above every finite value
    assert list(S.rank_avg([3.0, np.inf, 1.0])) == [2.0, 3.0, 1.0]


def _synthetic(g, sizes):
    K = np.concatenate([np.full(n, k) for k, n in zip([4, 8, 16, 32], sizes)]).astype(float)
    logn = 0.3 * np.log(K)[None] + g.normal(size=(4, len(K)))
    cov = np.column_stack([g.random(len(K)), g.random(len(K))])
    return K, logn, cov


def test_gates_match_independent_and_reproducible(c):
    g = np.random.default_rng(3)
    K, logn, cov = _synthetic(g, [6, 6, 6, 6])
    mu, sd = S.z_constants(logn)
    a = S.gate12(c, 0, logn, K, cov, mu, sd)
    b = I.gate12(c, 0, logn, K, cov)
    a2 = S.gate12(c, 0, logn, K, cov, mu, sd)
    for k in ('rho', 'partial'):
        assert abs(a[k] - b[k]) < 1e-10
        assert np.allclose(a[k + '_draws'], b[k + '_draws'], atol=1e-10, equal_nan=True)
        assert np.array_equal(a[k + '_draws'], a2[k + '_draws'], equal_nan=True)
    lo, hi, nn = S.ci(a['rho_draws'])
    lo2, hi2, nn2 = I.percentile_ci(b['rho_draws'])
    assert abs(lo - lo2) < 1e-10 and abs(hi - hi2) < 1e-10 and nn == nn2
    # Gate 3 with censored seeds
    tgt = np.r_[0:2, 6:8, 12:14, 18:20]
    ts = np.exp(g.normal(size=(len(tgt), 3)) + np.log(K[tgt])[:, None])
    ts[g.random(ts.shape) < 0.3] = np.inf
    a = S.gate3(c, 0, logn[:, tgt], K[tgt], mu, sd, ts)
    b = I.gate3(c, 0, logn[:, tgt], K[tgt], logn, ts)
    assert abs(a['rho'] - b['rho']) < 1e-10
    assert np.allclose(a['rho_draws'], b['rho_draws'], atol=1e-10, equal_nan=True)
    # Gate 5
    lr = g.normal(size=(4, 8)) * 0.2 + 0.1
    a = S.gate5(c, 0, lr, K[tgt])
    b = I.gate5(c, 0, lr, K[tgt])
    assert abs(a['ratio'] - b['ratio']) < 1e-10
    assert np.allclose(a['ratio_draws'], b['ratio_draws'], atol=1e-10)
    # Gate 4: 4 K levels x 5 m x 2 replicates, 4 probes, 66 pairs
    Kc = np.repeat([4, 8, 16, 32], 10).astype(float)
    m = np.tile(np.repeat([1, 2, 4, 8, 16], 2), 4)
    H = np.log2(m)
    cell = np.repeat(np.arange(20), 2)
    rep = np.tile([0, 1], 20)
    dist = 0.01 * H[:, None, None] + 0.005 * g.random((40, 4, 66))
    a = S.gate4(c, 0, dist, H, Kc, cell, rep)
    b = I.gate4(c, 0, dist, H, Kc, cell, rep)
    for k in ('rho_H', 'rho_K', 'gap'):
        assert abs(a[k] - b[k]) < 1e-10
    assert np.allclose(a['rho_H_draws'], b['rho_H_draws'], atol=1e-10, equal_nan=True)
    assert np.allclose(a['gap_draws'], b['gap_draws'], atol=1e-10, equal_nan=True)
