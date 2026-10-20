"""Unit tests 8 and 9 of the PROMPT: calibration and leakage, the slope regression, laws.py versus independent_fit.py,
leave-one-mix-out. Run: python -m pytest -q test_laws.py
"""
import numpy as np
import pytest

import common as C
import independent_fit as IFX
import laws as LW

c = C.load_config()


# 8. Calibration and leakage -----------------------------------------------------------------------------------------

def test_isotonic_monotone_and_matches_independent():
    rng = np.random.default_rng(0)
    x = rng.random(40)
    y = np.clip(x + rng.normal(0, 0.2, 40), 0, 1)
    x[5] = x[6]  # a tie
    a, b = LW.Isotonic(x, y), IFX.Isotonic(x, y)
    g = np.linspace(-0.2, 1.2, 301)
    assert np.all(np.diff(a(g)) >= -1e-15)
    assert np.max(np.abs(a(g) - b(g))) < 1e-10


def test_calibrator_never_reads_heldout():
    with pytest.raises(AssertionError, match='held-out'):
        LW.Isotonic([0.1, 0.2, 0.3], [0.05, 0.4, 1.0], rows=[0, 1, 2], heldout_rows=[2, 3])
    LW.Isotonic([0.1, 0.2], [0.05, 0.4], rows=[0, 1], heldout_rows=[2, 3])


def test_slope_recovers_planted():
    rng = np.random.default_rng(1)
    K = np.repeat([4, 8, 16, 32], 4).astype(float)
    alphas = np.array([0.05, 0.15, 0.4, 1.0])
    y = (0.2 + 0.6 * alphas)[None, :] - 0.05 * np.log2(K)[:, None] + rng.normal(0, 0.01, (16, 4))
    x = np.repeat(np.log2(K), 4)
    g = np.tile(np.arange(4), 16)
    s1 = LW.fe_slope(y.ravel(), x, g)
    s2 = IFX.fe_slope(y.ravel(), x, g)
    assert abs(s1 + 0.05) < 0.005
    assert abs(s1 - s2) < 1e-10


def test_spearman_ties_match():
    x = np.array([1, 2, 2, 3, 4, 4, 4, 5.0])
    y = np.array([0.1, 0.1, 0.3, 0.2, 0.5, 0.4, 0.6, 0.9])
    assert abs(LW.spearman(x, y) - IFX.spearman(x, y)) < 1e-12
    r = LW.spearman_rows(np.stack([x, x[::-1]]), np.stack([y, y]))
    assert abs(r[0] - LW.spearman(x, y)) < 1e-12 and abs(r[1] - LW.spearman(x[::-1], y)) < 1e-12


# 9. Laws ------------------------------------------------------------------------------------------------------------

def _synthetic_rows(cu, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for K, Kb in ((4, 20), (8, 56), (16, 144), (32, 352)):
        for a in (0.05, 0.15, 0.4, 1.0):
            for T in cu['T_checkpoints']:
                eps = 2.0 ** (3.0 + 0.3 * np.log2(Kb) - 0.4 * (0.8 * np.log2(a) + np.log2(T)) * 0.5 +
                              rng.normal(0, 0.05))
                rows.append(dict(seed=1, target_id=f'k{K}', K=K, K_bits=Kb, alpha=a, T=T, excess_bits=eps,
                                 mix='m1' if a == 0.05 else 'm2' if a == 0.15 else 'm3' if a == 0.4 else 'm4'))
    return rows


def test_laws_vs_independent_fit():
    L, IF, cu = LW.upstream(c, 'dfa')
    rows = _synthetic_rows(cu)
    fit = [r for r in rows if LW.is_fit(r, cu, 'dfa')]
    ho = [r for r in rows if not LW.is_fit(r, cu, 'dfa')]
    a = LW.fit_and_score(L, cu, fit, ho, L.fit)
    b = LW.fit_and_score(L, cu, fit, ho, IF.fit)
    assert abs(a['cost'] - b['cost']) <= 1e-4 * abs(a['cost'])
    fa, fb = np.array(a['forecast']), np.array(b['forecast'])
    assert np.max(np.abs(fa - fb) / np.abs(fa)) <= 1e-4


def test_lomo_never_trains_on_left_out():
    L, IF, cu = LW.upstream(c, 'dfa')
    rows = _synthetic_rows(cu)
    Tm = max(r['T'] for r in rows)
    for m in ('m1', 'm2', 'm3', 'm4'):
        fit, ho = LW.lomo_split(rows, m, Tm)
        assert not any(r['mix'] == m for r in fit)
        assert not any(r['T'] == Tm for r in fit)
        assert all(r['mix'] == m or r['T'] == Tm for r in ho)
        assert len(fit) + len(ho) == len(rows)
