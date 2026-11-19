"""Unit tests 1-4 of PROMPT.md: arithmetic, dynamic program, sampler and formats."""
import numpy as np
from scipy.stats import chisquare

import carry

SHARES = [0.0084, 0.1956, 0.3341, 0.2331, 0.1235, 0.0587, 0.0272, 0.0125, 0.0069]  # PROTOCOL 2.3 (percent / 100)


def column_walk(a, b, digits):
    carry_in, run, best = 0, 0, 0
    for place in range(digits):
        carry_in = (a // 10 ** place % 10 + b // 10 ** place % 10 + carry_in) // 10
        run = run + 1 if carry_in else 0
        best = max(best, run)
    return best


def test_arithmetic():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 10 ** 8, 10 ** 5), rng.integers(0, 10 ** 8, 10 ** 5)
    assert all(carry.column_addition(int(x), int(y), 8) == x + y for x, y in zip(a, b))
    a3, b3 = np.meshgrid(np.arange(1000), np.arange(1000), indexing='ij')
    assert all(carry.column_addition(int(x), int(y), 3) == x + y for x, y in zip(a3.ravel(), b3.ravel()))
    assert carry.carry_chain(np.array([123, 109, 199, 999]), np.array([234, 1, 1, 1]), 3).tolist() == [0, 1, 2, 3]
    walk = np.array([column_walk(int(x), int(y), 8) for x, y in zip(a, b)])
    assert np.array_equal(carry.carry_chain(a, b, 8), walk)


def brute_4_digits():
    counts = np.zeros(5, np.int64)
    b = np.arange(10 ** 4)
    for a in range(10 ** 4):
        counts += np.bincount(carry.carry_chain(np.full(10 ** 4, a), b, 4), minlength=5)
    return counts.tolist()


def test_dynamic_program():
    for d in (1, 2, 3):
        assert carry.stratum_counts(d) == carry.brute_counts(d)
    assert carry.stratum_counts(4) == brute_4_digits()
    c8 = carry.stratum_counts(8)
    assert sum(c8) == 10 ** 16
    assert [round(x / 1e16, 4) for x in c8] == SHARES


def test_sampler_exact_and_uniform():
    rng = np.random.default_rng(2)
    a3, b3 = np.meshgrid(np.arange(1000), np.arange(1000), indexing='ij')
    chain = carry.carry_chain(a3.ravel(), b3.ravel(), 3)
    for c in range(4):
        a, b = carry.sample_stratum(rng, c, 10 ** 6, 3)
        assert np.all(carry.carry_chain(a, b, 3) == c)
        members = np.flatnonzero(chain == c)
        observed = np.bincount(a * 1000 + b, minlength=10 ** 6)[members]
        assert observed.sum() == 10 ** 6
        assert chisquare(observed).pvalue > 0.001


def test_formats():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 10 ** 8, 1000), rng.integers(0, 10 ** 8, 1000)
    for fmt in ('plain', 'reversed'):
        tok = carry.encode(a, b, fmt)
        assert tok.shape == (1000, 29)
        da, db, s = carry.decode(tok, fmt)
        assert np.array_equal(da, a) and np.array_equal(db, b) and np.array_equal(s, a + b)
        places = carry.answer_place(fmt)
        digits = (np.asarray(a + b)[:, None] // 10 ** places) % 10
        assert np.array_equal(tok[:, carry.answer_positions()], digits)
    # Reversed output: the carry into column i is [a_{i-1} + b_{i-1} - s_{i-1} >= 9], a local function of seen tokens.
    a3, b3 = np.meshgrid(np.arange(1000), np.arange(1000), indexing='ij')
    a3, b3 = a3.ravel(), b3.ravel()
    s = a3 + b3
    carry_in = np.zeros_like(a3)
    for i in range(1, 4):
        col = a3 // 10 ** (i - 1) % 10 + b3 // 10 ** (i - 1) % 10 + carry_in
        carry_in = col // 10
        local = (a3 // 10 ** (i - 1) % 10 + b3 // 10 ** (i - 1) % 10 - s // 10 ** (i - 1) % 10) >= 9
        assert np.array_equal(carry_in.astype(bool), local)
