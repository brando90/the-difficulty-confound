"""Unit tests for dfa.py: minimization against brute force, known sizes, canonical form, exact covariates."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import dfa as A

C = json.loads((Path(__file__).parent / 'config.json').read_text())


def random_dfa(rng, n):
    return A.DFA(rng.integers(0, n, (n, 2)).astype(np.int64), rng.random(n) < 0.5)


def partition_sets(blocks):
    return {frozenset(np.flatnonzero(blocks == b).tolist()) for b in np.unique(blocks)}


def brute_equivalent(d1, d2):
    """Compare labels on every string of length < n1 + n2 by expanding the string tree level by level."""
    s1, s2 = np.zeros(1, np.int64), np.zeros(1, np.int64)
    for _ in range(d1.n + d2.n):
        if (d1.accept[s1] != d2.accept[s2]).any():
            return False
        s1 = np.concatenate([d1.trans[s1, 0], d1.trans[s1, 1]])
        s2 = np.concatenate([d2.trans[s2, 0], d2.trans[s2, 1]])
    return True


def test_hopcroft_equals_moore_and_equivalent():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        d = A.reachable(random_dfa(rng, int(rng.integers(2, 11))))
        assert partition_sets(A.hopcroft_partition(d)) == partition_sets(A.moore_partition(d))
        h = A.hopcroft_minimize(d)
        assert h.key() == A.moore_minimize(d).key()
        assert A.equivalent(h, d) and brute_equivalent(h, d)


def test_inequivalent_detected():
    rng = np.random.default_rng(2)
    found = 0
    for _ in range(200):
        d1, d2 = A.hopcroft_minimize(random_dfa(rng, 5)), A.hopcroft_minimize(random_dfa(rng, 5))
        assert A.equivalent(d1, d2) == brute_equivalent(d1, d2) == (d1.key() == d2.key())
        found += d1.key() != d2.key()
    assert found > 100


def make(trans, accept):
    return A.DFA(np.array(trans, np.int64), np.array(accept, bool))


def test_known_minimal_sizes():
    cases = [
        (make([[0, 0]], [1]), 1),                                                    # all strings
        (make([[1, 0], [0, 1]], [1, 0]), 2),                                          # even number of a's
        (make([[1, 0], [1, 2], [1, 0]], [0, 0, 1]), 3),                               # ends with "ab"
        (make([[1, 1], [2, 2], [3, 3], [0, 0]], [1, 0, 0, 0]), 4),                    # length mod 4 == 0
        (make([[1, 2], [0, 3], [3, 0], [2, 1]], [1, 0, 0, 0]), 4),                    # even a's and even b's
        (make([[1, 0], [0, 1], [3, 2], [2, 3]], [1, 0, 1, 0]), 2),                    # redundant copy collapses
    ]
    for d, n in cases:
        assert A.hopcroft_minimize(d).n == n
        assert A.moore_minimize(d).n == n


def test_canonical_form_invariant_under_relabeling():
    rng = np.random.default_rng(3)
    for _ in range(200):
        d = random_dfa(rng, 8)
        perm = np.r_[0, 1 + rng.permutation(7)]  # keep start at 0
        inv = np.argsort(perm)
        d2 = A.DFA(perm[d.trans[inv]], d.accept[inv])
        assert A.canonical_form(d).key() == A.canonical_form(d2).key()


@pytest.mark.parametrize('n', [4, 8, 16, 32])
def test_sampler_exact_size(n):
    rng = np.random.default_rng(n)
    for _ in range(5):
        d, draws = A.sample_minimal_dfa(rng, n)
        assert d.n == n and A.moore_minimize(d).n == n and draws >= 1
        assert A.canonical_form(d).key() == d.key()


def test_covariates_match_monte_carlo():
    rng = np.random.default_rng(4)
    lmin, lmax = C['length_min'], C['length_max']
    for i in range(20):
        d, _ = A.sample_minimal_dfa(rng, [4, 8, 16, 32][i % 4])
        mc = A.monte_carlo_covariates(d, np.random.default_rng(100 + i), 20000, lmin, lmax)
        acc, se = mc['acceptance']
        assert abs(A.acceptance_rate(d, lmin, lmax) - acc) < 4 * se + 1e-12
        sens, se = mc['sensitivity']
        assert abs(A.sensitivity(d, lmin, lmax) - sens) < 4 * se + 1e-12
        # visitation: Monte Carlo state counts over positions 0..l
        L = np.random.default_rng(200 + i).integers(lmin, lmax + 1, 20000)
        bits = np.random.default_rng(300 + i).integers(0, 2 ** 32, 20000, dtype=np.uint64)
        state = np.zeros(20000, np.int64)
        counts = np.bincount(state, minlength=d.n).astype(float)
        for p in range(lmax):
            c = ((bits >> np.uint64(p)) & np.uint64(1)).astype(np.int64)
            state = np.where(p < L, d.trans[state, c], state)
            counts += np.bincount(state[p < L], minlength=d.n)
        freq = counts / counts.sum()
        # visit_freq averages per-length totals, while pooled counts weight longer strings more: compare the pooled form
        p_t = A.state_distributions(d, lmax)
        pooled = sum(p_t[: l + 1].sum(0) for l in range(lmin, lmax + 1))
        pooled /= pooled.sum()
        assert np.abs(freq - pooled).max() < 0.01
        assert abs(A.visit_freq(d, lmin, lmax).sum() - 1) < 1e-12


def test_complement_detection():
    rng = np.random.default_rng(5)
    d, _ = A.sample_minimal_dfa(rng, 8)
    comp = A.canonical_form(A.complement(d))
    assert comp.n == 8 and not A.equivalent(d, comp)
    assert A.equivalent(A.complement(comp), d)


def test_k_bits():
    assert [A.k_bits(n) for n in C['n_levels']] == [20, 56, 144, 352]


def test_tail_exponent_power_law():
    f = np.arange(1, 33, dtype=float) ** -1.5
    assert math.isclose(A.visit_tail_exponent(f / f.sum()), -1.5, abs_tol=1e-9)
