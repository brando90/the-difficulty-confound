"""Unit test 7 of the PROMPT: rewiring, exact agreement versus Monte Carlo, kappa, re-minimization.
Run: python -m pytest -q test_graded.py
"""
import numpy as np

import common as C
import graded as GR

c = C.load_config()
A = GR.A_()


def _random_min_dfa(seed, n):
    rng = np.random.default_rng(seed)
    return A.sample_minimal_dfa(rng, n)[0]


def test_rewire_changes_exactly_r():
    T = _random_min_dfa(1, 8)
    rng = np.random.default_rng(2)
    for r in (1, 2, 4):
        for _ in range(20):
            d = GR.rewire(T, r, rng)
            assert int((d.trans != T.trans).sum()) == r
            assert np.array_equal(d.accept, T.accept)
    d = GR.rewire(T, 'all', rng)
    assert d.n == T.n and np.array_equal(d.accept, T.accept)


def _uniform_mc(d1, d2, n, seed):
    rng = np.random.default_rng(seed)
    L = rng.integers(8, 25, n)
    bits = rng.integers(0, 2 ** 32, n, dtype=np.uint64) & ((np.uint64(1) << L.astype(np.uint64)) - np.uint64(1))
    return float((A.run_dfa(d1, bits, L) == A.run_dfa(d2, bits, L)).mean())


def test_exact_agreement_vs_mc():
    rng = np.random.default_rng(3)
    for k in range(3):
        T = _random_min_dfa(10 + k, 8)
        d = A.hopcroft_minimize(GR.rewire(T, 2, rng))
        ex = GR.agreement_exact(T, d, 8, 24)
        mc = _uniform_mc(T, d, 2 ** 16, 20 + k)
        assert abs(ex - mc) < 0.01, (ex, mc)
    T = _random_min_dfa(30, 8)
    assert abs(GR.agreement_exact(T, T, 8, 24) - 1.0) < 1e-12
    assert abs(GR.agreement_exact(T, A.complement(T), 8, 24)) < 1e-12


def test_kappa_balanced():
    for a in (0.3, 0.5, 0.8, 0.97):
        rho, a0 = GR.kappa(a, 0.5, 0.37)
        assert abs(a0 - 0.5) < 1e-12
        assert abs(rho - max(0.0, 2 * a - 1)) < 1e-12


def test_reminimized_K_brute_force():
    rng = np.random.default_rng(4)
    for k in range(5):
        T = A.DFA(rng.integers(0, 5, (5, 2)).astype(np.int64), rng.random(5) < 0.5)
        for r in (1, 2, 'all'):
            d = GR.rewire(T, r, rng)
            assert A.hopcroft_minimize(d).n == _brute_min_states(d)


def _brute_min_states(d):
    """Number of Myhill-Nerode classes of the reachable states, by comparing suffix behaviour on all strings up to
    length n (enough for an n-state automaton)."""
    reach = {0}
    frontier = [0]
    while frontier:
        s = frontier.pop()
        for a in range(2):
            t = int(d.trans[s, a])
            if t not in reach:
                reach.add(t)
                frontier.append(t)
    n = d.n
    words = [()]
    for L in range(1, n + 1):
        words += [tuple((w >> i) & 1 for i in range(L)) for w in range(2 ** L)]

    def sig(s):
        out = []
        for w in words:
            q = s
            for a in w:
                q = int(d.trans[q, a])
            out.append(bool(d.accept[q]))
        return tuple(out)

    return len({sig(s) for s in reach})
