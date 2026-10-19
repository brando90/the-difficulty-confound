"""Unit tests 1-8 of PROMPT.md: forward algorithm, process rank, description length, entropy rate, mixture, alpha
schedule, common random numbers and eval/train disjointness."""
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import hmm as H

HERE = Path(__file__).resolve().parent


def rand_hmm(rng, n, V, q=6):
    """A random dense-ish HMM in the language when n >= 3 and V >= 4; for tiny cases build directly."""
    if n >= 3 and V >= 4:
        return H.sample_candidate(rng, n, V, q)
    succ = np.array([np.sort(rng.choice(n, min(3, n), replace=False)) for _ in range(n)])
    etok = np.array([np.sort(rng.choice(V, min(4, V), replace=False)) for _ in range(n)])
    trans = np.array([H.quantize(rng.dirichlet(np.ones(succ.shape[1])), q) for _ in range(n)])
    emit = np.array([H.quantize(rng.dirichlet(np.ones(etok.shape[1])), q) for _ in range(n)])
    return dict(n=n, V=V, q=q, succ=succ, trans=trans, etok=etok, emit=emit)


def brute_prob(h, x):
    A, O = H.matrices(h)
    n = h['n']
    tot = 0.0
    for path in itertools.product(range(n), repeat=len(x)):
        p = 1.0 / n
        for t, s in enumerate(path):
            p *= O[s, x[t]]
            if t + 1 < len(x):
                p *= A[s, path[t + 1]]
        tot += p
    return tot


def test_forward_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, V = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        h = rand_hmm(rng, n, V)
        A, O = H.matrices(h)
        x = H.sample_sequences(h, rng, 1, 6)
        cl = H.forward_logprobs(h, x)[0]
        assert abs(2.0 ** (-cl.sum()) - brute_prob(h, x[0])) <= 1e-12
        cl2 = H.forward_logspace(h, x)[0]
        assert np.max(np.abs(cl - cl2)) < 1e-9


def exact_hankel_rank(h):
    """Fraction rank of the Hankel matrix over all strings of length <= n-1 on each side."""
    A, O = H.int_matrices(h)
    n, V, q = h['n'], h['V'], h['q']
    Af = [[Fraction(int(a), 2 ** q) for a in r] for r in A]
    Of = [[Fraction(int(o), 2 ** q) for o in r] for r in O]
    strings = [()]
    for L in range(1, n):
        strings += list(itertools.product(range(V), repeat=L))

    def prob(s):
        v = [Fraction(1, n)] * n
        for x in s:
            v = [sum(v[i] * Of[i][x] * Af[i][j] for i in range(n)) for j in range(n)]
        return sum(v)
    Hm = [[prob(u + w) for w in strings] for u in strings]
    rank, rows = 0, [r[:] for r in Hm]
    cols = len(rows[0])
    for col in range(cols):
        piv = next((r for r in range(rank, len(rows)) if rows[r][col] != 0), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for r in range(len(rows)):
            if r != rank and rows[r][col] != 0:
                f = rows[r][col] / rows[rank][col]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


def test_process_rank_equals_brute_force_hankel():
    rng = np.random.default_rng(1)
    for _ in range(12):
        n, V = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        if n == 4 and V == 4:
            V = 3  # keeps the brute-force Hankel small
        h = rand_hmm(rng, n, V)
        assert H.process_rank(h)[0] == exact_hankel_rank(h)


def test_lumpable_detected():
    rng = np.random.default_rng(2)
    h = H.sample_candidate(rng, 6, 16, 6)
    # make states 4 and 5 identical in emissions and outgoing rows: an equivalent 5-state chain exists
    for k in ('succ', 'trans', 'etok', 'emit'):
        h[k][5] = h[k][4]
    assert H.process_rank(h)[0] < 6


def test_modp_and_fraction_agree():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(3, 9))
        h = H.sample_candidate(rng, n, int(rng.integers(4, 12)), 6)
        assert H.process_rank(h, exact_recheck_max_n=-1)[0] == H.process_rank_exact(h)


def test_description_length():
    rng = np.random.default_rng(4)
    for _ in range(1000):
        n = int(rng.choice([3, 4, 5, 8, 16, 32]))
        V = int(rng.choice([8, 64, 150]))
        h = H.sample_candidate(rng, n, V, 6)
        b = H.encode(h)
        assert len(b) == H.K_bits(n, V, 6)
        assert H.same(H.decode(b, n, V, 6), h)
    assert [H.K_bits(n, 64, 6) for n in (4, 8, 16, 32)] == [240, 504, 1056, 2208]
    assert [H.K_bits(n, 150, 6) for n in (8, 16, 32, 64, 128)] == [568, 1184, 2464, 5120, 10624]


def test_entropy_rate_visible_chain():
    rng = np.random.default_rng(5)
    n = 6
    h = H.sample_candidate(rng, n, 64, 6)
    h['etok'] = np.array([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3] for i in range(n)])
    h['emit'] = np.array([[61, 1, 1, 1]] * n)  # still a function of the state: tokens 4i..4i+3 reveal state i
    A, O = H.matrices(h)
    pi = H.stationary(h)
    with np.errstate(divide='ignore', invalid='ignore'):
        hA = -np.nansum(np.where(A > 0, A * np.log2(A), 0.0), 1)
        hO = -np.nansum(np.where(O > 0, O * np.log2(O), 0.0), 1)
    exact = float(pi @ (hA + hO))  # visible state: rate = H(next state | state) + H(token | state)
    est, se = H.entropy_rate(h, np.random.default_rng(6), 100, 10000, 100)
    assert abs(est - exact) <= 3 * se, (est, exact, se)


def test_mixture_properties():
    rng = np.random.default_rng(7)
    tgt = H.sample_candidate(rng, 8, 64, 6)
    ds = [H.sample_candidate(rng, n, 64, 6) for n in (4, 4, 8, 8, 16, 16, 32, 32)]
    docs = H.sample_docs(tgt, 99, 0, range(512), 128)
    oc = H.forward_logprobs(tgt, docs)
    assert np.array_equal(H.mixture_logprobs([tgt], [1.0], docs), oc)
    for a in (0.05, 0.15, 0.4):
        pr = [a] + [(1 - a) / 8] * 8
        mx = H.mixture_logprobs([tgt] + ds, pr, docs)
        reg = (mx - oc).sum(1)
        assert np.all(reg <= math.log2(1 / a) + 1e-9)
        assert reg.mean() >= -3 * reg.std(ddof=1) / math.sqrt(len(reg))
        mx2 = H.mixture_logspace([tgt] + ds, pr, docs[:16])
        assert np.max(np.abs(mx[:16] - mx2)) < 1e-9


def test_alpha_schedule_exact():
    for a in (0.05, 0.15, 0.4, 1.0):
        cids, js = H.schedule(13101, 0, [10, 11, 12, 13, 14, 15, 16, 17], a, 20000)
        cum = np.cumsum(cids == 0)
        m = np.arange(1, 20001)
        assert np.array_equal(cum, np.floor(m * a + 1e-9).astype(int))
        if a < 1:
            cnt = [int((cids == d).sum()) for d in range(10, 18)]
            assert max(cnt) - min(cnt) <= 1
        for c in np.unique(cids):
            assert np.array_equal(js[cids == c], np.arange((cids == c).sum()))


def test_common_random_numbers():
    rng = np.random.default_rng(8)
    cm = {0: H.sample_candidate(rng, 8, 64, 6), **{i: H.sample_candidate(rng, 4, 64, 6) for i in range(10, 18)}}
    docs = {}
    for a in (0.05, 0.4):
        cids, js = H.schedule(13101, 0, list(range(10, 18)), a, 400)
        toks = np.concatenate(list(H.stream_batches(cm, 13101, cids, js, 128, 8)))
        docs[a] = {(int(c), int(j)): toks[k] for k, (c, j) in enumerate(zip(cids, js))}
    common = set(docs[0.05]) & set(docs[0.4])
    assert len(common) > 30
    for key in common:
        assert np.array_equal(docs[0.05][key], docs[0.4][key])


def test_eval_train_disjoint_reduced():
    rng = np.random.default_rng(9)
    h = H.sample_candidate(rng, 4, 64, 6)
    ev = {H.doc_hash(d) for d in H.sample_docs(h, 20261020, 0, range(512), 128)}
    tr = {H.doc_hash(d) for s in (13101, 13202) for d in H.sample_docs(h, s, 0, range(2048), 128)}
    assert not ev & tr


def test_quantize():
    rng = np.random.default_rng(10)
    for _ in range(500):
        k = int(rng.integers(2, 6))
        v = H.quantize(rng.dirichlet(np.ones(k)), 6)
        assert v.sum() == 64 and v.min() >= 1
