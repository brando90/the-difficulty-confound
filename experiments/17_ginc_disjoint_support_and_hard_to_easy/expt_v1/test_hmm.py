"""Experiment 17 additions to the HMM language: token regions and the far (shifted) copies."""
import numpy as np

import hmm as H


def test_region_candidate_stays_in_region():
    rng = np.random.default_rng(0)
    for n in (4, 32):
        h = H.sample_candidate_region(rng, n, 64, 6, 0, 32)
        assert h['V'] == 64 and h['etok'].min() >= 0 and h['etok'].max() < 32
        A, O = H.matrices(h)
        assert np.allclose(A.sum(1), 1) and np.allclose(O.sum(1), 1) and O[:, 32:].sum() == 0


def test_shift_preserves_law_up_to_relabeling():
    rng = np.random.default_rng(1)
    h = H.sample_candidate_region(rng, 8, 64, 6, 0, 32)
    g = H.shift_tokens(h, 32)
    assert g['etok'].min() >= 32 and np.array_equal(g['trans'], h['trans']) and np.array_equal(g['emit'], h['emit'])
    docs = H.sample_docs(h, 5, 3, range(16), 128)
    docs_g = H.sample_docs(g, 5, 3, range(16), 128)
    assert np.array_equal(docs + 32, docs_g)
    assert np.allclose(H.forward_logprobs(h, docs), H.forward_logprobs(g, docs_g))
    assert H.process_rank(h)[0] == H.process_rank(g)[0]
    # a far concept assigns probability zero to region-A documents
    assert np.isinf(H.forward_logprobs(g, docs)[:, 0]).all()


def test_mixture_with_impossible_components():
    rng = np.random.default_rng(2)
    h = H.sample_candidate_region(rng, 4, 64, 6, 0, 32)
    far = [H.shift_tokens(H.sample_candidate_region(rng, 4, 64, 6, 0, 32), 32) for _ in range(3)]
    docs = H.sample_docs(h, 1, 0, range(8), 128)
    a = 0.05
    m = H.mixture_logprobs([h] + far, [a] + [(1 - a) / 3] * 3, docs)
    o = H.forward_logprobs(h, docs)
    # the first token identifies the target exactly: eps_id = log2(1/alpha) at t = 1, zero afterwards
    assert np.allclose(m[:, 0] - o[:, 0], np.log2(1 / a)) and np.allclose(m[:, 1:], o[:, 1:])
