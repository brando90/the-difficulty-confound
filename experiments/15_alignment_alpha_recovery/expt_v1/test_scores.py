"""Unit tests 1-6 and 10 of the PROMPT: Eq. 2, min-max, compression, stripping, frozen gradient, Task2Vec fixed head,
determinism. Run: python -m pytest -q test_scores.py
"""
import gzip  # noqa: F401  (gzip is what scores.ncd uses)
import os

import numpy as np
import pytest
import torch

import common as C
import scores as S

c = C.load_config()


# 1. Eq. 2 -----------------------------------------------------------------------------------------------------------

def test_eq2_by_hand():
    t = np.array([3.0, 5.0, 2.0, 4.0])
    s = np.array([0.5, 1.0, 0.0, 0.25])
    assert abs(S.eq2_exact(t, s) - (1.5 + 5 + 0 + 1) / 14) < 1e-12


def test_eq2_affine_equal_lengths():
    rng = np.random.default_rng(0)
    sT, sD = rng.random(1000), rng.random((3, 1000)) * 0.5
    t = np.full(1000, 7.0)
    mT, mD = sT.mean(), sD.mean()
    for a in (0.05, 0.15, 0.4, 1.0):
        nT = round(a * 1200)
        n = np.array([nT] + [(1200 - nT) / 3] * 3)
        mts = np.r_[np.mean(t * sT), [np.mean(t * x) for x in sD]]
        mt = np.full(4, 7.0)
        v = S.eq2_stratified(n, mts, mt)
        assert abs(v - (mD + (mT - mD) * (nT / 1200))) < 1e-12


def test_stratified_equals_exact_on_whole_file():
    rng = np.random.default_rng(1)
    sizes = [50, 70, 30]
    t = rng.integers(10, 26, sum(sizes)).astype(float)
    s = rng.random(sum(sizes))
    # the multiset: every unit of every task once (n_u = file size)
    mts, mt = S.block_means(t, s, sizes)
    assert abs(S.eq2_stratified(sizes, mts, mt) - S.eq2_exact(t, s)) < 1e-12


# 2. Min-max ---------------------------------------------------------------------------------------------------------

def test_minmax_range_and_constant():
    x = S.minmax(np.array([3.0, -1.0, 2.0, 7.0]))
    assert x.min() == 0.0 and x.max() == 1.0 and np.all((x >= 0) & (x <= 1))
    with pytest.raises(ValueError, match='constant'):
        S.minmax(np.full(5, 2.0))


def test_minmax_or_zero_constant_pool():
    # Deviation 2: a score constant over the pool maps to 0; otherwise identical to minmax
    assert np.array_equal(S.minmax_or_zero(np.full(5, 1.0)), np.zeros(5))
    x = np.array([3.0, -1.0, 2.0, 7.0])
    assert np.array_equal(S.minmax_or_zero(x), S.minmax(x))
    assert np.array_equal(S.minmax_or_zero(np.array([0.5, 1.0]), 1.0, 1.0), np.zeros(2))


# 3. Compression -----------------------------------------------------------------------------------------------------

def test_ncd_self_and_random():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 256, 1024, dtype=np.uint8).tobytes()
    y = rng.integers(0, 256, 1024, dtype=np.uint8).tobytes()
    assert S.ncd(c, x, x) < 0.1  # gzip header and the back-reference cost a few bytes
    assert S.ncd(c, x, y) > 0.95


def test_zstd_dictionary_only_references():
    refs = [(('ref', 'T', i), bytes([97 + i % 2]) * 12 + b'>Y') for i in range(400)]
    S.train_zstd_dictionary(c, refs, 1)
    with pytest.raises(AssertionError):
        S.train_zstd_dictionary(c, refs + [(('pool', 'T', 0), b'abab>Y')], 1)


# 4. Stripping -------------------------------------------------------------------------------------------------------

def _dfa_raw(n=50):
    rng = np.random.default_rng(3)
    return dict(bits=rng.integers(0, 2 ** 24, n).astype(np.uint32), lens=rng.integers(8, 25, n).astype(np.uint8),
                y=rng.integers(0, 2, n))


def _dsl_raw(n=50):
    rng = np.random.default_rng(4)
    return dict(x=rng.integers(0, 1024, n).astype(np.uint16), yw=rng.integers(0, 1024, n).astype(np.uint16))


@pytest.mark.parametrize('fam,raw,tok', [('dfa', _dfa_raw(), 31), ('dsl', _dsl_raw(), 39)])
def test_stripping(fam, raw, tok):
    lo = c['serialization']['task_token_first_byte']
    s = C.serialize(c, fam, raw)
    k = C.serialize(c, fam, raw, token=tok)
    for a, b in zip(s, k):
        assert max(a) < lo
        assert b == bytes([lo + tok]) + a
    assert all(len(a) == t for a, t in zip(s, C.n_tokens(fam, raw)))


# 5. Frozen gradient -------------------------------------------------------------------------------------------------

def _double_spec(fam):
    import task2vec as t2v
    C.mods(c)
    spec = t2v.Spec(C.load_probe(c, fam, c['probes'][0]), fam)
    spec.body = {k: v.double() for k, v in spec.body.items()}
    spec.head_w, spec.head_b = spec.head_w.double(), spec.head_b.double()
    return spec


@pytest.mark.parametrize('fam', ['dfa', 'dsl'])
def test_frozen_gradient_finite_difference(fam):
    spec = _double_spec(fam)
    names = S.grad_param_names(spec)
    assert not any(n.startswith(('tok', 'pos', 'ln_f')) for n in names)
    assert any(n.startswith('blocks.') for n in names) and 'head.weight' in names
    raw = _dfa_raw(3) if fam == 'dfa' else _dsl_raw(3)
    ex = C.probe_input(c, fam, raw)
    prm = S.sel_params(spec)
    _, flat = next(S.per_example_flat_grads(spec, ex, prm=prm))
    rng = np.random.default_rng(5)
    sizes = [prm[k].numel() for k in sorted(prm)]
    offs = np.r_[0, np.cumsum(sizes)]
    X = torch.from_numpy(ex['X'][:1])
    last = None if ex['last'] is None else torch.from_numpy(ex['last'][:1])
    y = torch.from_numpy(ex['y'][:1])
    checked = 0
    for _ in range(12):
        j = int(rng.integers(0, offs[-1]))
        ki = int(np.searchsorted(offs, j, side='right') - 1)
        k = sorted(prm)[ki]
        g = float(flat[0, j])
        if abs(g) < 1e-6:
            continue
        h = 1e-6
        p1 = {kk: v.clone() for kk, v in prm.items()}
        p2 = {kk: v.clone() for kk, v in prm.items()}
        p1[k].view(-1)[j - offs[ki]] += h
        p2[k].view(-1)[j - offs[ki]] -= h
        fd = (S._loss(spec, p1, X, last, y) - S._loss(spec, p2, X, last, y)).item() / (2 * h)
        assert abs(fd - g) <= 1e-4 * abs(g) + 1e-9, (k, fd, g)
        checked += 1
    assert checked >= 5
    one = dict(X=ex['X'][:1], y=ex['y'][:1], last=None if ex['last'] is None else ex['last'][:1])
    cos = S.per_example_grad_cos(spec, one, flat[:1].numpy())
    assert abs(cos[0, 0] - 1.0) < 1e-12


# 6. Task2Vec fixed head: Fisher linear in the mix --------------------------------------------------------------------

def test_fixed_head_fisher_linear():
    spec = _double_spec('dfa')
    prm = S.sel_params(spec)
    names = sorted(prm)
    A = C.probe_input(c, 'dfa', _dfa_raw(6))
    B = C.probe_input(c, 'dfa', dict(_dfa_raw(12), bits=_dfa_raw(12)['bits'] ^ 5))
    B = {k: (None if v is None else v[6:12]) for k, v in B.items()}

    def fisher(ex):
        """Exact diagonal Fisher, expectation over the model's labels, averaged over examples."""
        X = torch.from_numpy(ex['X'])
        last = torch.from_numpy(ex['last'])
        tot = None
        for i in range(len(X)):
            z = S._forward_logits(spec, prm, X[i:i + 1], last[i:i + 1])
            p = torch.softmax(z, -1)[0].detach()
            for lab in range(2):
                pp = {k: v.clone().requires_grad_(True) for k, v in prm.items()}
                lp = torch.log_softmax(S._forward_logits(spec, pp, X[i:i + 1], last[i:i + 1]), -1)[0, lab]
                g = torch.autograd.grad(lp, [pp[k] for k in names])
                f = p[lab] * torch.cat([x.reshape(-1) for x in g]) ** 2
                tot = f if tot is None else tot + f
        return tot / len(X)

    width = max(A['X'].shape[1], B['X'].shape[1])
    pad = lambda e: dict(X=np.pad(e['X'], ((0, 0), (0, width - e['X'].shape[1]))), last=e['last'], y=e['y'])
    A, B = pad(A), pad(B)
    AB = {k: np.concatenate([A[k], B[k]]) for k in A}
    fa, fb, fab = fisher(A), fisher(B), fisher(AB)
    assert torch.max(torch.abs(fab - 0.5 * (fa + fb))) <= 1e-6 * torch.max(torch.abs(fab))


# 10. Determinism ----------------------------------------------------------------------------------------------------

def test_determinism():
    torch.set_num_threads(1)
    raw = _dfa_raw(40)
    xs = C.serialize(c, 'dfa', raw)
    refs = [(('ref', 'T', i), b) for i, b in enumerate(C.serialize(c, 'dfa', _dfa_raw(400)))]
    d1 = S.train_zstd_dictionary(c, refs, 7)
    d2 = S.train_zstd_dictionary(c, refs, 7)
    assert d1 == d2
    a1, a2 = S.score_a(c, d1, xs), S.score_a(c, d2, xs)
    assert np.array_equal(a1[0], a2[0])
    import task2vec as t2v
    model = C.load_probe(c, 'dfa', c['probes'][0])
    spec = t2v.Spec(model, 'dfa')
    ex = C.probe_input(c, 'dfa', raw)
    assert np.array_equal(S.pooled_features(model, 'dfa', ex), S.pooled_features(model, 'dfa', ex))
    G = S.mean_gradient(spec, ex)[None]
    assert np.array_equal(S.per_example_grad_cos(spec, ex, G), S.per_example_grad_cos(spec, ex, G))
    c14 = C.mods(c)['e14']['cfg']
    e1 = t2v.embed_batch_e3(c14, spec, ex, (15, 'test'))['f']
    e2 = t2v.embed_batch_e3(c14, spec, ex, (15, 'test'))['f']
    assert np.array_equal(np.asarray(e1), np.asarray(e2))


def test_b1_label_blind():
    """The probe input carries no label: permuting or dropping labels cannot change b1 (why b1_perm = b1_inp = b1)."""
    raw = _dfa_raw(30)
    e1 = C.probe_input(c, 'dfa', raw)
    e2 = C.probe_input(c, 'dfa', dict(raw, y=raw['y'][::-1]))
    assert np.array_equal(e1['X'], e2['X'])
    rd = _dsl_raw(30)
    assert np.array_equal(C.probe_input(c, 'dsl', rd)['X'], C.probe_input(c, 'dsl', dict(rd, yw=rd['yw'][::-1]))['X'])


if __name__ == '__main__':
    os.sys.exit(pytest.main([__file__, '-q']))
