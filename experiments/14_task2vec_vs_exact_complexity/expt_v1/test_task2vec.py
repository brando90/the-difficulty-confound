"""Unit tests 1-6 of PROMPT.md for task2vec.py and probe.py, plus the exact-rewrite checks (functional forward, fast
gradients versus direct autograd through the nn.Module). Run: python -m pytest -q test_task2vec.py"""
import copy
import warnings

import numpy as np
import pytest
import torch
from scipy.spatial import distance

import probe as P
import task2vec as V
import tasks14 as T

warnings.filterwarnings('ignore', message='.*[Ss]parse.*')
torch.set_num_threads(1)


@pytest.fixture(scope='module')
def env():
    c = T.load_config()
    roster = T.build_roster(c)[0]
    return c, roster


def _small(ex, n):
    return {k: (None if v is None else v[:n]) for k, v in ex.items()}


# ---------------------------------------------------------------- 1. Fisher correctness on logistic regression

def _logreg(n=2048, d=8, seed=0):
    g = np.random.default_rng(seed)
    X = g.normal(size=(n, d))
    theta = g.normal(size=d) * 0.7
    p1 = 1 / (1 + np.exp(-(X @ theta)))
    y = np.zeros(n, np.int64)
    y[g.permutation(n)[: n // 2]] = 1               # label-balanced: the class-balanced sampler is uniform
    return torch.from_numpy(X), torch.from_numpy(p1), torch.from_numpy(y)


def _rows(X, chunk=300):
    def it():
        for i in range(0, len(X), chunk):
            idx = torch.arange(i, min(i + chunk, len(X)))
            yield idx, {'w': X[idx].float()}
    return it()


def _cfg(c, draws_per_epoch=None, draws_per_example=None):
    c = copy.deepcopy(c)
    if draws_per_epoch:
        for e in ('E0', 'E1', 'E2'):
            c['estimators'][e]['draws_per_epoch'] = draws_per_epoch
    if draws_per_example:
        c['estimators']['E2']['draws_per_example'] = draws_per_example
    return c


def test_fisher_logistic_regression(env):
    c, _ = env
    X, p1, y = _logreg()
    n, d = X.shape
    fim = ((p1 * (1 - p1))[:, None] * X ** 2).mean(0).numpy()
    shapes = {'w': (d,)}
    # E2, 200 label draws per example: within 2% of the analytic diagonal FIM
    cc = _cfg(c, draws_per_example=200)
    _, _, _, labs = V.draw_plan(cc, 'dfa', y, p1, ('t', 1), ('E2',))
    w = ((p1[None] - labs.double()) ** 2).sum(0)
    e2 = V.fisher_from_rows(_rows(X), n, shapes, [dict(kind='per_example', weight=w, denom=n * 200)])[0]['w']
    assert np.max(np.abs(e2.numpy() / fim - 1)) < 0.02
    # E0 x batch_size: within 5% (many batches so the Monte Carlo error is small)
    cc = _cfg(c, draws_per_epoch=64 * 6000)
    B = cc['estimators']['E0']['batch_size']
    idx, nb, lab, _ = V.draw_plan(cc, 'dfa', y, p1, ('t', 2), ('E0', 'E1'))
    jobs = [dict(kind='batch', ex_idx=idx, batch_of=torch.arange(len(idx)) // B,
                 coef=(p1[idx] - lab[e].double()) / B, n_batches=nb) for e in ('E0', 'E1')]
    e0, e1 = V.fisher_from_rows(_rows(X), n, shapes, jobs)
    assert np.max(np.abs(e0['w'].numpy() * B / fim - 1)) < 0.05
    # E1 equals the hand computation of the squared batch-mean gradients
    hand = np.zeros(d)
    Xn, pn, yn = X.numpy(), p1.numpy(), y.numpy()
    for b in range(nb):
        s = idx[b * B:(b + 1) * B].numpy()
        g = ((pn[s] - yn[s])[:, None] * Xn[s]).mean(0)
        hand += g ** 2
    hand /= nb
    assert np.max(np.abs(e1['w'].numpy() / hand - 1)) < 1e-6
    # per-example form with true labels equals mean_x (y - p)^2 x_i^2
    pe = V.fisher_from_rows(_rows(X), n, shapes, [dict(kind='per_example', weight=(p1 - y) ** 2, denom=n)])[0]['w']
    ref = (((yn - pn) ** 2)[:, None] * Xn ** 2).mean(0)
    assert np.max(np.abs(pe.numpy() / ref - 1)) < 1e-6


# ---------------------------------------------------------------- 2. aggregation

class _Mods(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(5, 3)
        self.lin = torch.nn.Linear(3, 4)
        self.ln = torch.nn.LayerNorm(4)
        self.inner = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.GELU(), torch.nn.Linear(6, 2))
        self.head = torch.nn.Linear(2, 2)


def test_aggregation_matches_reference_order():
    m = _Mods()
    g = torch.Generator().manual_seed(0)
    grad2 = {k: torch.rand(v.shape, generator=g, dtype=torch.float64) for k, v in m.named_parameters()}
    f = V.aggregate(V._layout(m), grad2)
    ref = []
    for name, mod in m.named_modules():                 # reference extract_embedding, literally
        if mod is m.head:
            continue
        if hasattr(mod, 'weight') and isinstance(mod.weight, torch.nn.Parameter):
            g2 = grad2[name + '.weight'].numpy()
            ref.append(g2.reshape(g2.shape[0], -1).mean(axis=1))
    ref = np.concatenate(ref)
    assert f.shape == (5 + 4 + 4 + 6 + 2,)
    assert np.allclose(f, ref, rtol=0, atol=1e-15)


# ---------------------------------------------------------------- 3. distance

def test_symmetric_cosine_matches_reference():
    g = np.random.default_rng(1)
    for _ in range(100):
        a, b = g.random(50) + 1e-3, g.random(50) * g.random() * 10 + 1e-3
        h0, h1 = a / (a + b + 1e-8), b / (a + b + 1e-8)        # task_similarity.get_scaled_hessian
        assert abs(V.sym_cosine(a, b) - distance.cosine(h0, h1)) < 1e-12
        assert abs(V.sym_cosine(a, a)) < 1e-12


# ---------------------------------------------------------------- forward and gradient rewrites

@pytest.mark.parametrize('fam', ['dfa', 'dsl'])
def test_functional_forward_equals_module(env, fam):
    c, roster = env
    m = P.build_probe(c, fam, 7).eval()
    ex = _small(T.example_set(c, roster, fam, T.embedded_tasks(roster, fam)[3]), 256)
    spec = V.Spec(m, fam)
    a = spec.features(ex)
    b = P.features(m, fam, ex)
    assert float((a - b).abs().max()) < 1e-5
    if fam == 'dfa':      # right padding does not change the causal read-out at the last real character
        X, last = torch.from_numpy(ex['X']), torch.from_numpy(ex['last'])
        for L in torch.unique(last).tolist():
            g = torch.nonzero(last == L).view(-1)
            u = V.features_fn(spec, spec.body, X[g, :L + 1])          # unpadded rows, read at the end
            assert float((u - a[g]).abs().max()) < 1e-5


def _direct(c, model, fam, ex, seed_parts, ests):
    """E0/E1/E2/E3 by plain autograd through the nn.Module, with the draws of task2vec.py."""
    spec = V.Spec(model, fam)
    y, X = torch.from_numpy(ex['y']), torch.from_numpy(ex['X'])
    last = None if ex['last'] is None else torch.from_numpy(ex['last'])
    wn = [k for k, _ in model.named_parameters() if k.endswith('.weight') and not k.startswith('head.')]
    prm = dict(model.named_parameters())

    def gw(W, b, sel, yy, position_mean):
        with torch.no_grad():
            model.head.weight.copy_(W)
            model.head.bias.copy_(b)
        z = model.head(P.body(model, fam, X[sel], None if last is None else last[sel]))
        if position_mean:
            loss = torch.nn.functional.cross_entropy(z.reshape(-1, 2), yy.reshape(-1))
        else:
            loss = P.per_example_nll(fam, z, yy).mean()
        return dict(zip(wn, torch.autograd.grad(loss, [prm[k] for k in wn])))

    def acc_sq(pieces, denom):
        a = {k: 0 for k in wn}
        for g in pieces:
            for k in wn:
                a[k] = a[k] + g[k].double() ** 2
        return V.aggregate(spec.layout, {k: v / denom for k, v in a.items()})

    H = spec.features(ex)
    W, b = V.fit_head_e0(c, fam, spec, H, y, seed_parts)
    p1 = V._p1(H, W, b)
    idx, nb, lab, labs2 = V.draw_plan(c, fam, y, p1, seed_parts, ests)
    B = c['estimators']['E0']['batch_size']
    out = {}
    for e in ('E0', 'E1'):
        if e in ests:
            out[e] = acc_sq((gw(W, b, idx[i * B:(i + 1) * B], lab[e][i * B:(i + 1) * B], False) for i in range(nb)),
                            nb)
    if 'E2' in ests:
        out['E2'] = acc_sq((gw(W, b, torch.tensor([i]), labs2[d][i:i + 1], False)
                            for d in range(len(labs2)) for i in range(len(y))), len(y) * len(labs2))
    if 'E3' in ests:
        e3 = c['estimators']['E3']
        W3, b3 = spec.head_w.clone(), spec.head_b.clone()
        V.finetune_head_diversity(W3, b3, H, y, e3)
        p3 = V._p1(H, W3, b3)
        l3 = V.sample_labels(p3, V._gen(*seed_parts, 'E3'))
        B3 = e3['batch_size']
        nb3 = -(-len(y) // B3)
        out['E3'] = acc_sq((gw(W3, b3, torch.arange(i * B3, min((i + 1) * B3, len(y))),
                               l3[i * B3:(i + 1) * B3], True) for i in range(nb3)), nb3)
    return out


@pytest.mark.parametrize('fam', ['dfa', 'dsl'])
def test_fast_gradients_equal_direct_autograd(env, fam):
    c, roster = env
    c = _cfg(c, draws_per_epoch=1280)
    m = P.build_probe(c, fam, 11).eval()
    ex = _small(T.example_set(c, roster, fam, T.embedded_tasks(roster, fam)[40]), 64)
    sp = (11, fam, 'unit')
    fast = V.embed_task(c, V.Spec(m, fam), ex, sp)
    ref = _direct(c, m, fam, ex, sp, ('E0', 'E1', 'E2', 'E3'))
    for e in ('E0', 'E1', 'E2', 'E3'):
        rel = np.max(np.abs(fast[e]['f'] - ref[e])) / np.max(np.abs(ref[e]))
        assert rel < 1e-5, (e, rel)


# ---------------------------------------------------------------- 4. rescaled probe

@pytest.mark.parametrize('fam', ['dfa', 'dsl'])
def test_rescaled_probe_preserves_function_changes_e0(env, fam):
    c, roster = env
    m = P.build_probe(c, fam, 5).eval()
    r = P.rescaled_copy(m, fam)
    ex = _small(T.example_set(c, roster, fam, T.embedded_tasks(roster, fam)[10]), 256)
    assert P.max_output_diff(m, r, fam, ex) < 1e-5
    sp = (5, fam, 'resc')
    a = V.embed_task(c, V.Spec(m, fam), ex, sp, ('E0',))['E0']['f']
    b = V.embed_task(c, V.Spec(r, fam), ex, sp, ('E0',))['E0']['f']
    assert np.max(np.abs(a - b) / (np.abs(a) + 1e-30)) > 1e-2


# ---------------------------------------------------------------- 5. head fitting on cached features

@pytest.mark.parametrize('fam', ['dfa', 'dsl'])
def test_head_fit_cached_equals_full_model(env, fam):
    c, roster = env
    m = P.build_probe(c, fam, 9).eval()
    ex = _small(T.example_set(c, roster, fam, T.embedded_tasks(roster, fam)[20]), 512)
    spec = V.Spec(m, fam)
    y = torch.from_numpy(ex['y'])
    X = torch.from_numpy(ex['X'])
    last = None if ex['last'] is None else torch.from_numpy(ex['last'])
    e0 = dict(c['estimators']['E0'], draws_per_epoch=1000, head_epochs=3)
    W, b = spec.head_w.clone(), spec.head_b.clone()
    V.fit_head_reference(fam, W, b, spec.features(ex), y, torch.Generator().manual_seed(3), e0)
    # the same recipe through the frozen module with torch.optim.Adam
    mm = copy.deepcopy(m)
    for p in mm.parameters():
        p.requires_grad_(False)
    mm.head.weight.requires_grad_(True)
    mm.head.bias.requires_grad_(True)
    opt = torch.optim.Adam(mm.head.parameters(), lr=e0['head_lr'], weight_decay=e0['head_weight_decay'])
    gen = torch.Generator().manual_seed(3)
    B = e0['batch_size']
    for _ in range(e0['head_epochs']):
        idx = V.balanced_draws(y, e0['draws_per_epoch'], gen, fam)
        for i in range(len(idx) // B):
            s = idx[i * B:(i + 1) * B]
            loss = P.per_example_nll(fam, mm.head(P.body(mm, fam, X[s], None if last is None else last[s])), y[s]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert float((mm.head.weight - W).abs().max()) < 1e-6
    assert float((mm.head.bias - b).abs().max()) < 1e-6
    # E3's AdamW fine-tuning likewise
    e3 = c['estimators']['E3']
    W3, b3 = spec.head_w.clone(), spec.head_b.clone()
    V.finetune_head_diversity(W3, b3, spec.features(ex), y, e3)
    m3 = copy.deepcopy(m)
    for p in m3.parameters():
        p.requires_grad_(False)
    m3.head.weight.requires_grad_(True)
    m3.head.bias.requires_grad_(True)
    opt = torch.optim.AdamW(m3.head.parameters(), lr=e3['head_lr'], eps=e3['head_eps'],
                            weight_decay=e3['head_weight_decay'])
    for _ in range(e3['head_epochs']):
        for i in range(0, len(y), e3['batch_size']):
            s = slice(i, i + e3['batch_size'])
            z = m3.head(P.body(m3, fam, X[s], None if last is None else last[s]))
            loss = torch.nn.functional.cross_entropy(z.reshape(-1, 2), y[s].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert float((m3.head.weight - W3).abs().max()) < 1e-6
    assert float((m3.head.bias - b3).abs().max()) < 1e-6


# ---------------------------------------------------------------- 6. determinism

def test_determinism_bit_for_bit(env):
    c, roster = env
    for fam in ('dfa', 'dsl'):
        m = P.build_probe(c, fam, 13).eval()
        ex = _small(T.example_set(c, roster, fam, T.embedded_tasks(roster, fam)[30]), 256)
        a = V.embed_task(c, V.Spec(m, fam), ex, (13, fam, 'det'))
        b = V.embed_task(c, V.Spec(m, fam), ex, (13, fam, 'det'))
        for e in a:
            assert np.array_equal(a[e]['f'], b[e]['f']) and a[e]['head_loss'] == b[e]['head_loss']


def test_sample_batch_shares_and_single_task(env):
    c, roster = env
    tasks = T.embedded_tasks(roster, 'dsl')[:3]
    exs = [T.example_set(c, roster, 'dsl', t) for t in tasks]
    g = torch.Generator().manual_seed(0)
    b = V.sample_batch(exs, [0.5, 0.25, 0.25], 512, g)
    assert b['X'].shape == (512, 11) and b['y'].shape == (512, 10)
    s = V.sample_batch(exs, [0.0, 1.0, 0.0], 512, g, single_task=True)
    rows = {r.tobytes() for r in np.concatenate([exs[1]['X'], exs[1]['y']], 1)}
    assert all(r.tobytes() in rows for r in np.concatenate([s['X'], s['y']], 1))
