"""Per-example scores, the controls and the stratified Eq. 2 estimator (PROTOCOL Sections 2.2-2.4).

Measures: a (zstd dictionary compression ratio), a' (dictionary gain), a2 (ZIP-FIT gzip NCD), b1 (probe embedding
max cosine), b2 (character 3-gram TF-IDF max cosine), c and c2 (50/50 combinations, formed at analysis time from the
normalized components), d (frozen-gradient cosine), e and e2 (Task2Vec mix-to-target distance, set level), the
label-agreement oracle and the random score. Controls: task token kept (leak), permuted-label references (perm),
input tokens only (inp).

Arithmetic mirrored from outside references (reimplemented, no code copied):
  ZIP-FIT, arXiv 2410.18194, Algorithm 1: NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y)) with gzip, the
  similarity of an example being 1 - mean NCD over target examples.
  LESS, arXiv 2402.04333 (https://github.com/princeton-nlp/LESS): cosine between an example's loss gradient and the
  mean target gradient. Ours is a simplified frozen-gradient cosine: one checkpoint, plain gradients, no warm-up, no
  Adam preconditioning, no low-rank adapters, no random projection.
  The alignment-scaling paper with Varun (https://github.com/madanva/alignment-scaling-laws, private, not readable
  from this machine): Eq. 2 token-weighted alignment and Eq. 3 0.5 * SBERT + 0.5 * ZIP; measure a uses the
  PROTOCOL's literal reading |x| / C_dict(x).
"""
import gzip
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import zstandard
from torch.func import grad, vmap

import common as C

# ---------------------------------------------------------------- normalization and Eq. 2


def minmax(s):
    """Min-max over the pool; a constant score raises instead of dividing by zero."""
    s = np.asarray(s, np.float64)
    lo, hi = float(s.min()), float(s.max())
    if not hi > lo:
        raise ValueError('constant score: min-max normalization undefined')
    return (s - lo) / (hi - lo)


def eq2_exact(t, s):
    """Eq. 2 on an explicit multiset: sum t s / sum t."""
    t = np.asarray(t, np.float64)
    return float(np.sum(t * np.asarray(s, np.float64)) / np.sum(t))


def eq2_stratified(n_u, mean_ts_u, mean_t_u):
    """Stratified Eq. 2: sum_u n_u mean(t s)_u / sum_u n_u mean(t)_u."""
    n_u = np.asarray(n_u, np.float64)
    return float(np.dot(n_u, mean_ts_u) / np.dot(n_u, mean_t_u))


def block_means(t, s, sizes):
    """Per-block means of t * s and t (blocks of the pool in layout order)."""
    t = np.asarray(t, np.float64)
    s = np.asarray(s, np.float64)
    edges = np.r_[0, np.cumsum(sizes)]
    ts = np.array([np.mean(t[a:b] * s[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    tt = np.array([np.mean(t[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    return ts, tt


# ---------------------------------------------------------------- compression


def zstd_dict_size(c, ref_bytes):
    z = c['zstd']
    B = sum(len(b) for b in ref_bytes)
    return min(z['max_dict_bytes'], z['dict_granule'] * (B // z['dict_fraction_divisor']))


def train_zstd_dictionary(c, samples, dict_id):
    """samples: list of (provenance tag, bytes). Only reference examples may train the dictionary."""
    for tag, _ in samples:
        assert tag[0] == 'ref', f'non-reference example {tag} passed to the dictionary trainer'
    data = [b for _, b in samples]
    d = zstandard.train_dictionary(zstd_dict_size(c, data), data, level=c['zstd']['level'], dict_id=int(dict_id),
                                   threads=0)
    return d.as_bytes()


def zstd_sizes(c, dict_bytes, xs):
    kw = dict(level=c['zstd']['level'], write_content_size=False, write_checksum=False, write_dict_id=False)
    if dict_bytes is not None:
        kw['dict_data'] = zstandard.ZstdCompressionDict(dict_bytes)
    cctx = zstandard.ZstdCompressor(**kw)
    return np.array([len(cctx.compress(x)) for x in xs], np.float64)


def score_a(c, dict_bytes, xs):
    """(a) |x| / C_dict(x) and (a') C_none(x) / C_dict(x)."""
    cd = zstd_sizes(c, dict_bytes, xs)
    c0 = zstd_sizes(c, None, xs)
    ln = np.array([len(x) for x in xs], np.float64)
    return ln / cd, c0 / cd


def gz(c, b):
    return len(gzip.compress(b, compresslevel=c['gzip_level'], mtime=0))


def ncd(c, x, y, cx=None, cy=None):
    cx = gz(c, x) if cx is None else cx
    cy = gz(c, y) if cy is None else cy
    return (gz(c, x + y) - min(cx, cy)) / max(cx, cy)


def score_a2(c, xs, refs):
    """ZIP-FIT: 1 - mean over references of NCD_gzip(x, r)."""
    cr = [gz(c, r) for r in refs]
    out = np.empty(len(xs))
    for i, x in enumerate(xs):
        cx = gz(c, x)
        out[i] = 1.0 - np.mean([ncd(c, x, r, cx, crj) for r, crj in zip(refs, cr)])
    return out


# ---------------------------------------------------------------- TF-IDF


def score_b2(c, ref_bytes, pool_bytes, chunk=8192):
    from sklearn.feature_extraction.text import TfidfVectorizer
    n = c['tfidf_ngram']
    docs = [b.decode('latin-1') for b in ref_bytes] + [b.decode('latin-1') for b in pool_bytes]
    vec = TfidfVectorizer(analyzer='char', ngram_range=(n, n), lowercase=False)
    Xall = vec.fit_transform(docs)
    R = Xall[: len(ref_bytes)]
    P = Xall[len(ref_bytes):]
    dense = Xall.shape[1] <= 4096
    Rd = R.toarray() if dense else R
    out = np.empty(P.shape[0])
    for i in range(0, P.shape[0], chunk):
        blk = P[i:i + chunk]
        S = (blk.toarray() @ Rd.T) if dense else (blk @ R.T).toarray()
        out[i:i + chunk] = S.max(1)
    return out, Xall.shape[1]


def max_cosine(P, R, chunk=8192):
    P = np.asarray(P, np.float64)
    R = np.asarray(R, np.float64)
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-300)
    Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-300)
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = (Pn[i:i + chunk] @ Rn.T).max(1)
    return out


# ---------------------------------------------------------------- probe embedding (b1)


@torch.no_grad()
def pooled_features(model, fam, ex, chunk=2048):
    """Mean over the real positions (task-token position included) of the final hidden states after the final layer
    norm. DFA rows are right-padded; the model is causal, so positions up to `last` do not see the padding."""
    torch.set_num_threads(1)
    X = torch.from_numpy(ex['X'])
    last = None if ex['last'] is None else torch.from_numpy(ex['last'])
    out = []
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk]
        if fam == 'dfa':
            lb = last[i:i + chunk]
            L = int(lb.max()) + 1
            xb = xb[:, :L]
            h = model.tok(xb) + model.pos(torch.arange(L))
        else:
            L = xb.shape[1]
            h = model.tok_emb(xb) + model.pos_emb(torch.arange(L))
        for b in model.blocks:
            h = b(h)
        h = model.ln_f(h).double()
        if fam == 'dfa':
            m = (torch.arange(L)[None, :] <= lb[:, None]).double()
        else:
            m = torch.ones(len(xb), L, dtype=torch.float64)
        out.append((h * m[..., None]).sum(1) / m.sum(1, keepdim=True))
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- frozen gradient (d)


def grad_param_names(spec):
    """Transformer-block and head parameters (weights and biases); embedding tables and the final norm excluded."""
    return sorted([k for k in spec.body if k.startswith('blocks.')] + ['head.weight', 'head.bias'])


def sel_params(spec):
    p = dict(spec.body)
    p['head.weight'], p['head.bias'] = spec.head_w, spec.head_b
    return {k: p[k].detach() for k in grad_param_names(spec)}


def _forward_logits(spec, params, X, last):
    """Logits through Experiment 14's functional transcription of the source model (task2vec.features_fn, unit-tested
    against the source module there); DFA reads the last real character (causal, padding-safe)."""
    import task2vec as t2v
    prm = dict(spec.body)
    prm.update({k: v for k, v in params.items() if not k.startswith('head.')})
    h = t2v.features_fn(spec, prm, X, last)
    return h @ params['head.weight'].T + params['head.bias']


def _loss(spec, params, X, last, y):
    """Label cross-entropy in nats, summed over examples (DSL: summed over the 10 output bits too)."""
    z = _forward_logits(spec, params, X, last)
    return F.cross_entropy(z.reshape(-1, 2), y.reshape(-1), reduction='sum')


def mean_gradient(spec, ex, chunk=1024):
    """Gradient of the mean per-example loss (= mean of per-example gradients), flattened in sorted name order."""
    torch.set_num_threads(1)
    prm = sel_params(spec)
    names = sorted(prm)
    acc = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in prm.items()}
    X = torch.from_numpy(ex['X'])
    last = None if ex['last'] is None else torch.from_numpy(ex['last'])
    y = torch.from_numpy(ex['y'])
    n = len(X)
    for i in range(0, n, chunk):
        xb = X[i:i + chunk]
        lb = None if last is None else last[i:i + chunk]
        if lb is not None:
            xb = xb[:, : int(lb.max()) + 1]
        p = {k: v.clone().requires_grad_(True) for k, v in prm.items()}
        loss = _loss(spec, p, xb, lb, y[i:i + chunk])
        g = torch.autograd.grad(loss, [p[k] for k in names])
        for k, gk in zip(names, g):
            acc[k] += gk.double()
    return torch.cat([acc[k].reshape(-1) / n for k in names]).numpy()


def per_example_flat_grads(spec, ex, chunk=256, prm=None):
    """Yields (slice, flat per-example gradients (n, P) float64) chunk by chunk; nothing is stored."""
    torch.set_num_threads(1)
    prm = sel_params(spec) if prm is None else prm
    names = sorted(prm)

    def one(p, x, l, yy):
        return _loss(spec, p, x[None], None if l is None else l[None], yy[None])

    X = torch.from_numpy(ex['X'])
    last = None if ex['last'] is None else torch.from_numpy(ex['last'])
    y = torch.from_numpy(ex['y'])
    if last is None:
        gfn = vmap(grad(lambda p, x, yy: one(p, x, None, yy)), in_dims=(None, 0, 0))
    else:
        gfn = vmap(grad(one), in_dims=(None, 0, 0, 0))
    for i in range(0, len(X), chunk):
        sl = slice(i, i + chunk)
        if last is None:
            g = gfn(prm, X[sl], y[sl])
        else:
            lb = last[sl]
            g = gfn(prm, X[sl, : int(lb.max()) + 1], lb, y[sl])
        yield sl, torch.cat([g[k].reshape(len(g[k]), -1) for k in names], 1).double()


def per_example_grad_cos(spec, ex, G, chunk=256):
    """cos(g(x), G_j) for every example x and every row G_j (G: (m, P) float64). Gradients are never stored."""
    Gt = torch.from_numpy(np.asarray(G, np.float64))
    Gn = Gt / Gt.norm(dim=1, keepdim=True)
    out = np.empty((len(ex['X']), len(G)))
    for sl, flat in per_example_flat_grads(spec, ex, chunk):
        flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-300)
        out[sl] = (flat @ Gn.T).numpy()
    return out


# ---------------------------------------------------------------- Task2Vec (e, e2)


def embed_fixed_head(c14, t2v, spec, ex, seed_parts, W, b):
    """E3 Fisher embedding of one batch with a given (fixed) head: labels sampled from the head's predictive
    distribution (generator (..., 'E3')), sequential batches of 8, loss averaged over positions; mirrors
    task2vec.embed_batch_e3 without its head fine-tuning."""
    e = c14['estimators']['E3']
    fam = spec.fam
    H = spec.features(ex)
    n = len(ex['y'])
    gen = t2v._gen(*seed_parts, 'E3')
    p1 = t2v._p1(H, W, b)
    lab = t2v.sample_labels(p1, gen)
    B = e['batch_size']
    nb = -(-n // B)
    if fam == 'dfa':
        batch_of = torch.arange(n) // B
        sizes = torch.bincount(batch_of, minlength=nb).to(p1.dtype)
        coef = (p1 - lab.to(p1.dtype)) / sizes[batch_of]
        g2 = t2v.dfa_fisher(spec, W, b, ex, [dict(kind='batch', ex_idx=torch.arange(n), batch_of=batch_of, coef=coef,
                                                  n_batches=nb)])[0]
    else:
        assert n % B == 0
        bt = torch.arange(n).view(nb, B)
        g2 = t2v.dsl_fisher(spec, W, b, ex, bt, t2v._cot(p1[bt], lab[bt], B * lab.shape[-1]).unsqueeze(1))[0]
    hl, he = t2v.head_fit_stats(H, torch.from_numpy(ex['y']), W, b)
    return dict(f=t2v.aggregate(spec.layout, g2), head_loss=hl, head_error=he)


def fixed_head(c14, t2v, spec, ex_ref):
    """e2's head: the probe head fine-tuned on the references by the E3 recipe (diversity-paper fine-tuning)."""
    H = spec.features(ex_ref)
    W, b = spec.head_w.clone(), spec.head_b.clone()
    t2v.finetune_head_diversity(W, b, H, torch.from_numpy(ex_ref['y']), c14['estimators']['E3'])
    return W, b


def batch_gen(fam, target_id, name, i):
    import tasks14
    return torch.Generator().manual_seed(tasks14.task_seed_int(15, fam, target_id, name, 'batch', i))


def mix_batches(c, t2v, ex_by_task, shares, fam, target_id, name):
    n, B = c['task2vec_batches'], c['task2vec_batch_size']
    return [t2v.sample_batch(ex_by_task, shares, B, batch_gen(fam, target_id, name, i)) for i in range(n)]


def ref_batches(c, ex_ref):
    B = c['task2vec_batch_size']
    out = []
    for i in range(c['task2vec_batches']):
        sl = slice(i * B, (i + 1) * B)
        out.append(dict(X=ex_ref['X'][sl], last=None if ex_ref['last'] is None else ex_ref['last'][sl],
                        y=ex_ref['y'][sl]))
    return out


def alpha_hat_e(c0, c1, cS):
    return float(np.clip((c0 - cS) / (c0 - c1), 0.0, 1.0))


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0


def finite(x):
    return bool(np.all(np.isfinite(np.asarray(x, np.float64))))


def log2(x):
    return math.log2(x)
