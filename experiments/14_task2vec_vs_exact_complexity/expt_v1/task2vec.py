"""Task2Vec estimators E0-E3, aggregation, the symmetric cosine distance and the diversity coefficient.

Public API (Experiment 15 imports it; keep these names and signatures):
  Spec(model, fam)                                  frozen probe as parameters + a functional forward
  embed_task(c, spec, ex, seed_parts, estimators=('E0','E1','E2','E3')) -> {est: dict(f, head_loss, head_error)}
  embed_batch_e3(c, spec, ex, seed_parts) -> dict(f, head_loss, head_error)     (diversity batches)
  layout(spec) -> [(module name, start, end)] of the embedding coordinates
  sym_cosine(fa, fb, eps=1e-8); pairwise_sym_cosine(F); diversity(embs); cross_diversity(a, b); rao_q(shares, F)
  sample_batch(ex_by_task, shares, n, gen, single_task=False)

Arithmetic mirrored (reimplemented, no code copied):
  awslabs/aws-cv-task2vec task2vec.py: `_get_loader` (class-balanced WeightedRandomSampler, 10,000 draws, batch 64,
  drop_last), `_fit_classifier` (Adam lr 4e-4, weight decay 1e-4, 10 epochs on cached features),
  `montecarlo_fisher` (labels sampled from the model, squared gradient of the batch-mean loss accumulated per
  parameter and divided by the number of batches) and `extract_embedding` (per module with a weight, except the
  classifier: grad2.reshape(rows, -1).mean(1), concatenated in named_modules order; biases ignored);
  task_similarity.py `cosine` (cosine distance of h0/(h0+h1+1e-8) and h1/(h0+h1+1e-8)).
  brando90/beyond-scale-div-coeff src/diversity/task2vec.py: `_finetune_classifier` (AdamW lr 5e-5, eps 1e-8,
  weight decay 1e-4 on the output layer, 5 epochs, batch 8, sequential loader, true labels, loss averaged over all
  positions) and `montecarlo_fisher_autoregressive` (sequential batches of 8, a label sampled at every position,
  loss averaged over positions, squared gradients divided by the number of batches); div_coeff.py
  `get_diversity_coefficient` (mean pairwise distance of batch embeddings).

How gradients are computed (exact rewrites, unit-tested against the direct computation):
  * The forward is a functional transcription of the source model (explicit attention; same weights, same math;
    max logit difference to the source module < 1e-5, unit-tested), so torch.func can vectorize it. Only `.weight`
    tensors are differentiated, since biases never enter the embedding.
  * Every output is a 2-way softmax, so the gradient of the cross-entropy at one output position is
    (p_1 - y) * grad(s) with s = logit_1 - logit_0. For the DFA family (one output position) the per-example
    gradients grad(s(x)) of the 2,048 distinct examples are computed once per fitted head (torch.func vmap over
    right-padded rows read at the last real character, exact because the model is causal), and every batch
    gradient is the weighted sum sum_{draws in batch} c * (p_1 - y_draw) * grad(s(x_draw)) (fisher_from_rows).
    The DSL family (10 output positions) computes each batch gradient directly (one forward and one
    vector-Jacobian product per batch and label set), many batches at once with vmap (dsl_fisher).
  * E0, E1 and E2 share one fitted head ("head fit: as E0"), seeded by (..., 'head'). E0 and E1 use the same
    class-balanced draws (generator (..., 'E0'); E0's sampled labels come from it too, E1 uses the true labels);
    E2 draws its labels from (..., 'E2'). E3 fits its own head.
Every random draw uses a torch.Generator seeded from seed_parts (tasks14.task_seed_int), so any embedding can be
recomputed bit for bit on one thread.
"""
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import grad, vjp, vmap

import tasks14 as T

warnings.filterwarnings('ignore', message='Sparse CSR tensor support is in beta')
NAMES = dict(dfa=dict(tok='tok', pos='pos', out='proj'), dsl=dict(tok='tok_emb', pos='pos_emb', out='out'))


def _gen(*parts):
    return torch.Generator().manual_seed(T.task_seed_int(*parts))


# ---------------------------------------------------------------- the frozen probe as a function

class Spec:
    """Parameters of a frozen probe split into body (feature extractor) and head, plus its functional forward."""

    def __init__(self, model, fam):
        self.fam = fam
        sd = {k: v.detach().clone().float() for k, v in model.named_parameters()}
        self.body = {k: v for k, v in sd.items() if not k.startswith('head.')}
        self.head_w = sd['head.weight']
        self.head_b = sd['head.bias']
        self.n_layers = len(model.blocks)
        self.n_heads = model.blocks[0].h
        self.d = self.head_w.shape[1]
        self.layout = _layout(model)
        self.n_pos = 1 if fam == 'dfa' else 10

    def feats(self, prm, X, last=None):
        return features_fn(self, prm, X, last)

    @torch.no_grad()
    def features(self, ex, chunk=1024):
        X = torch.from_numpy(ex['X'])
        last = None if ex['last'] is None else torch.from_numpy(ex['last'])
        out = []
        for i in range(0, len(X), chunk):
            out.append(self.feats(self.body, X[i:i + chunk], None if last is None else last[i:i + chunk]))
        return torch.cat(out)


def features_fn(spec, prm, X, last=None):
    """Hidden states fed to the head. DFA (causal): position `last` of each row (right padding after it cannot
    change it); DSL (bidirectional): positions 1..10."""
    nm = NAMES[spec.fam]
    B, L = X.shape
    d, nh = spec.d, spec.n_heads
    dh = d // nh
    h = F.embedding(X, prm[f"{nm['tok']}.weight"]) + prm[f"{nm['pos']}.weight"][:L]
    mask = torch.ones(L, L, dtype=torch.bool).tril() if spec.fam == 'dfa' else None
    for i in range(spec.n_layers):
        p = f'blocks.{i}.'
        a = F.layer_norm(h, (d,), prm[p + 'ln1.weight'], prm[p + 'ln1.bias'])
        q, k, v = F.linear(a, prm[p + 'qkv.weight'], prm[p + 'qkv.bias']).split(d, dim=-1)
        q, k, v = (t.reshape(B, L, nh, dh).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        if mask is not None:
            att = att.masked_fill(~mask, float('-inf'))
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, L, d)
        h = h + F.linear(o, prm[p + nm['out'] + '.weight'], prm[p + nm['out'] + '.bias'])
        a = F.layer_norm(h, (d,), prm[p + 'ln2.weight'], prm[p + 'ln2.bias'])
        h = h + F.linear(F.gelu(F.linear(a, prm[p + 'fc1.weight'], prm[p + 'fc1.bias'])),
                         prm[p + 'fc2.weight'], prm[p + 'fc2.bias'])
    if spec.fam == 'dfa':
        if last is None:
            h = h[:, -1]
        else:
            h = h.gather(1, last.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
    else:
        h = h[:, 1:]
    return F.layer_norm(h, (d,), prm['ln_f.weight'], prm['ln_f.bias'])


def _layout(model):
    out, start = [], 0
    for name, mod in model.named_modules():
        if mod is model.head:
            continue
        w = getattr(mod, 'weight', None)
        if isinstance(w, torch.nn.Parameter):
            out.append((name, start, start + w.shape[0]))
            start += w.shape[0]
    return out


def layout(spec):
    return spec.layout


def aggregate(lay, grad2):
    """Reference `extract_embedding`: per module (except the head) with a weight, the accumulated squared gradient
    reshaped to (rows, -1) and averaged over columns, concatenated in named_modules order (lay = layout(spec))."""
    hs = []
    for name, a, b in lay:
        g = grad2[f'{name}.weight']
        hs.append(g.reshape(g.shape[0], -1).mean(1))
    return torch.cat(hs).numpy()


# ---------------------------------------------------------------- samplers and labels

def balanced_draws(y, n, gen, fam):
    """Reference `_get_loader`: weights 1 / class_count[label] / n_classes, n draws with replacement. A task with more
    than one label token (DSL, 10 output bits) is sampled uniformly."""
    if fam == 'dfa':
        yy = y.numpy()
        n_cls = int(yy.max()) + 1
        cc = np.eye(n_cls)[yy].sum(0)
        w = 1.0 / cc[yy] / n_cls
        w = torch.from_numpy(w / w.sum())
    else:
        w = torch.full((len(y),), 1.0 / len(y), dtype=torch.float64)
    return torch.multinomial(w, n, replacement=True, generator=gen)


def sample_labels(p1, gen):
    """One label per output position from the 2-way predictive distribution (p1 = P(label 1)), like
    torch.multinomial(softmax(logits), 1) in the reference."""
    p = torch.stack([1 - p1, p1], -1).reshape(-1, 2)
    return torch.multinomial(p, 1, generator=gen).view(p1.shape)


# ---------------------------------------------------------------- head fitting (closed-form gradients)

def _head_grad(H, y, W, b, norm):
    """Gradient of sum(cross-entropy) / norm for logits H @ W.T + b (H: (B, d) or (B, P, d))."""
    z = H @ W.T + b
    g = torch.softmax(z, -1)
    g = g - F.one_hot(y, 2).to(g.dtype)
    g = g / norm
    gW = g.reshape(-1, 2).T @ H.reshape(-1, H.shape[-1])
    return gW, g.reshape(-1, 2).sum(0)


class _Adam:
    """torch.optim.Adam / AdamW single-tensor arithmetic (defaults betas (0.9, 0.999))."""

    def __init__(self, params, lr, weight_decay, eps=1e-8, decoupled=False):
        self.p, self.lr, self.wd, self.eps, self.dec = params, lr, weight_decay, eps, decoupled
        self.m = [torch.zeros_like(x) for x in params]
        self.v = [torch.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2 = 0.9, 0.999
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for x, g, m, v in zip(self.p, grads, self.m, self.v):
            if self.dec:
                x.mul_(1 - self.lr * self.wd)
            elif self.wd != 0:
                g = g.add(x, alpha=self.wd)
            m.lerp_(g, 1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)
            x.addcdiv_(m, denom, value=-self.lr / bc1)


def fit_head_reference(fam, W, b, H, y, gen, e):
    """Reference `_fit_classifier`: Adam on the head, class-balanced draws_per_epoch draws per epoch, batch
    batch_size with drop_last, head_epochs epochs, per-example loss averaged over the batch."""
    opt = _Adam([W, b], e['head_lr'], e['head_weight_decay'])
    B = e['batch_size']
    for _ in range(e['head_epochs']):
        idx = balanced_draws(y, e['draws_per_epoch'], gen, fam)
        for i in range(len(idx) // B):
            s = idx[i * B:(i + 1) * B]
            opt.step(_head_grad(H[s], y[s], W, b, B))


def finetune_head_diversity(W, b, H, y, e):
    """Diversity-paper `_finetune_classifier`: AdamW on the head, sequential batches of batch_size, true labels,
    cross-entropy averaged over every output position of the batch, head_epochs epochs."""
    opt = _Adam([W, b], e['head_lr'], e['head_weight_decay'], eps=e['head_eps'], decoupled=True)
    B = e['batch_size']
    for _ in range(e['head_epochs']):
        for i in range(0, len(H), B):
            hb, yb = H[i:i + B], y[i:i + B]
            opt.step(_head_grad(hb, yb, W, b, yb.numel()))


@torch.no_grad()
def head_fit_stats(H, y, W, b):
    """Mean per-example cross-entropy in nats (summed over output positions) and error rate (per position)."""
    z = H @ W.T + b
    nll = F.cross_entropy(z.reshape(-1, 2), y.reshape(-1), reduction='none').view(y.shape)
    if nll.dim() > 1:
        nll = nll.sum(-1)
    return float(nll.mean()), float((z.argmax(-1) != y).float().mean())


# ---------------------------------------------------------------- gradients

def _weights(spec):
    """Only `.weight` tensors enter the embedding (biases ignored, as in the reference), so only they are
    differentiated; biases are held constant."""
    return {k: v for k, v in spec.body.items() if k.endswith('.weight')}


def _with(spec, wts):
    prm = dict(spec.body)
    prm.update(wts)
    return prm


def _s_fn(spec, W, b):
    dw, db = W[1] - W[0], b[1] - b[0]

    def s(wts, x, last):
        return (spec.feats(_with(spec, wts), x[None], last[None])[0] @ dw) + db
    return s


def dfa_rows(spec, W, b, X, last, chunk=128):
    """Yield (example indices, dict name -> (n, numel) per-example grad(s) of every weight), chunks of `chunk`
    examples in index order. Rows are right-padded; the model is causal, so reading position `last` is the unpadded
    forward (unit-tested)."""
    gfn = vmap(grad(_s_fn(spec, W, b)), in_dims=(None, 0, 0))
    wts = _weights(spec)
    for i in range(0, len(X), chunk):
        idx = torch.arange(i, min(i + chunk, len(X)))
        L = int(last[idx].max()) + 1
        g = gfn(wts, X[idx, :L], last[idx])
        yield idx, {k: v.reshape(len(idx), -1) for k, v in g.items()}


def dfa_fisher(spec, W, b, ex, jobs, chunk=128):
    """DFA batch and per-example Fisher accumulations from one pass over per-example gradients grad(s(x)); see
    fisher_from_rows."""
    X = torch.from_numpy(ex['X'])
    last = torch.from_numpy(ex['last'])
    shapes = {k: v.shape for k, v in _weights(spec).items()}
    return fisher_from_rows(dfa_rows(spec, W, b, X, last, chunk), len(X), shapes, jobs)


def fisher_from_rows(rows_iter, n, shapes, jobs):
    """Fisher accumulations from per-example gradients grad(s(x)) of a 2-way softmax output (s = logit_1 - logit_0).
    rows_iter yields (example indices, dict name -> (len, numel) rows); shapes: name -> parameter shape.

    jobs: list of dicts, each either
      dict(kind='batch', ex_idx (n_draws,), batch_of (n_draws,), coef (n_draws,), n_batches)
          -> mean over batches of (sum_{draws in batch} coef * grad s(x))^2
      dict(kind='per_example', weight (n_examples,), denom)
          -> sum_x weight_x * grad s(x)^2 / denom
    Batch gradients are accumulated per chunk of examples as a sparse (batches x examples) coefficient matrix times
    the chunk's rows (float32, at most batch_size terms per batch gradient); squares and sums are float64. Returns one grad2 dict per job."""
    acc = []
    for j in jobs:
        lead = (j['n_batches'],) if j['kind'] == 'batch' else ()
        dt = torch.float32 if j['kind'] == 'batch' else torch.float64
        acc.append({k: torch.zeros(lead + (math.prod(v),), dtype=dt) for k, v in shapes.items()})
    pos = torch.full((n,), -1, dtype=torch.long)
    for idx, rows in rows_iter:
        pos[idx] = torch.arange(len(idx))
        for j, a in zip(jobs, acc):
            if j['kind'] == 'batch':
                sel = torch.nonzero(pos[j['ex_idx']] >= 0).view(-1)
                if len(sel) == 0:
                    continue
                C = torch.sparse_coo_tensor(torch.stack([j['batch_of'][sel], pos[j['ex_idx'][sel]]]),
                                            j['coef'][sel].float(), (j['n_batches'], len(idx)),
                                            check_invariants=True).coalesce()
                C = C.to_sparse_csr()
                for k, r in rows.items():
                    a[k].addmm_(C, r)
            else:
                w = j['weight'][idx].float()
                for k, r in rows.items():
                    a[k] += (w @ r.square()).double()
        pos[idx] = -1
    out = []
    for j, a in zip(jobs, acc):
        if j['kind'] == 'batch':
            out.append({k: v.double().square_().sum(0).div_(j['n_batches']).view(shapes[k]) for k, v in a.items()})
        else:
            out.append({k: (v / j['denom']).view(shapes[k]) for k, v in a.items()})
    return out


def _cot(p1, y, norm):
    """Gradient of cross-entropy / norm with respect to the 2 logits: softmax - one_hot(y), divided by norm."""
    d = (p1 - y.to(p1.dtype)) / norm
    return torch.stack([-d, d], -1)


def dsl_fisher(spec, W, b, ex, batches, cots, chunk_examples=128):
    """DSL: for each of J cotangent sets, the mean over batches of the squared gradient of the batch loss.
    batches (n_b, B) example indices (equal sizes); cots (n_b, J, B, 10, 2) gradients of the batch loss with respect
    to the logits (see _cot). One forward per batch serves all J sets (torch.func vjp); many batches per call with
    vmap. Returns J grad2 dicts (weights only)."""
    X = torch.from_numpy(ex['X'])
    wts = _weights(spec)

    def logits(w, Xb):
        return spec.feats(_with(spec, w), Xb) @ W.T + b

    def one(w, Xb, cb):
        _, f = vjp(lambda w_: logits(w_, Xb), w)
        return vmap(f)(cb)[0]

    gfn = vmap(one, in_dims=(None, 0, 0))
    nb, B = batches.shape
    nj = cots.shape[1]
    acc = [{k: torch.zeros(v.shape, dtype=torch.float64) for k, v in wts.items()} for _ in range(nj)]
    step = max(1, chunk_examples // B)
    for i in range(0, nb, step):
        g = gfn(wts, X[batches[i:i + step]], cots[i:i + step])
        for j in range(nj):
            for k in acc[j]:
                acc[j][k] += g[k][:, j].double().square_().sum(0)
    return [{k: v / nb for k, v in a.items()} for a in acc]


# ---------------------------------------------------------------- estimators

def _p1(H, W, b):
    with torch.no_grad():
        return torch.softmax(H @ W.T + b, -1)[..., 1]


def fit_head_e0(c, fam, spec, H, y, seed_parts):
    """The head shared by E0, E1 and E2: the probe's head refit by the reference recipe, generator (..., 'head')."""
    W, b = spec.head_w.clone(), spec.head_b.clone()
    fit_head_reference(fam, W, b, H, y, _gen(*seed_parts, 'head'), c['estimators']['E0'])
    return W, b


def draw_plan(c, fam, y, p1, seed_parts, ref):
    """Random draws of E0/E1/E2: class-balanced example draws idx (nb * batch_size,) and their labels per estimator
    (E0 sampled from p1, E1 true), and E2's label draws (draws_per_example, n[, 10])."""
    E = c['estimators']
    e0 = E['E0']
    idx, nb, lab, labs2 = None, 0, {}, None
    if 'E0' in ref or 'E1' in ref:
        gen = _gen(*seed_parts, 'E0')
        idx = balanced_draws(y, e0['draws_per_epoch'], gen, fam)
        nb = len(idx) // e0['batch_size']
        idx = idx[: nb * e0['batch_size']]
        if 'E0' in ref:
            lab['E0'] = sample_labels(p1[idx], gen)
        if 'E1' in ref:
            lab['E1'] = y[idx]
    if 'E2' in ref:
        gen = _gen(*seed_parts, 'E2')
        labs2 = torch.stack([sample_labels(p1, gen) for _ in range(E['E2']['draws_per_example'])])
    return idx, nb, lab, labs2


def embed_task(c, spec, ex, seed_parts, estimators=('E0', 'E1', 'E2', 'E3')):
    """Task2Vec embeddings of one fixed example set under the frozen probe `spec` (PROTOCOL Section 2.4).
    E0 and E1 use the same class-balanced draws (generator (..., 'E0')); E0 samples its labels from the same
    generator, E1 uses the true labels."""
    torch.set_num_threads(c['threads'])
    fam = spec.fam
    y = torch.from_numpy(ex['y'])
    n = len(y)
    H = spec.features(ex)
    out = {}
    ref = [e for e in ('E0', 'E1', 'E2') if e in estimators]
    if ref:
        E = c['estimators']
        e0 = E['E0']
        for k in ('draws_per_epoch', 'batch_size', 'head_lr', 'head_weight_decay', 'head_epochs'):
            assert E['E1'][k] == e0[k] and E['E2'][k] == e0[k]
        W, b = fit_head_e0(c, fam, spec, H, y, seed_parts)
        hl, he = head_fit_stats(H, y, W, b)
        p1 = _p1(H, W, b)
        Bsz = e0['batch_size']
        idx, nb, lab, labs2 = draw_plan(c, fam, y, p1, seed_parts, ref)
        bl = [e for e in ('E0', 'E1') if e in ref]
        res = {}
        if fam == 'dfa':
            jobs = [dict(kind='batch', ex_idx=idx, batch_of=torch.arange(len(idx)) // Bsz,
                         coef=(p1[idx] - lab[e].to(p1.dtype)) / Bsz, n_batches=nb) for e in bl]
            if 'E2' in ref:
                w = ((p1[None] - labs2.to(p1.dtype)) ** 2).sum(0)
                jobs.append(dict(kind='per_example', weight=w, denom=n * len(labs2)))
            res = dict(zip(bl + (['E2'] if 'E2' in ref else []), dfa_fisher(spec, W, b, ex, jobs)))
        else:
            if bl:
                bt = idx.view(nb, Bsz)
                cots = torch.stack([_cot(p1[bt], lab[e].view(nb, Bsz, -1), Bsz) for e in bl], 1)
                res.update(zip(bl, dsl_fisher(spec, W, b, ex, bt, cots)))
            if 'E2' in ref:
                # per-example gradients (batches of one example), one cotangent set per draw
                cots = torch.stack([_cot(p1, l, 1.0) for l in labs2], 1).unsqueeze(2)       # (n, d, 1, 10, 2)
                g = dsl_fisher(spec, W, b, ex, torch.arange(n).view(n, 1), cots)
                res['E2'] = {k: sum(gd[k] for gd in g) / len(g) for k in g[0]}
        for est in ref:
            out[est] = dict(f=aggregate(spec.layout, res[est]), head_loss=hl, head_error=he)
    if 'E3' in estimators:
        out['E3'] = embed_batch_e3(c, spec, ex, seed_parts, H=H)
    return out


def embed_batch_e3(c, spec, ex, seed_parts, H=None):
    """E3, the diversity-paper pipeline: head fine-tuned on the data in order, then sequential batches of 8 with a
    label sampled at every output position, loss averaged over positions."""
    torch.set_num_threads(c['threads'])
    e = c['estimators']['E3']
    fam = spec.fam
    y = torch.from_numpy(ex['y'])
    n = len(y)
    H = spec.features(ex) if H is None else H
    W, b = spec.head_w.clone(), spec.head_b.clone()
    finetune_head_diversity(W, b, H, y, e)
    hl, he = head_fit_stats(H, y, W, b)
    gen = _gen(*seed_parts, 'E3')
    p1 = _p1(H, W, b)
    lab = sample_labels(p1, gen)
    B = e['batch_size']
    nb = -(-n // B)
    if fam == 'dfa':
        batch_of = torch.arange(n) // B
        sizes = torch.bincount(batch_of, minlength=nb).to(p1.dtype)
        coef = (p1 - lab.to(p1.dtype)) / sizes[batch_of]
        g2 = dfa_fisher(spec, W, b, ex, [dict(kind='batch', ex_idx=torch.arange(n), batch_of=batch_of, coef=coef,
                                              n_batches=nb)])[0]
    else:
        assert n % B == 0
        bt = torch.arange(n).view(nb, B)
        g2 = dsl_fisher(spec, W, b, ex, bt, _cot(p1[bt], lab[bt], B * lab.shape[-1]).unsqueeze(1))[0]
    return dict(f=aggregate(spec.layout, g2), head_loss=hl, head_error=he)


# ---------------------------------------------------------------- distances and diversity

def sym_cosine(fa, fb, eps=1e-8):
    fa = np.asarray(fa, np.float64)
    fb = np.asarray(fb, np.float64)
    ha, hb = fa / (fa + fb + eps), fb / (fa + fb + eps)
    return float(1.0 - np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb)))


def pairwise_sym_cosine(F_, eps=1e-8):
    n = len(F_)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = sym_cosine(F_[i], F_[j], eps)
    return D


def diversity(embs, eps=1e-8):
    n = len(embs)
    d = pairwise_sym_cosine(embs, eps)[np.triu_indices(n, 1)]
    return dict(div=float(d.mean()), pair_distances=d)


def cross_diversity(embs_a, embs_b, eps=1e-8):
    return float(np.mean([sym_cosine(a, b, eps) for a in embs_a for b in embs_b]))


def rao_q(shares, F_, eps=1e-8):
    pi = np.asarray(shares, np.float64)
    return float(pi @ pairwise_sym_cosine(F_, eps) @ pi)


def sample_batch(ex_by_task, shares, n, gen, single_task=False):
    """Interleaved (default): each example i.i.d.: a task by its share, then a uniformly random example of that task.
    Single-task: one task by its share, then n of its examples drawn without replacement."""
    sh = torch.tensor(shares, dtype=torch.float64)
    if single_task:
        j = int(torch.multinomial(sh, 1, generator=gen))
        ex = ex_by_task[j]
        idx = torch.randperm(len(ex['y']), generator=gen)[:n].numpy()
        return dict(X=ex['X'][idx], last=None if ex['last'] is None else ex['last'][idx], y=ex['y'][idx])
    tj = torch.multinomial(sh, n, replacement=True, generator=gen).numpy()
    u = torch.rand(n, generator=gen, dtype=torch.float64).numpy()
    width = max(e['X'].shape[1] for e in ex_by_task)
    X = np.zeros((n, width), np.int64)
    y = np.zeros((n,) + ex_by_task[0]['y'].shape[1:], np.int64)
    last = None if ex_by_task[0]['last'] is None else np.zeros(n, np.int64)
    for i, j in enumerate(tj):
        e = ex_by_task[j]
        k = int(np.floor(u[i] * len(e['y'])))
        X[i, : e['X'].shape[1]] = e['X'][k]
        y[i] = e['y'][k]
        if last is not None:
            last[i] = e['last'][k]
    return dict(X=X, last=last, y=y)
