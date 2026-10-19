"""Probe networks: the imported Experiment 11/12 transformer with a null task token, pretrained with task dropout.

Public API (Experiment 15 imports it; keep these names): build_probe, pretrain_probe, load_probe, save_probe,
probe_path, features, body, head_logits, per_example_nll, rescaled_copy, evaluate_probe.

Neither source model ties its input embedding to its output layer (Experiment 11: `tok` and `head`; Experiment 12:
`tok_emb` and `head` are separate modules), so the probe is the source model unchanged except for a vocabulary with
one more task token (the null token). `head` is the Task2Vec classifier; everything else is the feature extractor.

DFA probes see right-padded batches of mixed length: the model is causal, so the hidden state at the last real
character (position l of [token, c_1..c_l]) does not depend on padding after it, and `body` reads it there. This is
exactly the source model's forward on the unpadded string (unit-tested).
"""
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import tasks14 as T


def probe_path(out, fam, seed):
    return Path(out) / 'probes' / f'{fam}_{seed}.npz'


def build_probe(c, fam, seed=None):
    m = T.exp_modules(c)[fam]
    if seed is not None:
        torch.manual_seed(seed)
    return m['model'].build_model(m['cfg'], c[fam]['n_probe_tokens'] + 1)


def save_probe(c, fam, model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    T.exp_modules(c)[fam]['model'].export_npz(model, tmp)
    tmp.replace(path)


def load_probe(c, fam, path):
    m = T.exp_modules(c)[fam]
    return m['model'].load_npz(build_probe(c, fam), path)


# ---------------------------------------------------------------- forward pieces

def body(model, fam, X, last=None):
    """Final hidden states fed to the head. DFA: (B, d) at position `last`; DSL: (B, 10, d) at positions 1..10."""
    if fam == 'dfa':
        L = int(last.max()) + 1
        X = X[:, :L]
        h = model.tok(X) + model.pos(torch.arange(L))
        for b in model.blocks:
            h = b(h)
        h = h[torch.arange(len(X)), last]
        return model.ln_f(h)
    h = model.tok_emb(X) + model.pos_emb(torch.arange(X.shape[1]))
    for b in model.blocks:
        h = b(h)
    return model.ln_f(h[:, 1:])


def head_logits(head, H):
    return head(H)


def per_example_nll(fam, logits, y):
    """Nats per example. DFA: cross-entropy of the 2-way label; DSL: sum over the 10 output bits (Experiment 12's
    training loss is this, averaged over the batch)."""
    if fam == 'dfa':
        return F.cross_entropy(logits, y, reduction='none')
    return F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction='none').view(y.shape).sum(-1)


@torch.no_grad()
def features(model, fam, ex, chunk=1024):
    X = torch.from_numpy(ex['X'])
    last = None if ex['last'] is None else torch.from_numpy(ex['last'])
    out = []
    for i in range(0, len(X), chunk):
        out.append(body(model, fam, X[i:i + chunk], None if last is None else last[i:i + chunk]))
    return torch.cat(out)


# ---------------------------------------------------------------- pretraining

def _dfa_pretrain_stream(c, roster, seed, n):
    """n fresh training-partition examples, 64 per batch, one length per batch (Experiment 11's length schedule),
    tasks dealt evenly over the 64 probe tasks, task dropout 0.5 (null token otherwise)."""
    m = T.exp_modules(c)['dfa']
    c11, D, A = m['cfg'], m['data'], m['gen']
    probes = sorted((t for t in roster['dfa'] if t['role'] == 'probe_pretrain'), key=lambda t: t['probe_token'])
    B = c11['batch_size']
    rng = np.random.default_rng([seed, 1])
    Ls = D.lengths(c11)
    sched = D.length_schedule(rng, n // B, Ls)
    nt = len(probes)
    base, extra = divmod(n, nt)
    reps = np.full(nt, base)
    reps[rng.permutation(nt)[:extra]] += 1
    task_of = np.repeat(np.arange(nt), reps)[rng.permutation(n)]
    lens = np.repeat(sched, B)
    bits = np.zeros(n, np.uint64)
    for L in Ls:
        idx = np.flatnonzero(lens == L)
        bits[idx] = D.sample_strings(rng, len(idx), int(L), 'train', c11)
    y = np.zeros(n, np.int64)
    dfas = [A.DFA.from_json(t['definition']) for t in probes]
    for j in range(nt):
        idx = np.flatnonzero(task_of == j)
        y[idx] = A.run_dfa(dfas[j], bits[idx], lens[idx])
    keep = rng.random(n) < (1.0 - c['task_dropout'])
    tok = np.where(keep, task_of, T.null_token(c, 'dfa'))
    X, last = T._dfa_tokens(c, bits, lens, tok)
    return dict(X=X, last=last, y=y, task=task_of, keep=keep, batch=B)


def _dsl_pretrain_stream(c, roster, seed, n):
    """n examples over the 80 probe tasks' training pairs (512 each; about 1.6 passes, PROTOCOL 2.3): tasks dealt
    evenly, each task's examples from Experiment 12's example_stream, then interleaved; task dropout 0.5."""
    m = T.exp_modules(c)['dsl']
    c12, D = m['cfg'], m['data']
    probes = sorted((t for t in roster['dsl'] if t['role'] == 'probe_pretrain'), key=lambda t: t['probe_token'])
    rng = np.random.default_rng([seed, 1])
    nt = len(probes)
    base, extra = divmod(n, nt)
    reps = np.full(nt, base)
    reps[rng.permutation(nt)[:extra]] += 1
    bx, by, task_of = [], [], []
    for j, t in enumerate(probes):
        s = D.example_stream(c12, t, int(reps[j]), seed, 'train')
        bx.append(s['bits_x'])
        by.append(s['bits_y'])
        task_of.append(np.full(int(reps[j]), j))
    order = rng.permutation(n)
    bx, by, task_of = (np.concatenate(a)[order] for a in (bx, by, task_of))
    keep = rng.random(n) < (1.0 - c['task_dropout'])
    tok = np.where(keep, task_of, T.null_token(c, 'dsl'))
    ex = T.encode(c, 'dsl', dict(bits_x=bx, y=by), token_index=tok)
    return dict(ex, task=task_of, keep=keep, batch=c12['batch_size'])


def pretrain_probe(c, roster, fam, seed, n=None, log=None):
    """Experiment 11/12 optimizer (AdamW, lr, betas, weight decay), warmup schedule, gradient clip and batch size;
    n = probe_pretrain_examples fresh examples, one pass; returns (model, record)."""
    torch.set_num_threads(c['threads'])
    n = n or c['probe_pretrain_examples']
    src = T.exp_modules(c)[fam]['cfg']
    t0 = time.time()
    st = _dfa_pretrain_stream(c, roster, seed, n) if fam == 'dfa' else _dsl_pretrain_stream(c, roster, seed, n)
    model = build_probe(c, fam, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=src['lr'], betas=tuple(src['betas']),
                            weight_decay=src['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / src['warmup_steps']))
    X, y = torch.from_numpy(st['X']), torch.from_numpy(st['y'])
    last = None if st['last'] is None else torch.from_numpy(st['last'])
    B = st['batch']
    curve = []
    model.train()
    run = 0.0
    for step in range(n // B):
        sl = slice(step * B, (step + 1) * B)
        H = body(model, fam, X[sl], None if last is None else last[sl])
        loss = per_example_nll(fam, model.head(H), y[sl]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), src['grad_clip'])
        opt.step()
        sched.step()
        run += float(loss)
        if (step + 1) % 64 == 0:
            curve.append(dict(T=(step + 1) * B, train_bits=run / 64 / math.log(2)))
            run = 0.0
    model.eval()
    rec = dict(family=fam, seed=seed, examples=n, steps=n // B, parameters=sum(p.numel() for p in model.parameters()),
               curve=curve, task_token_kept=float(st['keep'].mean()), seconds=round(time.time() - t0, 3))
    return model, rec


@torch.no_grad()
def evaluate_probe(c, roster, fam, model):
    """Held-out loss (bits) and accuracy per probe-pretraining task, with its task token and with the null token,
    using Experiment 11/12's evaluate on their held-out sets (DFA: the shared 4,096-string test set; DSL: the 512
    test inputs)."""
    m = T.exp_modules(c)[fam]
    probes = sorted((t for t in roster[fam] if t['role'] == 'probe_pretrain'), key=lambda t: t['probe_token'])
    out = []
    if fam == 'dfa':
        tb, tl = m['data'].build_test_set(m['cfg']) if 'test' not in _CACHE else _CACHE['test']
        _CACHE['test'] = (tb, tl)
        for t in probes:
            lab = m['gen'].run_dfa(m['gen'].DFA.from_json(t['definition']), tb, tl).astype(np.uint8)
            r = {}
            for name, tok in (('task', t['probe_token']), ('null', T.null_token(c, fam))):
                ev = m['model'].evaluate(model, tok, tb, tl, lab)
                r[name] = dict(bits=ev['excess_bits'], accuracy=ev['accuracy'])
            out.append(dict(task_id=t['task_id'], K=t['K'], **r))
    else:
        split = m['data'].make_split(m['cfg']) if 'split' not in _CACHE else _CACHE['split']
        _CACHE['split'] = split
        tx = split['test_idx'].astype(np.int64)
        for t in probes:
            tab = np.asarray(t['truth_table'], np.int64)
            r = {}
            for name, tok in (('task', t['probe_token']), ('null', T.null_token(c, fam))):
                ev = m['model'].evaluate(model, m['data'].to_bits(tx), tok, m['data'].to_bits(tab[tx]))
                r[name] = dict(bits=ev['excess_bits'], accuracy=ev['accuracy'])
            out.append(dict(task_id=t['task_id'], K=t['K'], **r))
    return out


_CACHE = {}


def rescaled_copy(model, fam, cfac=2.0):
    """Function-preserving rescaling (PROTOCOL Section 3): query weight and bias x c, key weight and bias / c,
    value weight and bias x c, attention-output weight / c, in every block. qkv rows: 0:d query, d:2d key, 2d:3d
    value (both sources split the fused projection this way)."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for b in m.blocks:
            d = b.qkv.weight.shape[1]
            b.qkv.weight[:d] *= cfac
            b.qkv.bias[:d] *= cfac
            b.qkv.weight[d:2 * d] /= cfac
            b.qkv.bias[d:2 * d] /= cfac
            b.qkv.weight[2 * d:] *= cfac
            b.qkv.bias[2 * d:] *= cfac
            outp = b.proj if fam == 'dfa' else b.out
            outp.weight /= cfac
    return m


@torch.no_grad()
def max_output_diff(model_a, model_b, fam, ex):
    Ha = model_a.head(features(model_a, fam, ex))
    Hb = model_b.head(features(model_b, fam, ex))
    return float((Ha - Hb).abs().max())
