"""The decoder-only transformer language model of PROTOCOL Section 2.6, npz export and float64 evaluation.

Public API: build_model, param_count, nonembedding_count, save_npz, load_npz, code_lengths.
Tokens 0..V-1 are the vocabulary and V is the beginning-of-sequence token. The input is [BOS, x_1..x_127] and the
model predicts x_1..x_128 (128 learned positions).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, d_mlp):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d)

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (t.view(B, L, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.h)
        mask = torch.ones(L, L, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float('-inf')).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, vocab, d, n_layers, h, d_mlp, max_pos):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_pos, d)
        self.blocks = nn.ModuleList([Block(d, h, d_mlp) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        return self.ln_f(h) @ self.tok.weight.T  # tied output embedding, no bias


def param_count(m):
    return sum(p.numel() for p in m.parameters())


def nonembedding_count(m):
    return param_count(m) - m.tok.weight.numel() - m.pos.weight.numel()


def build_model(mc):
    m = LM(mc['vocab'], mc['d_model'], mc['n_layers'], mc['n_heads'], mc['d_mlp'], mc['max_positions'])
    if 'expected_params' in mc:
        assert param_count(m) == mc['expected_params'], param_count(m)
        assert nonembedding_count(m) == mc['expected_nonembedding'], nonembedding_count(m)
    return m


def inputs_targets(docs, bos):
    """LongTensors x = [BOS, x_1..x_{L-1}] and y = x_1..x_L for docs (B, L)."""
    d = torch.from_numpy(np.asarray(docs, np.int64))
    x = torch.cat([torch.full((d.shape[0], 1), bos, dtype=torch.long), d[:, :-1]], 1)
    return x, d


def save_npz(model, path):
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    with open(path, 'wb') as f:
        np.savez(f, **arrays)


def load_npz(model, path):
    with np.load(path, allow_pickle=False) as z:
        model.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files})
    return model


@torch.no_grad()
def code_lengths(model, docs, bos, batch=64):
    """Per-token -log2 p(x_t | x_<t) (B, L) in float64, on a float64 copy of the model, deterministic."""
    torch.use_deterministic_algorithms(True)
    m64 = _float64_copy(model)
    out = []
    for s in range(0, len(docs), batch):
        x, y = inputs_targets(docs[s:s + batch], bos)
        lp = F.log_softmax(m64(x), -1)
        out.append((-lp.gather(-1, y[..., None])[..., 0] / math.log(2.0)).numpy())
    return np.concatenate(out).astype(np.float64)


def _float64_copy(model):
    import copy
    m = copy.deepcopy(model).double()
    m.eval()
    return m
