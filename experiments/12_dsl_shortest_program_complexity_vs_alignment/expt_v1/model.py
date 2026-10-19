"""The 103,618-parameter bidirectional transformer, the 83,988-parameter MLP arm, npz export and evaluation.

Public API (Experiments 14 and 15 import it; keep these names): build_model, build_mlp, n_params, encode,
export_npz, load_npz, evaluate.

Input: [task token, x_0, ..., x_9] with token ids 0 and 1 for bit values and 2 + t for task token t. Output: logits
(B, 10, 2), output bit j read at the position of x_j (positions 1..10) by one shared 2-way head.

Module names: tok_emb, pos_emb, blocks.{0,1}.{ln1, qkv, out, ln2, fc1, fc2}, ln_f, head. qkv is one Linear(64, 192)
whose output rows 0:64 are the query, 64:128 the key and 128:192 the value, each head-major (4 heads x 16 dims:
rows 16h..16h+15 of a block belong to head h). Attention is torch.nn.functional.scaled_dot_product_attention with no
causal mask. Experiment 14 rescales query, key and value by slicing qkv.weight and qkv.bias this way.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORD_BITS = 10


class Block(nn.Module):
    def __init__(self, d, h, d_mlp):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d)

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        q, k, v = (t.view(B, L, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.out(a.transpose(1, 2).reshape(B, L, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Transformer(nn.Module):
    def __init__(self, vocab, d, n_layers, h, d_mlp, max_pos):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_pos, d)
        self.blocks = nn.ModuleList([Block(d, h, d_mlp) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h[:, 1:]))


class MLP(nn.Module):
    """One-hot task token and the 10 input bits, two GELU hidden layers, 20 logits read as (B, 10, 2)."""

    def __init__(self, n_task_tokens, hidden):
        super().__init__()
        self.n_task_tokens = n_task_tokens
        dims = [n_task_tokens + WORD_BITS] + list(hidden)
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(dims[-1], 2 * WORD_BITS)

    def forward(self, x):
        task = F.one_hot(x[:, 0] - 2, self.n_task_tokens).float()
        h = torch.cat([task, x[:, 1:].float()], dim=1)
        for lyr in self.layers:
            h = F.gelu(lyr(h))
        return self.head(h).view(-1, WORD_BITS, 2)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(c, n_task_tokens=40):
    m = Transformer(2 + n_task_tokens, c['d_model'], c['n_layers'], c['n_heads'], c['d_mlp'], c['max_positions'])
    if n_task_tokens == c['n_task_tokens']:
        assert n_params(m) == c['expected_parameters'], n_params(m)
    elif n_task_tokens == c['n_task_tokens'] + 1:
        assert n_params(m) == c['expected_parameters_null_token'], n_params(m)
    return m


def build_mlp(c, n_task_tokens=40):
    m = MLP(n_task_tokens, c['mlp']['hidden'])
    if n_task_tokens == c['n_task_tokens']:
        assert n_params(m) == c['mlp']['expected_parameters'], n_params(m)
    return m


def encode(task_token, bits_x):
    """LongTensor (B, 11): [2 + task token, x_0..x_9]. task_token is a scalar or one per row."""
    bits_x = np.asarray(bits_x, np.int64)
    tok = np.broadcast_to(np.asarray(task_token, np.int64).reshape(-1, 1) + 2, (len(bits_x), 1))
    return torch.from_numpy(np.ascontiguousarray(np.concatenate([tok, bits_x], axis=1)))


def export_npz(model, path):
    """float32 numpy arrays keyed by state_dict name; no pickle. Writes to `path` exactly."""
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    with open(path, 'wb') as f:
        np.savez(f, **arrays)


def load_npz(model, path):
    with np.load(path, allow_pickle=False) as z:
        model.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files})
    return model


@torch.no_grad()
def evaluate(model, bits_x, task_token, bits_y):
    """One batch in the given order. Per example: sum over the 10 output bits of -log2 p(correct bit), from a
    float32 forward, accumulated in float64. Returns mean bits, accuracy (all 10 bits right under greedy decoding),
    bit accuracy, per-output-bit mean bits (10) and per-example bits (float32)."""
    was = model.training
    model.eval()
    logits = model(encode(task_token, bits_x)).float()  # (n, 10, 2)
    y = torch.from_numpy(np.asarray(bits_y, np.int64))
    nll = -F.log_softmax(logits, dim=-1).gather(-1, y[..., None])[..., 0]  # (n, 10) nats, float32
    bits = nll.numpy().astype(np.float64) / math.log(2.0)
    per_ex = bits.sum(1)
    right = (logits.argmax(-1) == y).numpy()
    if was:
        model.train()
    return dict(excess_bits=float(per_ex.mean()), accuracy=float(right.all(1).mean()),
                bit_accuracy=float(right.mean()), excess_bits_by_output_bit=bits.mean(0).tolist(),
                per_example_bits=per_ex.astype(np.float32))
