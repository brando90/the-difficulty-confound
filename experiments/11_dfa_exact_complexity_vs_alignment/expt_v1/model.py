"""The 104,002-parameter decoder-only transformer, the optional 84,290-parameter LSTM, npz export and evaluation.

Public API: build_model, build_lstm, encode_batch, export_npz, load_npz, evaluate.
Tokens: a = 0, b = 1, task token k = 2 + k. Input is [task token, c_1..c_l]; output is a 2-way logit at the last
position.
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


class Transformer(nn.Module):
    def __init__(self, vocab, d, n_layers, h, d_mlp, max_pos):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_pos, d)
        self.blocks = nn.ModuleList([Block(d, h, d_mlp) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h[:, -1]))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.tok = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(self.tok(x))
        return self.head(out[:, -1])


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(c, n_task_tokens=32):
    m = Transformer(2 + n_task_tokens, c['d_model'], c['n_layers'], c['n_heads'], c['d_mlp'], c['max_positions'])
    if n_task_tokens == 32:
        assert n_params(m) == c['expected_parameters'], n_params(m)
    return m


def build_lstm(c, n_task_tokens=32):
    m = LSTMClassifier(2 + n_task_tokens, c['lstm']['embed'], c['lstm']['hidden'])
    if n_task_tokens == 32:
        assert n_params(m) == c['lstm']['expected_parameters'], n_params(m)
    return m


def encode_batch(task_token, bits, length):
    """LongTensor (B, length + 1) for strings that all have the same length."""
    bits = np.asarray(bits, np.uint64)
    chars = ((bits[:, None] >> np.arange(length, dtype=np.uint64)) & np.uint64(1)).astype(np.int64)
    tok = np.broadcast_to(np.asarray(task_token, np.int64).reshape(-1, 1) + 2, (len(bits), 1))
    return torch.from_numpy(np.concatenate([tok, chars], axis=1))


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
def evaluate(model, task_token, test_bits, test_lens, test_labels):
    """Mean held-out bits, accuracy, per-length bits and per-string bits (test set ordered by length, index).

    One batch per length; float32 forward; -log2 p(correct) accumulated in float64.
    """
    model.eval()
    per_string = np.empty(len(test_bits), np.float32)
    correct = np.zeros(len(test_bits), bool)
    by_len = []
    for L in np.unique(test_lens):
        idx = np.flatnonzero(test_lens == L)
        logits = model(encode_batch(task_token, test_bits[idx], int(L)))
        logp = F.log_softmax(logits.float(), dim=-1)
        y = torch.from_numpy(test_labels[idx].astype(np.int64))
        nll = -logp.gather(1, y[:, None])[:, 0]
        per_string[idx] = (nll / math.log(2.0)).numpy().astype(np.float32)
        correct[idx] = (logits.argmax(-1) == y).numpy()
        by_len.append(float(per_string[idx].astype(np.float64).mean()))
    model.train()
    return dict(excess_bits=float(per_string.astype(np.float64).mean()), accuracy=float(correct.mean()),
                excess_bits_by_length=by_len, per_string_bits=per_string)
