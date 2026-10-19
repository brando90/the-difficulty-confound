"""Small causal transformer that writes the sum one digit at a time.

Prompt tokens: a (hundreds, tens, ones) '+' b (hundreds, tens, ones) '='.
Answer tokens: the 4-digit sum, most significant digit first (the order people write it).
Training and conditional bits use teacher forcing; accuracy uses greedy decoding.
Digit tensors outside this module stay least-significant-first, as in Experiments 01-09.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PLUS, EQUALS = 10, 11
PROMPT_LEN, ANSWER_LEN = 8, 4
MAX_LEN = PROMPT_LEN + ANSWER_LEN - 1


def prompt_tokens(a_digits_lsb, b_digits_lsb):
    """(N, 3) least-significant-first digits -> (N, 8) prompt tokens, most significant first."""
    n = len(a_digits_lsb)
    return np.concatenate(
        [a_digits_lsb[:, ::-1], np.full((n, 1), PLUS), b_digits_lsb[:, ::-1], np.full((n, 1), EQUALS)], 1
    ).astype(np.int64)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h, mask):
        b, t, w = h.shape
        q, k, v = self.qkv(self.ln1(h)).reshape(b, t, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(w // self.heads)
        att = att.masked_fill(mask[:t, :t], float('-inf')).softmax(-1)
        h = h + self.proj((att @ v).transpose(1, 2).reshape(b, t, w))
        return h + self.ff(self.ln2(h))


class AdditionTransformer(nn.Module):
    def __init__(self, width, layers=2, heads=4):
        super().__init__()
        assert width % heads == 0
        self.tok = nn.Embedding(12, width)
        self.pos = nn.Embedding(MAX_LEN, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 10)
        self.register_buffer('mask', torch.triu(torch.ones(MAX_LEN, MAX_LEN, dtype=torch.bool), 1), persistent=False)

    def _logits(self, seq):
        t = seq.shape[1]
        h = self.tok(seq) + self.pos(torch.arange(t))
        for block in self.blocks:
            h = block(h, self.mask)
        return self.head(self.norm(h))

    def forward(self, prompt, y_lsb):
        """Teacher-forced logits for the 4 answer digits, returned least-significant-first: (B, 4, 10)."""
        answer_msb = y_lsb.flip(1)
        z = self._logits(torch.cat([prompt, answer_msb[:, : ANSWER_LEN - 1]], 1))
        return z[:, PROMPT_LEN - 1 :].flip(1)

    @torch.no_grad()
    def generate(self, prompt):
        """Greedy decoding; returns (B, 4) digits least-significant-first."""
        seq = prompt
        for _ in range(ANSWER_LEN):
            nxt = self._logits(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
        return seq[:, PROMPT_LEN:].flip(1)


def model(config):
    return AdditionTransformer(config['hidden_width'], config.get('layers', 2), config.get('heads', 4))
