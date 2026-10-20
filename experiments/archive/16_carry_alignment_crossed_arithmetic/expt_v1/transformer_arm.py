"""Arm 16b: fresh 8-digit addition with nanoGPT-style decoders (PROTOCOL 2.2-2.6).

Training stream. Example j of stratum c for training seed s is element j mod B of block j // B, where block k is drawn
by the exact within-stratum sampler from NumPy PCG64 seeded by SeedSequence([s, c, k]) (B = stream_block). Example j
is therefore the same in every run of that seed, whatever alpha or model size (common random numbers). Slot k of the
schedule is a target example iff floor((k+1) alpha) - floor(k alpha) == 1; non-target slots cycle through the other
training levels in a seeded permutation. Test pairs and their commuted twins are rejected and counted.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import carry


# --- model --------------------------------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.heads = heads

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class Decoder(nn.Module):
    """Pre-LayerNorm decoder, learned positions, GELU MLP of width 4d, tied embeddings, no dropout."""

    def __init__(self, layers, d, heads, vocab=13, block=32):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(block, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.tok.weight.T


def build(t, size):
    s = t['sizes'][size]
    return Decoder(s['layers'], s['d'], s['heads'], len(t['vocab']), t['block'])


def param_counts(net):
    total = sum(p.numel() for p in net.parameters())
    emb = net.tok.weight.numel() + net.pos.weight.numel()
    return total, total - emb


# --- data ---------------------------------------------------------------------------------------------------

class StratumStream:
    """Example j of stratum c for training seed s, in blocks, with test rejection."""

    def __init__(self, seed, c, block, reject, digits=8):
        self.seed, self.c, self.block, self.reject, self.digits = seed, c, block, reject, digits
        self.F = carry.completion_table(digits, c)
        self.cache, self.rejections = {}, 0

    def _block(self, k):
        if k not in self.cache:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.c, k]))
            a, b = carry.sample_stratum(rng, self.c, self.block, self.digits, self.F)
            bad = np.isin(a * 10 ** self.digits + b, self.reject)
            while bad.any():  # expected rate below 1e-10; every rejection is counted
                self.rejections += int(bad.sum())
                a2, b2 = carry.sample_stratum(rng, self.c, int(bad.sum()), self.digits, self.F)
                a[bad], b[bad] = a2, b2
                bad = np.isin(a * 10 ** self.digits + b, self.reject)
            self.cache = {k: (a, b)}  # keep one block: streams are read in order
        return self.cache[k]

    def take(self, j0, n):
        """Examples j0..j0+n-1, read block by block."""
        out_a, out_b, j = [], [], j0
        while j < j0 + n:
            k, off = divmod(j, self.block)
            m = min(self.block - off, j0 + n - j)
            a, b = self._block(k)
            out_a.append(a[off:off + m])
            out_b.append(b[off:off + m])
            j += m
        return np.concatenate(out_a), np.concatenate(out_b)


def schedule(alpha, target, others, seed, k0, n):
    """Stratum of training slots k0..k0+n-1: Bresenham target slots; others cycle through a seeded permutation."""
    k = np.arange(k0, k0 + n, dtype=np.int64)
    num = int(round(alpha * 10000))  # exact integer arithmetic: alpha is a multiple of 1e-4 on every grid
    assert abs(num / 10000 - alpha) < 1e-12
    targets_before = (k * num) // 10000
    is_target = ((k + 1) * num) // 10000 - targets_before == 1
    perm = np.random.default_rng([seed, 16]).permutation(others)
    return np.where(is_target, target, perm[(k - targets_before) % len(perm)])


def test_set(t, split_seed):
    """2,000 pairs per stratum c = 0..8 from the split seed; pairs and commuted twins are unique."""
    pairs = {}
    seen = set()
    for c in range(t['digits'] + 1):
        rng = np.random.default_rng(np.random.SeedSequence([split_seed, c, 0x7E57]))
        F_c = carry.completion_table(t['digits'], c)
        a, b = carry.sample_stratum(rng, c, t['test_per_stratum'], t['digits'], F_c)
        for i in range(len(a)):
            while (int(a[i]), int(b[i])) in seen or (int(b[i]), int(a[i])) in seen:
                a2, b2 = carry.sample_stratum(rng, c, 1, t['digits'], F_c)
                a[i], b[i] = a2[0], b2[0]
            seen.add((int(a[i]), int(b[i])))
        pairs[c] = (a, b)
    keys = np.array(sorted({p[0] * 10 ** t['digits'] + p[1] for p in seen} | {p[1] * 10 ** t['digits'] + p[0] for p in seen}))
    return pairs, keys


# --- training and evaluation --------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(net, a, b, fmt, device, batch=2000):
    """float64 teacher-forced bits per answer, per-digit bits (by answer position), and greedy exact match."""
    net64 = net.double().eval()
    pos = torch.as_tensor(carry.answer_positions(), device=device)
    bits_all, correct_all = [], []
    for off in range(0, len(a), batch):
        tok = torch.as_tensor(carry.encode(a[off:off + batch], b[off:off + batch], fmt), device=device)
        logits = net64(tok[:, :-1])
        lp = F.log_softmax(logits[:, pos - 1], -1)
        bits = -lp.gather(-1, tok[:, pos].unsqueeze(-1)).squeeze(-1) / math.log(2)
        seq = tok[:, :pos[0]].clone()
        for _ in range(len(pos)):
            seq = torch.cat([seq, net64(seq)[:, -1].argmax(-1, keepdim=True)], 1)
        correct = (seq[:, pos[0]:] == tok[:, pos]).all(1)
        bits_all.append(bits.cpu().numpy())
        correct_all.append(correct.cpu().numpy())
    net.float().train()
    return np.concatenate(bits_all), np.concatenate(correct_all)


def train_run(t, size, fmt, c_target, alpha, seed, lr, T_checkpoints, test_pairs, reject, device, ckpt_dir=None,
              log=print, run_meta=None):
    """One 16b run; returns metric rows (every checkpoint, every stratum) and run information."""
    torch.manual_seed(seed)
    net = build(t, size).to(device)
    decay = [p for p in net.parameters() if p.dim() >= 2]
    no_decay = [p for p in net.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=t['weight_decay']), dict(params=no_decay, weight_decay=0.0)],
                            lr=lr, betas=tuple(t['betas']))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / t['warmup']))
    levels = t['train_levels']
    others = [lv for lv in levels if lv != c_target]
    streams = {lv: StratumStream(seed, lv, t['stream_block'], reject) for lv in levels}
    used = {lv: 0 for lv in levels}
    pos = torch.as_tensor(carry.answer_positions(), device=device)
    bs = t['batch_size']
    steps = max(T_checkpoints) // bs
    amp = device.type == 'cuda'
    rows, started, examples = [], time.monotonic(), 0
    for step in range(1, steps + 1):
        strata = schedule(alpha, c_target, others, seed, examples, bs)
        a, b = np.empty(bs, np.int64), np.empty(bs, np.int64)
        for lv in levels:
            m = strata == lv
            if m.any():
                a[m], b[m] = streams[lv].take(used[lv], int(m.sum()))
                used[lv] += int(m.sum())
        tok = torch.as_tensor(carry.encode(a, b, fmt), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = net(tok[:, :-1])
        loss = F.cross_entropy(logits[:, pos - 1].float().reshape(-1, logits.shape[-1]), tok[:, pos].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), t['clip'])
        opt.step()
        sched.step()
        examples += bs
        if examples in T_checkpoints:
            state = {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()}
            if ckpt_dir is not None:
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                np.savez(Path(ckpt_dir) / f'T_{examples}.npz', **state)
            for c_eval, (ta, tb) in test_pairs.items():
                bits, correct = evaluate(net, ta, tb, fmt, device, t['eval_batch'])
                rows.append(metric_row(t, size, fmt, c_target, alpha, seed, examples, step, c_eval, bits, correct,
                                       param_counts(net)[1]))
            log(f'step {step} T {examples} loss {loss.item():.4f} {time.monotonic() - started:.0f}s')
    info = dict(seconds=time.monotonic() - started, examples=examples, used=used,
                rejections=int(sum(s.rejections for s in streams.values())), **(run_meta or {}))
    return rows, info


def metric_row(t, size, fmt, c_target, alpha, seed, T, step, c_eval, bits, correct, n_nonembedding):
    places = carry.answer_place(fmt)
    row = dict(seed=seed, target_id=f'c{c_target}', K=c_target + 1, K_bits='', alpha=alpha, T=T, step=step,
               excess_bits=float(bits.sum(1).mean()), accuracy=float(correct.mean()), n_test=int(len(bits)),
               arm='transformer', format=fmt, N=n_nonembedding, size=size, c=c_target, s=c_target + 1,
               eval_stratum=f'c{c_eval}', eval_c=c_eval)
    for j in range(9):  # digit_bits_j is the answer digit of decimal place j
        row[f'digit_bits_{j}'] = float(bits[:, list(places).index(j)].mean())
    return row


# --- shard map (PROTOCOL 2.6) -------------------------------------------------------------------------------

def shard(t, i):
    """Shard id -> run. Primary 0-179: 60*i_N + 12*i_c + 3*i_alpha + i_seed; control 180-239 at control_size."""
    primary = t['primary_format_final'] or t['primary_format']
    control = 'reversed' if primary == 'plain' else 'plain'
    if i < 180:
        i_n, r = divmod(i, 60)
        size, fmt = t['size_order'][i_n], primary
    elif i < t['n_shards']:
        r, size, fmt = i - 180, t['control_size'], control
    else:
        raise ValueError(i)
    i_c, r = divmod(r, 12)
    i_a, i_s = divmod(r, 3)
    return dict(shard=i, size=size, format=fmt, c=t['train_levels'][i_c], alpha=t['alpha_grid'][i_a], seed=t['seeds'][i_s])


def pilot_job(t, j):
    i_size, i_lr = divmod(j, 3)
    size = t['size_order'][i_size]
    mult = t['pilot_lr_multipliers'][i_lr]
    return dict(job=j, size=size, lr_multiplier=mult, lr=t['sizes'][size]['lr'] * mult, c=t['pilot_c'],
                alpha=t['pilot_alpha'], seed=t['pilot_seed'], format=t['primary_format'])


def pilot_select(t, results):
    """Frozen rule (PROTOCOL 2.8): per size, the multiplier with the lowest pilot excess at 2^22 (ties within
    pilot_tie_bits go to 1); swap formats if the best 6M run is above format_swap_threshold_bits."""
    chosen = {}
    for size in t['size_order']:
        rr = sorted((r for r in results if r['size'] == size), key=lambda r: r['excess_bits'])
        best = rr[0]
        one = next(r for r in rr if r['lr_multiplier'] == 1.0)
        chosen[size] = 1.0 if one['excess_bits'] - best['excess_bits'] <= t['pilot_tie_bits'] else best['lr_multiplier']
    best6 = min(r['excess_bits'] for r in results if r['size'] == t['control_size'])
    swap = best6 > t['format_swap_threshold_bits']
    return dict(lr_multiplier=chosen, lr_selected={s: t['sizes'][s]['lr'] * m for s, m in chosen.items()},
                best_6M_excess=best6, primary_format_final='reversed' if swap else 'plain')
