"""Bit encoding, the shared train/test partition, data files, the exact alignment mix and example streams.

Public API (Experiments 14 and 15 import it; keep these names): to_bits, from_bits, make_split, build_data,
octave_intervals, assemble_mix, example_stream, sample_inputs, partition_inputs.
Importing this module has no side effects.
"""
import hashlib
from pathlib import Path

import numpy as np

WORD_BITS = 10
N_INPUTS = 1 << WORD_BITS


def to_bits(x):
    """uint8 (n, 10) with bit i of x in column i."""
    x = np.asarray(x).astype(np.int64)
    return ((x[:, None] >> np.arange(WORD_BITS)) & 1).astype(np.uint8)


def from_bits(b):
    b = np.asarray(b).astype(np.int64)
    return (b << np.arange(WORD_BITS)).sum(-1).astype(np.uint16)


def make_split(c):
    """perm of 0..1023 from default_rng([task_seed, attempt]), redrawn until every input bit is 1 in 236..276 of
    perm[:512]. Train = perm[:n_train], test = perm[n_train:]. The rep768 arm uses perm[:768] and perm[768:]."""
    lo, hi = c['train_bit_count_range']
    attempt = 0
    while True:
        perm = np.random.default_rng([c['task_seed'], attempt]).permutation(N_INPUTS)
        counts = to_bits(perm[: c['n_train']]).sum(0)
        if counts.min() >= lo and counts.max() <= hi:
            break
        attempt += 1
    return dict(perm=perm.astype(np.uint16), train_idx=perm[: c['n_train']].astype(np.uint16),
                test_idx=perm[c['n_train']:].astype(np.uint16), attempts=attempt + 1,
                train_bit_counts=counts.astype(int).tolist())


def partition_inputs(split, partition='train', n_train=512):
    perm = split['perm']
    return perm[:n_train] if partition == 'train' else perm[n_train:]


def build_data(c, tasks, split, out_dir):
    """data/<task_id>_{train,test}.npz for every task: x, y (uint16), bits_x, bits_y (uint8), in split order."""
    d = Path(out_dir) / 'data'
    d.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        tab = np.asarray(t['truth_table'], np.uint16)
        for part, x in (('train', split['train_idx']), ('test', split['test_idx'])):
            y = tab[x]
            tmp = d / f"{t['task_id']}_{part}.npz.tmp"
            with open(tmp, 'wb') as f:
                np.savez(f, x=x, y=y, bits_x=to_bits(x), bits_y=to_bits(y))
            tmp.replace(d / f"{t['task_id']}_{part}.npz")


def octave_intervals(T_checkpoints):
    """[(start, end)] half-open example ranges [0, T_1), [T_1, T_2), ... (examples numbered from 0)."""
    edges = [0] + list(T_checkpoints)
    return list(zip(edges[:-1], edges[1:]))


def assemble_mix(c, tasks, target_id, alpha, seed, n_train=None, T_checkpoints=None):
    """Exact target-share schedule for one cell (PROTOCOL Section 2.5). Pure and deterministic.

    Returns dict: task_token (per example), input_index (per example, into perm[:n_train]), counts[T][task_id],
    target_passes[T], data_order_seed. Distractors are the 20 tasks with role 'distractor', ordered by token.
    """
    n_train = int(n_train or c['n_train'])
    T_checkpoints = list(T_checkpoints or c['T_checkpoints'])
    by_id = {t['task_id']: t for t in tasks}
    target = by_id[target_id]
    distractors = sorted((t for t in tasks if t['role'] == 'distractor'), key=lambda t: t['task_token'])
    dos = [int(seed), int(target['task_token']), int(round(100 * alpha))]
    rng = np.random.default_rng(dos)
    T_max, B = T_checkpoints[-1], c['batch_size']
    assert T_max % B == 0 and all(T % B == 0 for T in T_checkpoints)
    d_tokens = np.array([t['task_token'] for t in distractors], np.int64)
    slot_task = np.empty(T_max, np.int64)
    cum = np.zeros(len(d_tokens), np.int64)  # running distractor counts, so cumulative counts also stay within +-1
    for a, b in octave_intervals(T_checkpoints):
        L = b - a
        k = int(np.floor(alpha * L + 0.5))
        is_target = np.zeros(L, bool)
        is_target[rng.choice(L, k, replace=False)] = True
        rest = L - k
        base, extra = divmod(rest, len(d_tokens))
        reps = np.full(len(d_tokens), base)
        perm = rng.permutation(len(d_tokens))
        reps[perm[np.argsort(cum[perm], kind='stable')][:extra]] += 1
        cum += reps
        seg = np.full(L, target['task_token'], np.int64)
        seg[~is_target] = np.repeat(d_tokens, reps)[rng.permutation(rest)]
        slot_task[a:b] = seg
    input_index = np.empty(T_max, np.int64)
    for tok in [int(target['task_token'])] + d_tokens.tolist():
        sel = np.flatnonzero(slot_task == tok)
        passes = -(-len(sel) // n_train)
        order = np.concatenate([rng.permutation(n_train) for _ in range(passes)]) if passes else np.empty(0, np.int64)
        input_index[sel] = order[: len(sel)]
    tok2id = {t['task_token']: t['task_id'] for t in tasks if t['task_token'] is not None}
    counts, passes = {}, {}
    for T in T_checkpoints:
        tk, cnt = np.unique(slot_task[:T], return_counts=True)
        counts[str(T)] = {tok2id[int(k)]: int(v) for k, v in zip(tk, cnt)}
        passes[str(T)] = counts[str(T)].get(target_id, 0) / n_train
    return dict(task_token=slot_task, input_index=input_index, counts=counts, target_passes=passes,
                data_order_seed=dos, n_train=n_train)


def _task_hash(task_id):
    return int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], 'big')


def example_stream(c, task, n, seed, partition='train', split=None):
    """First n examples of successive permutation passes over the partition (content repeats after one pass).

    rng = default_rng([seed, h]) with h the first 4 bytes of SHA-256(task_id) as an unsigned big-endian integer.
    Returns dict x, y (uint16), bits_x, bits_y (uint8).
    """
    split = split or make_split(c)
    inp = partition_inputs(split, partition, c['n_train'])
    rng = np.random.default_rng([int(seed), _task_hash(task['task_id'])])
    passes = -(-n // len(inp))
    idx = np.concatenate([rng.permutation(len(inp)) for _ in range(passes)])[:n]
    x = inp[idx]
    y = np.asarray(task['truth_table'], np.uint16)[x]
    return dict(x=x, y=y, bits_x=to_bits(x), bits_y=to_bits(y))


def sample_inputs(rng, n, partition, split, n_train=512):
    """n uniform draws with replacement from the partition's inputs (uint16)."""
    inp = partition_inputs(split, partition, n_train)
    return inp[rng.integers(0, len(inp), n)]
