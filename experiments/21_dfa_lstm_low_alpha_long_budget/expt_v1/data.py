"""Strings, the hash train/test partition, task pools, the shared test set and the exact alignment mix.

Public API (Experiment 15 imports assemble_mix): to_ascii, in_test_partition, sample_strings,
build_test_set, build_pool, labels_for, assemble_mix, octave_intervals, length_schedule.
"""
import hashlib

import numpy as np

from dfa import DFA, run_dfa

TEST_STREAM, POOL_STREAM = 1, 2


def to_ascii(bits, length):
    """ASCII bytes per string: character i is 'b' if bit i is set, else 'a'."""
    bits = np.asarray(bits, np.uint64)
    shifts = np.arange(length, dtype=np.uint64)
    chars = np.where((bits[:, None] >> shifts) & np.uint64(1), ord('b'), ord('a')).astype(np.uint8)
    return [row.tobytes() for row in chars]


def in_test_partition(bits, length, c):
    """True where the first byte of SHA-256 of the ASCII string is below test_hash_byte_below."""
    return np.array([hashlib.sha256(s).digest()[0] < c['test_hash_byte_below'] for s in to_ascii(bits, length)],
                    bool)


def sample_strings(rng, count, length, partition, c):
    """`count` uniform strings of `length`, rejection-sampled into partition 'train' or 'test' (with replacement)."""
    want_test = partition == 'test'
    out = []
    have = 0
    while have < count:
        k = (count - have) * (10 if want_test else 2) + 16
        cand = rng.integers(0, 2 ** length, k, dtype=np.uint64)
        keep = cand[in_test_partition(cand, length, c) == want_test]
        out.append(keep)
        have += len(keep)
    return np.concatenate(out)[:count].astype(np.uint32)


def lengths(c):
    return list(range(c['length_min'], c['length_max'] + 1))


def build_test_set(c):
    """Shared test set ordered by length then index: 241 per length 8..23 and 240 at 24 (4,096 total)."""
    Ls = lengths(c)
    base, extra = divmod(c['test_size'], len(Ls))
    bits, lens = [], []
    for i, L in enumerate(Ls):
        cnt = base + (1 if i < extra else 0)
        rng = np.random.default_rng([c['task_seed'], TEST_STREAM, L])
        bits.append(sample_strings(rng, cnt, L, 'test', c))
        lens.append(np.full(cnt, L, np.uint8))
    return np.concatenate(bits), np.concatenate(lens)


def build_pool(c, task):
    """Task-specific train-partition pool: pool_per_length strings per length, ordered by length then index."""
    bits, lens = [], []
    for L in lengths(c):
        rng = np.random.default_rng([c['task_seed'], POOL_STREAM, task['task_token'], L])
        bits.append(sample_strings(rng, c['pool_per_length'], L, 'train', c))
        lens.append(np.full(c['pool_per_length'], L, np.uint8))
    return np.concatenate(bits), np.concatenate(lens)


def labels_for(task, bits, lens):
    return run_dfa(DFA.from_json(task['definition']), bits, lens).astype(np.uint8)


def octave_intervals(T_checkpoints):
    """[(start, end)] half-open example ranges: [0, T_1), [T_1, T_2), ... (examples numbered from 0)."""
    edges = [0] + list(T_checkpoints)
    return list(zip(edges[:-1], edges[1:]))


def length_schedule(rng, n_batches, Ls):
    base, extra = divmod(n_batches, len(Ls))
    plus = set(rng.permutation(len(Ls))[:extra].tolist())
    sched = np.concatenate([np.full(base + (1 if i in plus else 0), L, np.int64) for i, L in enumerate(Ls)])
    return sched[rng.permutation(len(sched))]


def assemble_mix(c, tasks, target_id, alpha, seed, T_checkpoints=None):
    """Exact target-share schedule for one cell (PROTOCOL Section 2.3). Pure and deterministic.

    Returns dict with per-example arrays task_token, pool_index (into that task's pool, which is ordered by
    length then index), batch_length (per batch), and counts[T][task_id] = examples of that task in [0, T).
    """
    T_checkpoints = list(T_checkpoints or c['T_checkpoints'])
    by_id = {t['task_id']: t for t in tasks}
    target = by_id[target_id]
    distractors = sorted((t for t in tasks if t['role'] == 'distractor'), key=lambda t: t['task_token'])
    rng = np.random.default_rng([seed, target['task_token'], int(round(100 * alpha))])
    T_max, B = T_checkpoints[-1], c['batch_size']
    assert T_max % B == 0 and all(T % B == 0 for T in T_checkpoints)
    slot_task = np.empty(T_max, np.int64)
    d_tokens = np.array([t['task_token'] for t in distractors])
    for a, b in octave_intervals(T_checkpoints):
        L = b - a
        k = int(np.floor(alpha * L + 0.5))
        is_target = np.zeros(L, bool)
        is_target[rng.choice(L, k, replace=False)] = True
        rest = L - k
        base, extra = divmod(rest, len(d_tokens))
        plus = rng.permutation(len(d_tokens))[:extra]
        reps = np.full(len(d_tokens), base)
        reps[plus] += 1
        deal = np.repeat(d_tokens, reps)[rng.permutation(rest)]
        seg = np.full(L, target['task_token'], np.int64)
        seg[~is_target] = deal
        slot_task[a:b] = seg
    Ls = lengths(c)
    batch_length = length_schedule(rng, T_max // B, Ls)
    ex_length = np.repeat(batch_length, B)
    P = c['pool_per_length']
    used = [target] + distractors
    pool_index = np.empty(T_max, np.int64)
    reuse = 0
    for t in used:
        for li, L in enumerate(Ls):
            perm = rng.permutation(P)
            sel = np.flatnonzero((slot_task == t['task_token']) & (ex_length == L))
            if len(sel) > P:  # only the 2^17 pilot can exhaust a pool; later passes use fresh permutations
                reuse += len(sel) - P
                reps = -(-len(sel) // P)
                perm = np.concatenate([perm] + [np.random.default_rng([seed, t['task_token'], L, r]).permutation(P)
                                                for r in range(1, reps)])
            pool_index[sel] = li * P + perm[: len(sel)]
    counts = {}
    for T in T_checkpoints:
        tok, cnt = np.unique(slot_task[:T], return_counts=True)
        tok2id = {t['task_token']: t['task_id'] for t in tasks}
        counts[str(T)] = {tok2id[int(k)]: int(v) for k, v in zip(tok, cnt)}
    return dict(task_token=slot_task, pool_index=pool_index, batch_length=batch_length,
                counts=counts, pool_reuse=reuse, data_order_seed=[seed, target['task_token'], int(round(100 * alpha))])
