"""Shared pieces for Experiment 19: Experiment 11's code and tasks, the train/test partition as a lookup table,
DFA labels on the GPU and the exact target-share schedule.

Experiment 11's dfa.py, data.py, model.py, config.json and results/tasks.json are used unchanged (imported from
../../11_dfa_exact_complexity_vs_alignment/expt_v1).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E11 = HERE.parents[1] / '11_dfa_exact_complexity_vs_alignment' / 'expt_v1'


def _load(name):
    sys.path.insert(0, str(E11))
    spec = importlib.util.spec_from_file_location(f'e11_{name}', E11 / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DFA11 = _load('dfa')
D11 = _load('data')
M11 = _load('model')


def load_c11():
    return json.loads((E11 / 'config.json').read_text())


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def load_tasks():
    return json.loads((E11 / 'results' / 'tasks.json').read_text())


def test_mask_for_length(L, c):
    """Boolean array over all 2^L strings (index = bit pattern, bit i = character i): True = test partition."""
    N = 2 ** L
    out = np.empty(N, bool)
    shifts = np.arange(L, dtype=np.uint64)
    thr = c['test_hash_byte_below']
    step = 1 << 16
    for a in range(0, N, step):
        idx = np.arange(a, min(N, a + step), dtype=np.uint64)
        chars = np.where((idx[:, None] >> shifts) & np.uint64(1), ord('b'), ord('a')).astype(np.uint8)
        buf = chars.tobytes()
        out[a:a + len(idx)] = [hashlib.sha256(buf[i * L:(i + 1) * L]).digest()[0] < thr for i in range(len(idx))]
    return out


def _mask_worker(args):
    L, c = args
    return L, np.packbits(test_mask_for_length(L, c))


def test_masks(c, cache_path=None, workers=1):
    """{L: bool array} for every training length, cached as packed bits in cache_path (an .npz)."""
    Ls = list(range(c['length_min'], c['length_max'] + 1))
    if cache_path is not None and Path(cache_path).exists():
        with np.load(cache_path) as z:
            return {L: np.unpackbits(z[f'L{L}'])[: 2 ** L].astype(bool) for L in Ls}
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as p:
            packed = dict(p.map(_mask_worker, [(L, c) for L in sorted(Ls, reverse=True)]))
    else:
        packed = dict(_mask_worker((L, c)) for L in Ls)
    if cache_path is not None:
        tmp = str(cache_path) + '.tmp.npz'
        np.savez(tmp, **{f'L{L}': packed[L] for L in Ls})
        Path(tmp).replace(cache_path)
    return {L: np.unpackbits(packed[L])[: 2 ** L].astype(bool) for L in Ls}


class TaskTables:
    """Transition and accept tables of all 32 tasks, padded to 32 states, for vectorized labels on a device."""

    def __init__(self, tasks, device):
        nt = len(tasks)
        S = max(t['definition']['n_states'] for t in tasks)
        trans = np.zeros((nt, S, 2), np.int64)
        acc = np.zeros((nt, S), np.int64)
        for t in tasks:
            d = DFA11.DFA.from_json(t['definition'])
            trans[t['task_token'], :d.n] = d.trans
            acc[t['task_token'], :d.n] = d.accept
        self.S = S
        self.trans = torch.from_numpy(trans.reshape(-1)).to(device)
        self.acc = torch.from_numpy(acc.reshape(-1)).to(device)

    def labels(self, task_token, bits, L):
        """task_token, bits: int64 tensors of one shape; every string has length L. Returns int64 labels."""
        state = torch.zeros_like(bits)
        base = task_token * self.S
        for i in range(L):
            c = (bits >> i) & 1
            state = self.trans[(base + state) * 2 + c]
        return self.acc[base + state]


def target_counts(alpha, B, n_steps):
    """Target examples per step so that the running count after T examples is round(alpha * T) at every step."""
    s = np.arange(n_steps + 1)
    cum = np.floor(alpha * B * s + 0.5).astype(np.int64)
    return np.diff(cum)


def tokens(task_token, bits, L):
    """LongTensor (..., L + 1): [2 + task token, c_1..c_L] with c_i = bit i (a = 0, b = 1)."""
    ch = (bits[..., None] >> torch.arange(L, device=bits.device)) & 1
    return torch.cat([(task_token + 2)[..., None], ch], dim=-1)
