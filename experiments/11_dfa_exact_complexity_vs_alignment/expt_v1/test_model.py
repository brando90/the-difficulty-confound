"""Unit tests for model.py and run.train_cell determinism."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import data as D
import dfa as A
import model as M
import run as R

C = json.loads((Path(__file__).parent / 'config.json').read_text())
torch.set_num_threads(1)


def test_parameter_counts():
    assert M.n_params(M.build_model(C)) == 104002
    assert M.n_params(M.build_lstm(C)) == 84290


def test_npz_round_trip(tmp_path):
    torch.manual_seed(0)
    m = M.build_model(C)
    M.export_npz(m, tmp_path / 'w.npz')
    m2 = M.load_npz(M.build_model(C), tmp_path / 'w.npz')
    x = M.encode_batch(np.full(5, 3), np.arange(5), 12)
    assert (m(x) - m2(x)).abs().max().item() == 0.0


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp('exp11')
    tasks, _, tb, tl, pools = R.build_all(C)
    R.write_data(out, tasks, tb, tl, pools)
    return out, R.load_data(out)


def test_train_cell_deterministic(prepared):
    out, data = prepared
    kw = dict(data=data, T_checkpoints=[512, 1280])
    runs = []
    for seed, sub in ((11101, 'a'), (11101, 'b'), (11202, 'c')):
        R.train_cell(C, data['tasks'], 'dfa_n08_t2', 0.4, seed, out / sub, **kw)
        z = np.load(out / sub / 'checkpoints' / R.cell_name(seed, 'dfa_n08_t2', 0.4) / 'T_1280.npz')
        runs.append({k: z[k] for k in z.files})
    assert all(np.array_equal(runs[0][k], runs[1][k]) for k in runs[0])
    assert any(not np.array_equal(runs[0][k], runs[2][k]) for k in runs[0])


def test_evaluate_matches_naive_loop(prepared):
    _, data = prepared
    torch.manual_seed(1)
    m = M.build_model(C)
    tok = 5
    lab = data['test_labels'][tok]
    ev = M.evaluate(m, tok, data['test_bits'], data['test_lens'], lab)
    idx = np.random.default_rng(0).choice(4096, 300, replace=False)
    with torch.no_grad():
        for i in idx:
            logits = m(M.encode_batch(np.array([tok]), data['test_bits'][i:i + 1], int(data['test_lens'][i])))
            bits = -torch.log_softmax(logits, -1)[0, int(lab[i])].item() / math.log(2)
            assert abs(bits - float(ev['per_string_bits'][i])) < 1e-5  # float32 batch-vs-single kernels
    assert abs(ev['excess_bits'] - ev['per_string_bits'].astype(np.float64).mean()) < 1e-12
