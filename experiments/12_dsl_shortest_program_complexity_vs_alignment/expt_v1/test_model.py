"""Unit tests for model.py and run.train_cell determinism."""
import json
import math
from pathlib import Path

import numpy as np
import torch

import data as D
import model as M
import run as R
from test_data import toy_tasks

C = json.loads((Path(__file__).parent / 'config.json').read_text())
torch.set_num_threads(1)


def test_parameter_counts():
    assert M.n_params(M.build_model(C)) == 103618
    assert M.n_params(M.build_model(C, 41)) == 103682
    assert M.n_params(M.build_mlp(C)) == 83988


def test_qkv_layout_scaling_invariance():
    torch.manual_seed(0)
    m = M.build_model(C)
    x = M.encode(np.arange(8) % 40, D.to_bits(np.arange(8) * 97 % 1024))
    with torch.no_grad():
        ref = m(x).clone()
        d = C['d_model']
        for blk in m.blocks:
            blk.qkv.weight[:d] *= 2.0
            blk.qkv.bias[:d] *= 2.0
            blk.qkv.weight[d:2 * d] *= 0.5
            blk.qkv.bias[d:2 * d] *= 0.5
        out = m(x)
    assert (out - ref).abs().max().item() < 1e-5
    with torch.no_grad():  # scaling the value rows does change the output, so the slices are not interchangeable
        for blk in m.blocks:
            blk.qkv.weight[2 * d:] *= 2.0
        assert (m(x) - ref).abs().max().item() > 1e-3


def test_bidirectional():
    torch.manual_seed(1)
    m = M.build_model(C)
    b = D.to_bits(np.array([0, 512]))  # differ only in input bit 9
    with torch.no_grad():
        out = m(M.encode(3, b))
    assert (out[0, 0] - out[1, 0]).abs().max().item() > 0  # position 1 (output bit 0) sees input bit 9


def test_npz_round_trip(tmp_path):
    torch.manual_seed(0)
    m = M.build_model(C)
    M.export_npz(m, tmp_path / 'w.npz')
    m2 = M.load_npz(M.build_model(C), tmp_path / 'w.npz')
    x = M.encode(5, D.to_bits(np.arange(64)))
    with torch.no_grad():
        assert (m(x) - m2(x)).abs().max().item() == 0.0


def test_train_cell_deterministic(tmp_path):
    tasks = toy_tasks()
    kw = dict(T_checkpoints=[640, 1280])  # 20 steps
    runs = []
    for seed, sub in ((12101, 'a'), (12101, 'b'), (12202, 'c')):
        R.train_cell(C, tasks, 'toy_target_2', 0.4, seed, tmp_path / sub, **kw)
        z = np.load(tmp_path / sub / 'checkpoints' / R.cell_name(seed, 'toy_target_2', 0.4) / 'T_1280.npz')
        runs.append({k: z[k] for k in z.files})
    assert all(np.array_equal(runs[0][k], runs[1][k]) for k in runs[0])
    assert any(not np.array_equal(runs[0][k], runs[2][k]) for k in runs[0])
    pa = np.load(tmp_path / 'a' / 'predictions' / f"{R.cell_name(12101, 'toy_target_2', 0.4)}.npz")
    pb = np.load(tmp_path / 'b' / 'predictions' / f"{R.cell_name(12101, 'toy_target_2', 0.4)}.npz")
    assert np.array_equal(pa['test'], pb['test']) and np.array_equal(pa['train'], pb['train'])


def test_evaluate_matches_naive_loop():
    torch.manual_seed(2)
    m = M.build_model(C)
    tab = np.array(toy_tasks()[4]['truth_table'], np.uint16)
    x = np.arange(0, 1024, 7)
    bx, by = D.to_bits(x), D.to_bits(tab[x])
    ev = M.evaluate(m, bx, 4, by)
    with torch.no_grad():
        for i in range(len(x)):
            lp = torch.log_softmax(m(M.encode(4, bx[i:i + 1])).float(), -1)[0]
            b = -sum(lp[j, int(by[i, j])].item() for j in range(10)) / math.log(2)
            assert abs(b - float(ev['per_example_bits'][i])) < 1e-5  # float32 batch-versus-single kernels
    assert abs(ev['excess_bits'] - ev['per_example_bits'].astype(np.float64).mean()) < 1e-5
    # the batch statistic itself is reproduced exactly by the same float64 accumulation
    with torch.no_grad():
        lp = torch.log_softmax(m(M.encode(4, bx)).float(), -1)
    nll = -lp.gather(-1, torch.from_numpy(by.astype(np.int64))[..., None])[..., 0].numpy().astype(np.float64)
    assert abs(ev['excess_bits'] - (nll / math.log(2)).sum(1).mean()) < 1e-12
