"""Experiment 13's model tests on Experiment 17's config: parameter counts and the exact npz round trip."""
import json
from pathlib import Path

import numpy as np
import torch

import model as M

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())


def test_small_counts():
    m = M.build_model(CFG['model'])
    assert M.param_count(m) == 354240 and M.nonembedding_count(m) == 335712


def test_npz_round_trip(tmp_path):
    torch.manual_seed(0)
    m = M.build_model(CFG['model'])
    M.save_npz(m, tmp_path / 'w.npz')
    torch.manual_seed(1)
    m2 = M.load_npz(M.build_model(CFG['model']), tmp_path / 'w.npz')
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    docs = np.random.default_rng(0).integers(0, 64, size=(8, 128))
    assert np.array_equal(M.code_lengths(m, docs, 64), M.code_lengths(m2, docs, 64))


def test_code_lengths_are_causal_and_normalized():
    torch.manual_seed(0)
    m = M.build_model(CFG['model'])
    docs = np.random.default_rng(1).integers(0, 64, size=(4, 128))
    a = M.code_lengths(m, docs, 64)
    docs2 = docs.copy()
    docs2[:, 100:] = (docs2[:, 100:] + 1) % 64
    b = M.code_lengths(m, docs2, 64)
    assert np.array_equal(a[:, :100], b[:, :100])
    x, _ = M.inputs_targets(docs, 64)
    with torch.no_grad():
        p = torch.softmax(m.double()(x), -1).sum(-1)
    assert torch.allclose(p, torch.ones_like(p), atol=1e-12)
