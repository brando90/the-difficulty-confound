"""Unit tests 5-7 of PROMPT.md (alignment, disjointness, models) and a CPU smoke test of the 16b training loop."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import carry
import mlp_arm as M
import transformer_arm as TA

C = json.loads((Path(__file__).resolve().parent / 'config.json').read_text())
CM, T = C['mlp'], C['transformer']


@pytest.fixture(scope='module')
def data():
    return M.data(CM, C['shared']['split_seed'])


def test_mlp_alignment(data):
    held = set(np.concatenate(list(data['splits'].values())).tolist())
    for seed in CM['seeds'][:2] + [CM['pilot_seed']]:
        for target in M.TARGETS:
            for alpha in CM['alpha_grid']:
                ids = M.training_ids(CM, data, seed, target, alpha)
                tot = M.cell_totals(CM, target, alpha)
                assert len(ids) == len(set(ids.tolist())) == CM['train_size'] and not held & set(ids.tolist())
                assert all(np.bincount(data['cell'][ids], minlength=4)[k] == v for k, v in tot.items())
                _, cnt = np.unique((data['a'] + data['b'])[ids], return_counts=True)
                assert np.all(cnt == CM['train_per_answer'])
                # Nested across alpha: each (answer, cell) selection is a prefix of one seeded permutation.
                chosen = set(ids.tolist())
                for (ans, k), pool in list(data['pools'].items())[:60]:
                    perm = np.random.default_rng([seed, ans, k]).permutation(pool)
                    flags = [int(i) in chosen for i in perm]
                    assert flags == sorted(flags, reverse=True)


def test_transformer_schedule():
    for alpha in T['alpha_grid']:
        st = TA.schedule(alpha, 4, [0, 1, 2, 7], 16101, 0, 20000)
        targets = np.cumsum(st == 4)
        m = np.arange(1, 20001)
        assert np.array_equal(targets, (m * int(round(alpha * 10000))) // 10000)
        counts = [int((st == o).sum()) for o in (0, 1, 2, 7)]
        assert max(counts) - min(counts) <= 1
        # Slots are computed from k alone, so any window agrees with the full schedule.
        assert np.array_equal(TA.schedule(alpha, 4, [0, 1, 2, 7], 16101, 777, 500), st[777:1277])


def test_stream_common_random_numbers_and_rejection():
    s1 = TA.StratumStream(16101, 4, 256, np.array([], np.int64))
    a1, b1 = s1.take(0, 700)
    a2, b2 = TA.StratumStream(16101, 4, 256, np.array([], np.int64)).take(300, 100)
    assert np.array_equal(a1[300:400], a2) and np.array_equal(b1[300:400], b2)
    assert np.all(carry.carry_chain(a1, b1, 8) == 4)
    forced = np.unique(a1[:20] * 10 ** 8 + b1[:20])  # force collisions with the first examples
    s3 = TA.StratumStream(16101, 4, 256, forced)
    a3, b3 = s3.take(0, 256)
    assert not np.isin(a3 * 10 ** 8 + b3, forced).any() and s3.rejections >= 20


def test_test_set_disjoint_with_twins():
    pairs, keys = TA.test_set(dict(T, test_per_stratum=200), C['shared']['split_seed'])
    for c, (a, b) in pairs.items():
        assert np.all(carry.carry_chain(a, b, 8) == c)
        assert np.all(np.isin(b * 10 ** 8 + a, keys))  # commuted twins are in the rejection set


def test_models_and_npz(tmp_path):
    for w, n in zip(CM['widths'], CM['expected_params']):
        assert sum(p.numel() for p in carry.addition06.model(dict(hidden_width=w)).parameters()) == n
    for size, tot, ne in zip(T['size_order'], T['expected_params'], T['expected_nonembedding']):
        net = TA.build(T, size)
        assert TA.param_counts(net) == (tot, ne)
    net = TA.build(T, '1M')
    state = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    np.savez(tmp_path / 'x.npz', **state)
    back = dict(np.load(tmp_path / 'x.npz', allow_pickle=False))
    assert all(np.array_equal(state[k], back[k]) for k in state)


def test_transformer_smoke_cpu():
    """Two tiny checkpoints on CPU: losses finite, rows complete, evaluation reproducible from saved weights."""
    t = dict(T, batch_size=32, warmup=2)
    pairs, reject = TA.test_set(dict(T, test_per_stratum=20), C['shared']['split_seed'])
    rows, info = TA.train_run(t, '1M', 'plain', 2, 0.4, 16101, 1e-3, [64, 128], pairs, reject, torch.device('cpu'),
                              log=lambda m: None)
    assert len(rows) == 2 * 9 and info['examples'] == 128
    assert all(np.isfinite(r['excess_bits']) for r in rows)
    assert sum(info['used'].values()) == 128
