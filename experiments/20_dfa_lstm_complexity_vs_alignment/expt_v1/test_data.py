"""Unit tests for data.py: partition, pools, labels and the exact alignment mix."""
import json
from pathlib import Path

import numpy as np
import pytest

import data as D
import dfa as A

C = json.loads((Path(__file__).parent / 'config.json').read_text())


@pytest.fixture(scope='module')
def built():
    tasks, _ = A.select_tasks(C)
    tb, tl = D.build_test_set(C)
    pools = {t['task_id']: D.build_pool(C, t) for t in tasks}
    return tasks, tb, tl, pools


def test_counts_and_disjointness(built):
    tasks, tb, tl, pools = built
    Ls = D.lengths(C)
    assert len(tb) == 4096
    assert [int((tl == L).sum()) for L in Ls] == [241] * 16 + [240]
    test_set = set(zip(tb.tolist(), tl.tolist()))
    for b, l in pools.values():
        assert len(b) == 69632
        assert all(int((l == L).sum()) == 4096 for L in Ls)
        assert not test_set & set(zip(b.tolist(), l.tolist()))
        assert all((b[l == L] < 2 ** L).all() for L in Ls)
        assert not D.in_test_partition(b[l == 20], 20, C).any()


def test_labels_match_python_loop(built):
    tasks, _, _, pools = built
    for t in tasks:
        d = t['definition']
        b, l = pools[t['task_id']]
        idx = np.random.default_rng(t['task_token']).choice(len(b), 2000, replace=False)
        y = D.labels_for(t, b[idx], l[idx])
        for k, i in enumerate(idx):
            s = 0
            for p in range(int(l[i])):
                s = d['transitions'][s][(int(b[i]) >> p) & 1]
            assert y[k] == (s in d['accept'])


def test_assemble_mix(built):
    tasks, _, _, pools = built
    target = 'dfa_n16_t2'
    tok = next(t['task_token'] for t in tasks if t['task_id'] == target)
    for alpha in C['alphas']:
        m = D.assemble_mix(C, tasks, target, alpha, C['seeds'][0])
        for a, b in D.octave_intervals(C['T_checkpoints']):
            assert int((m['task_token'][a:b] == tok).sum()) == int(np.floor(alpha * (b - a) + 0.5))
            dc = np.unique(m['task_token'][a:b][m['task_token'][a:b] != tok], return_counts=True)[1]
            if len(dc):
                assert len(dc) == 16 and dc.max() - dc.min() <= 1
        for T in C['T_checkpoints']:
            assert abs(m['counts'][str(T)][target] - alpha * T) <= 4
        bl = np.unique(m['batch_length'], return_counts=True)[1]
        assert set(bl.tolist()) <= {60, 61} and bl.sum() == 1024
        ex_len = np.repeat(m['batch_length'], 64)
        pairs = set()
        for i in range(len(ex_len)):
            key = (int(m['task_token'][i]), int(m['pool_index'][i]))
            assert key not in pairs
            pairs.add(key)
        assert (m['pool_index'] // 4096 + 8 == ex_len).all()
        assert m['pool_reuse'] == 0
        # Varun's Eq. 2: token-weighted share (task token + characters) vs example-weighted share
        w = ex_len + 1
        tw = w[m['task_token'] == tok].sum() / w.sum()
        assert abs(tw - (m['task_token'] == tok).mean()) < 0.01
    m1 = D.assemble_mix(C, tasks, target, 0.4, C['seeds'][0])
    m2 = D.assemble_mix(C, tasks, target, 0.4, C['seeds'][0])
    m3 = D.assemble_mix(C, tasks, target, 0.4, C['seeds'][1])
    assert all((m1[k] == m2[k]).all() for k in ('task_token', 'pool_index', 'batch_length'))
    assert (m1['task_token'] != m3['task_token']).any() and (m1['pool_index'] != m3['pool_index']).any()
