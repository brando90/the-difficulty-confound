"""Unit tests for data.py: the split, data files, the exact alignment mix and example streams."""
import json
from pathlib import Path

import numpy as np
import pytest

import data as D
import dsl

C = json.loads((Path(__file__).parent / 'config.json').read_text())


def toy_tasks():
    """20 targets and 20 distractors with real DSL programs (no BFS needed); K is the program length here."""
    rng = np.random.default_rng(5)
    tasks, seen = [], set()
    while len(tasks) < 40:
        prog = [dsl.NAMES[i] for i in rng.integers(0, 12, rng.integers(2, 7))]
        tab = dsl.apply_program(prog)
        if tab.tobytes() in seen:
            continue
        seen.add(tab.tobytes())
        i = len(tasks)
        role = 'target' if i < 20 else 'distractor'
        tasks.append(dict(task_id=f'toy_{role}_{i}', role=role, task_token=i, K=len(prog), K_bits=dsl.k_bits(len(prog)),
                          definition=dict(primitives=prog), truth_table=tab.tolist()))
    return tasks


def test_split_and_files(tmp_path):
    split = D.make_split(C)
    tr, te = set(split['train_idx'].tolist()), set(split['test_idx'].tolist())
    assert not tr & te and tr | te == set(range(1024)) and len(tr) == 512
    lo, hi = C['train_bit_count_range']
    assert all(lo <= v <= hi for v in split['train_bit_counts'])
    tasks = toy_tasks()
    D.build_data(C, tasks, split, tmp_path)
    for t in tasks[:5]:
        tab = dsl.apply_program(t['definition']['primitives'])
        for part, idx in (('train', split['train_idx']), ('test', split['test_idx'])):
            z = np.load(tmp_path / 'data' / f"{t['task_id']}_{part}.npz")
            assert np.array_equal(z['x'], idx) and np.array_equal(z['y'], tab[idx])
            assert np.array_equal(D.from_bits(z['bits_x']), z['x']) and np.array_equal(D.from_bits(z['bits_y']), z['y'])


@pytest.mark.parametrize('alpha', C['alphas'])
def test_assemble_mix(alpha):
    tasks = toy_tasks()
    tid = 'toy_target_3'
    m = D.assemble_mix(C, tasks, tid, alpha, C['seeds'][0])
    tok = 3
    for a, b in D.octave_intervals(C['T_checkpoints']):
        seg = m['task_token'][a:b]
        assert (seg == tok).sum() == int(np.floor(alpha * (b - a) + 0.5))
        dc = np.bincount(seg[seg != tok], minlength=40)[20:40]
        assert dc.max() - dc.min() <= 1
        if alpha < 1.0:  # cumulative distractor counts at every checkpoint are also equal +-1
            cc = np.bincount(m['task_token'][:b][m['task_token'][:b] != tok], minlength=40)[20:40]
            assert cc.max() - cc.min() <= 1
    for T in C['T_checkpoints']:
        for k in [tok] + list(range(20, 40)):
            sel = np.flatnonzero(m['task_token'][:T] == k)
            if not len(sel):
                continue
            reps = np.bincount(m['input_index'][sel], minlength=512)
            n = len(sel)
            assert reps.min() >= n // 512 and reps.max() <= -(-n // 512)
    # token-weighted alpha (every example has the same 11 tokens) equals example-weighted alpha
    T = C['T_checkpoints'][-1]
    ex_alpha = m['counts'][str(T)][tid] / T
    tok_alpha = 11 * m['counts'][str(T)][tid] / (11 * sum(m['counts'][str(T)].values()))
    assert ex_alpha == tok_alpha
    m2 = D.assemble_mix(C, tasks, tid, alpha, C['seeds'][0])
    assert np.array_equal(m['task_token'], m2['task_token']) and np.array_equal(m['input_index'], m2['input_index'])
    m3 = D.assemble_mix(C, tasks, tid, alpha, C['seeds'][1])
    assert not np.array_equal(m['input_index'], m3['input_index'])


def test_example_stream():
    tasks = toy_tasks()
    split = D.make_split(C)
    s = D.example_stream(C, tasks[0], 1500, 7, split=split)
    assert sorted(s['x'][:512].tolist()) == sorted(split['train_idx'].tolist())
    assert sorted(s['x'][512:1024].tolist()) == sorted(split['train_idx'].tolist())
    assert np.array_equal(s['y'], np.array(tasks[0]['truth_table'], np.uint16)[s['x']])
    s2 = D.example_stream(C, tasks[0], 1500, 7, split=split)
    assert np.array_equal(s['x'], s2['x'])
    t = D.example_stream(C, tasks[0], 512, 7, partition='test', split=split)
    assert sorted(t['x'].tolist()) == sorted(split['test_idx'].tolist())
