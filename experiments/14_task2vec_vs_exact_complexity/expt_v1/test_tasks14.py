"""Unit test 7 of PROMPT.md: role counts, disjointness, recomputed K, label noise. Run: python -m pytest -q
test_tasks14.py"""
import numpy as np
import pytest

import tasks14 as T


@pytest.fixture(scope='module')
def env():
    c = T.load_config()
    roster, logs = T.build_roster(c)
    return c, roster


def test_roles_disjoint_and_k(env):
    c, roster = env
    ok, rep = T.check_roster(c, roster)
    assert ok, rep
    assert rep['dfa']['n_embedded'] == 96 and rep['dsl']['n_embedded'] == 120
    assert rep['dfa']['n_noise_tasks'] * len(c['noise_rates']) == 32
    assert rep['dsl']['n_noise_tasks'] * len(c['noise_rates']) == 40


def test_dfa_complements_also_distinct(env):
    c, roster = env
    A = T.exp_modules(c)['dfa']['gen']
    keys = set()
    for t in roster['dfa']:
        d = A.DFA.from_json(t['definition'])
        keys.add(A.canonical_form(d).key())
    for t in roster['dfa']:
        if t['source'] == 'exp14':
            comp = A.canonical_form(A.complement(A.DFA.from_json(t['definition']))).key()
            assert comp not in keys


def test_inputs_null_token_and_tv(env):
    c, roster = env
    ok, rep = T.check_inputs(c, roster)
    assert ok, rep


@pytest.mark.parametrize('fam', ['dfa', 'dsl'])
def test_label_noise_exact_count(env, fam):
    c, roster = env
    for t in T.noise_tasks(c, roster, fam):
        clean = T.example_set(c, roster, fam, t)
        for r in c['noise_rates']:
            ex = T.noisy_example_set(c, roster, fam, t, r)
            k = int(round(r * c['examples_per_task']))
            y0, y1 = clean['y'].reshape(len(clean['y']), -1), ex['y'].reshape(len(ex['y']), -1)
            changed = np.flatnonzero((y0 != y1).any(1))
            assert len(changed) == k
            assert np.array_equal(changed, ex['noise_positions'])
            assert np.array_equal(ex['X'], clean['X'])


def test_example_sets_fixed_size(env):
    c, roster = env
    for fam in T.FAMILIES:
        for t in T.embedded_tasks(roster, fam)[::7]:
            ex = T.example_set(c, roster, fam, t)
            assert len(ex['y']) == c['examples_per_task'] == len(ex['X'])
