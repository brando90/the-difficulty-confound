"""Unit tests for dsl.py: primitives, BFS against brute force, covariates and the task selector."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import data as D
import dsl
import proxies as PX

C = json.loads((Path(__file__).parent / 'config.json').read_text())
N = 1024


def bit(x, i):
    return (x >> i) & 1


def naive(name, x):
    """Plain per-bit loop for each primitive (bit 0 is the least significant)."""
    b = [bit(x, i) for i in range(10)]
    if name == 'NOT':
        o = [1 - v for v in b]
    elif name == 'REV':
        o = [b[9 - i] for i in range(10)]
    elif name == 'ROTL1':
        o = [b[(i - 1) % 10] for i in range(10)]
    elif name == 'ROTR1':
        o = [b[(i + 1) % 10] for i in range(10)]
    elif name == 'SHL1':
        o = [0] + b[:9]
    elif name == 'SHR1':
        o = b[1:] + [0]
    elif name == 'GRAY':
        o = [b[i] ^ (b[i + 1] if i < 9 else 0) for i in range(10)]
    elif name == 'PREFIX_XOR':
        o = [sum(b[i:]) % 2 for i in range(10)]
    elif name == 'SWAP_HALVES':
        o = b[5:] + b[:5]
    elif name == 'SWAP_ADJ':
        o = [b[i + 1] if i % 2 == 0 else b[i - 1] for i in range(10)]
    elif name == 'INC':
        return (x + 1) % N
    elif name == 'NEG':
        return (-x) % N
    return sum(v << i for i, v in enumerate(o))


def test_primitives_match_naive_loop():
    for p, name in enumerate(dsl.NAMES):
        assert [naive(name, x) for x in range(N)] == dsl.PRIM[p].tolist(), name


def test_identities():
    I = dsl.IDENTITY
    ap = dsl.apply_program
    assert np.array_equal(ap(['GRAY', 'PREFIX_XOR']), I) and np.array_equal(ap(['PREFIX_XOR', 'GRAY']), I)
    assert np.array_equal(ap(['NEG']), ap(['NOT', 'INC']))  # NEG = INC o NOT (NOT applied first)
    for n in ('NOT', 'REV', 'SWAP_HALVES', 'SWAP_ADJ'):
        assert np.array_equal(ap([n, n]), I), n
    assert np.array_equal(ap(['ROTL1', 'ROTR1']), I)


def test_hand_values():
    sens = {n: dsl.features(dsl.apply_program([n]))['sensitivity'] for n in dsl.NAMES}
    for n in ('NOT', 'REV', 'SWAP_HALVES'):
        assert abs(sens[n] - 0.1) < 1e-12
    assert abs(sens['GRAY'] - 0.19) < 1e-12 and abs(sens['PREFIX_XOR'] - 0.55) < 1e-12
    assert round(sens['INC'], 4) == 0.1396 and round(sens['NEG'], 4) == 0.1396
    f = dsl.features(dsl.apply_program(['SHL1']))
    assert f['n_depends'] == 9 and abs(f['output_balance'] - 0.45) < 1e-12
    assert [dsl.k_bits(k) for k in range(1, 7)] == [5, 11, 14, 20, 23, 27]


@pytest.fixture(scope='module')
def bfs():
    return dsl.bfs_enumerate(C)


def test_bfs_level_counts(bfs):
    assert bfs['level_counts'] == [12, 112, 988, 8543, 73146, 622876] == C['bfs_level_counts']
    assert bfs['peak_rss_gb'] < C['bfs_ram_limit_gb']


def test_brute_force_matches_bfs_up_to_5(bfs):
    bf = dsl.brute_force(5)
    assert len(bf) == 1 + sum(C['bfs_level_counts'][:5])
    assert sum(12 ** L for L in range(6)) == 271453
    bad = 0
    for h, (K, prog, ns, tab) in bf.items():
        k2, g = dsl.exact_k(tab, bfs)
        bad += not (k2 == K and tuple(int(p) for p in bfs['canonical'][g] if p >= 0) == prog
                    and int(bfs['n_shortest'][g]) == ns and np.array_equal(dsl.apply_program(prog), tab))
    assert bad == 0
    # every BFS function with K <= 5 is in the brute force set
    assert int((bfs['K'] <= 5).sum()) == len(bf)


def test_sensitivity_equals_fourier():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.integers(0, N, N).astype(np.uint16)
        assert abs(dsl.features(t)['sensitivity'] - dsl.fourier_sensitivity(t)) < 1e-12


def gf2_affine(t):
    """Direct check: f(x ^ y) ^ f(0) == (f(x) ^ f(0)) ^ (f(y) ^ f(0)) for all x, y."""
    t = np.asarray(t, np.int64)
    g = t ^ t[0]
    x = np.arange(N)
    return bool(np.all(g[x[:, None] ^ x[None, :]] == (g[:, None] ^ g[None, :])))


def test_affine(bfs):
    small = np.flatnonzero(bfs['K'] <= 4)
    flags = dsl.is_affine_table(bfs['tables'][small])
    for i, g in enumerate(small):
        prog = [p for p in bfs['canonical'][g] if p >= 0]
        direct = gf2_affine(bfs['tables'][g])
        assert flags[i] == direct
        if not {10, 11} & set(prog):
            assert direct
    rng = np.random.default_rng(1)
    for _ in range(200):
        t = rng.integers(0, N, N).astype(np.uint16)
        assert dsl.is_affine_table(t) == gf2_affine(t)


def test_local_rules_naive():
    rng = np.random.default_rng(2)
    tabs = [rng.integers(0, N, N).astype(np.uint16) for _ in range(40)]
    tabs += [dsl.apply_program(p) for p in (['INC'], ['NEG', 'GRAY'], ['PREFIX_XOR', 'INC', 'REV'],
                                            ['SHL1', 'INC'], ['SWAP_ADJ', 'NEG'], ['INC', 'INC'], ['NOT'],
                                            ['GRAY'], ['REV', 'INC'], ['INC', 'SWAP_HALVES', 'INC'])]
    for t in tabs:
        rules = {}
        for x in range(N):
            J = tuple(tuple(bit(int(t[x]) ^ int(t[x ^ (1 << i)]), j) for j in range(10)) for i in range(10))
            rules[J] = rules.get(J, 0) + 1
        p = np.array(list(rules.values())) / N
        H = max(0.0, float(-np.sum(p * np.log2(p))))
        lr = dsl.local_rules(t, C['rare_rule_count'])
        assert abs(lr['local_rule_entropy'] - H) < 1e-12 and lr['n_local_rules'] == len(rules)
        assert abs(lr['rare_rule_mass'] - sum(v for v in rules.values() if v < C['rare_rule_count']) / N) < 1e-12
        if gf2_affine(t):
            assert lr['local_rule_entropy'] == 0.0


def test_select_tasks(bfs):
    split = D.make_split(C)
    tasks, log = dsl.select_tasks(C, bfs, split['train_idx'], PX.proxies)
    roles = {}
    for t in tasks:
        roles[t['role']] = roles.get(t['role'], 0) + 1
    assert roles == dict(target=20, distractor=20, probe_pretrain=80, fresh=80)
    tabs = [np.array(t['truth_table'], np.uint16).tobytes() for t in tasks]
    comps = [(np.array(t['truth_table'], np.uint16) ^ np.uint16(1023)).tobytes() for t in tasks]
    assert len(set(tabs)) == 200 and not set(tabs) & set(comps)
    for role in ('target', 'distractor'):
        ts = [t for t in tasks if t['role'] == role]
        for k in C['K_levels']:
            lv = sorted((t for t in ts if t['K'] == k), key=lambda t: t['sensitivity_stratum'])
            assert [t['sensitivity_stratum'] for t in lv] == ['S1', 'S2', 'S3', 'S4']
            assert [t['is_affine'] for t in lv] == [ct == 'affine' for ct in C['cell_types']]
        for t in ts:
            tab = np.array(t['truth_table'], np.uint16)
            assert dsl.train_identifiable(tab, t['K'], bfs, split['train_idx'])
            assert np.array_equal(dsl.apply_program(t['definition']['primitives']), tab)
            assert dsl.exact_k(tab, bfs)[0] == t['K'] and t['K_bits'] == dsl.k_bits(t['K'])
    assert sorted(t['task_token'] for t in tasks if t['role'] == 'target') == list(range(20))
    assert sorted(t['task_token'] for t in tasks if t['role'] == 'distractor') == list(range(20, 40))
    again, _ = dsl.select_tasks(C, bfs, split['train_idx'], PX.proxies)
    assert json.dumps(again, sort_keys=True) == json.dumps(tasks, sort_keys=True)
    assert math.isfinite(tasks[0]['bdm_rowmajor'])
