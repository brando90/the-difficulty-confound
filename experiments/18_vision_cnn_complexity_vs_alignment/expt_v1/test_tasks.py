import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as TK, data as D
import numpy as np


def test_k():
    assert TK.slot_classes('par_a') == (2, 1) and TK.k_bits('par_a') == 1
    assert TK.slot_classes('ge5_xor') == (2, 2)
    assert TK.slot_classes('id_b') == (1, 10)
    for t in ('sum10', 'diff10', 'a3b10', '3ab10', 'a2b7_10', 'a_lt_b', 'sum19', 'diff19', 'a3b', '3ab', 'a7b'):
        assert TK.slot_classes(t) == (10, 10), t
    assert TK.slot_classes('a_mod3') == (2, 1) and TK.slot_classes('a7_xor_b2') == (2, 2)
    for pool in TK.POOLS.values():
        for t in TK.H_TARGETS['easy'] + TK.H_TARGETS['hard']:
            assert t not in pool


def test_alpha_mask():
    for a in (0.05, 0.15, 0.4, 1.0):
        m = D.aligned_mask(10000, a)
        for n in (1, 7, 20, 999, 10000):
            assert m[:n].sum() == int(np.floor(n * a + 1e-9)), (a, n)


if __name__ == '__main__':
    test_k(); test_alpha_mask(); print('ok')
