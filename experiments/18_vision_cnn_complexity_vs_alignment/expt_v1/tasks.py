"""Task library: label functions f(a, b) of the class indices a (left image) and b (right image) of a pair.

Exact complexity K_bits (PROTOCOL Section 2.2): the label depends on the pair only through per-slot statistics, and
the network sees the two images separately (a shared per-image tower), so the information it must extract is the sum
over slots of log2 of the number of classes of that slot's minimal sufficient partition. Two class indices d, d' are in
the same left-slot class iff f(d, b) = f(d', b) for every b (and symmetrically for the right slot). Computed by brute
force over the 10 x 10 table; tests assert the hand-derived values."""
import math

import numpy as np

FUNCS = {
    # main grid, 4 levels x 2 targets
    'par_a': lambda a, b: a % 2,
    'b_ge5': lambda a, b: (b >= 5).astype(np.int64),
    'par_sum': lambda a, b: (a + b) % 2,
    'ge5_xor': lambda a, b: ((a >= 5) ^ (b >= 5)).astype(np.int64),
    'id_a': lambda a, b: a,
    'id_b': lambda a, b: b,
    'sum10': lambda a, b: (a + b) % 10,
    'diff10': lambda a, b: (a - b) % 10,
    # hard pool and hard-to-easy targets (arm H)
    'a3b10': lambda a, b: (a + 3 * b) % 10,
    '3ab10': lambda a, b: (3 * a + b) % 10,
    'a_mod3': lambda a, b: (a % 3 == 0).astype(np.int64),
    'a7_xor_b2': lambda a, b: ((a >= 7) ^ (b <= 2)).astype(np.int64),
    'a2b7_10': lambda a, b: (a + 7 * b + 2) % 10,
    'a_lt_b': lambda a, b: (a < b).astype(np.int64),
}

MAIN_LEVELS = [('L1', ['par_a', 'b_ge5']), ('L2', ['par_sum', 'ge5_xor']), ('L3', ['id_a', 'id_b']),
               ('L4', ['sum10', 'diff10'])]
POOLS = {
    'easy': ['par_a', 'b_ge5', 'par_sum', 'ge5_xor'],
    'hard': ['sum10', 'diff10', 'a3b10', '3ab10'],
    'mixed': ['par_a', 'ge5_xor', 'sum10', 'a3b10'],
}
H_TARGETS = {'easy': ['a_mod3', 'a7_xor_b2'], 'hard': ['a2b7_10', 'a_lt_b']}


def table(name):
    a, b = np.meshgrid(np.arange(10), np.arange(10), indexing='ij')
    return np.asarray(FUNCS[name](a, b), np.int64)


def n_classes(name):
    return int(table(name).max()) + 1


def slot_classes(name):
    t = table(name)
    left = len({tuple(t[d, :]) for d in range(10)})
    right = len({tuple(t[:, d]) for d in range(10)})
    return left, right


def k_bits(name):
    l, r = slot_classes(name)
    return math.log2(l) + math.log2(r)


def chance_error(name):
    """Error of the best constant guess under uniform independent a, b (1 - largest class prior)."""
    t = table(name)
    return 1.0 - np.bincount(t.ravel()).max() / t.size


def label(name, a, b):
    return np.asarray(FUNCS[name](np.asarray(a, np.int64), np.asarray(b, np.int64)), np.int64)


def info(name):
    return dict(task_id=name, K_bits=k_bits(name), slot_classes=list(slot_classes(name)), n_classes=n_classes(name),
                chance_error=chance_error(name))
