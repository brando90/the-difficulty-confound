"""Exact arithmetic for Experiment 16: chain-length strata, an exact within-stratum sampler, and token formats.

The oracle (column_addition, carry_chain) is imported read-only from Experiment 06 and hash-checked.
Chain length c is the longest run of consecutive columns that produce a carry, counting the carry out of the top
column. It is a structural demand, not Kolmogorov complexity.

Dynamic program. After columns 0..i-1 (least significant first) the state is (carry out, current run, longest run).
F[i, carry, run, mx] counts digit assignments for columns i..digits-1 that end with longest run exactly c. Counts are
int64 (at most 10**16 < 2**63) and sampling uses integer draws, so within-stratum sampling is exactly uniform.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p / 'experiments').is_dir())
EXP06 = REPO / 'experiments/archive/06_compositional_carry_marginal_matched/expt_v1'
EXP08 = REPO / 'experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1'
# SHA-256 of the read-only imports (Experiment 06 addition.py is byte-identical to Experiments 02 and 08).
IMPORT_SHA256 = {
    'exp06_addition.py': '766123242a12da2ee6973f03ade252880983720ed9c65db88932c01decef2289',
    'exp08_run.py': '097e1e1c7f438fca6d55e100356be57611dc40442436c3b7e0360fb32127c8e7',
}
TOKENS = {str(d): d for d in range(10)} | {'+': 10, '=': 11, '$': 12}
PLUS, EQ, DOLLAR = 10, 11, 12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(name, path, expected):
    assert sha(path) == expected, f'{path} changed (expected {expected})'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


addition06 = _load('exp06_addition', EXP06 / 'addition.py', IMPORT_SHA256['exp06_addition.py'])
column_addition = addition06.column_addition
carry_chain = addition06.carry_chain


def import_hashes():
    return {'exp06_addition.py': sha(EXP06 / 'addition.py'), 'exp08_run.py': sha(EXP08 / 'run.py')}


# --- dynamic program ----------------------------------------------------------------------------------------

PAIRS_WITH_SUM = np.array([t + 1 if t <= 9 else 19 - t for t in range(19)], dtype=np.int64)


def completion_table(digits, c):
    """F[i, carry, run, mx] for target chain length c (int64)."""
    F = np.zeros((digits + 1, 2, digits + 1, digits + 1), dtype=np.int64)
    F[digits, :, :, c] = 1
    for i in range(digits - 1, -1, -1):
        for carry in (0, 1):
            for run in range(i + 1):
                for mx in range(i + 1):
                    total = 0
                    for t in range(19):
                        co = int(t + carry >= 10)
                        r2 = (run + 1) * co
                        total += int(PAIRS_WITH_SUM[t]) * int(F[i + 1, co, r2, max(mx, r2)])
                    F[i, carry, run, mx] = total
    return F


def stratum_counts(digits):
    """Exact number of ordered pairs (a, b), 0 <= a, b < 10**digits, with each chain length 0..digits."""
    return [int(completion_table(digits, c)[0, 0, 0, 0]) for c in range(digits + 1)]


def brute_counts(digits):
    a, b = np.meshgrid(np.arange(10 ** digits), np.arange(10 ** digits), indexing='ij')
    return np.bincount(carry_chain(a.ravel(), b.ravel(), digits), minlength=digits + 1).tolist()


def sample_stratum(rng, c, n, digits, F=None):
    """n ordered pairs drawn exactly uniformly from stratum c (least significant column first)."""
    F = completion_table(digits, c) if F is None else F
    carry = np.zeros(n, dtype=np.int64)
    run = np.zeros(n, dtype=np.int64)
    mx = np.zeros(n, dtype=np.int64)
    a = np.zeros(n, dtype=np.int64)
    b = np.zeros(n, dtype=np.int64)
    t_all = np.arange(19)
    for i in range(digits):
        co = (t_all[None, :] + carry[:, None]) >= 10
        r2 = (run[:, None] + 1) * co
        m2 = np.maximum(mx[:, None], r2)
        w = PAIRS_WITH_SUM[None, :] * F[i + 1, co.astype(np.int64), r2, m2]
        cum = np.cumsum(w, axis=1)
        total = cum[:, -1]
        assert np.array_equal(total, F[i, carry, run, mx]), 'dynamic program inconsistent'
        u = rng.integers(0, total)
        t = (cum > u[:, None]).argmax(axis=1)
        lo = np.maximum(0, t - 9)
        ai = lo + rng.integers(0, PAIRS_WITH_SUM[t])
        bi = t - ai
        a += ai * 10 ** i
        b += bi * 10 ** i
        carry = (t + carry >= 10).astype(np.int64)
        run = (run + 1) * carry
        mx = np.maximum(mx, run)
    assert np.all(mx == c)
    return a, b


# --- formats ------------------------------------------------------------------------------------------------

def digits_msb(x, width):
    return (np.asarray(x)[:, None] // 10 ** np.arange(width - 1, -1, -1)) % 10


def encode(a, b, fmt, digits=8):
    """Token ids: plain `$ a7..a0 + b7..b0 = s8..s0 $`, reversed `$ a7..a0 + b7..b0 = s0..s8 $` (29 tokens)."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    s = digits_msb(a + b, digits + 1)
    if fmt == 'reversed':
        s = s[:, ::-1]
    elif fmt != 'plain':
        raise ValueError(fmt)
    n = len(a)
    col = lambda v: np.full((n, 1), v, dtype=np.int64)
    return np.concatenate([col(DOLLAR), digits_msb(a, digits), col(PLUS), digits_msb(b, digits), col(EQ), s, col(DOLLAR)], 1)


def answer_positions(digits=8):
    """Token positions of the answer digits; the model predicts token p from positions < p."""
    start = 2 * digits + 3
    return np.arange(start, start + digits + 1)


def answer_place(fmt, digits=8):
    """Decimal place (power of ten) of the digit at each answer position."""
    places = np.arange(digits, -1, -1)
    return places if fmt == 'plain' else places[::-1]


def decode(tokens, fmt, digits=8):
    tokens = np.asarray(tokens)
    a = (tokens[:, 1:1 + digits] * 10 ** np.arange(digits - 1, -1, -1)).sum(1)
    b = (tokens[:, 2 + digits:2 + 2 * digits] * 10 ** np.arange(digits - 1, -1, -1)).sum(1)
    ans = tokens[:, answer_positions(digits)]
    s = (ans * 10 ** answer_place(fmt, digits)).sum(1)
    return a, b, s


# --- covariates ---------------------------------------------------------------------------------------------

def stratum_covariates(a, b, digits):
    """Input-digit total-variation distance from uniform digits, answer-digit entropy, mean number of carries."""
    da = digits_msb(a, digits)
    db = digits_msb(b, digits)
    hist = np.bincount(np.concatenate([da.ravel(), db.ravel()]), minlength=10) / (2 * digits * len(a))
    ans = digits_msb(np.asarray(a) + np.asarray(b), digits + 1)
    ph = np.bincount(ans.ravel(), minlength=10) / ans.size
    ph = ph[ph > 0]
    carry = np.zeros(len(a), np.int64)
    n_carries = np.zeros(len(a), np.int64)
    for place in range(digits):
        carry = ((np.asarray(a) // 10 ** place) % 10 + (np.asarray(b) // 10 ** place) % 10 + carry) // 10
        n_carries += carry
    return dict(input_digit_tv=float(0.5 * np.abs(hist - 0.1).sum()), answer_digit_entropy_bits=float(-(ph * np.log2(ph)).sum()),
                mean_carries=float(n_carries.mean()))


def chain_exposure(shares, digits):
    """Share of a mix's examples containing a chain of length >= k, for k = 1..digits (shares: {c: share})."""
    return {k: float(sum(v for c, v in shares.items() if c >= k)) for k in range(1, digits + 1)}
