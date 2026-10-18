"""Deterministic finite automata over {a, b}: sampling, exact minimization and exact covariates.

Public API (Experiments 14 and 15 import these names): DFA, sample_minimal_dfa, hopcroft_minimize,
moore_minimize, canonical_form, equivalent, complement, run_dfa, acceptance_by_length, acceptance_rate,
sensitivity, visit_freq, visit_tail_exponent, k_bits, select_tasks.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DFA:
    trans: np.ndarray   # int64, shape (n, 2); trans[s, c] for c in {0 = a, 1 = b}
    accept: np.ndarray  # bool, shape (n,)

    @property
    def n(self):
        return len(self.accept)

    def key(self):
        """Bytes that identify the automaton exactly (use on canonical forms for language identity)."""
        return self.trans.astype(np.int64).tobytes() + self.accept.astype(np.uint8).tobytes()

    def to_json(self):
        return dict(n_states=int(self.n), alphabet=['a', 'b'], start=0,
                    transitions=self.trans.astype(int).tolist(),
                    accept=[int(s) for s in np.flatnonzero(self.accept)])

    @staticmethod
    def from_json(d):
        n = d['n_states']
        acc = np.zeros(n, bool)
        acc[d['accept']] = True
        return DFA(np.array(d['transitions'], np.int64).reshape(n, 2), acc)


def k_bits(n):
    """Exact code length of an n-state DFA in the description language L_DFA (PROTOCOL Section 1)."""
    return n * (2 * math.ceil(math.log2(n)) + 1) if n > 1 else 1


def bfs_relabel(dfa, blocks=None):
    """Quotient by `blocks` (state -> block id; identity if None) and relabel reachable blocks in BFS order.

    BFS visits symbols a then b, so two minimal DFAs of the same language get identical arrays.
    """
    n = dfa.n
    if blocks is None:
        blocks = np.arange(n)
    nb = int(blocks.max()) + 1
    btrans = np.full((nb, 2), -1, np.int64)
    bacc = np.zeros(nb, bool)
    for s in range(n):
        b = blocks[s]
        for c in range(2):
            t = blocks[dfa.trans[s, c]]
            assert btrans[b, c] in (-1, t), 'blocks are not a congruence'
            btrans[b, c] = t
        bacc[b] = dfa.accept[s]
    order = {int(blocks[0]): 0}
    queue = [int(blocks[0])]
    for b in queue:
        for c in range(2):
            t = int(btrans[b, c])
            if t not in order:
                order[t] = len(order)
                queue.append(t)
    m = len(order)
    trans = np.zeros((m, 2), np.int64)
    acc = np.zeros(m, bool)
    for b, i in order.items():
        trans[i] = [order[int(btrans[b, 0])], order[int(btrans[b, 1])]]
        acc[i] = bacc[b]
    return DFA(trans, acc)


def reachable(dfa):
    return bfs_relabel(dfa)


def hopcroft_partition(dfa):
    """Hopcroft (1971) partition refinement; returns block id per state (states assumed reachable)."""
    n = dfa.n
    inv = [[[] for _ in range(n)] for _ in range(2)]
    for s in range(n):
        for c in range(2):
            inv[c][int(dfa.trans[s, c])].append(s)
    acc = frozenset(np.flatnonzero(dfa.accept).tolist())
    rej = frozenset(range(n)) - acc
    P = [b for b in (acc, rej) if b]
    W = [min(P, key=len)] if len(P) == 2 else []
    while W:
        A = W.pop()
        for c in range(2):
            X = set()
            for q in A:
                X.update(inv[c][q])
            if not X:
                continue
            newP = []
            for Y in P:
                i = Y & X
                d = Y - X
                if i and d:
                    newP += [frozenset(i), frozenset(d)]
                    if Y in W:
                        W.remove(Y)
                        W += [frozenset(i), frozenset(d)]
                    else:
                        W.append(frozenset(i) if len(i) <= len(d) else frozenset(d))
                else:
                    newP.append(Y)
            P = newP
    blocks = np.zeros(n, np.int64)
    for k, Y in enumerate(sorted(P, key=min)):
        for s in Y:
            blocks[s] = k
    return blocks


def moore_partition(dfa):
    """Brute-force Moore refinement: split by (class, class of a-successor, class of b-successor) until stable."""
    cls = dfa.accept.astype(np.int64)
    k = len(set(cls.tolist()))
    while True:
        sig = [(int(cls[s]), int(cls[dfa.trans[s, 0]]), int(cls[dfa.trans[s, 1]])) for s in range(dfa.n)]
        ids = {}
        new = np.array([ids.setdefault(x, len(ids)) for x in sig], np.int64)
        if len(ids) == k:
            # renumber by smallest member, as hopcroft_partition does
            first = {}
            for s in range(dfa.n):
                first.setdefault(int(new[s]), s)
            rank = {b: i for i, b in enumerate(sorted(first, key=first.get))}
            return np.array([rank[int(b)] for b in new], np.int64)
        cls, k = new, len(ids)


def hopcroft_minimize(dfa):
    r = reachable(dfa)
    return bfs_relabel(r, hopcroft_partition(r))


def moore_minimize(dfa):
    r = reachable(dfa)
    return bfs_relabel(r, moore_partition(r))


def canonical_form(dfa):
    """Minimal DFA in BFS canonical order: equal for two DFAs iff they accept the same language."""
    return hopcroft_minimize(dfa)


def complement(dfa):
    return DFA(dfa.trans.copy(), ~dfa.accept)


def equivalent(d1, d2):
    """Language equivalence by breadth-first search of the product automaton."""
    seen = {(0, 0)}
    queue = [(0, 0)]
    for u, v in queue:
        if d1.accept[u] != d2.accept[v]:
            return False
        for c in range(2):
            p = (int(d1.trans[u, c]), int(d2.trans[v, c]))
            if p not in seen:
                seen.add(p)
                queue.append(p)
    return True


def run_dfa(dfa, bits, lengths):
    """Vectorized labels: bit i of `bits` is character i (a = 0)."""
    bits = np.asarray(bits, np.uint64)
    lengths = np.asarray(lengths, np.int64)
    state = np.zeros(len(bits), np.int64)
    for i in range(int(lengths.max()) if len(lengths) else 0):
        c = ((bits >> np.uint64(i)) & np.uint64(1)).astype(np.int64)
        state = np.where(i < lengths, dfa.trans[state, c], state)
    return dfa.accept[state]


def _step_matrix(dfa):
    M = np.zeros((dfa.n, dfa.n))
    for c in range(2):
        np.add.at(M, (np.arange(dfa.n), dfa.trans[:, c]), 0.5)
    return M


def state_distributions(dfa, t_max):
    """p[t] = distribution of the state after t uniform characters, t = 0..t_max."""
    M = _step_matrix(dfa)
    p = np.zeros((t_max + 1, dfa.n))
    p[0, 0] = 1.0
    for t in range(1, t_max + 1):
        p[t] = p[t - 1] @ M
    return p


def acceptance_by_length(dfa, lmin, lmax):
    p = state_distributions(dfa, lmax)
    return np.array([p[l] @ dfa.accept for l in range(lmin, lmax + 1)])


def acceptance_rate(dfa, lmin, lmax):
    return float(acceptance_by_length(dfa, lmin, lmax).mean())


def sensitivity(dfa, lmin, lmax):
    """Exact mean over inputs of the fraction of single-character substitutions that flip the label."""
    p = state_distributions(dfa, lmax)
    t0, t1 = dfa.trans[:, 0], dfa.trans[:, 1]
    D = [(dfa.accept[:, None] != dfa.accept[None, :]).astype(float)]
    for _ in range(lmax):
        prev = D[-1]
        D.append(0.5 * (prev[np.ix_(t0, t0)] + prev[np.ix_(t1, t1)]))
    out = []
    for l in range(lmin, lmax + 1):
        s = sum(p[i] @ D[l - i - 1][t0, t1] for i in range(l))
        out.append(s / l)
    return float(np.mean(out))


def visit_freq(dfa, lmin, lmax):
    p = state_distributions(dfa, lmax)
    v = np.mean([p[: l + 1].sum(0) for l in range(lmin, lmax + 1)], axis=0)
    return v / v.sum()


def visit_tail_exponent(freq):
    """OLS slope b of log frequency on log rank (states with zero frequency excluded)."""
    f = np.sort(np.asarray(freq))[::-1]
    f = f[f > 0]
    rank = np.arange(1, len(f) + 1)
    return float(np.polyfit(np.log(rank), np.log(f), 1)[0])


def monte_carlo_covariates(dfa, rng, n_strings, lmin, lmax):
    """Monte Carlo acceptance and sensitivity with standard errors (for cross-checks)."""
    L = rng.integers(lmin, lmax + 1, n_strings)
    bits = rng.integers(0, 2 ** 32, n_strings, dtype=np.uint64) & ((np.uint64(1) << L.astype(np.uint64)) - np.uint64(1))
    y = run_dfa(dfa, bits, L)
    flips = np.zeros(n_strings)
    for i in range(lmax):
        m = i < L
        y2 = run_dfa(dfa, bits ^ (np.uint64(1) << np.uint64(i)), L)
        flips += m & (y2 != y)
    f = flips / L
    return dict(acceptance=(y.mean(), y.std() / np.sqrt(n_strings)),
                sensitivity=(f.mean(), f.std() / np.sqrt(n_strings)))


def sample_minimal_dfa(rng, n, overprovision=1.25):
    """Draw random DFAs until one has exactly n minimal states; returns (canonical DFA, draws)."""
    m = math.ceil(overprovision * n) + 1
    draws = 0
    while True:
        draws += 1
        trans = rng.integers(0, m, (m, 2)).astype(np.int64)
        acc = rng.random(m) < 0.5
        d = hopcroft_minimize(DFA(trans, acc))
        if d.n == n:
            return d, draws


def covariates(dfa, c):
    lmin, lmax = c['length_min'], c['length_max']
    by_len = acceptance_by_length(dfa, lmin, lmax)
    vf = visit_freq(dfa, lmin, lmax)
    return dict(acceptance_rate=float(by_len.mean()), acceptance_by_length=by_len.tolist(),
                sensitivity=sensitivity(dfa, lmin, lmax), visit_freq=vf.tolist(),
                visit_tail_exponent=visit_tail_exponent(vf))


def select_tasks(c):
    """Sensitivity-stratified selection of 4 targets and 4 distractors per level (PROTOCOL Section 2.2)."""
    lmin, lmax = c['length_min'], c['length_max']
    lo, hi = c['accept_range']
    plo, phi = c['per_length_accept_range']
    tasks, log = [], {}
    for n in c['n_levels']:
        rng = np.random.default_rng([c['task_seed'], n])
        kept, keys = [], set()
        counts = dict(dfa_draws=0, accepted_minimal=0, rejected_overall_acceptance=0,
                      rejected_per_length_acceptance=0, rejected_duplicate_or_complement=0)
        while len(kept) < c['candidates_per_level']:
            d, draws = sample_minimal_dfa(rng, n, c['sampler_overprovision'])
            counts['dfa_draws'] += draws
            counts['accepted_minimal'] += 1
            by_len = acceptance_by_length(d, lmin, lmax)
            if not lo <= by_len.mean() <= hi:
                counts['rejected_overall_acceptance'] += 1
                continue
            if not ((by_len >= plo) & (by_len <= phi)).all():
                counts['rejected_per_length_acceptance'] += 1
                continue
            k, kc = d.key(), canonical_form(complement(d)).key()
            if k in keys or kc in keys:
                counts['rejected_duplicate_or_complement'] += 1
                continue
            keys.add(k)
            kept.append((sensitivity(d, lmin, lmax), d))
        kept.sort(key=lambda x: x[0])
        nbins = c['targets_per_level']
        size = len(kept) // nbins
        for q in range(nbins):
            pick = rng.choice(size, 2, replace=False)
            for role, idx in (('target', pick[0]), ('distractor', pick[1])):
                sens, d = kept[q * size + int(idx)]
                tasks.append(dict(dfa=d, n=n, role=role, quartile=q + 1, sensitivity=sens))
        log[f'n{n:02d}'] = counts
    targets = [t for t in tasks if t['role'] == 'target']
    distractors = [t for t in tasks if t['role'] == 'distractor']
    out = []
    for token, t in enumerate(targets + distractors):
        letter = 't' if t['role'] == 'target' else 'd'
        tid = f"dfa_n{t['n']:02d}_{letter}{t['quartile']}"
        cov = covariates(t['dfa'], c)
        out.append(dict(task_id=tid, family='dfa', role=t['role'], K=t['n'], K_bits=k_bits(t['n']),
                        definition=t['dfa'].to_json(), sensitivity_quartile=t['quartile'],
                        task_token=token, **cov, visit_tail_steepness=-cov['visit_tail_exponent']))
    return out, log
