"""The concept language L_HMM of PROTOCOL Section 2: sparse single-chain HMMs, exact process rank, description
length, forward-algorithm code lengths, the Bayes mixture, entropy rates, covariates and the document stream.

Pure numpy plus Python integers / fractions; no torch.

A concept h is a dict: n, V, q, succ (n x 3 sorted successor states), trans (n x 3 integer numerators summing to
2**q), etok (n x 4 sorted emission tokens), emit (n x 4 integer numerators summing to 2**q). The current state emits
x_t, then transitions; the initial state is uniform.
"""
import hashlib
import math
from fractions import Fraction

import numpy as np

P = 2 ** 61 - 1


# ---------------------------------------------------------------- the language

def quantize(p, q):
    """Integer numerators >= 1 summing to 2**q: reserve one unit per entry, then largest-remainder rounding."""
    k = len(p)
    free = 2 ** q - k
    raw = np.asarray(p, float) * free
    base = np.floor(raw).astype(np.int64)
    rem = free - int(base.sum())
    order = np.argsort(-(raw - base), kind='stable')
    base[order[:rem]] += 1
    return base + 1


def sample_candidate(rng, n, V, q, n_succ=3, n_emit=4):
    succ, trans, etok, emit = [], [], [], []
    for _ in range(n):
        succ.append(np.sort(rng.choice(n, n_succ, replace=False)))
        etok.append(np.sort(rng.choice(V, n_emit, replace=False)))
        trans.append(quantize(rng.dirichlet(np.ones(n_succ)), q))
        emit.append(quantize(rng.dirichlet(np.ones(n_emit)), q))
    return dict(n=int(n), V=int(V), q=int(q), succ=np.array(succ, np.int64), trans=np.array(trans, np.int64),
                etok=np.array(etok, np.int64), emit=np.array(emit, np.int64))


def to_json(h):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in h.items()}


def from_json(d):
    return dict(n=int(d['n']), V=int(d['V']), q=int(d['q']), succ=np.array(d['succ'], np.int64),
                trans=np.array(d['trans'], np.int64), etok=np.array(d['etok'], np.int64),
                emit=np.array(d['emit'], np.int64))


def same(h1, h2):
    return all(h1[k] == h2[k] for k in ('n', 'V', 'q')) and all(
        np.array_equal(h1[k], h2[k]) for k in ('succ', 'trans', 'etok', 'emit'))


def int_matrices(h):
    """A_int (n x n) and O_int (n x V), integer numerators over 2**q."""
    n, V = h['n'], h['V']
    A = np.zeros((n, n), np.int64)
    O = np.zeros((n, V), np.int64)
    for i in range(n):
        A[i, h['succ'][i]] = h['trans'][i]
        O[i, h['etok'][i]] = h['emit'][i]
    return A, O


def matrices(h):
    A, O = int_matrices(h)
    return A / 2.0 ** h['q'], O / 2.0 ** h['q']


def strongly_connected_aperiodic(h):
    n = h['n']
    adj = [set(map(int, h['succ'][i])) for i in range(n)]
    radj = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            radj[j].add(i)

    def reach(g):
        seen, st = {0}, [0]
        while st:
            u = st.pop()
            for v in g[u]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        return len(seen) == n
    if not (reach(adj) and reach(radj)):
        return False
    level, frontier, g = {0: 0}, [0], 0
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in level:
                    level[v] = level[u] + 1
                    nxt.append(v)
        frontier = nxt
    for u in range(n):
        for v in adj[u]:
            g = math.gcd(g, level[u] + 1 - level[v])
    return g == 1


# ---------------------------------------------------------------- exact process rank (PROTOCOL Section 2.2)

def _krylov_modp(start, step_fns, n, p=P):
    """Dimension mod p of the closure of span{start} under the linear maps step_fns (integer vectors)."""
    basis = []  # (pivot, vector normalized so vector[pivot] == 1); later vectors are 0 at earlier pivots

    def reduce(v):
        v = v % p
        for piv, b in basis:
            if v[piv]:
                v = (v - v[piv] * b) % p
        return v

    def add(v):
        v = reduce(v)
        nz = np.flatnonzero(v != 0)
        if not len(nz):
            return None
        piv = int(nz[0])
        v = (v * pow(int(v[piv]), p - 2, p)) % p
        basis.append((piv, v))
        return v
    queue = [add(np.array([int(x) for x in start], dtype=object))]
    while queue and len(basis) < n:
        v = queue.pop(0)
        for f in step_fns:
            w = add(f(v))
            if w is not None:
                queue.append(w)
                if len(basis) == n:
                    break
    return len(basis)


def _krylov_exact(start, step_fns, n):
    """Same closure with exact Fraction elimination."""
    basis = []

    def add(v):
        v = list(v)
        for piv, b in basis:
            if v[piv] != 0:
                c = v[piv]
                v = [a - c * bb for a, bb in zip(v, b)]
        nz = [i for i, a in enumerate(v) if a != 0]
        if not nz:
            return None
        piv = nz[0]
        c = v[piv]
        v = [a / c for a in v]
        basis.append((piv, v))
        return v
    queue = [add([Fraction(int(x)) for x in start])]
    while queue and len(basis) < n:
        v = queue.pop(0)
        for f in step_fns:
            w = add(f(v))
            if w is not None:
                queue.append(w)
                if len(basis) == n:
                    break
    return len(basis)


def _maps(h, exact):
    A, O = int_matrices(h)
    n, V = h['n'], h['V']
    toks = [x for x in range(V) if O[:, x].any()]
    if exact:
        Al = [[int(a) for a in row] for row in A]
        Ol = [[int(o) for o in O[:, x]] for x in toks]

        def fwd(o):
            return lambda v: [sum(v[i] * o[i] * Al[i][j] for i in range(n)) for j in range(n)]

        def bwd(o):
            return lambda v: [o[i] * sum(Al[i][j] * v[j] for j in range(n)) for i in range(n)]
        return [fwd(o) for o in Ol], [bwd(o) for o in Ol]
    Ao = A.astype(object)
    Os = [O[:, x].astype(object) for x in toks]
    fwd = [(lambda v, o=o: (v * o).dot(Ao)) for o in Os]
    bwd = [(lambda v, o=o: o * Ao.dot(v)) for o in Os]
    return fwd, bwd


def process_rank(h, p=P, exact_recheck_max_n=32):
    """Rank of the Hankel matrix: min of the forward (pi^T M_u) and backward (M_u 1) Krylov span dimensions, with
    M_x = diag(O[:, x]) A in integer arithmetic. Returns (rank, method)."""
    n = h['n']
    fwd, bwd = _maps(h, exact=False)
    r = min(_krylov_modp(np.ones(n, np.int64), fwd, n, p), _krylov_modp(np.ones(n, np.int64), bwd, n, p))
    if r == n:
        return r, 'modp'
    if n <= exact_recheck_max_n:
        fe, be = _maps(h, exact=True)
        return min(_krylov_exact([1] * n, fe, n), _krylov_exact([1] * n, be, n)), 'fraction'
    return r, 'modp'


def process_rank_exact(h):
    fe, be = _maps(h, exact=True)
    n = h['n']
    return min(_krylov_exact([1] * n, fe, n), _krylov_exact([1] * n, be, n))


# ---------------------------------------------------------------- description length

def clog2(x):
    return int(math.ceil(math.log2(x))) if x > 1 else 0


def K_bits(n, V, q):
    return n * (3 * clog2(n) + 4 * clog2(V) + 5 * q)


def encode(h):
    n, V, q = h['n'], h['V'], h['q']
    bn, bv = clog2(n), clog2(V)
    out = []
    for i in range(n):
        out += [format(int(s), f'0{bn}b') for s in h['succ'][i]]
        out += [format(int(t), f'0{bv}b') for t in h['etok'][i]]
        out += [format(int(k) - 1, f'0{q}b') for k in h['trans'][i][:2]]
        out += [format(int(m) - 1, f'0{q}b') for m in h['emit'][i][:3]]
    return ''.join(out)


def decode(bits, n, V, q):
    bn, bv = clog2(n), clog2(V)
    pos = 0

    def take(w):
        nonlocal pos
        v = int(bits[pos:pos + w], 2) if w else 0
        pos += w
        return v
    succ, etok, trans, emit = [], [], [], []
    for _ in range(n):
        succ.append([take(bn) for _ in range(3)])
        etok.append([take(bv) for _ in range(4)])
        t = [take(q) + 1 for _ in range(2)]
        trans.append(t + [2 ** q - sum(t)])
        e = [take(q) + 1 for _ in range(3)]
        emit.append(e + [2 ** q - sum(e)])
    assert pos == len(bits)
    return dict(n=n, V=V, q=q, succ=np.array(succ, np.int64), trans=np.array(trans, np.int64),
                etok=np.array(etok, np.int64), emit=np.array(emit, np.int64))


# ---------------------------------------------------------------- sampling

def _walk(h, start, r_emit, r_trans):
    """Vectorized sampling given the start states (B,) and integer draws in [0, 2**q) (B, L) for emission/transition."""
    ce = np.cumsum(h['emit'], 1)[:, :-1]
    ct = np.cumsum(h['trans'], 1)[:, :-1]
    s = start.copy()
    B, L = r_emit.shape
    out = np.empty((B, L), np.int64)
    for t in range(L):
        k = (r_emit[:, t, None] >= ce[s]).sum(1)
        out[:, t] = h['etok'][s, k]
        kt = (r_trans[:, t, None] >= ct[s]).sum(1)
        s = h['succ'][s, kt]
    return out


def doc_draws(seed, concept_id, j, n, L, q):
    rng = np.random.default_rng([int(seed), int(concept_id), int(j)])
    start = int(rng.integers(0, n))
    r = rng.integers(0, 2 ** q, size=(2, L))
    return start, r[0], r[1]


def sample_docs(h, seed, concept_id, js, L):
    """The documents j in js of concept concept_id under seed: PCG64 seeded by (seed, concept_id, j)."""
    js = list(js)
    st = np.empty(len(js), np.int64)
    re = np.empty((len(js), L), np.int64)
    rt = np.empty((len(js), L), np.int64)
    for i, j in enumerate(js):
        st[i], re[i], rt[i] = doc_draws(seed, concept_id, j, h['n'], L, h['q'])
    return _walk(h, st, re, rt)


def sample_sequences(h, rng, B, L):
    st = rng.integers(0, h['n'], size=B)
    re = rng.integers(0, 2 ** h['q'], size=(B, L))
    rt = rng.integers(0, 2 ** h['q'], size=(B, L))
    return _walk(h, st, re, rt)


def doc_hash(doc):
    return hashlib.sha256(np.ascontiguousarray(doc, dtype=np.int16).tobytes()).hexdigest()


# ---------------------------------------------------------------- exact code lengths

def forward_logprobs(h, docs):
    """Scaled forward algorithm in float64: per-token code lengths -log2 p(x_t | x_<t), shape (B, L)."""
    A, O = matrices(h)
    docs = np.asarray(docs)
    B, L = docs.shape
    a = np.full((B, h['n']), 1.0 / h['n'])
    out = np.empty((B, L))
    for t in range(L):
        e = O[:, docs[:, t]].T
        ae = a * e
        p = ae.sum(1)
        with np.errstate(divide='ignore'):
            out[:, t] = -np.log2(p)  # +inf once a document is impossible under this concept
        a = (ae / np.where(p > 0, p, 1.0)[:, None]) @ A
    return out


def mixture_from_codes(codes, prior):
    """Exact Bayes-mixture code lengths from per-concept code lengths (list of (B, L)) and the prior weights."""
    if len(codes) == 1:
        return np.array(codes[0], dtype=np.float64, copy=True)  # prior 1 on one concept: the oracle exactly
    ln2 = math.log(2.0)
    pref = [-np.cumsum(c, 1) * ln2 for c in codes]  # natural-log prefix likelihoods
    lp = np.log(np.asarray(prior, float))
    stacked = np.stack([lp[i] + pref[i] for i in range(len(codes))])  # (C, B, L)
    m = stacked.max(0)
    lmix = m + np.log(np.exp(stacked - m).sum(0))
    prev = np.concatenate([np.zeros((lmix.shape[0], 1)), lmix[:, :-1]], 1)
    return -(lmix - prev) / ln2


def mixture_logprobs(concepts, prior, docs):
    return mixture_from_codes([forward_logprobs(h, docs) for h in concepts], prior)


def posterior_after(codes, prior, t):
    """Posterior weight of concept 0 after t tokens, per document."""
    ln2 = math.log(2.0)
    lp = np.log(np.asarray(prior, float))
    s = np.stack([lp[i] - codes[i][:, :t].sum(1) * ln2 for i in range(len(codes))])
    s = s - s.max(0)
    w = np.exp(s)
    return w[0] / w.sum(0)


def forward_logspace(h, docs):
    """Independent log-space forward algorithm (verify.py): code lengths from log alpha recursions."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return _forward_logspace(h, docs)


def _forward_logspace(h, docs):
    A, O = matrices(h)
    with np.errstate(divide='ignore'):
        lA, lO = np.log(A), np.log(O)
    docs = np.asarray(docs)
    B, L = docs.shape
    la = np.full((B, h['n']), -math.log(h['n']))
    out = np.empty((B, L))

    def lse(x, axis):
        m = np.max(x, axis=axis, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))
    prev = np.zeros(B)
    for t in range(L):
        joint = la + lO[:, docs[:, t]].T
        tot = lse(joint, 1)
        out[:, t] = np.where(np.isfinite(prev), -(tot - prev) / math.log(2.0), np.inf)
        prev = tot
        la = lse(joint[:, :, None] + lA[None], 1)
    return out


def mixture_logspace(concepts, prior, docs):
    """Independent mixture: log P_mix(x_1:t) = logsumexp_c(log prior_c + log P_c(x_1:t)), differenced."""
    ln2 = math.log(2.0)
    tot = np.stack([math.log(w) - np.cumsum(forward_logspace(h, docs), 1) * ln2 for h, w in zip(concepts, prior)])
    m = tot.max(0)
    l = m + np.log(np.exp(tot - m).sum(0))
    out = np.empty_like(l)
    out[:, 0] = -l[:, 0] / ln2
    out[:, 1:] = -(l[:, 1:] - l[:, :-1]) / ln2
    return out


# ---------------------------------------------------------------- entropy rate and covariates

def entropy_rate(h, rng, n_seqs=100, L=10000, burn=100):
    """Mean of -log2 p(x_t | x_<t) over n_seqs sequences of L tokens, positions 1..burn discarded; SE = sd/sqrt."""
    x = sample_sequences(h, rng, n_seqs, L)
    cl = forward_logprobs(h, x)[:, burn:]
    per = cl.mean(1)
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(n_seqs))


def stationary(h):
    A, _ = matrices(h)
    w, v = np.linalg.eig(A.T)
    i = int(np.argmin(np.abs(w - 1)))
    pi = np.real(v[:, i])
    pi = pi / pi.sum()
    return np.maximum(pi, 0) / np.maximum(pi, 0).sum()


def visit_tail_exponent(pi):
    s = np.sort(pi)[::-1]
    r = np.arange(1, len(s) + 1)
    return float(np.polyfit(np.log(r), np.log(s), 1)[0])


def expected_visits_rarest(h, pi, L=128):
    A, _ = matrices(h)
    i = int(np.argmin(pi))
    d = np.full(h['n'], 1.0 / h['n'])
    tot = 0.0
    for _ in range(L):
        tot += d[i]
        d = d @ A
    return float(tot)


def emission_support(h):
    return set(int(t) for t in np.unique(h['etok']))


def emission_overlap(h1, h2):
    a, b = emission_support(h1), emission_support(h2)
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- the alpha schedule and the document stream

def is_target_slot(k, alpha):
    return math.floor((k + 1) * alpha + 1e-12) - math.floor(k * alpha + 1e-12) == 1


def alpha_flags(n_docs, alpha):
    """Exact Bresenham rule with rational alpha (alpha given to 2 decimals), so floor(m * alpha) is exact."""
    num = int(round(alpha * 100))
    k = np.arange(n_docs, dtype=np.int64)
    return ((k + 1) * num) // 100 - (k * num) // 100 == 1


def schedule(seed, target_cid, distractor_cids, alpha, n_docs):
    """(concept ids, per-concept document indices) for the first n_docs documents of a run's training stream."""
    flags = alpha_flags(n_docs, alpha)
    perm = np.random.default_rng([int(seed), 7777]).permutation(len(distractor_cids))
    pool = [distractor_cids[i] for i in perm]
    cids = np.empty(n_docs, np.int64)
    js = np.empty(n_docs, np.int64)
    counts = {}
    d = 0
    for k in range(n_docs):
        if flags[k]:
            c = target_cid
        else:
            c = pool[d % len(pool)]
            d += 1
        cids[k] = c
        js[k] = counts.get(c, 0)
        counts[c] = js[k] + 1
    return cids, js


def stream_batches(concepts_by_id, seed, cids, js, L, batch_docs):
    """Token arrays (batch_docs, L) in stream order."""
    for s in range(0, len(cids), batch_docs):
        cc, jj = cids[s:s + batch_docs], js[s:s + batch_docs]
        out = np.empty((len(cc), L), np.int64)
        for c in np.unique(cc):
            m = cc == c
            out[m] = sample_docs(concepts_by_id[int(c)], seed, int(c), jj[m], L)
        yield out
