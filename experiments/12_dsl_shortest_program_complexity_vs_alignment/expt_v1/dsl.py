"""The 12-primitive bit-word language, exact shortest-program complexity by breadth-first search, covariates and the
stratified task selector (PROTOCOL Sections 1-2.4).

Public API (Experiments 14 and 15 import it; keep these names): PRIM, NAMES, apply_program, k_bits, bfs_enumerate,
load_or_run_bfs, exact_k, brute_force, features, fourier_sensitivity, is_affine_table, local_rules,
train_identifiable, identifiability_table, select_tasks.

Importing this module has no side effects beyond building the 12 lookup tables.
K is exact only for this language: these 12 primitives on 10-bit words with the Elias-gamma length code.
"""
import math
import resource
import time
from pathlib import Path

import numpy as np

WORD_BITS = 10
N = 1 << WORD_BITS
MASK = N - 1
NAMES = ['NOT', 'REV', 'ROTL1', 'ROTR1', 'SHL1', 'SHR1', 'GRAY', 'PREFIX_XOR', 'SWAP_HALVES', 'SWAP_ADJ', 'INC', 'NEG']
_X = np.arange(N, dtype=np.int64)


def _rev(x):
    y = np.zeros_like(x)
    for i in range(WORD_BITS):
        y |= ((x >> i) & 1) << (WORD_BITS - 1 - i)
    return y


def _prefix_xor(x):
    y = x.copy()
    for s in (1, 2, 4, 8):
        y ^= y >> s
    return y


def _build_prim():
    x = _X
    t = {
        'NOT': x ^ MASK,
        'REV': _rev(x),
        'ROTL1': ((x << 1) | (x >> 9)) & MASK,
        'ROTR1': ((x >> 1) | (x << 9)) & MASK,
        'SHL1': (x << 1) & MASK,
        'SHR1': x >> 1,
        'GRAY': x ^ (x >> 1),
        'PREFIX_XOR': _prefix_xor(x),
        'SWAP_HALVES': ((x & 31) << 5) | (x >> 5),
        'SWAP_ADJ': ((x & 341) << 1) | ((x >> 1) & 341),
        'INC': (x + 1) % N,
        'NEG': (-x) % N,
    }
    return np.stack([t[n] for n in NAMES]).astype(np.uint16)


PRIM = _build_prim()  # (12, 1024) uint16, row p is primitive p's lookup table
IDENTITY = np.arange(N, dtype=np.uint16)


def _index(p):
    return NAMES.index(p) if isinstance(p, str) else int(p)


def apply_program(program):
    """Truth table (uint16, 1024) of x -> p_K(...p_1(x)) for a sequence of primitive names or indices."""
    t = IDENTITY.copy()
    for p in program:
        t = PRIM[_index(p)][t]
    return t


def k_bits(K):
    """Elias gamma code of K plus the index of the program among the 12^K sequences of length K."""
    K = int(K)
    return 2 * int(math.floor(math.log2(K))) + 1 + int(math.ceil(K * math.log2(12) - 1e-12))


# ---------------------------------------------------------------- hashing

def _hash_weights(n_words, seed):
    return np.random.default_rng([seed, n_words]).integers(1, 2 ** 63, n_words, dtype=np.uint64) * np.uint64(2) + \
        np.uint64(1)


def _mix64(z):
    """splitmix64 finalizer, a bijection on uint64 (in place)."""
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z


def row_hash(rows, seed=20261012, block=8192):
    """64-bit hash of each row (any unsigned dtype, row bytes a multiple of 8).

    Each 64-bit word is XORed with a position key and mixed by splitmix64, and the mixed words are summed. Rows that
    differ in any word get different hashes unless a true 64-bit collision occurs, which callers detect by exact
    comparison.
    """
    rows = np.ascontiguousarray(rows)
    words = rows.view(np.uint64).reshape(len(rows), -1)
    w = _hash_weights(words.shape[1], seed)
    out = np.empty(len(rows), np.uint64)
    with np.errstate(over='ignore'):
        for s in range(0, len(rows), block):
            z = words[s:s + block] ^ w
            out[s:s + block] = _mix64(z).sum(axis=1, dtype=np.uint64)
    return out


def _peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


# ---------------------------------------------------------------- breadth-first search

def bfs_enumerate(c, max_K=None):
    """Exact BFS over distinct functions (PROTOCOL Section 2.2).

    Returns dict: tables (uint16, (N, 1024)), K (int8), canonical (int8, (N, max_K), -1 padded), n_shortest (int64),
    level_counts (new functions per level 1..max_K), seconds, peak_rss_gb, hash (uint64 per table), level_start.
    Children are generated in (parent order, primitive index) order and the first hit is kept, so each canonical
    program is the lexicographically smallest shortest program (parents are themselves in canonical-program order).
    Equality is exact: a 64-bit hash for sorting, then full-row comparison; a hash collision raises.
    """
    max_K = int(max_K or c['max_K'])
    seed = c.get('hash_seed', 20261012)
    chunk = int(c['bfs_chunk_parents'])
    t0 = time.time()
    cap = sum(12 ** L for L in range(max_K + 1))
    buf = np.empty((cap, N), np.uint16)  # virtual; only rows written become resident
    can = np.full((cap, max_K), -1, np.int8)
    nsh = np.zeros(cap, np.int64)
    buf[0] = IDENTITY
    nsh[0] = 1
    starts = [0, 1]
    seen_h = row_hash(buf[:1], seed)
    seen_id = np.zeros(1, np.int64)
    for L in range(1, max_K + 1):
        p0, p1 = starts[L - 1], starts[L]
        base = n_tot = p1
        for s in range(p0, p1, chunk):
            e = min(s + chunk, p1)
            n = e - s
            cand = PRIM[:, buf[s:e]].transpose(1, 0, 2).reshape(n * 12, N)  # (parent, primitive) order
            par = np.repeat(np.arange(s, e), 12)
            prim = np.tile(np.arange(12, dtype=np.int8), n)
            h = row_hash(cand, seed)
            pos = np.minimum(np.searchsorted(seen_h, h), len(seen_h) - 1)
            found = seen_h[pos] == h
            fid = np.full(len(h), -1, np.int64)
            fid[found] = seen_id[pos[found]]
            fi = np.flatnonzero(found)
            for a in range(0, len(fi), 65536):  # exact comparison inside equal-hash groups
                idx = fi[a:a + 65536]
                if not np.array_equal(buf[fid[idx]], cand[idx]):
                    raise RuntimeError(f'64-bit hash collision with unequal rows at level {L}')
            nf = np.flatnonzero(~found)
            if len(nf):
                uq, first, inv = np.unique(h[nf], return_index=True, return_inverse=True)
                inv = inv.reshape(-1)
                for a in range(0, len(nf), 65536):
                    sl = slice(a, a + 65536)
                    if not np.array_equal(cand[nf[sl]], cand[nf[first[inv[sl]]]]):
                        raise RuntimeError(f'64-bit hash collision among new functions at level {L}')
                order = np.argsort(first, kind='stable')  # emission order = first occurrence
                rank = np.empty(len(order), np.int64)
                rank[order] = np.arange(len(order))
                k = len(order)
                fid[nf] = n_tot + rank[inv]
                keep = nf[first[order]]
                buf[n_tot:n_tot + k] = cand[keep]
                can[n_tot:n_tot + k] = can[par[keep]]
                can[n_tot:n_tot + k, L - 1] = prim[keep]
                seen_h = np.concatenate([seen_h, uq[order]])
                seen_id = np.concatenate([seen_id, n_tot + np.arange(k)])
                srt = np.argsort(seen_h, kind='stable')
                seen_h, seen_id = seen_h[srt], seen_id[srt]
                n_tot += k
            hit = fid >= base  # pairs that reach a function of this level count toward its shortest programs
            np.add.at(nsh, fid[hit], nsh[par[hit]])
            del cand
        starts.append(n_tot)
    n_all = starts[-1]
    Ks = np.concatenate([np.full(starts[L + 1] - starts[L], L, np.int8) for L in range(max_K + 1)])
    out = dict(tables=buf[:n_all], K=Ks, canonical=can[:n_all].copy(), n_shortest=nsh[:n_all].copy(),
               level_counts=[int(starts[L + 1] - starts[L]) for L in range(1, max_K + 1)],
               level_start=np.array(starts, np.int64))
    out['hash'] = row_hash(out['tables'], seed)
    srt = np.argsort(out['hash'], kind='stable')
    out['hash_sorted'], out['hash_order'] = out['hash'][srt], srt
    out['seconds'] = round(time.time() - t0, 1)
    out['peak_rss_gb'] = round(_peak_rss_gb(), 3)
    if out['peak_rss_gb'] >= c['bfs_ram_limit_gb']:
        raise RuntimeError(f"BFS peak RSS {out['peak_rss_gb']} GB over the {c['bfs_ram_limit_gb']} GB limit")
    return out


def load_or_run_bfs(c, cache=None):
    """BFS from the cache file when present (results/bfs_cache.npz, git-ignored), else run it and write the cache."""
    keys = ('tables', 'K', 'canonical', 'n_shortest', 'level_counts', 'level_start', 'hash', 'hash_sorted',
            'hash_order')
    if cache is not None and Path(cache).exists():
        with np.load(cache, allow_pickle=False) as z:
            b = {k: z[k] for k in keys}
            b['level_counts'] = b['level_counts'].tolist()
            b['seconds'] = float(z['seconds'])
            b['peak_rss_gb'] = float(z['peak_rss_gb'])
            b['from_cache'] = True
        return b
    b = bfs_enumerate(c)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(cache) + '.tmp')
        with open(tmp, 'wb') as f:
            np.savez(f, **{k: np.asarray(b[k]) for k in keys}, seconds=b['seconds'], peak_rss_gb=b['peak_rss_gb'])
        tmp.replace(cache)
    b['from_cache'] = False
    return b


def exact_k(table, bfs, seed=20261012):
    """(K, global index) of a truth table by lookup, or (None, None) if it is not in the BFS (K > max_K)."""
    table = np.asarray(table, np.uint16).reshape(1, N)
    h = row_hash(table, seed)[0]
    i = np.searchsorted(bfs['hash_sorted'], h)
    if i < len(bfs['hash_sorted']) and bfs['hash_sorted'][i] == h:
        g = int(bfs['hash_order'][i])
        if np.array_equal(bfs['tables'][g], table[0]):
            return int(bfs['K'][g]), g
        raise RuntimeError('hash collision in exact_k')
    return None, None


def program_of(bfs, g):
    return [NAMES[p] for p in bfs['canonical'][g] if p >= 0]


def brute_force(max_len, seed=20261012):
    """Enumerate all programs of length <= max_len in lexicographic order (by length, then from p_1).

    Returns dict hash -> (K, canonical program tuple, n_shortest, table) for every distinct function.
    """
    out = {}
    prev_tab = IDENTITY[None, :].copy()
    prev_prog = np.zeros((1, 0), np.int8)
    h0 = row_hash(prev_tab, seed)[0]
    out[int(h0)] = (0, (), 1, prev_tab[0])
    for L in range(1, max_len + 1):
        n = len(prev_tab)
        tab = PRIM[:, prev_tab].transpose(1, 0, 2).reshape(n * 12, N)
        prog = np.concatenate([np.repeat(prev_prog, 12, axis=0), np.tile(np.arange(12, dtype=np.int8), n)[:, None]],
                              axis=1)
        h = row_hash(tab, seed)
        uq, first, cnt = np.unique(h, return_index=True, return_counts=True)
        for k, f, m in zip(uq.tolist(), first.tolist(), cnt.tolist()):
            if k not in out:
                out[k] = (L, tuple(int(p) for p in prog[f]), int(m), tab[f].copy())
        prev_tab, prev_prog = tab, prog
    return out


# ---------------------------------------------------------------- covariates

_POP = np.array([bin(i).count('1') for i in range(N)], np.int64)


def bits_of(table):
    """(…, 1024, 10) uint8 output bits, bit j in column j."""
    t = np.asarray(table).astype(np.int64)
    return ((t[..., None] >> np.arange(WORD_BITS)) & 1).astype(np.uint8)


def flip_matrix(tables):
    """F[..., i, x] = f(x) XOR f(x XOR 2^i): which output bits flip when input bit i flips."""
    t = np.asarray(tables).astype(np.int64)
    return np.stack([t ^ t[..., _X ^ (1 << i)] for i in range(WORD_BITS)], axis=-2)


def is_affine_table(tables):
    """True where f(x) XOR f(0) is linear over GF(2) (vectorized over leading axes)."""
    t = np.asarray(tables).astype(np.int64)
    g = t ^ t[..., :1]
    pred = np.zeros_like(g)
    for i in range(WORD_BITS):
        pred ^= ((_X >> i) & 1) * g[..., [1 << i]]
    return np.all(pred == g, axis=-1)


def local_rules(table, rare_count=16):
    """Local-rule entropy (bits), number of distinct rules and rare-rule mass (PROTOCOL Section 2.3)."""
    F = flip_matrix(table)  # (10, 1024): row i is the bits of output that flip with input bit i
    keys = np.ascontiguousarray(F.T.astype(np.uint16))  # (1024, 10): the 10x10 bit matrix J(x), row per input bit
    _, cnt = np.unique(keys, axis=0, return_counts=True)
    p = cnt / N
    H = float(-np.sum(p * np.log2(p)))
    rare = float(cnt[cnt < rare_count].sum() / N)
    return dict(local_rule_entropy=H if H > 0 else 0.0, n_local_rules=int(len(cnt)), rare_rule_mass=rare)


def quick_features(tables, block=512):
    """Vectorized balance, n_depends, sensitivity and affine indicator for a (n, 1024) block of tables."""
    tables = np.asarray(tables)
    parts = []
    for s in range(0, len(tables), block):
        t = tables[s:s + block]
        F = flip_matrix(t)  # (n, 10, 1024)
        parts.append(dict(output_balance=_POP[t.astype(np.int64)].sum(-1) / (N * WORD_BITS),
                          n_depends=(F != 0).any(-1).sum(-1),
                          sensitivity=_POP[F].sum((-1, -2)) / (N * WORD_BITS * WORD_BITS),
                          is_affine=is_affine_table(t)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def features(table, rare_count=16):
    """Every exact covariate of one truth table (PROTOCOL Section 2.3)."""
    table = np.asarray(table, np.uint16)
    q = quick_features(table[None])
    B = bits_of(table)
    out = dict(output_balance=float(q['output_balance'][0]), output_bit_balance=B.mean(0).astype(float).tolist(),
               n_depends=int(q['n_depends'][0]), sensitivity=float(q['sensitivity'][0]),
               is_affine=bool(q['is_affine'][0]))
    out.update(local_rules(table, rare_count))
    return out


def fourier_sensitivity(table):
    """(1/100) sum_j sum_S |S| ghat_j(S)^2 by the Walsh-Hadamard transform (O'Donnell's influence formula)."""
    B = bits_of(table).astype(np.float64)  # (1024, 10)
    g = 1.0 - 2.0 * B
    W = g.copy()
    h = 1
    while h < N:  # fast Walsh-Hadamard transform along axis 0
        W = W.reshape(N // (2 * h), 2, h, WORD_BITS)
        a, b = W[:, 0].copy(), W[:, 1].copy()
        W[:, 0], W[:, 1] = a + b, a - b
        W = W.reshape(N, WORD_BITS)
        h *= 2
    W /= N
    deg = _POP[np.arange(N)].astype(np.float64)
    return float(np.sum(deg[:, None] * W ** 2) / (WORD_BITS * WORD_BITS))


def stratum_of(s, strata):
    for k, (lo, hi) in enumerate(strata):
        if lo <= s < hi or (k == len(strata) - 1 and s == hi):
            return k
    return -1


# ---------------------------------------------------------------- identifiability

def identifiability_table(bfs, train_idx, seed=20261012):
    """Per function: whether another function with K' < K (shorter) or K' == K (equal) has the same restriction."""
    tr = np.ascontiguousarray(bfs['tables'][:, np.asarray(train_idx)])
    rh = row_hash(tr, seed)
    _, inv, cnt = np.unique(rh, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    multi = np.flatnonzero(cnt[inv] > 1)
    # exact check inside equal-hash groups: restrictions must be truly equal
    grp_first = {}
    for i in multi.tolist():
        g = int(inv[i])
        if g in grp_first:
            if not np.array_equal(tr[i], tr[grp_first[g]]):
                raise RuntimeError('hash collision among restricted truth tables')
        else:
            grp_first[g] = i
    maxK = int(bfs['K'].max())
    lev = np.zeros((len(cnt), maxK + 1), np.int64)
    np.add.at(lev, (inv, bfs['K'].astype(np.int64)), 1)
    K = bfs['K'].astype(np.int64)
    cum = np.cumsum(lev, axis=1)
    shorter = (K > 0) & (cum[inv, np.maximum(K - 1, 0)] > 0)
    equal = lev[inv, K] > 1
    return dict(shorter_clash=shorter, equal_clash=equal)


def train_identifiable(table, K, bfs, train_idx):
    """True if no other function with K' <= K has the same restriction to train_idx (direct comparison)."""
    table = np.asarray(table, np.uint16)
    m = bfs['K'] <= K
    sub = bfs['tables'][m][:, np.asarray(train_idx)]
    same = np.all(sub == table[np.asarray(train_idx)], axis=1)
    full = np.all(bfs['tables'][m] == table, axis=1)
    return not np.any(same & ~full)


# ---------------------------------------------------------------- selection

def select_tasks(c, bfs, train_idx, proxies_fn=None):
    """Sensitivity- and type-stratified selection (PROTOCOL Section 2.4). Returns (tasks, selection_log)."""
    strata, types = c['strata'], c['cell_types']
    lo_b, hi_b = c['balance_range']
    ident = identifiability_table(bfs, train_idx)
    chosen = {}  # truth-table bytes of chosen tasks and their complements
    slots_by_level = {}
    log = dict(rejections={}, eligible_by_cell={}, scanned={}, identifiability_clashes={})
    for K in c['K_levels']:
        idx = np.flatnonzero(bfs['K'] == K)
        order = np.random.default_rng([c['task_seed'], K]).permutation(idx)
        tgt = [None] * len(strata)
        dis = [None] * len(strata)
        probe = [[] for _ in strata]
        fresh = [[] for _ in strata]
        rej = dict(balance=0, depends=0, identifiability=0, distinctness=0, no_stratum=0)
        cells = {f'S{k + 1}_{t}': 0 for k in range(len(strata)) for t in ('affine', 'nonaffine')}
        clashes = dict(shorter=0, equal=0)
        scanned = 0

        def full():
            return (all(x is not None for x in tgt) and all(x is not None for x in dis)
                    and all(len(p) == c['probe_per_stratum'] for p in probe)
                    and all(len(f) == c['fresh_per_stratum'] for f in fresh))
        for s in range(0, len(order), c['selection_chunk']):
            if full():
                break
            ids = order[s:s + c['selection_chunk']]
            q = quick_features(bfs['tables'][ids])
            for j, g in enumerate(ids.tolist()):
                if full():
                    break
                scanned += 1
                if not (lo_b <= q['output_balance'][j] <= hi_b):
                    rej['balance'] += 1
                    continue
                if q['n_depends'][j] < c['min_depends']:
                    rej['depends'] += 1
                    continue
                sh, eq = bool(ident['shorter_clash'][g]), bool(ident['equal_clash'][g])
                clashes['shorter'] += sh
                clashes['equal'] += eq
                if sh or eq:
                    rej['identifiability'] += 1
                    continue
                tab = bfs['tables'][g]
                if tab.tobytes() in chosen:
                    rej['distinctness'] += 1
                    continue
                k = stratum_of(float(q['sensitivity'][j]), strata)
                if k < 0:
                    rej['no_stratum'] += 1
                    continue
                aff = bool(q['is_affine'][j])
                cells[f"S{k + 1}_{'affine' if aff else 'nonaffine'}"] += 1
                typ_ok = (types[k] == 'affine') == aff
                if typ_ok and tgt[k] is None:
                    tgt[k] = g
                elif typ_ok and dis[k] is None:
                    dis[k] = g
                elif len(probe[k]) < c['probe_per_stratum']:
                    probe[k].append(g)
                elif len(fresh[k]) < c['fresh_per_stratum']:
                    fresh[k].append(g)
                else:
                    continue
                chosen[tab.tobytes()] = g
                chosen[(tab ^ np.uint16(MASK)).astype(np.uint16).tobytes()] = g
        if not full():
            raise RuntimeError(f'level {K} exhausted before every slot was filled')
        slots_by_level[K] = dict(t=tgt, d=dis, p=[g for k in range(len(strata)) for g in probe[k]],
                                 f=[g for k in range(len(strata)) for g in fresh[k]],
                                 p_stratum=[k for k in range(len(strata)) for _ in probe[k]],
                                 f_stratum=[k for k in range(len(strata)) for _ in fresh[k]])
        log['rejections'][str(K)] = rej
        log['eligible_by_cell'][str(K)] = cells
        log['scanned'][str(K)] = scanned
        log['identifiability_clashes'][str(K)] = clashes
    # assemble task dicts
    tasks = []
    n_lv = len(c['K_levels'])
    for li, K in enumerate(c['K_levels']):
        sl = slots_by_level[K]
        rows = [(f'dsl_k{K}_t{k + 1}', 'target', g, k, li * len(strata) + k) for k, g in enumerate(sl['t'])]
        rows += [(f'dsl_k{K}_d{k + 1}', 'distractor', g, k, n_lv * len(strata) + li * len(strata) + k)
                 for k, g in enumerate(sl['d'])]
        rows += [(f'dsl_k{K}_p{i + 1:02d}', 'probe_pretrain', g, sl['p_stratum'][i], None) for i, g in enumerate(sl['p'])]
        rows += [(f'dsl_k{K}_f{i + 1:02d}', 'fresh', g, sl['f_stratum'][i], None) for i, g in enumerate(sl['f'])]
        for tid, role, g, k, tok in rows:
            tab = bfs['tables'][g]
            ft = features(tab, c['rare_rule_count'])
            t = dict(task_id=tid, family='dsl', role=role, task_token=tok, K=int(K), K_bits=k_bits(K),
                     definition=dict(primitives=program_of(bfs, g), word_bits=WORD_BITS,
                                     order='x -> p_K(...p_1(x))'),
                     truth_table=tab.astype(int).tolist(), n_shortest_programs=int(bfs['n_shortest'][g]),
                     sensitivity_stratum=f'S{k + 1}', bfs_index=int(g))
            t.update(ft)
            t['visit_tail_exponent'] = None
            tasks.append(t)
    # proxies and maximum agreement with any other selected task
    if proxies_fn is not None:
        for t in tasks:
            t.update(proxies_fn(np.array(t['truth_table'], np.uint16), c))
    B = np.stack([bits_of(np.array(t['truth_table'], np.uint16)).reshape(-1) for t in tasks]).astype(np.float32)
    agree = (B @ B.T + (1 - B) @ (1 - B).T) / B.shape[1]
    np.fill_diagonal(agree, -1)
    for i, t in enumerate(tasks):
        t['max_agreement'] = float(agree[i].max())
    return tasks, log
