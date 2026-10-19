"""Experiment 11/12 imports (hash-checked), the Experiment 14 task roster, fixed example sets, label noise, mixes.

Public API (Experiment 15 may import it; keep these names): load_config, exp_modules, build_roster,
example_set, noisy_example_set, coverage_mixes, exp_mixes, bound_example_set, task_seed_int.

Experiment 11/12 code is imported from their folders by file path (both have a model.py and a data.py) and never
copied or modified. Before use, the SHA-256 of every imported file must equal the value in that experiment's
results/manifest.json.

Token layout of the probes (PROTOCOL Section 2.3): DFA probe: bit tokens 0, 1; probe-pretraining task tokens 2..65
(64 tasks); null token 66 (task token 64 in Experiment 11's numbering, where token = 2 + task token). DSL probe: bit
tokens 0, 1; probe-pretraining task tokens 2..81 (80 tasks); null token 82 (task token 80). Every embedding input
carries the null token at position 0.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILIES = ('dfa', 'dsl')


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def task_seed_int(*parts):
    """A 63-bit integer seed from any parts, e.g. (seed, family, task id, probe, estimator)."""
    s = '|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], 'big') >> 1


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_MODS = {}


def exp_dirs(c):
    return dict(dfa=(HERE / c['exp11_dir']).resolve(), dsl=(HERE / c['exp12_dir']).resolve())


def exp_modules(c=None):
    """dict family -> dict(dir, cfg, gen (dfa.py / dsl.py), data, model, manifest, hashes). Hash-checked."""
    if _MODS:
        return _MODS
    c = c or load_config()
    dirs = exp_dirs(c)
    for fam, srcs, gen_name in (('dfa', c['exp11_sources'], 'dfa'), ('dsl', c['exp12_sources'], 'dsl')):
        d = dirs[fam]
        man = json.loads((d / 'results' / 'manifest.json').read_text())
        hashes = {}
        for s in srcs:
            h = sha256_file(d / s)
            if man['sources'].get(s) != h:
                raise RuntimeError(f'{fam}: {s} SHA-256 {h} differs from its manifest {man["sources"].get(s)}')
            hashes[s] = h
        gen = _load(gen_name, d / f'{gen_name}.py')  # Experiment 11 data.py does `from dfa import ...`
        data = _load(f'e_{fam}_data', d / 'data.py')
        model = _load(f'e_{fam}_model', d / 'model.py')
        _MODS[fam] = dict(dir=d, cfg=json.loads((d / 'config.json').read_text()), gen=gen, data=data, model=model,
                          manifest=man, hashes=hashes)
    return _MODS


def null_token(c, fam):
    """Task-token index of the null token (the embedding row is 2 + this)."""
    return c[fam]['n_probe_tokens']


# ---------------------------------------------------------------- roster

def _dfa_new_tasks(c, old_tasks):
    """64 probe-pretraining and 64 fresh DFA tasks with Experiment 11's sampler and filters (PROTOCOL Section 2.2).

    Per level: an rng seeded (task_sampling_seed, n) draws candidates with dfa.sample_minimal_dfa and Experiment 11's
    acceptance filters until 64 are kept, rejecting any language (or complement) already used by Experiment 11 or by
    an earlier candidate, exactly as Experiment 11's select_tasks rejects duplicates. Candidates are sorted by exact
    sensitivity and cut into 4 quartile bins of 16; from each bin 8 are drawn without replacement, the first 4 for
    probe pretraining and the next 4 fresh. This mirrors Experiment 12's 4-per-stratum probe and fresh roles.
    """
    m = exp_modules(c)['dfa']
    A, c11 = m['gen'], m['cfg']
    lmin, lmax = c11['length_min'], c11['length_max']
    lo, hi = c11['accept_range']
    plo, phi = c11['per_length_accept_range']
    used = set()
    for t in old_tasks:
        d = A.DFA.from_json(t['definition'])
        used.add(A.canonical_form(d).key())
        used.add(A.canonical_form(A.complement(d)).key())
    out, log = [], {}
    cc = c['dfa']
    for n in c11['n_levels']:
        rng = np.random.default_rng([c['task_sampling_seed'], n])
        kept = []
        counts = dict(dfa_draws=0, accepted_minimal=0, rejected_overall_acceptance=0,
                      rejected_per_length_acceptance=0, rejected_duplicate_or_complement=0)
        while len(kept) < cc['candidates_per_level']:
            d, draws = A.sample_minimal_dfa(rng, n, c11['sampler_overprovision'])
            counts['dfa_draws'] += draws
            counts['accepted_minimal'] += 1
            by_len = A.acceptance_by_length(d, lmin, lmax)
            if not lo <= by_len.mean() <= hi:
                counts['rejected_overall_acceptance'] += 1
                continue
            if not ((by_len >= plo) & (by_len <= phi)).all():
                counts['rejected_per_length_acceptance'] += 1
                continue
            k, kc = d.key(), A.canonical_form(A.complement(d)).key()
            if k in used or kc in used:
                counts['rejected_duplicate_or_complement'] += 1
                continue
            used.add(k)
            used.add(kc)
            kept.append((A.sensitivity(d, lmin, lmax), d))
        kept.sort(key=lambda x: x[0])
        size = len(kept) // 4
        npq, nfq = cc['probe_per_quartile'], cc['fresh_per_quartile']
        probe, fresh = [], []
        for q in range(4):
            pick = rng.choice(size, npq + nfq, replace=False)
            probe += [(q + 1, kept[q * size + int(i)][1]) for i in pick[:npq]]
            fresh += [(q + 1, kept[q * size + int(i)][1]) for i in pick[npq:]]
        for role, lst, letter in (('probe_pretrain', probe, 'p'), ('fresh', fresh, 'f')):
            for i, (q, d) in enumerate(lst):
                cov = A.covariates(d, c11)
                out.append(dict(task_id=f'dfa_n{n:02d}_{letter}{i + 1:02d}', family='dfa', role=role, K=n,
                                K_bits=A.k_bits(n), definition=d.to_json(), sensitivity_quartile=q,
                                task_token=None, **cov, visit_tail_steepness=-cov['visit_tail_exponent']))
        log[f'n{n:02d}'] = counts
    return out, log


def label_entropy(t):
    def h(p):
        p = float(np.clip(p, 0.0, 1.0))
        return 0.0 if p in (0.0, 1.0) else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    if t['family'] == 'dfa':
        return h(t['acceptance_rate'])
    return float(np.mean([h(p) for p in t['output_bit_balance']]))


def build_roster(c=None):
    """dict family -> list of task dicts with roles target / distractor / probe_pretrain / fresh, plus the selection
    log. Probe tokens: probe_pretrain tasks get probe_token 0..n-1 in level then id order."""
    c = c or load_config()
    mods = exp_modules(c)
    roster, logs = {}, {}
    old11 = json.loads((mods['dfa']['dir'] / 'results' / 'tasks.json').read_text())
    new11, log11 = _dfa_new_tasks(c, old11)
    roster['dfa'] = [dict(t, source='exp11') for t in old11] + [dict(t, source='exp14') for t in new11]
    logs['dfa'] = log11
    t12 = json.loads((mods['dsl']['dir'] / 'results' / 'tasks.json').read_text())
    roster['dsl'] = [dict(t, source='exp12') for t in t12]
    logs['dsl'] = dict(note='probe_pretrain and fresh tasks read from Experiment 12 tasks.json')
    for fam in FAMILIES:
        probes = sorted((t for t in roster[fam] if t['role'] == 'probe_pretrain'), key=lambda t: (t['K'], t['task_id']))
        tok = {t['task_id']: i for i, t in enumerate(probes)}
        for t in roster[fam]:
            t['probe_token'] = tok.get(t['task_id'])
            t['label_entropy'] = label_entropy(t)
            t['embedded'] = t['role'] in ('target', 'distractor', 'fresh')
    return roster, logs


def embedded_tasks(roster, fam):
    """Targets, distractors and fresh tasks in a fixed order: by K, then role (target, distractor, fresh), then id."""
    order = dict(target=0, distractor=1, fresh=2)
    return sorted((t for t in roster[fam] if t['embedded']), key=lambda t: (t['K'], order[t['role']], t['task_id']))


def noise_tasks(c, roster, fam):
    """The first noise_tasks_per_level fresh tasks per K level (id order)."""
    out = []
    for K in sorted({t['K'] for t in roster[fam]}):
        fr = sorted((t for t in roster[fam] if t['role'] == 'fresh' and t['K'] == K), key=lambda t: t['task_id'])
        out += fr[: c['noise_tasks_per_level']]
    return out


# ---------------------------------------------------------------- example sets

def _dfa_tokens(c, bits, lens, token_index):
    """LongTensor-ready int64 (n, max_len + 1) right-padded with 0: [2 + token, c_1..c_l, 0...]; last = l."""
    n = len(bits)
    Lm = int(lens.max())
    X = np.zeros((n, Lm + 1), np.int64)
    X[:, 0] = 2 + np.asarray(token_index, np.int64)
    sh = np.arange(Lm, dtype=np.uint64)
    chars = ((np.asarray(bits, np.uint64)[:, None] >> sh) & np.uint64(1)).astype(np.int64)
    chars[np.arange(Lm)[None, :] >= np.asarray(lens)[:, None]] = 0
    X[:, 1:] = chars
    return X, np.asarray(lens, np.int64)


def _new_dfa_inputs(c, task_index, n):
    """n inputs from Experiment 11's training-partition sampler: length uniform on 8..24, then a uniform string."""
    m = exp_modules(c)['dfa']
    c11 = m['cfg']
    rng = np.random.default_rng([c['task_sampling_seed'], 4, task_index])
    lens = rng.integers(c11['length_min'], c11['length_max'] + 1, n)
    bits = np.zeros(n, np.uint64)
    for L in np.unique(lens):
        idx = np.flatnonzero(lens == L)
        bits[idx] = m['data'].sample_strings(rng, len(idx), int(L), 'train', c11)
    return bits.astype(np.uint32), lens.astype(np.uint8)


_EX = {}


def raw_examples(c, roster, fam, task):
    """The fixed 2,048 training examples of a task: dict(bits/x, lens, y) in the fixed order (PROTOCOL 2.2)."""
    key = (fam, task['task_id'])
    if key in _EX:
        return _EX[key]
    n = c['examples_per_task']
    m = exp_modules(c)[fam]
    if fam == 'dfa':
        if task['source'] == 'exp11':
            z = np.load(m['dir'] / 'results' / 'data' / f"{task['task_id']}_train.npz")
            perm = np.random.default_rng(c['task_sampling_seed']).permutation(len(z['bits']))[:n]
            out = dict(bits=z['bits'][perm], lens=z['length'][perm], y=z['label'][perm].astype(np.int64))
        else:
            idx = sorted(t['task_id'] for t in roster['dfa'] if t['source'] == 'exp14').index(task['task_id'])
            bits, lens = _new_dfa_inputs(c, idx, n)
            y = m['gen'].run_dfa(m['gen'].DFA.from_json(task['definition']), bits, lens).astype(np.int64)
            out = dict(bits=bits, lens=lens, y=y)
    else:
        s = m['data'].example_stream(m['cfg'], task, n, c['task_sampling_seed'], 'train')
        out = dict(x=s['x'], bits_x=s['bits_x'].astype(np.int64), y=s['bits_y'].astype(np.int64))
    _EX[key] = out
    return out


def encode(c, fam, raw, token_index=None):
    """dict(X int64 (n, L), last int64 (n,) or None, y int64) with the null token unless token_index is given."""
    tok = null_token(c, fam) if token_index is None else token_index
    if fam == 'dfa':
        X, last = _dfa_tokens(c, raw['bits'], raw['lens'], np.broadcast_to(tok, (len(raw['bits']),)))
        return dict(X=X, last=last, y=np.asarray(raw['y'], np.int64))
    bx = np.asarray(raw['bits_x'], np.int64)
    X = np.concatenate([np.broadcast_to(2 + np.asarray(tok, np.int64).reshape(-1, 1), (len(bx), 1)), bx], axis=1)
    return dict(X=np.ascontiguousarray(X), last=None, y=np.asarray(raw['y'], np.int64))


def example_set(c, roster, fam, task):
    return encode(c, fam, raw_examples(c, roster, fam, task))


def noisy_example_set(c, roster, fam, task, rate):
    """Exactly round(rate * 2048) labels changed at positions drawn without replacement (seeded per task and rate).
    DFA: the accept/reject label is flipped. DSL: the 10-bit output word is replaced by a uniformly random different
    word."""
    ex = example_set(c, roster, fam, task)
    n = len(ex['y'])
    k = int(round(rate * n))
    rng = np.random.default_rng(task_seed_int(c['task_sampling_seed'], fam, task['task_id'], 'noise', rate))
    pos = rng.choice(n, k, replace=False)
    y = ex['y'].copy()
    if fam == 'dfa':
        y[pos] = 1 - y[pos]
    else:
        D = exp_modules(c)['dsl']['data']
        word = D.from_bits(y[pos]).astype(np.int64)
        new = (word + rng.integers(1, 1024, k)) % 1024
        y[pos] = D.to_bits(new).astype(np.int64)
    return dict(ex, y=y, noise_positions=np.sort(pos))


# ---------------------------------------------------------------- mixes

def coverage_mixes(c, roster, fam):
    """m in coverage_m fresh tasks of one K level, uniform shares, 2 replicates (PROTOCOL Section 2.5)."""
    out = []
    for K in sorted({t['K'] for t in roster[fam]}):
        fr = sorted((t['task_id'] for t in roster[fam] if t['role'] == 'fresh' and t['K'] == K))
        for m in c['coverage_m']:
            for r in range(c['coverage_replicates']):
                rng = np.random.default_rng([c['task_sampling_seed'], 7, K, m, r])
                pick = sorted(fr[i] for i in rng.choice(len(fr), m, replace=False))
                out.append(dict(mix_id=f'{fam}_cov_K{K}_m{m:02d}_r{r}', kind='coverage', K=K, m=m, replicate=r,
                                H=float(np.log2(m)), tasks=pick, shares=[1.0 / m] * m))
    return out


def exp_mixes(c, roster, fam):
    """Every Experiment 11/12 training mix, one per (target, alpha), shares from mixes.json at the last checkpoint of
    the first training seed (shares are identical across seeds up to +-1 example)."""
    m = exp_modules(c)[fam]
    mixes = json.loads((m['dir'] / 'results' / 'mixes.json').read_text())
    if fam == 'dsl':
        mixes = mixes['main']
    seed0 = m['cfg']['seeds'][0]
    Tmax = str(m['cfg']['T_checkpoints'][-1])
    out = []
    for t in sorted((t for t in roster[fam] if t['role'] == 'target'), key=lambda t: t['task_token']):
        for a in m['cfg']['alphas']:
            rec = mixes[f"{seed0}_{t['task_id']}_{a:.2f}"]
            cnt = rec['counts'][Tmax]
            ids = sorted(cnt)
            tot = sum(cnt.values())
            shares = [cnt[i] / tot for i in ids]
            H = float(-sum(p * np.log2(p) for p in shares if p > 0))
            out.append(dict(mix_id=f"{fam}_exp_{t['task_id']}_a{a:.2f}", kind='exp', K=t['K'], alpha=a,
                            target_id=t['task_id'], H=H, tasks=ids, shares=shares))
    return out


def bound_example_set(c, fam, which, n, rng):
    """Reference bounds (diversity paper Sect. 2.4). lower: every example is the same input of one fixed non-special
    token (DFA: 'a' repeated lower_bound_length times; DSL: the all-zero word) with label 0. upper: uniformly random
    tokens over the family vocabulary without task tokens (DFA: lengths uniform on 8..24, uniform bits over all
    strings; DSL: uniform 10-bit words) with uniformly random labels. The null token stays at position 0."""
    if fam == 'dfa':
        c11 = exp_modules(c)['dfa']['cfg']
        if which == 'lower':
            L = c['dfa']['lower_bound_length']
            raw = dict(bits=np.zeros(n, np.uint64), lens=np.full(n, L, np.int64), y=np.zeros(n, np.int64))
        else:
            lens = rng.integers(c11['length_min'], c11['length_max'] + 1, n)
            bits = rng.integers(0, 2 ** 32, n, dtype=np.uint64) & ((np.uint64(1) << lens.astype(np.uint64)) - np.uint64(1))
            raw = dict(bits=bits, lens=lens, y=rng.integers(0, 2, n))
    else:
        if which == 'lower':
            raw = dict(bits_x=np.zeros((n, 10), np.int64), y=np.zeros((n, 10), np.int64))
        else:
            raw = dict(bits_x=rng.integers(0, 2, (n, 10)), y=rng.integers(0, 2, (n, 10)))
    return encode(c, fam, raw)


# ---------------------------------------------------------------- roster file and checks

def save_roster(path, roster, logs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(dict(roster=roster, selection_log=logs), indent=1))
    tmp.replace(path)


def load_roster(path):
    d = json.loads(Path(path).read_text())
    return d['roster'], d['selection_log']


EXPECTED_ROLES = dict(dfa=dict(probe_pretrain=16, target=4, distractor=4, fresh=16),
                      dsl=dict(probe_pretrain=16, target=4, distractor=4, fresh=16))


def check_roster(c, roster):
    """PROTOCOL Section 2.2 checks: role counts per K level, pairwise distinct languages / functions across every
    task of a family, exact K recomputed with the imported Experiment 11/12 code. Returns (ok, report dict)."""
    mods = exp_modules(c)
    rep, ok = {}, True
    for fam in FAMILIES:
        levels = sorted({t['K'] for t in roster[fam]})
        counts = {f'{role}@K={K}': sum(1 for t in roster[fam] if t['role'] == role and t['K'] == K)
                  for role in EXPECTED_ROLES[fam] for K in levels}
        good_counts = all(counts[f'{r}@K={K}'] == n for r, n in EXPECTED_ROLES[fam].items() for K in levels)
        n_levels_ok = len(levels) == (4 if fam == 'dfa' else 5)
        if fam == 'dfa':
            A = mods['dfa']['gen']
            keys, k_ok = [], []
            for t in roster[fam]:
                d = A.DFA.from_json(t['definition'])
                cf_ = A.canonical_form(d)
                keys.append(cf_.key())
                k_ok.append(cf_.n == t['K'])
        else:
            D = mods['dsl']['gen']
            bfs = D.load_or_run_bfs(mods['dsl']['cfg'], mods['dsl']['dir'] / 'results' / 'bfs_cache.npz')
            keys, k_ok = [], []
            for t in roster[fam]:
                tab = np.asarray(t['truth_table'], np.uint16)
                keys.append(tab.tobytes())
                k_ok.append(D.exact_k(tab, bfs)[0] == t['K'])
            del bfs
        distinct = len(set(keys)) == len(keys)
        rep[fam] = dict(levels=levels, role_counts=counts, role_counts_ok=good_counts, n_levels_ok=n_levels_ok,
                        n_tasks=len(keys), pairwise_distinct=distinct, k_recomputed_ok=all(k_ok),
                        n_k_mismatch=int(len(k_ok) - sum(k_ok)), n_embedded=len(embedded_tasks(roster, fam)),
                        n_noise_tasks=len(noise_tasks(c, roster, fam)))
        ok &= good_counts and n_levels_ok and distinct and all(k_ok)
    return ok, rep


def check_inputs(c, roster):
    """Embedding inputs carry the null token at position 0 and only bit tokens elsewhere (never a task token), and
    every task's inputs come from one input distribution. DSL: all tasks' 2,048 inputs are the same multiset (four
    passes over the shared 512 training inputs), so the total-variation (TV) distance between task input
    distributions is exactly 0. DFA: every input is a train-partition string from Experiment 11's sampler (length
    uniform on 8..24, then a uniform string), the distribution of every Experiment 11 pool and every new task, so the
    TV distance between task input distributions is 0 by construction; the check confirms every input is in the
    train partition and in the length range, and reports the largest empirical TV distance of the length histograms
    between tasks (sampling noise only)."""
    mods = exp_modules(c)
    rep, ok = {}, True
    for fam in FAMILIES:
        null = 2 + null_token(c, fam)
        tok_ok, multisets, hists, part_ok = True, set(), [], True
        for t in embedded_tasks(roster, fam):
            ex = example_set(c, roster, fam, t)
            X = ex['X']
            tok_ok &= bool((X[:, 0] == null).all() and (X[:, 1:] <= 1).all() and (X[:, 1:] >= 0).all())
            if fam == 'dsl':
                multisets.add(np.sort(raw_examples(c, roster, fam, t)['x']).tobytes())
            else:
                raw = raw_examples(c, roster, fam, t)
                c11 = mods['dfa']['cfg']
                lens = np.asarray(raw['lens'], np.int64)
                part_ok &= bool((lens >= c11['length_min']).all() and (lens <= c11['length_max']).all())
                for L in np.unique(lens):
                    part_ok &= not mods['dfa']['data'].in_test_partition(raw['bits'][lens == L], int(L), c11).any()
                hists.append(np.bincount(lens, minlength=c11['length_max'] + 1)[c11['length_min']:] / len(lens))
        r = dict(null_token_only=tok_ok)
        if fam == 'dsl':
            r['distinct_input_multisets'] = len(multisets)
            r['tv_distance'] = 0.0 if len(multisets) == 1 else None
            ok &= tok_ok and len(multisets) == 1
        else:
            H = np.array(hists)
            tv = max(0.5 * np.abs(H[i] - H[j]).sum() for i in range(len(H)) for j in range(i + 1, len(H)))
            r.update(train_partition_and_length_range=part_ok, tv_distance_by_construction=0.0,
                     max_empirical_length_tv_between_tasks=float(tv))
            ok &= tok_ok and part_ok
        rep[fam] = r
    return ok, rep
