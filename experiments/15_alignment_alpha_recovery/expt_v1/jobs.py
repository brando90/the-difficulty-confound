"""Scoring jobs for the Experiment 11/12 mixes (one process each, resume-safe: every job writes one part file
atomically; a finished part is never recomputed)."""
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C
import mixes15 as MX
import scores as S

NONPROBE = ['a', 'aprime', 'a_leak', 'a_perm', 'a_inp', 'a2', 'b2', 'oracle', 'random']
PROBE = ['b1', 'b1_leak', 'b1_perm', 'b1_inp', 'd', 'd_perm']


# ---------------------------------------------------------------- references and scored units on disk

def save_refs_units(out, fam, tasks, R, U):
    targets = [t['task_id'] for t in tasks if t['role'] == 'target']
    ids = [t['task_id'] for t in tasks]
    ref = {f'{fam}__targets': np.array(targets)}
    for k in R[targets[0]]['raw']:
        ref[f'{fam}__{k}'] = np.stack([R[t]['raw'][k] for t in targets])
    ref[f'{fam}__file_index'] = np.stack([R[t]['file_index'] for t in targets])
    uni = {f'{fam}__tasks': np.array(ids)}
    for k in U[ids[0]]['raw']:
        uni[f'{fam}__{k}'] = np.stack([U[t]['raw'][k] for t in ids])
    uni[f'{fam}__file_index'] = np.stack([U[t]['file_index'] for t in ids])
    return ref, uni


class Setup:
    """References, scored units, pools and mix counts of one arm: dfa (Experiment 11), dsl (Experiment 12) or graded
    (the DFA graded-relevance arm; its 'alphas' are the mixes m1-m5)."""

    def __init__(self, c, arm, out):
        self.c, self.arm, self.out = c, arm, Path(out)
        self.fam = 'dfa' if arm == 'graded' else arm
        self.code = dict(dfa=11, dsl=12, graded=15)[arm]
        if arm == 'graded':
            G = json.loads((self.out / 'graded' / 'graded_tasks_full.json').read_text())
            self.G = G
            self.tasks = sorted(G['tasks'], key=lambda t: t['task_token'])
            self.counts = {(k.rsplit('_', 1)[0], k.rsplit('_', 1)[1]): v for k, v in G['counts_mean'].items()}
            self.count_spread = G['counts_spread']
            self.alphas = sorted(G['mix_names'])
        else:
            self.tasks = C.load_tasks(c, arm)
            self.counts, self.count_spread = MX.tmax_counts(c, arm)
            self.alphas = sorted(MX.upstream_cfg(c, arm)['alphas'])
        self.by_id = {t['task_id']: t for t in self.tasks}
        zr = np.load(self.out / 'references.npz')
        zu = np.load(self.out / 'scored_units.npz')
        fields = ['bits', 'lens', 'y'] if self.fam == 'dfa' else ['x', 'yw']
        self.targets = [str(x) for x in zr[f'{arm}__targets']]
        self.ids = [str(x) for x in zu[f'{arm}__tasks']]
        assert self.ids == [t['task_id'] for t in self.tasks]
        self.refs = {t: {k: zr[f'{arm}__{k}'][i] for k in fields} for i, t in enumerate(self.targets)}
        self.ref_file_index = {t: zr[f'{arm}__file_index'][i] for i, t in enumerate(self.targets)}
        self.units = {t: {k: zu[f'{arm}__{k}'][i] for k in fields} for i, t in enumerate(self.ids)}
        self.unit_file_index = {t: zu[f'{arm}__file_index'][i] for i, t in enumerate(self.ids)}
        self.nU = C.n_examples(self.units[self.ids[0]])

    def layout(self, target):
        if self.arm == 'graded':
            return [target] + [t['task_id'] for t in self.tasks if t.get('of_target') == target]
        return MX.pool_layout(self.tasks, target)

    def pool(self, target, n_per_task=None):
        lay = self.layout(target)
        if n_per_task is None:
            return C.concat([self.units[u] for u in lay])
        return C.concat([C.take(self.units[u], np.arange(n_per_task)) for u in lay])

    def n_u(self, target, alpha):
        key = (target, alpha if self.arm == 'graded' else float(alpha))
        cnt = self.counts[key]
        return np.array([cnt.get(u, 0.0) for u in self.layout(target)])

    def token(self, task_id):
        return self.by_id[task_id]['task_token']

    def perm_refs(self, target):
        """References with labels randomly permuted (seed 20261015)."""
        r = dict(self.refs[target])
        rng = np.random.default_rng([self.c['task_sampling_seed'], self.code, 4, self.token(target)])
        p = rng.permutation(C.n_examples(r))
        key = 'y' if self.fam == 'dfa' else 'yw'
        r[key] = r[key][p]
        return r

    def ncd_ref_index(self, target):
        rng = np.random.default_rng([self.c['task_sampling_seed'], self.code, 5, self.token(target)])
        return np.sort(rng.choice(self.c['references_per_target'], self.c['ncd_references'], replace=False))


def ref_samples(target, xs):
    return [(('ref', target, i), b) for i, b in enumerate(xs)]


def dict_path(out, arm, target, variant):
    return Path(out) / 'dicts' / f'{arm}__{target}__{variant}.zdict'


# ---------------------------------------------------------------- non-probe measures (one job per target)

CONTROLS = dict(dfa=True, dsl=True, graded=False)


def nonprobe_scores(c, st, target, out=None, only=None):
    fam = st.fam
    tok = st.token(target)
    lay = st.layout(target)
    pool = st.pool(target)
    refs = st.refs[target]
    res, timing = {}, {}
    dict_id = 15000 + 100 * (st.code - 10) + tok
    controls = CONTROLS[st.arm]

    def want(k):
        return (only is None or k in only) and (controls or k in ('a', 'aprime', 'a2', 'b2', 'oracle', 'random'))

    pool_bytes = C.serialize(c, fam, pool)
    variants = dict(a=lambda: (C.serialize(c, fam, refs), pool_bytes))
    if controls:
        variants.update(
            a_leak=lambda: (C.serialize(c, fam, refs, token=tok),
                            C.serialize(c, fam, pool, token=np.repeat([st.token(u) for u in lay], st.nU))),
            a_perm=lambda: (C.serialize(c, fam, st.perm_refs(target)), pool_bytes),
            a_inp=lambda: (C.serialize(c, fam, refs, input_only=True), C.serialize(c, fam, pool, input_only=True)))
    for v, mk in variants.items():
        if not (want(v) or (v == 'a' and want('aprime'))):
            continue
        t0 = time.time()
        rb, pb = mk()
        d = train_or_load_dict(c, out, st.arm, target, v, rb, dict_id)
        a, ap = S.score_a(c, d, pb)
        res[v] = a
        if v == 'a':
            res['aprime'] = ap
        timing[v] = time.time() - t0
    if want('a2'):
        t0 = time.time()
        ri = st.ncd_ref_index(target)
        rb = C.serialize(c, fam, C.take(refs, ri))
        res['a2'] = S.score_a2(c, C.serialize(c, fam, st.pool(target, c['ncd_units_per_task'])), rb)
        timing['a2'] = time.time() - t0
    if want('b2'):
        t0 = time.time()
        res['b2'], vocab = S.score_b2(c, C.serialize(c, fam, refs), pool_bytes)
        timing['b2'] = time.time() - t0
        timing['b2_vocab'] = vocab
    if want('oracle'):
        lab = C.labels_for(c, fam, st.by_id[target], pool)
        key = 'y' if fam == 'dfa' else 'yw'
        res['oracle'] = (np.asarray(pool[key]).astype(np.int64) == lab).astype(np.float64)
    if want('random'):
        rng = np.random.default_rng([c['task_sampling_seed'], st.code, 3, tok])
        res['random'] = rng.random(C.n_examples(pool))
    return res, timing


def train_or_load_dict(c, out, arm, target, variant, ref_bytes, dict_id):
    p = None if out is None else dict_path(out, arm, target, variant)
    if p is not None and p.exists():
        return p.read_bytes()
    d = S.train_zstd_dictionary(c, ref_samples(target, ref_bytes), dict_id)
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(p) + '.tmp')
        tmp.write_bytes(d)
        tmp.replace(p)
    return d


def job_nonprobe(args):
    arm, target, out = args
    c = C.load_config()
    torch.set_num_threads(c['threads'])
    st = Setup(c, arm, out)
    t0 = time.time()
    res, timing = nonprobe_scores(c, st, target, out)
    p = Path(out) / 'parts' / part_name('nonprobe', args)
    C.atomic_savez(p, **{k: np.asarray(v, np.float32) for k, v in res.items()},
                   timing=np.array(json.dumps(timing)))
    return dict(job=p.name, seconds=time.time() - t0, timing=timing)


# ---------------------------------------------------------------- probe measures (one job per arm x probe)

def probe_scores(c, st, probe, targets=None, unit_index=None):
    """b1 and d for every target's pool (main arms also: b1 with the task token kept, d with permuted-label
    references). unit_index: optional dict task -> unit indices to score (verification samples)."""
    fam = st.fam
    controls = CONTROLS[st.arm]
    import task2vec as t2v
    model = C.load_probe(c, fam, probe)
    model.eval()
    spec = t2v.Spec(model, fam)
    targets = targets or st.targets
    sel = {u: (np.arange(st.nU) if unit_index is None else np.asarray(unit_index[u])) for u in st.ids}
    timing = {}
    t0 = time.time()
    feat, feat_leak = {}, {}
    for u in st.ids:
        raw = C.take(st.units[u], sel[u])
        feat[u] = S.pooled_features(model, fam, C.probe_input(c, fam, raw))
        if controls:
            feat_leak[u] = S.pooled_features(model, fam, C.probe_input(c, fam, raw, token=st.token(u)))
    timing['unit_features'] = time.time() - t0
    t0 = time.time()
    G, Gp, rf, rfl = [], [], {}, {}
    for t in targets:
        rf[t] = S.pooled_features(model, fam, C.probe_input(c, fam, st.refs[t]))
        G.append(S.mean_gradient(spec, C.probe_input(c, fam, st.refs[t])))
        if controls:
            rfl[t] = S.pooled_features(model, fam, C.probe_input(c, fam, st.refs[t], token=st.token(t)))
            Gp.append(S.mean_gradient(spec, C.probe_input(c, fam, st.perm_refs(t))))
    timing['ref_features_and_mean_gradients'] = time.time() - t0
    t0 = time.time()
    Gall = np.stack(G + Gp)
    cos = {}
    n_grad = 0
    for u in st.ids:
        cos[u] = S.per_example_grad_cos(spec, C.probe_input(c, fam, C.take(st.units[u], sel[u])), Gall)
        n_grad += len(sel[u])
    timing['unit_gradients'] = time.time() - t0
    timing['n_unit_gradients'] = n_grad
    nT = len(targets)
    res = {k: [] for k in (('b1', 'b1_leak', 'd', 'd_perm') if controls else ('b1', 'd'))}
    for ti, t in enumerate(targets):
        lay = st.layout(t)
        res['b1'].append(S.max_cosine(np.concatenate([feat[u] for u in lay]), rf[t]))
        res['d'].append(np.concatenate([cos[u][:, ti] for u in lay]))
        if controls:
            res['b1_leak'].append(S.max_cosine(np.concatenate([feat_leak[u] for u in lay]), rfl[t]))
            res['d_perm'].append(np.concatenate([cos[u][:, nT + ti] for u in lay]))
    return {k: np.stack(v) for k, v in res.items()}, timing


def job_probe(args):
    arm, probe, out = args[:3]
    targets = args[3] if len(args) > 3 else None  # the pilot scores 2 targets per family only
    c = C.load_config()
    torch.set_num_threads(c['threads'])
    st = Setup(c, arm, out)
    t0 = time.time()
    res, timing = probe_scores(c, st, probe, targets=targets)
    if CONTROLS[arm]:
        # the probe never reads the label (its input is the task token and the input symbols), so permuting the
        # references' labels or dropping the label leaves b1 unchanged; test_scores.py asserts this on real data
        res['b1_perm'] = res['b1']
        res['b1_inp'] = res['b1']
    p = Path(out) / 'parts' / part_name('probe', args)
    C.atomic_savez(p, **{k: np.asarray(v, np.float32) for k, v in res.items()}, timing=np.array(json.dumps(timing)))
    return dict(job=p.name, seconds=time.time() - t0, timing=timing)


# ---------------------------------------------------------------- Task2Vec (one job per arm x probe x target)

def t2v_compositions(st, target):
    """[(name, task ids, shares, c0 name)]: every mix; its c0 (the mix's distractors only, same relative shares);
    c1 (target units only). Main arms share one c0 (all distractors at the alpha = 0.05 counts' relative shares)."""
    lay = st.layout(target)
    comp = []
    if st.arm == 'graded':
        for m in st.alphas:
            n = st.n_u(target, m)
            keep = [i for i in range(len(lay)) if n[i] > 0]
            comp.append((m, [lay[i] for i in keep], n[keep] / n[keep].sum(), f'c0_{m}'))
            dk = [i for i in keep if i > 0]
            comp.append((f'c0_{m}', [lay[i] for i in dk], n[dk] / n[dk].sum(), None))
    else:
        for a in st.alphas:
            n = st.n_u(target, a)
            keep = [i for i in range(len(lay)) if n[i] > 0]
            comp.append((f'alpha{a:g}', [lay[i] for i in keep], n[keep] / n[keep].sum(), 'c0'))
        n = st.n_u(target, min(st.alphas))[1:]
        comp.append(('c0', lay[1:], n / n.sum(), None))
    comp.append(('c1', [target], np.array([1.0]), None))
    return comp


def t2v_scores(c, st, probe, target):
    import task2vec as t2v
    c14 = C.mods(c)['e14']['cfg']
    fam = st.fam
    model = C.load_probe(c, fam, probe)
    model.eval()
    spec = t2v.Spec(model, fam)
    comps = t2v_compositions(st, target)
    need = sorted(set(u for _, ids, _, _ in comps for u in ids))
    ex_units = {u: C.probe_input(c, fam, st.units[u]) for u in need}
    ex_ref = C.probe_input(c, fam, st.refs[target])
    W, b = S.fixed_head(c14, t2v, spec, ex_ref)
    emb = {'e': {}, 'e2': {}}
    hl = {'e': [], 'e2': []}

    def embed(name, batches):
        for est in ('e', 'e2'):
            out = []
            for i, bt in enumerate(batches):
                parts = (15, st.arm, target, probe, est, name, i)
                r = t2v.embed_batch_e3(c14, spec, bt, parts) if est == 'e' else \
                    S.embed_fixed_head(c14, t2v, spec, bt, parts, W, b)
                out.append(r['f'])
                hl[est].append(r['head_loss'])
            emb[est][name] = out

    t0 = time.time()
    embed('ref', S.ref_batches(c, ex_ref))
    for name, ids, shares, _ in comps:
        embed(name, S.mix_batches(c, t2v, [ex_units[u] for u in ids], list(shares), st.arm, target, name))
    out = dict(n_batch_embeddings=2 * c['task2vec_batches'] * (len(comps) + 1), embed_seconds=time.time() - t0)
    for est in ('e', 'e2'):
        cd = {name: t2v.cross_diversity(emb[est][name], emb[est]['ref']) for name, _, _, _ in comps}
        ah = {name: S.alpha_hat_e(cd[c0], cd['c1'], cd[name]) for name, _, _, c0 in comps if c0 is not None}
        out[est] = dict(cross_div=cd, alpha_hat=ah, mean_head_loss=float(np.mean(hl[est])))
    return out


def job_t2v(args):
    arm, probe, target, out = args
    c = C.load_config()
    torch.set_num_threads(c['threads'])
    st = Setup(c, arm, out)
    t0 = time.time()
    res = t2v_scores(c, st, probe, target)
    res['seconds'] = time.time() - t0
    p = Path(out) / 'parts' / part_name('t2v', args)
    C.atomic_write_text(p, json.dumps(res))
    return dict(job=p.name, seconds=res['seconds'])


# ---------------------------------------------------------------- merge parts into scores/

ARMS = ('dfa', 'dsl', 'graded')


def nonprobe_measures(arm):
    return NONPROBE if CONTROLS[arm] else ['a', 'aprime', 'a2', 'b2', 'oracle', 'random']


def probe_measures(arm):
    return PROBE if CONTROLS[arm] else ['b1', 'd']


def merge_scores(c, out, arm, probes=None):
    st = Setup(c, arm, out)
    probes = probes or c['probes']
    d = Path(out) / 'scores'
    d.mkdir(exist_ok=True)
    for m in nonprobe_measures(arm):
        arr = np.stack([np.load(Path(out) / 'parts' / part_name('nonprobe', (arm, t)))[m] for t in st.targets])
        C.atomic_savez(d / f'{arm}_{m}_all.npz', raw=arr, targets=np.array(st.targets))
    for p in probes:
        z = np.load(Path(out) / 'parts' / part_name('probe', (arm, p)))
        for m in probe_measures(arm):
            C.atomic_savez(d / f'{arm}_{m}_{p}.npz', raw=z[m], targets=np.array(st.targets))
    t2 = {}
    for p in probes:
        for t in st.targets:
            t2[f'{p}|{t}'] = json.loads((Path(out) / 'parts' / part_name('t2v', (arm, p, t))).read_text())
    C.atomic_write_text(d / f'{arm}_task2vec.json', json.dumps(t2))


def arm_targets(c, out, arm):
    if arm == 'graded':
        G = json.loads((Path(out) / 'graded' / 'graded_tasks_full.json').read_text())
        return [t['task_id'] for t in sorted(G['tasks'], key=lambda t: t['task_token']) if t['role'] == 'target']
    return [t['task_id'] for t in C.load_tasks(c, arm) if t['role'] == 'target']


def all_jobs(c, out, arms=ARMS, probes=None, targets=None):
    """Probe jobs first (longest), then Task2Vec, then the non-probe jobs."""
    probes = probes or c['probes']
    jobs = []
    for arm in arms:
        jobs += [('probe', (arm, p, str(out))) for p in probes]
    for arm in arms:
        tt = (targets or {}).get(arm) or arm_targets(c, out, arm)
        jobs += [('t2v', (arm, p, t, str(out))) for p in probes for t in tt]
    for arm in arms:
        tt = (targets or {}).get(arm) or arm_targets(c, out, arm)
        jobs += [('nonprobe', (arm, t, str(out))) for t in tt]
    return jobs


def part_name(kind, args):
    if kind == 'probe':
        return f'{args[0]}__p{args[1]}__probe.npz'
    if kind == 'nonprobe':
        return f'{args[0]}__{args[1]}__nonprobe.npz'
    return f'{args[0]}__p{args[1]}__{args[2]}__t2v.json'


RUNNERS = dict(probe=job_probe, nonprobe=job_nonprobe, t2v=job_t2v)


def run_job(kind_args):
    kind, args = kind_args
    t0 = time.time()
    r = RUNNERS[kind](args)
    r['kind'] = kind
    r['wall'] = time.time() - t0
    return r


# ---------------------------------------------------------------- exact Eq. 2 on full mixes (measure c)

def eq2_check(c, out, fam, probes=None, log=print):
    """Exact Eq. 2 over full training multisets versus the stratified estimator, measure c, one seed."""
    st = Setup(c, fam, out)
    cfg = MX.upstream_cfg(c, fam)
    M = C.mods(c)[fam]
    seed = c['eq2_check']['dfa_seed' if fam == 'dfa' else 'dsl_seed']
    if fam == 'dfa':
        tg = []
        for K in cfg['n_levels']:
            tg.append(next(t for t in st.targets if st.by_id[t]['K'] == K))
    else:
        tg = list(st.targets)
    probes = probes or c['probes']
    files = {t['task_id']: C.load_train_file(c, fam, t['task_id']) for t in st.tasks}
    rows = []
    for t in tg:
        ti = st.targets.index(t)
        a_raw = np.load(Path(out) / 'scores' / f'{fam}_a_all.npz')['raw'][ti].astype(np.float64)
        dct = dict_path(out, fam, t, 'a').read_bytes()
        mixes = []
        for alpha in st.alphas:
            mix = M['data'].assemble_mix(cfg, st.tasks, t, alpha, seed)
            idx = mix['pool_index'] if fam == 'dfa' else mix['input_index']
            mixes.append((alpha, mix['task_token'], idx))
        # distinct (task token, file index) across this target's mixes
        keys = np.unique(np.concatenate([tok * (1 << 20) + idx for _, tok, idx in mixes]))
        ktok, kidx = keys >> 20, keys & ((1 << 20) - 1)
        tok2id = {st.token(u): u for u in st.ids}
        raws = C.concat([C.take(files[tok2id[int(k)]], kidx[ktok == k]) for k in np.unique(ktok)])
        order = np.concatenate([np.flatnonzero(ktok == k) for k in np.unique(ktok)])
        xs = C.serialize(c, fam, raws)
        a_full, _ = S.score_a(c, dct, xs)
        tt = C.n_tokens(fam, raws)
        lo_a, hi_a = a_raw.min(), a_raw.max()
        na = (a_full - lo_a) / (hi_a - lo_a)
        per_probe = {}
        for p in probes:
            b_raw = np.load(Path(out) / 'scores' / f'{fam}_b1_{p}.npz')['raw'][ti].astype(np.float64)
            model = C.load_probe(c, fam, p)
            model.eval()
            rf = S.pooled_features(model, fam, C.probe_input(c, fam, st.refs[t]))
            bf = S.max_cosine(S.pooled_features(model, fam, C.probe_input(c, fam, raws)), rf)
            nb = (bf - b_raw.min()) / (b_raw.max() - b_raw.min())
            sc = 0.5 * nb + 0.5 * na
            pos = np.empty(len(keys), np.int64)
            pos[order] = np.arange(len(keys))
            for alpha, tok, idx in mixes:
                k = tok * (1 << 20) + idx
                j = pos[np.searchsorted(keys, k)]
                per_probe.setdefault(alpha, []).append(S.eq2_exact(tt[j], sc[j]))
        for alpha in st.alphas:
            rows.append(dict(family=fam, target=t, alpha=alpha, seed=seed,
                             exact_c_per_probe=per_probe[alpha], exact_c=float(np.mean(per_probe[alpha]))))
        log(f'eq2 check {fam} {t} done')
    return rows
