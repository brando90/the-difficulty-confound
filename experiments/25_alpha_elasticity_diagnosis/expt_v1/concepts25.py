"""Experiment 25 concepts and exact references (PROTOCOL Section 2.1-2.2).

Targets and their disjoint pools are Experiment 24's: its unchanged code (concepts24.build_all) is run on its frozen
config, and the n = 8 and n = 32 targets (3 each) with their 4 disjoint distractors (tokens 150-299) are kept, with
Experiment 24's task ids and concept ids, so every document of these concepts is generated exactly as there.

The iid pool: for every disjoint distractor d, a one-state concept over d's tokens whose emission distribution is the
exact average token distribution of d's 128-token documents (start state uniform), rounded to multiples of 2^-12. Its
documents have d's unigram statistics and no sequential structure.

References: 1,024 held-out documents per target (Experiment 24's, same eval seed and concept ids) and 64 per
distractor; oracle code lengths; exact Bayes-mixture code lengths of {target} + pool under the training prior
(alpha, (1 - alpha)/4 each) for both pools and every alpha < 1; the posterior on the target after 16 tokens.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
E13 = ROOT / '13_ginc_hmm_complexity_vs_alignment' / 'expt_v1'
E24 = ROOT / '24_similarity_alignment_complexity' / 'expt_v1'
for p in (E13, E24, HERE):
    sys.path.insert(0, str(p))

import concepts24 as C24  # noqa: E402
import hmm as H  # noqa: E402
import run as R13  # noqa: E402

PREP_FILES = ['tasks.json', 'acceptance_log.json', 'oracle.npz', 'mixture.npz', 'posterior16.json']
POOLS = ['disjoint', 'iid']


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def exp24_config():
    return json.loads((E24 / 'config.json').read_text())


def sha256_file(p):
    return R13.sha256_file(p)


def doc_unigram(h, L):
    """Exact average token distribution over positions 1..L of h's documents (uniform start state)."""
    A, O = H.matrices(h)
    d = np.full(h['n'], 1.0 / h['n'])
    tot = np.zeros(h['V'])
    for _ in range(L):
        tot += d @ O
        d = d @ A
    return tot / L


def iid_concept(h, q, L):
    """One state, emitting h's document unigram rounded to multiples of 2^-q (every token of h's support kept)."""
    p = doc_unigram(h, L)
    toks = np.flatnonzero(p > 0)
    num = H.quantize(p[toks] / p[toks].sum(), q)
    g = dict(n=1, V=int(h['V']), q=int(q), succ=np.zeros((1, 1), np.int64), trans=np.array([[2 ** q]], np.int64),
             etok=toks[None, :].astype(np.int64), emit=num[None, :].astype(np.int64))
    tv = 0.5 * float(np.abs(num / 2 ** q - p[toks]).sum())
    return g, tv


def unigram_entropy(g):
    p = g['emit'][0] / 2 ** g['q']
    return float(-(p * np.log2(p)).sum())


def build_all(cfg, log=print, cfg24=None):
    """Exp 25's task list: the kept Experiment 24 targets and disjoint pools, then the iid pools (cid offset 1000)."""
    cfg24 = exp24_config() if cfg24 is None else cfg24
    tasks24, acc24 = C24.build_all(cfg24, log)
    cc = cfg['concepts']
    tg = [t for t in tasks24 if t['role'] == 'target' and t['K'] in cc['levels']]
    ids = {t['task_id'] for t in tg}
    pools = [t for t in tasks24 if t['role'] == 'distractor' and t['similarity'] == cc['pool_similarity']
             and t['target_id'] in ids]
    iid = []
    for p in pools:
        g, tv = iid_concept(H.from_json(p['definition']), cc['iid_q'], cc['doc_len'])
        iid.append(dict(task_id=f"{p['task_id']}_iid", cid=cc['iid_cid_offset'] + p['cid'], family='iid',
                        role='distractor', target_id=p['target_id'], similarity='iid', parent_id=p['task_id'], K=1,
                        K_bits=None, definition=H.to_json(g), support=int(g['etok'].shape[1]), unigram_tv=tv,
                        entropy_rate=unigram_entropy(g), parent_entropy_rate=p['entropy_rate']))
    tasks = tg + pools + iid
    acc = dict(h_ref=acc24['h_ref'], band_bits=acc24['band_bits'], exp24_concepts=len(tasks24),
               kept=dict(targets=len(tg), disjoint=len(pools), iid=len(iid)))
    return tasks, acc


def pool_of(tasks, target_id, pool):
    return sorted([t for t in tasks if t['role'] == 'distractor' and t['target_id'] == target_id
                   and t['similarity'] == pool], key=lambda t: t['task_id'])


def targets(tasks):
    return sorted([t for t in tasks if t['role'] == 'target'], key=lambda t: (t['K'], t['task_id']))


def prior_for(alpha, D):
    return [alpha] + [(1 - alpha) / D] * D if alpha < 1 else [1.0]


def eval_docs(cfg, h, cid, n_docs):
    return H.sample_docs(h, cfg['eval']['eval_seed'], cid, range(n_docs), cfg['concepts']['doc_len'])


def build_references(cfg, tasks, out):
    cm = {t['cid']: H.from_json(t['definition']) for t in tasks}
    (out / 'data').mkdir(parents=True, exist_ok=True)
    oracle, mixture, post = {}, {}, {}
    for t in tasks:
        nd = cfg['eval']['distractor_eval_docs'] if t['role'] == 'distractor' else cfg['eval']['eval_docs_per_target']
        docs = eval_docs(cfg, cm[t['cid']], t['cid'], nd)
        with open(out / 'data' / f"{t['task_id']}_test.npz", 'wb') as f:
            np.savez(f, docs=docs.astype(np.int16))
        oc = H.forward_logprobs(cm[t['cid']], docs)
        oracle[t['task_id']] = oc
        if t['role'] != 'target':
            continue
        for pool in POOLS:
            ps = pool_of(tasks, t['task_id'], pool)
            codes = [oc] + [H.forward_logprobs(cm[p['cid']], docs) for p in ps]
            for a in cfg['grid']['alphas']:
                if a >= 1:
                    continue
                pr = prior_for(a, len(ps))
                mixture[f"{t['task_id']}|{pool}|{a:.2f}"] = H.mixture_from_codes(codes, pr)
                post[f"{t['task_id']}|{pool}|{a:.2f}"] = float(H.posterior_after(codes, pr, 16).mean())
    with open(out / 'oracle.npz', 'wb') as f:
        np.savez(f, **oracle)
    with open(out / 'mixture.npz', 'wb') as f:
        np.savez(f, **mixture)
    R13.atomic_write_text(out / 'posterior16.json', json.dumps(post, indent=1, sort_keys=True))


def prep_hashes(prep):
    prep = Path(prep)
    files = PREP_FILES + sorted(f'data/{p.name}' for p in (prep / 'data').glob('*.npz'))
    return {f: sha256_file(prep / f) for f in files}


def prepare(cfg, out, log=print, cfg24=None):
    out.mkdir(parents=True, exist_ok=True)
    tasks, acc = build_all(cfg, log, cfg24)
    R13.atomic_write_text(out / 'tasks.json', json.dumps(tasks, indent=1))
    R13.atomic_write_text(out / 'acceptance_log.json', json.dumps(acc, indent=1))
    build_references(cfg, tasks, out)
    return tasks, acc


def build_prep(cfg, prep, log=print, cfg24=None):
    """Build into a fresh folder, then install atomically; an existing prep must have identical bytes."""
    prep = Path(prep)
    tmp = prep.parent / f'{prep.name}.tmp-{os.getpid()}'
    shutil.rmtree(tmp, ignore_errors=True)
    tasks, acc = prepare(cfg, tmp, log, cfg24)
    if prep.exists():
        a, b = prep_hashes(prep), prep_hashes(tmp)
        if a != b:
            raise SystemExit(f'rebuilt references differ from {prep}: '
                             f'{sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:5]}')
        shutil.rmtree(tmp)
    else:
        os.replace(tmp, prep)
    return tasks, acc


def load_refs(prep, tasks):
    prep = Path(prep)
    o = np.load(prep / 'oracle.npz')
    m = np.load(prep / 'mixture.npz')
    docs = {t['task_id']: np.load(prep / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64) for t in tasks}
    return dict(oracle={k: o[k] for k in o.files}, mixture={k: m[k] for k in m.files}, docs=docs)


def iid_matches_parent(g, h, L, tol):
    p = doc_unigram(h, L)
    q = np.zeros(h['V'])
    q[g['etok'][0]] = g['emit'][0] / 2 ** g['q']
    return 0.5 * float(np.abs(p - q).sum()) <= tol and set(np.flatnonzero(p > 0)) == set(g['etok'][0].tolist())


def mean_eps_id(prep):
    o, mx = np.load(Path(prep) / 'oracle.npz'), np.load(Path(prep) / 'mixture.npz')
    eid = {}
    for k in mx.files:
        tid, s, a = k.split('|')
        eid.setdefault(s, {}).setdefault(a, []).append(float((mx[k] - o[tid]).mean()))
    return {s: {a: float(np.mean(v)) for a, v in d.items()} for s, d in eid.items()}

