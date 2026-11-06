"""Experiment 24 concepts and exact references (PROTOCOL Sections 2.1-2.4).

Targets: Experiment 13's acceptance procedure (run.build_concepts, unchanged) on a new stream (arm_index 2), levels
n = 8, 32, 64, 3 targets and 1 pilot concept per level. Distractor pools: for every target (and the pilot concept) and
every similarity level, 4 distractors of the target's n, each from its own PCG64 stream seeded
(task_sampling_seed, pool_seed_tag, target cid, similarity index, d), accepted with the same checks as the targets
(strongly connected and aperiodic, certified process rank n, quick entropy rate within the targets' band of h_ref):
  copy20 / copy50  the target with round(f n) states (f = 0.2 / 0.5, at least 1) re-drawn: those states' successor
                   set, transition numerators, emission tokens and emission numerators come from a fresh candidate;
  family           an independent candidate of the same n on the same 150 tokens;
  disjoint         an independent candidate of the same n whose emission tokens are shifted to 150..299.
References: 1,024 held-out documents per target; oracle code lengths; exact Bayes-mixture code lengths of
{target} + pool under the training prior (alpha, (1 - alpha)/4 each) for every similarity level and alpha < 1.
"""
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E13 = HERE.parent.parent / '13_ginc_hmm_complexity_vs_alignment' / 'expt_v1'
sys.path.insert(0, str(E13))
sys.path.insert(0, str(HERE))

import hmm as H  # noqa: E402
import run as R13  # noqa: E402

PREP_FILES = ['tasks.json', 'acceptance_log.json', 'oracle.npz', 'mixture.npz', 'posterior16.json']


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def concept_config(cfg):
    """The keys run.build_concepts reads, from config['concepts']."""
    cc = dict(cfg['concepts'])
    cc['V'] = cc['V_concepts']
    return cc


def sha256_file(p):
    return R13.sha256_file(p)


def _qh(cc, h, key):
    r = np.random.default_rng(list(key))
    return H.entropy_rate(h, r, cc['quick_entropy_seqs'], cc['entropy_seq_len'], cc['entropy_burn_in'])[0]


def _rows_equal(a, b, i):
    return all(np.array_equal(a[k][i], b[k][i]) for k in ('succ', 'trans', 'etok', 'emit'))


def shared_fraction(target, h):
    if h['V'] != target['V']:
        return 0.0
    return float(np.mean([_rows_equal(target, h, i) for i in range(target['n'])]))


def emission_overlap(h1, h2):
    a, b = set(int(x) for x in np.unique(h1['etok'])), set(int(x) for x in np.unique(h2['etok']))
    return len(a & b) / len(a | b)


def propose(cc, rng, target, sim):
    """One candidate distractor for `target` at similarity level `sim`."""
    n, q, V = target['n'], target['q'], cc['V_concepts']
    fresh = H.sample_candidate(rng, n, V, q)
    if sim in ('copy20', 'copy50'):
        k = max(1, int(round(cc['redraw_fraction'][sim] * n)))
        idx = np.sort(rng.choice(n, k, replace=False))
        h = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in target.items()}
        for key in ('succ', 'trans', 'etok', 'emit'):
            h[key][idx] = fresh[key][idx]
        return h
    if sim == 'disjoint':
        fresh['etok'] = fresh['etok'] + cc['disjoint_token_offset']
        fresh['V'] = cc['V_model_tokens']
    return fresh


def make_pool(cc, target_task, sim, h_ref, band, log=print):
    """The 4 distractors of one (target, similarity level), deterministic."""
    target = H.from_json(target_task['definition'])
    s_idx = cc['similarity_levels'].index(sim)
    pool = []
    for d in range(cc['pool_size']):
        rng = np.random.default_rng([cc['task_sampling_seed'], cc['pool_seed_tag'], target_task['cid'], s_idx, d])
        for attempt in range(cc['max_pool_attempts']):
            h = propose(cc, rng, target, sim)
            if H.same(h, target) or any(H.same(h, p['h']) for p in pool):
                continue
            if not H.strongly_connected_aperiodic(h):
                continue
            r, how = H.process_rank(h)
            if r != h['n']:
                continue
            hq = _qh(cc, h, (cc['task_sampling_seed'], cc['pool_seed_tag'], target_task['cid'], s_idx, d, attempt, 1))
            if abs(hq - h_ref) > band:
                continue
            pool.append(dict(h=h, attempts=attempt + 1, quick_entropy_rate=hq, process_rank=int(r)))
            break
        else:
            raise RuntimeError(f'no {sim} distractor {d} for {target_task["task_id"]} in {cc["max_pool_attempts"]} attempts')
    log(f'{target_task["task_id"]} {sim}: attempts {[p["attempts"] for p in pool]}')
    return pool


def build_all(cfg, log=print):
    """Targets, pilot concepts and every distractor pool, as one task list (cids are list indices)."""
    cc = concept_config(cfg)
    tasks, acc_log = R13.build_concepts(cc, log)
    for t in tasks:
        t['similarity'] = None
    h_ref, band = acc_log['h_ref'], acc_log['band_bits']
    owners = [t for t in tasks if t['role'] == 'target'] + \
        [t for t in tasks if t['role'] == 'pilot' and t['K'] == cfg['pilot']['level']]
    pools_log = {}
    for t in owners:
        sims = cc['similarity_levels'] if t['role'] == 'target' else [cfg['pilot']['similarity']]
        ht = H.from_json(t['definition'])
        for sim in sims:
            pool = make_pool(cc, t, sim, h_ref, band, log)
            pools_log[f"{t['task_id']}|{sim}"] = [p['attempts'] for p in pool]
            for d, p in enumerate(pool):
                h = p['h']
                r = np.random.default_rng([cc['task_sampling_seed'], cc['pool_seed_tag'], t['cid'],
                                           cc['similarity_levels'].index(sim), d, 2])
                hr, se = H.entropy_rate(h, r, cc['entropy_seqs'], cc['entropy_seq_len'], cc['entropy_burn_in'])
                tasks.append(dict(
                    task_id=f"{t['task_id']}_{sim}_d{d + 1}", cid=len(tasks), family='hmm', role='distractor',
                    target_id=t['task_id'], similarity=sim, K=int(h['n']), K_bits=H.K_bits(h['n'], h['V'], h['q']),
                    definition=H.to_json(h), process_rank=p['process_rank'], attempts=p['attempts'],
                    redraw_fraction=cc['redraw_fraction'][sim], shared_fraction=shared_fraction(ht, h),
                    emission_overlap_with_target=emission_overlap(ht, h), entropy_rate=hr, entropy_rate_se=se,
                    quick_entropy_rate=p['quick_entropy_rate']))
    acc_log['pools'] = pools_log
    return tasks, acc_log


def pool_of(tasks, target_id, sim):
    return [t for t in tasks if t['role'] == 'distractor' and t['target_id'] == target_id and t['similarity'] == sim]


def prior_for(alpha, D):
    return [alpha] + [(1 - alpha) / D] * D if alpha < 1 else [1.0]


def eval_docs(cfg, h, cid, n_docs):
    return H.sample_docs(h, cfg['eval']['eval_seed'], cid, range(n_docs), cfg['concepts']['doc_len'])


def build_references(cfg, tasks, out):
    cm = {t['cid']: H.from_json(t['definition']) for t in tasks}
    (out / 'data').mkdir(parents=True, exist_ok=True)
    oracle, mixture, post = {}, {}, {}
    sims = cfg['concepts']['similarity_levels']
    for t in tasks:
        nd = cfg['eval']['distractor_eval_docs'] if t['role'] == 'distractor' else cfg['eval']['eval_docs_per_target']
        docs = eval_docs(cfg, cm[t['cid']], t['cid'], nd)
        with open(out / 'data' / f"{t['task_id']}_test.npz", 'wb') as f:
            np.savez(f, docs=docs.astype(np.int16))
        oc = H.forward_logprobs(cm[t['cid']], docs)
        oracle[t['task_id']] = oc
        if t['role'] == 'distractor':
            continue
        for sim in sims:
            pool = pool_of(tasks, t['task_id'], sim)
            if not pool:
                continue
            codes = [oc] + [H.forward_logprobs(cm[p['cid']], docs) for p in pool]
            for a in cfg['grid']['alphas']:
                if a >= 1:
                    continue
                pr = prior_for(a, len(pool))
                mixture[f"{t['task_id']}|{sim}|{a:.2f}"] = H.mixture_from_codes(codes, pr)
                post[f"{t['task_id']}|{sim}|{a:.2f}"] = float(H.posterior_after(codes, pr, 16).mean())
    with open(out / 'oracle.npz', 'wb') as f:
        np.savez(f, **oracle)
    with open(out / 'mixture.npz', 'wb') as f:
        np.savez(f, **mixture)
    R13.atomic_write_text(out / 'posterior16.json', json.dumps(post, indent=1, sort_keys=True))


def prep_hashes(prep):
    prep = Path(prep)
    files = PREP_FILES + sorted(f'data/{p.name}' for p in (prep / 'data').glob('*.npz'))
    return {f: sha256_file(prep / f) for f in files}


def prepare(cfg, out, log=print):
    out.mkdir(parents=True, exist_ok=True)
    tasks, acc_log = build_all(cfg, log)
    R13.atomic_write_text(out / 'tasks.json', json.dumps(tasks, indent=1))
    R13.atomic_write_text(out / 'acceptance_log.json', json.dumps(acc_log, indent=1))
    build_references(cfg, tasks, out)
    return tasks, acc_log


def build_prep(cfg, prep, log=print):
    """Build into a fresh folder, then install atomically; an existing prep must have identical bytes."""
    prep = Path(prep)
    tmp = prep.parent / f'{prep.name}.tmp-{os.getpid()}'
    shutil.rmtree(tmp, ignore_errors=True)
    tasks, acc_log = prepare(cfg, tmp, log)
    if prep.exists():
        a, b = prep_hashes(prep), prep_hashes(tmp)
        if a != b:
            raise SystemExit(f'rebuilt references differ from {prep}: '
                             f'{sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:5]}')
        shutil.rmtree(tmp)
    else:
        os.replace(tmp, prep)
    return tasks, acc_log


def load_refs(prep, tasks):
    o = np.load(prep / 'oracle.npz')
    m = np.load(prep / 'mixture.npz')
    docs = {t['task_id']: np.load(prep / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64) for t in tasks}
    return dict(oracle={k: o[k] for k in o.files}, mixture={k: m[k] for k in m.files}, docs=docs)
