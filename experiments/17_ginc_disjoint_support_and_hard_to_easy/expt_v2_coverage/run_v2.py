"""Experiment 17b runner (CPU): coverage-matched easy pools for Experiment 17's arm H (PROTOCOL_exp17b.md).

Experiment 17's easy pool (8 n=4 HMMs, 4 emission tokens per state) emits 11-15 of the 32 tokens per concept and
never emits token 18; the hard pool (n=32) emits 31-32 per concept. Two new n=4 pools separate state count from
token coverage:
  cov32 (primary)    8 n=4 HMMs with 8 emission tokens per state, the 4 states a disjoint partition of all 32 tokens:
                     every concept covers all 32 tokens, as the hard pool's concepts do.
  cov16 (secondary)  8 n=4 HMMs in Experiment 17's exact family (4 emission tokens per state), 16 distinct tokens per
                     concept (disjoint across states), in complementary pairs: every token in exactly 4 of 8 concepts.
Both pass Experiment 17's checks (strongly connected and aperiodic, process rank = n) and its entropy band around
the same h_ref. Targets, seeds, model, optimizer, alpha schedule and checkpoints are Experiment 17's; the hard, easy,
near and alpha = 1 runs are reused from expt_v1/results (concept ids 0-37 unchanged, new concepts get 38+).

Usage (venv python, from expt_v2_coverage/):
  run_v2.py --check-only                 new concepts + references into prep/, checks, results_check_v2.json
  run_v2.py --output results [--max-seconds S] [--workers W]   the 30 gated runs, resume-safe
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'expt_v1'
sys.path.insert(0, str(V1))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import hmm as H  # noqa: E402
import run as R  # noqa: E402

SOURCES_V2 = ['config_v2.json', 'run_v2.py']
ANALYSIS_SOURCES_V2 = ['analyze_v2.py', 'report_v2.py', 'verify_v2.py']
LOCK_DIR = HERE.parents[3]  # scratch/at343 (repo/experiments/17_.../expt_v2_coverage -> at343)


def load_c2():
    return json.loads((HERE / 'config_v2.json').read_text())


def k_bits_general(n, W, q, e):
    """Experiment 17's description length with e emission tokens per state: per state 3 successors, e tokens,
    2 + (e - 1) free probabilities of q bits. e = 4 gives H.K_bits."""
    return n * (3 * H.clog2(n) + e * H.clog2(W) + (2 + e - 1) * q)


# ---------------------------------------------------------------- concepts (PROTOCOL_exp17b Section 2)

def sample_with_tokens(rng, n, V, q, tok_by_state, n_succ=3):
    succ, trans, etok, emit = [], [], [], []
    for i in range(n):
        succ.append(np.sort(rng.choice(n, n_succ, replace=False)))
        etok.append(np.sort(np.asarray(tok_by_state[i], np.int64)))
        trans.append(H.quantize(rng.dirichlet(np.ones(n_succ)), q))
        emit.append(H.quantize(rng.dirichlet(np.ones(len(tok_by_state[i]))), q))
    return dict(n=int(n), V=int(V), q=int(q), succ=np.array(succ, np.int64), trans=np.array(trans, np.int64),
                etok=np.array(etok, np.int64), emit=np.array(emit, np.int64))


def candidate(c, c2, pool, slot, i):
    """Candidate i for a pool slot, from its own PCG64 stream (task_sampling_seed, 1717, pool code, slot, i)."""
    pc = c2['pools'][pool]
    n, W, q, lo = pc['n'], c['region_width'], c['q'], c['region_A']
    rng = np.random.default_rng([c['task_sampling_seed'], 1717, pc['code'], slot, i])
    if pool == 'cov32':
        perm = rng.permutation(W)
        e = pc['emit_per_state']
        tok = [perm[e * s:e * (s + 1)] + lo for s in range(n)]
    else:  # cov16: the pair's fixed token half, randomly partitioned over the n states
        half = np.random.default_rng([c['task_sampling_seed'], 1717, pc['code'], slot // 2, 0, 7]).permutation(W)
        mine = half[:W // 2] if slot % 2 == 0 else half[W // 2:]
        mine = rng.permutation(mine)
        e = pc['emit_per_state']
        tok = [mine[e * s:e * (s + 1)] + lo for s in range(n)]
    return sample_with_tokens(rng, n, c['V'], q, tok)


def exact_entropy_rate_disjoint(h):
    """Exact entropy rate (bits/token) of an HMM whose states emit disjoint token sets: each token reveals its state,
    so the rate is sum_i pi_i (H(A_i) + H(O_i)). Replaces Experiment 17's simulated quick estimate (same rule, no
    sampling noise; the simulated 100-sequence estimate is still recorded)."""
    A, O = H.matrices(h)
    pi = H.stationary(h)

    def ent(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    return float(sum(pi[k] * (ent(A[k]) + ent(O[k])) for k in range(h['n'])))


def build_cov_concepts(c, c2, v1_tasks, log=print):
    """Per pool and slot, candidates in generation order; accept the first that passes checks 1-2 and lies within the
    band of Experiment 17's h_ref (bands tried in order, one band per pool for all its slots). The band test
    uses the exact entropy rate (exact_entropy_rate_disjoint)."""
    acc_ref = v1_tasks[0]['acceptance']
    h_ref = acc_ref['h_ref']
    tasks, acc_log = [], {}
    cid = max(t['cid'] for t in v1_tasks) + 1
    for pool in c2['pool_order']:
        pc = c2['pools'][pool]
        cache = {}

        def get(slot, i):
            if (slot, i) not in cache:
                h = candidate(c, c2, pool, slot, i)
                ok, why, r = R._check12(h)
                qh = None
                if ok:
                    qh = exact_entropy_rate_disjoint(h)
                cache[(slot, i)] = (h, ok, r, qh)
            return cache[(slot, i)]
        chosen, band = None, None
        for b in c['entropy_bands']:
            pick = []
            for slot in range(c['pool_size']):
                got = None
                for i in range(c['max_candidates_per_level']):
                    h, ok, r, qh = get(slot, i)
                    if ok and abs(qh - h_ref) <= b:
                        got = i
                        break
                pick.append(got)
            if all(p is not None for p in pick):
                chosen, band = pick, b
                break
            log(f'{pool} band {b}: {sum(p is not None for p in pick)}/{c["pool_size"]} slots filled')
        if chosen is None:
            raise RuntimeError(f'{pool}: no entropy band fills every slot within the candidate cap')
        for slot, i in enumerate(chosen):
            h, ok, r, qh = get(slot, i)
            rr = np.random.default_rng([c['task_sampling_seed'], 1717, pc['code'], slot, i, 2])
            hr, se = H.entropy_rate(h, rr, c['entropy_seqs'], c['entropy_seq_len'], c['entropy_burn_in'])
            pi = H.stationary(h)
            tasks.append(dict(
                task_id=f'hmm_n{pc["n"]:03d}_{pool}_{slot + 1}', cid=cid, role=pool, K=pc['n'],
                K_bits=k_bits_general(pc['n'], c['region_width'], c['q'], pc['emit_per_state']),
                definition=H.to_json(h), process_rank=int(r), candidate_index=i,
                region=[int(h['etok'].min()), int(h['etok'].max())], entropy_rate=hr, entropy_rate_se=se,
                quick_entropy_rate=qh, min_stationary=float(pi.min()),
                acceptance=dict(band_bits=band, h_ref=h_ref)))
            cid += 1
        acc_log[pool] = dict(band_bits=band, h_ref=h_ref, accepted_indices=chosen, checked=len(cache),
                             passed_checks_1_2=sum(v[1] for v in cache.values()))
        log(f'{pool}: band {band}, accepted candidate indices {chosen}, {len(cache)} checked')
    return tasks, acc_log


def token_marginal(h):
    """Stationary probability of each of the V tokens."""
    A, O = H.matrices(h)
    return H.stationary(h) @ O


def coverage(tasks, W=32):
    """Token coverage of every target and pool (not gated; PROTOCOL_exp17b Section 5 'reported')."""
    out = {}
    groups = {}
    for t in tasks:
        if t['role'] in ('far', 'pilot'):
            continue
        key = t['role'] if t['role'] != 'target' else f"target_n{t['K']}"
        groups.setdefault(key, []).append(t)
    for key, ts in sorted(groups.items()):
        distinct = [len(set(np.array(t['definition']['etok']).ravel().tolist())) for t in ts]
        margs = np.array([token_marginal(H.from_json(t['definition']))[:W] for t in ts])
        avg = margs.mean(0)
        nz = avg[avg > 0]
        out[key] = dict(concepts=len(ts), distinct_tokens_per_concept=[min(distinct), max(distinct)],
                        union_tokens=int((avg > 0).sum()), missing_tokens=[int(x) for x in np.where(avg == 0)[0]],
                        min_token_share_of_pool=float(avg.min()),
                        token_perplexity_of_pool=float(2 ** (-(nz * np.log2(nz)).sum())),
                        mean_token_perplexity_per_concept=float(np.mean(
                            [2 ** (-(m[m > 0] * np.log2(m[m > 0])).sum()) for m in margs])))
    return out


# ---------------------------------------------------------------- references and grid

def ref_keys_v2(c, c2, tasks):
    return sorted((t['task_id'], pool, a) for t in R.by_role(tasks, 'target')
                  for pool in c2['pool_order'] for a in c['alpha_H'])


def build_references_v2(c, c2, v1_prep, tasks, new, out):
    """Copy Experiment 17's references and add the new pool concepts' held-out documents, their oracle codes, and
    the Bayes-mixture references of every (target, new pool, alpha_H)."""
    R.copy_prep(v1_prep, out)
    atomic = R.atomic_write_text
    atomic(out / 'tasks.json', json.dumps(tasks, indent=1))
    cm = R.concept_map(tasks)
    oracle = dict(np.load(out / 'oracle.npz'))
    mixture = dict(np.load(out / 'mixture.npz'))
    post = json.loads((out / 'posterior16.json').read_text())
    for t in new:
        docs = R.eval_docs(c, cm[t['cid']], t['cid'], c['pool_eval_docs'])
        with open(out / 'data' / f"{t['task_id']}_test.npz", 'wb') as f:
            np.savez(f, docs=docs.astype(np.int16))
        oracle[t['task_id']] = H.forward_logprobs(cm[t['cid']], docs)
    for tid, pool, a in ref_keys_v2(c, c2, tasks):
        docs = np.load(out / 'data' / f'{tid}_test.npz')['docs'].astype(np.int64)
        codes = [oracle[tid]] + [H.forward_logprobs(cm[d['cid']], docs) for d in R.by_role(tasks, pool)]
        pr = R.prior_for(a, len(codes) - 1)
        mixture[R.mix_key(tid, pool, a)] = H.mixture_from_codes(codes, pr)
        post[R.mix_key(tid, pool, a)] = float(H.posterior_after(codes, pr, 16).mean())
    with open(out / 'oracle.npz', 'wb') as f:
        np.savez(f, **oracle)
    with open(out / 'mixture.npz', 'wb') as f:
        np.savez(f, **mixture)
    atomic(out / 'posterior16.json', json.dumps(post, indent=1, sort_keys=True))


def grid_runs_v2(c, c2, tasks, seeds=None):
    seeds = c['seeds'] if seeds is None else seeds
    runs = []
    for s in seeds:
        for pool in c2['pool_order']:
            runs.append((s, pool, '-', 0.0))
            for t in R.by_role(tasks, 'target'):
                for a in c['alpha_H']:
                    runs.append((s, pool, t['task_id'], a))
    return runs


def check_only(c, c2, prep, log):
    v1_tasks = json.loads((V1 / 'results' / 'tasks.json').read_text())
    assert v1_tasks == json.loads((V1 / 'prep' / 'tasks.json').read_text())
    new, acc_log = build_cov_concepts(c, c2, v1_tasks, log)
    tasks = v1_tasks + new
    prep.mkdir(parents=True, exist_ok=True)
    build_references_v2(c, c2, V1 / 'prep', tasks, new, prep)
    R.atomic_write_text(prep / 'acceptance_log_v2.json', json.dumps(acc_log, indent=1))
    rec = dict(acceptance=acc_log, concepts=[], checks={})
    for t in new:
        h = H.from_json(t['definition'])
        toks = h['etok'].ravel()
        rec['concepts'].append(dict(task_id=t['task_id'], cid=t['cid'], K=t['K'], K_bits=t['K_bits'],
                                    process_rank=t['process_rank'], distinct_tokens=int(len(set(toks.tolist()))),
                                    entropy_rate=round(t['entropy_rate'], 4)))
        log(f"{t['task_id']:18s} cid={t['cid']} n={t['K']} K_bits={t['K_bits']} rank={t['process_rank']} "
            f"tokens {len(set(toks.tolist()))} h={t['entropy_rate']:.4f}+-{t['entropy_rate_se']:.4f}")
        assert t['process_rank'] == t['K'] and len(toks) == len(set(toks.tolist()))  # disjoint across states
        assert toks.min() >= c['region_A'] and toks.max() < c['region_A'] + c['region_width']
    cov32 = R.by_role(tasks, 'cov32')
    assert all(len(set(np.array(t['definition']['etok']).ravel().tolist())) == 32 for t in cov32)
    cov16 = R.by_role(tasks, 'cov16')
    cnt = np.zeros(32, int)
    for t in cov16:
        cnt[np.array(t['definition']['etok']).ravel()] += 1
    assert (cnt == len(cov16) // 2).all(), cnt
    rec['checks']['cov32_every_concept_all_32_tokens'] = True
    rec['checks']['cov16_every_token_in_exactly_4_concepts'] = True
    rec['checks']['v1_concepts_unchanged'] = tasks[:len(v1_tasks)] == v1_tasks
    # schedules exact and closed; the new pools' streams use only their own concepts
    nd = c['T_checkpoints'][-1] // c['doc_len']
    for s, pool, tid, a in grid_runs_v2(c, c2, tasks, seeds=c['seeds'][:1]):
        cids, js = R.stream_ids(tasks, s, pool, tid, a, nd)
        allowed = {t['cid'] for t in R.by_role(tasks, pool)}
        if tid != '-':
            tcid = next(t['cid'] for t in tasks if t['task_id'] == tid)
            assert np.array_equal(np.cumsum(cids == tcid), (np.arange(1, nd + 1) * int(round(a * 100))) // 100)
            allowed.add(tcid)
        assert set(np.unique(cids).tolist()) <= allowed
    rec['checks']['alpha_schedule_exact_and_pools_closed'] = True
    # eval documents of every concept never appear in the new pools' training streams (first 2^14 docs)
    eval_h = set()
    for t in tasks:
        for x in np.load(prep / 'data' / f"{t['task_id']}_test.npz")['docs']:
            eval_h.add(H.doc_hash(x))
    hits = 0
    for t in new:
        h = H.from_json(t['definition'])
        for s in c['seeds']:
            hits += sum(H.doc_hash(x) in eval_h for x in H.sample_docs(h, s, t['cid'], range(2 ** 14 // 8),
                                                                        c['doc_len']))
    assert hits == 0
    rec['checks']['eval_train_disjoint'] = dict(hits=hits)
    rec['checks']['expected_runs'] = len(grid_runs_v2(c, c2, tasks))
    rec['coverage'] = coverage(tasks)
    mx, o = np.load(prep / 'mixture.npz'), np.load(prep / 'oracle.npz')
    rec['eps_id_all_new'] = {k: float((mx[k] - o[k.split('|')[0]]).mean()) for k in mx.files
                             if k.split('|')[1] in c2['pool_order']}
    # smoke: one new-pool run trains and evaluates (2^13 tokens, not a gated result)
    refs = R.load_refs(prep, tasks)
    t0 = time.time()
    sm = R.train_run(c, tasks, (c['seeds'][0], 'cov32', '-', 0.0), prep, [2 ** 13], refs)
    rec['checks']['smoke_cov32_2^13'] = dict(finite=sm['finite'], seconds=round(time.time() - t0, 1),
                                             excess_all=[round(r['excess_bits'], 3) for r in sm['rows']
                                                         if r['slice'] == 'all'])
    rec['source_sha256'] = {s: R.sha256_file(HERE / s) for s in SOURCES_V2} | \
                           {f'expt_v1/{s}': R.sha256_file(V1 / s) for s in R.SOURCES}
    R.atomic_write_text(HERE / 'results_check_v2.json', json.dumps(rec, indent=1))
    log(json.dumps(rec['checks']))
    log(json.dumps(rec['coverage'], indent=1))
    return rec


def acquire_lock(task):
    others = [p.name for p in LOCK_DIR.glob('CPU_LOCK_*') if p.name != f'CPU_LOCK_{task}']
    if others:
        raise SystemExit(f'CPU lock held: {others}')
    (LOCK_DIR / f'CPU_LOCK_{task}').write_text(f'{task} Exp 17b coverage-matched grid (30 runs), started '
                                               f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')


def release_lock(task):
    p = LOCK_DIR / f'CPU_LOCK_{task}'
    if p.exists():
        p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--output', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--max-seconds', type=float, default=float('inf'))
    ap.add_argument('--lock-task', default='AT-453')
    args = ap.parse_args()
    c, c2 = R.load_config(), load_c2()
    if args.workers:
        c['workers'] = args.workers
    prep = HERE / 'prep'
    if args.check_only:
        check_only(c, c2, prep, R.log_to(HERE / 'check_v2.log'))
        return
    out = Path(args.output) if args.output else HERE / 'results'
    if not (out / 'tasks.json').exists():
        out.mkdir(parents=True, exist_ok=True)
        R.copy_prep(prep, out)
        shutil.copy2(prep / 'acceptance_log_v2.json', out / 'acceptance_log_v2.json')
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = grid_runs_v2(c, c2, tasks)
    log = R.log_to(HERE / 'grid_v2.log')
    acquire_lock(args.lock_task)
    try:
        missing, state = R.run_many(c, out, runs, c['T_checkpoints'], args.max_seconds, log)
    finally:
        release_lock(args.lock_task)
    recs = R.aggregate(out, runs)
    R.write_manifest(out, dict(grid=dict(runs=len(runs), finished=len(recs), missing=missing,
                                         diverged=[r['run'] for r in recs if not r['finite']],
                                         wall_seconds=state['elapsed_seconds'], chunks=state['chunks'])))
    man = json.loads((out / 'manifest.json').read_text())
    man['sources_v2'] = {s: R.sha256_file(HERE / s) for s in SOURCES_V2 + ANALYSIS_SOURCES_V2 if (HERE / s).exists()}
    man['v1_metrics_sha256'] = R.sha256_file(V1 / 'results' / 'metrics.csv')
    man['results_check_v2'] = json.loads((HERE / 'results_check_v2.json').read_text())
    R.atomic_write_text(out / 'manifest.json', json.dumps(man, indent=1))
    log(f'{len(recs)}/{len(runs)} runs finished; {len(missing)} missing; cumulative {state["elapsed_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
