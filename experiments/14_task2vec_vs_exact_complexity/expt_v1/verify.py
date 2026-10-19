"""Independent checks of a finished run (PROTOCOL Section 8). Prints PASS or the first failure.

  python verify.py [--results results] [--budget-seconds 3300]

1. Reloads every probe and recomputes its held-out pretraining losses (max abs diff < 1e-9 bits).
2. Recomputes the exact K of every task with the imported Experiment 11/12 code and re-checks role disjointness
   and inputs (tasks14.check_roster, check_inputs), and that tasks14.json equals a fresh build.
3. Recomputes a seeded 10% sample (verify_sample_seed) of all task and batch embeddings, plus every embedding behind
   a gate decision for the first probe seed (main E0 of every embedded task, 25%-noise E0 of the noise tasks,
   interleaved batches of every coverage mix), with the same seeds and 1 thread; max relative coordinate difference
   <= verify_rel_tol. A task is recomputed with the estimator list of its original job, and every returned
   estimator is compared. Recomputed parts go to results/verify/ so a stopped run resumes with the same command.
4. Recomputes every gate, bootstrap draw and reported statistic from the saved embeddings with independent_stats.py
   (analyze.py run with that backend) and compares with gates.json, analysis.json, bootstrap.npz, task_table.csv
   and diversity.csv to stats_tol.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import concurrent.futures as cf
import csv
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import analyze as AN
import independent_stats as I
import probe as P
import run as R
import tasks14 as T

HERE = Path(__file__).resolve().parent
GROUPS_PER_JOB = 4


class Fail(Exception):
    pass


def check(ok, msg):
    if not ok:
        raise Fail(msg)


# ---------------------------------------------------------------- 1-2 probes, K, roster

def check_probes(c, out, roster, log):
    worst = 0.0
    for fam in T.FAMILIES:
        for s in c['probe_seeds'] + [c['random_init_seed']]:
            rec = json.loads((out / 'probes' / f'probe_{fam}_{s}.json').read_text())
            m = P.load_probe(c, fam, P.probe_path(out, fam, s))
            m.eval()
            ev = P.evaluate_probe(c, roster, fam, m)
            check(len(ev) == len(rec['evaluation']), f'probe {fam} {s}: evaluation length differs')
            for a, b in zip(ev, rec['evaluation']):
                for k in ('task', 'null'):
                    d = abs(a[k]['bits'] - b[k]['bits'])
                    worst = max(worst, d)
                    check(d < 1e-9, f'probe {fam} {s} task {a["task_id"]} {k}: held-out bits differ by {d:.3g}')
    log(f'1. probes reloaded and re-evaluated: max abs diff {worst:.3g} bits')
    return worst


def check_tasks(c, out, roster, log):
    ok, rep = T.check_roster(c, roster)
    check(ok, f'roster check failed: {json.dumps(rep)[:400]}')
    ok, rep2 = T.check_inputs(c, roster)
    check(ok, f'input check failed: {json.dumps(rep2)[:400]}')
    fresh = T.build_roster(c)[0]
    check(json.dumps(fresh, sort_keys=True) == json.dumps(roster, sort_keys=True),
          'tasks14.json differs from a fresh build of the roster')
    log('2. exact K recomputed for every task, roles disjoint, inputs checked, roster rebuilds identically')
    return dict(roster=rep, inputs=rep2)


# ---------------------------------------------------------------- 3 embedding recomputation

def plan(c, out, roster):
    """Recomputation jobs: dicts with kind 'vembed' (groups of (fam, seed, variant, tid, rate, ests)) or 'vdiv'
    (fam, seed, mix_id, kind, batch indices)."""
    probes, emb, div = R.all_jobs(c, roster)
    rng = np.random.default_rng(c['verify_sample_seed'])
    # task embeddings: key -> group
    group_of, ests_of = {}, {}
    for j in emb:
        for tid, rate in j['items']:
            g = (j['fam'], j['seed'], j['variant'], tid, rate)
            ests_of[g] = tuple(j['ests'])
            for e in j['ests']:
                group_of[R.emb_key(j['fam'], tid, j['seed'], e, R.variant_name(j['variant'], rate))] = g
    keys = sorted(group_of)
    pick = rng.choice(len(keys), math.ceil(c['verify_sample_fraction'] * len(keys)), replace=False)
    sampled = {group_of[keys[i]] for i in pick}
    p0 = c['probe_seeds'][0]
    hi = max(c['noise_rates'])
    gate = set()
    for fam in T.FAMILIES:
        gate |= {(fam, p0, 'main', t['task_id'], None) for t in T.embedded_tasks(roster, fam)}
        gate |= {(fam, p0, 'noise', t['task_id'], hi) for t in T.noise_tasks(c, roster, fam)}
    groups = sorted(sampled | gate, key=lambda g: (g[0], g[1], g[2], g[3], -1 if g[4] is None else g[4]))
    # batch embeddings
    bkeys = []
    for j in div:
        for mix_id, kind in j['items']:
            bkeys += [R.div_key(j['fam'], j['seed'], mix_id, kind, i) for i in range(j['n_batches'])]
    bkeys = sorted(bkeys)
    bpick = {bkeys[i] for i in rng.choice(len(bkeys), math.ceil(c['verify_sample_fraction'] * len(bkeys)),
                                         replace=False)}
    for fam in T.FAMILIES:
        for m in T.coverage_mixes(c, roster, fam):
            bpick |= {R.div_key(fam, p0, m['mix_id'], 'interleaved', i) for i in range(c['diversity_batches'])}
    bgroups = {}
    for k in sorted(bpick):
        fam, seed, mix_id, kind, i = k.split('|')
        bgroups.setdefault((fam, int(seed), mix_id, kind), []).append(int(i))
    jobs = []
    for n, st in enumerate(range(0, len(groups), GROUPS_PER_JOB)):
        jobs.append(dict(kind='vembed', id=f'vemb_{n:04d}',
                         groups=[(*g, ests_of[g]) for g in groups[st:st + GROUPS_PER_JOB]]))
    for n, (g, idx) in enumerate(sorted(bgroups.items())):
        jobs.append(dict(kind='vdiv', id=f'vdiv_{n:04d}', fam=g[0], seed=g[1], mix_id=g[2], mix_kind=g[3],
                         batches=sorted(idx), n_batches=c['diversity_batches']))
    stats = dict(task_keys=len(keys), task_keys_sampled=len(pick), task_groups=len(groups),
                 task_groups_gate_only=len(gate - sampled), batch_keys=len(bkeys),
                 batch_keys_checked=len(bpick), jobs=len(jobs))
    return jobs, stats


def vpath(out, job):
    return out / 'verify' / f"{job['id']}.npz"


def verify_job(job):
    try:
        t0 = time.time()
        c, roster = R._W['c'], R._W['roster']
        keys, F_, hl = [], [], []
        if job['kind'] == 'vembed':
            for fam, seed, variant, tid, rate, ests in job['groups']:
                res = R.embed_one(c, roster, fam, seed, variant, tid, rate, ests)
                for e in ests:
                    keys.append(R.emb_key(fam, tid, seed, e, R.variant_name(variant, rate)))
                    F_.append(res[e]['f'])
                    hl.append(res[e]['head_loss'])
        else:
            fam, seed = job['fam'], job['seed']
            spec = R._spec(fam, seed, 'main')
            mix = R.mix_lookup(c, roster, fam)[job['mix_id']]
            want = set(job['batches'])
            for i, batch in enumerate(R.diversity_batches(c, roster, fam, mix, job['mix_kind'], job['n_batches'])):
                if i in want:
                    r = R.batch_embedding(c, spec, fam, seed, job['mix_id'], job['mix_kind'], i, batch)
                    keys.append(R.div_key(fam, seed, job['mix_id'], job['mix_kind'], i))
                    F_.append(r['f'])
                    hl.append(r['head_loss'])
        R._atomic_npz(vpath(R._W['out'], job), keys=np.array(keys), f=np.array(F_, np.float32),
                      head_loss=np.array(hl))
        return dict(id=job['id'], wall=time.time() - t0)
    except Exception:
        return dict(id=job['id'], error=traceback.format_exc())


def run_recompute(c, out, jobs, deadline, log):
    todo = [j for j in jobs if not vpath(out, j).exists()]
    log(f'3. {len(jobs) - len(todo)} of {len(jobs)} recomputation jobs already done; {len(todo)} to run')
    if not todo:
        return 0
    ctx = mp.get_context('spawn')
    failed = 0
    with cf.ProcessPoolExecutor(c['workers'], mp_context=ctx, initializer=R._init_worker,
                                initargs=(str(out),)) as ex:
        it, running, stopped = iter(todo), {}, False
        while True:
            while not stopped and len(running) < c['workers']:
                j = next(it, None) if time.time() < deadline else None
                if j is None:
                    stopped = True
                    break
                running[ex.submit(verify_job, j)] = j
            if not running:
                break
            fin, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                running.pop(f)
                r = f.result()
                if 'error' in r:
                    failed += 1
                    log(f"FAILED {r['id']}:\n{r['error']}")
                else:
                    log(f"done {r['id']} {r['wall']:.1f}s")
    check(failed == 0, f'{failed} recomputation jobs raised errors')
    return sum(1 for j in jobs if not vpath(out, j).exists())


def rel_diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), np.abs(b))
    d = np.abs(a - b)
    return float(np.max(np.where(den > 0, d / np.where(den > 0, den, 1.0), 0.0)))


def compare_recomputed(c, out, jobs, log):
    z = np.load(out / 'embeddings.npz', allow_pickle=False)
    saved = {k: i for i, k in enumerate(z['keys'].tolist())}
    sf, shl = z['f'], z['head_loss']
    bsaved = {}
    for fam in T.FAMILIES:
        for s in c['probe_seeds']:
            b = np.load(out / 'batch_embeddings' / f'{fam}_{s}.npz', allow_pickle=False)
            for k, f, h in zip(b['keys'].tolist(), b['f'], b['head_loss']):
                bsaved[k] = (f, h)
    worst, worst_key, n = 0.0, None, 0
    for j in jobs:
        r = np.load(vpath(out, j), allow_pickle=False)
        for k, f, h in zip(r['keys'].tolist(), r['f'], r['head_loss']):
            f0, h0 = (sf[saved[k]], shl[saved[k]]) if j['kind'] == 'vembed' else bsaved[k]
            d = max(rel_diff(f, f0), rel_diff([h], [h0]))
            n += 1
            if d > worst:
                worst, worst_key = d, k
            check(d <= c['verify_rel_tol'], f'embedding {k}: max relative coordinate difference {d:.3g}')
    log(f'3. {n} embeddings recomputed: max relative difference {worst:.3g} ({worst_key})')
    return n, worst


# ---------------------------------------------------------------- 4 statistics with independent_stats

class IndependentBackend:
    """The stats.py API used by analyze.py, implemented with independent_stats.py."""

    spearman = staticmethod(lambda x, y: I.spearman(x, y))
    partial_spearman = staticmethod(lambda x, y, Z: I.partial_spearman(x, y, Z))
    kendall_w = staticmethod(lambda M: I.kendall_w(M))
    ci = staticmethod(lambda d: I.percentile_ci(d))

    @staticmethod
    def z_constants(logn):
        _, mu, sd = I._zscores(np.asarray(logn, np.float64))
        return mu, sd

    @staticmethod
    def khat(logn, mu, sd):
        return ((logn - mu[:, None]) / sd[:, None]).sum(0) / logn.shape[0]

    @staticmethod
    def tstar_table(rows, target_ids, seeds, eps_star, alpha=1.0):
        curves = {}
        for r in rows:
            if abs(float(r['alpha']) - alpha) <= 1e-9:
                curves.setdefault((r['target_id'], int(r['seed'])), []).append((int(r['T']), float(r['excess_bits'])))
        out = np.zeros((len(target_ids), len(seeds)))
        for i, t in enumerate(target_ids):
            for j, s in enumerate(seeds):
                cv = sorted(curves[(t, int(s))])
                out[i, j] = I.t_star([a for a, _ in cv], [b for _, b in cv], eps_star)
        return out

    @staticmethod
    def gate12(c, fi, logn, K, cov, mu, sd):
        return I.gate12(c, fi, logn, K, cov)

    @staticmethod
    def gate3(c, fi, logn_t, K_t, mu, sd, ts):
        return I.gate3_z(c, fi, (logn_t - mu[:, None]) / sd[:, None], K_t, ts)

    @staticmethod
    def gate4(c, fi, dist, H, K, cell, rep):
        return I.gate4(c, fi, dist, H, K, cell, rep)

    @staticmethod
    def gate5(c, fi, logratio, K_n):
        return I.gate5(c, fi, logratio, K_n)


def compare_tree(a, b, tol, path, counter):
    if isinstance(b, dict):
        check(isinstance(a, dict) and set(a) == set(b), f'{path}: keys differ')
        for k in b:
            compare_tree(a[k], b[k], tol, f'{path}.{k}', counter)
    elif isinstance(b, (list, tuple)):
        check(len(a) == len(b), f'{path}: lengths differ')
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, tol, f'{path}[{i}]', counter)
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        check(a == b, f'{path}: {a!r} != {b!r}')
    else:
        x, y = float(a), float(b)
        if math.isinf(y) or math.isnan(y):
            check(x == y or (math.isnan(x) and math.isnan(y)), f'{path}: {x} != {y}')
        else:
            d = abs(x - y)
            counter['n'] += 1
            counter['max'] = max(counter['max'], d)
            check(d <= tol, f'{path}: stats.py {y!r} vs independent {x!r} (diff {d:.3g})')


def jsonable(o):
    return json.loads(json.dumps(o))


def check_stats(c, out, log):
    tol = c['stats_tol']
    AN.S = IndependentBackend
    st = AN.Store(c, out)
    gates = json.loads((out / 'gates.json').read_text())['families']
    analysis = json.loads((out / 'analysis.json').read_text())
    saved_boot = np.load(out / 'bootstrap.npz')
    counter = dict(n=0, max=0.0)
    boot, trows, drows = {}, [], []
    for fam in T.FAMILIES:
        g, A_, tr, dr, _ = AN.analyze_family(c, st, fam, boot)
        compare_tree(jsonable(g), gates[fam], tol, f'gates.{fam}', counter)
        compare_tree(jsonable(A_), analysis[fam], tol, f'analysis.{fam}', counter)
        trows += tr
        drows += dr
    check(set(boot) == set(saved_boot.files), 'bootstrap.npz arrays differ')
    for k in boot:
        a, b = np.nan_to_num(boot[k], nan=1e9), np.nan_to_num(saved_boot[k], nan=1e9)
        check(a.shape == b.shape, f'bootstrap {k}: shape differs')
        d = float(np.max(np.abs(a - b)))
        counter['n'] += a.size
        counter['max'] = max(counter['max'], d)
        check(d <= tol, f'bootstrap {k}: max diff {d:.3g}')
    for name, rows in (('task_table.csv', trows), ('diversity.csv', drows)):
        with open(out / name) as f:
            saved = list(csv.DictReader(f))
        check(len(saved) == len(rows), f'{name}: row count differs')
        for i, (r, s) in enumerate(zip(rows, saved)):
            for k, v in r.items():
                if isinstance(v, (float, np.floating)) and s[k] != '':
                    d = abs(float(v) - float(s[k])) if math.isfinite(float(v)) else (0.0 if float(s[k]) == v else 1.0)
                    counter['n'] += 1
                    check(d <= tol, f'{name} row {i} {k}: {s[k]} vs {v}')
                elif not isinstance(v, (float, np.floating)):
                    check(str(v) == s[k], f'{name} row {i} {k}: {s[k]!r} vs {v!r}')
    log(f'4. {counter["n"]} statistics recomputed with independent_stats.py: max abs diff {counter["max"]:.3g}')
    return counter


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--budget-seconds', type=float, default=3300)
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + a.budget_seconds
    torch.set_num_threads(1)
    c = T.load_config()
    out = HERE / a.results
    logf = open(out / 'verify.log', 'a')

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    report = {}
    try:
        roster = T.load_roster(out / 'tasks14.json')[0]
        jobs, report['plan'] = plan(c, out, roster)
        log(f"plan: {json.dumps(report['plan'])}")
        pending = run_recompute(c, out, jobs, deadline, log)
        if pending:
            log(f'INCOMPLETE: {pending} recomputation jobs left; rerun the same command to resume')
            sys.exit(2)
        report['probe_max_abs_diff_bits'] = check_probes(c, out, roster, log)
        check_tasks(c, out, roster, log)
        report['embeddings_checked'], report['embedding_max_rel_diff'] = compare_recomputed(c, out, jobs, log)
        report['stats'] = check_stats(c, out, log)
    except Fail as e:
        log(f'FAIL: {e}')
        print(f'FAIL: {e}')
        sys.exit(1)
    report['seconds_this_invocation'] = round(time.time() - t0, 1)
    R._atomic_json(out / 'verify.json', report)
    log('PASS')
    print('PASS')


if __name__ == '__main__':
    main()
