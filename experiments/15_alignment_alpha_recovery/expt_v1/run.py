"""Experiment 15 runner: references and scored units, every score, the graded-relevance arm and the Eq. 2 check.

Usage (from expt_v1/, venv python):
  run.py --check-only [--scratch DIR]       invariants, graded construction check, work counts; trains and scores nothing
  run.py --pilot                            disclosed pilot (seed 15909) into results_pilot/, with a wall-time projection
  run.py --output results [--stage refs|scores|graded|eq2|all] [--max-seconds S] [--resume]
                                            gated run; each invocation is bounded by --max-seconds (a chunk), the
                                            cumulative wall cap max_wall_seconds is tracked in run_state.json, and
                                            every finished job is a file, so a rerun with --resume continues
"""
import argparse
import concurrent.futures as cf
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import common as C  # noqa: E402
import graded as GR  # noqa: E402
import jobs as J  # noqa: E402
import mixes15 as MX  # noqa: E402

SOURCES = ['config.json', 'common.py', 'mixes15.py', 'scores.py', 'jobs.py', 'graded.py', 'run.py', 'laws.py',
           'independent_fit.py', 'analyze.py', 'verify.py', 'report.py', 'test_scores.py', 'test_graded.py',
           'test_laws.py', 'upstream_sha256.json', 'requirements.txt']


def now():
    return datetime.now().strftime('%m-%d-%Y %H:%M:%S')


class Log:
    def __init__(self, path=None):
        self.path = path

    def __call__(self, msg):
        line = f'[{now()}] {msg}'
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, 'a') as f:
                f.write(line + '\n')


# ---------------------------------------------------------------- references, units, graded construction

def ref_stats(c, fam, tasks, R, U):
    """Per target: exact-content overlap (pool examples whose input and label equal a reference's), reference
    coverage (pool inputs among reference inputs). Per family: input total-variation distance target vs distractors
    (DFA over lengths, DSL over input words) on the scored units."""
    rows = []
    lay_all = {t['task_id']: MX.pool_layout(tasks, t['task_id']) for t in tasks if t['role'] == 'target'}
    for tid, lay in lay_all.items():
        rk = set(C.content_keys(fam, R[tid]['raw']).tolist())
        rki = set(C.content_keys(fam, R[tid]['raw'], input_only=True).tolist())
        pool = C.concat([U[u]['raw'] for u in lay])
        pk = C.content_keys(fam, pool)
        pki = C.content_keys(fam, pool, input_only=True)
        rows.append(dict(family=fam, target=tid, exact_content_overlap=float(np.isin(pk, list(rk)).mean()),
                         reference_coverage=float(np.isin(pki, list(rki)).mean())))
    tv = []
    for t in tasks:
        if t['role'] != 'target':
            continue
        dists = [u for u in lay_all[t['task_id']][1:]]
        if fam == 'dfa':
            def hist(raw):
                return np.bincount(raw['lens'].astype(int), minlength=25) / C.n_examples(raw)
        else:
            def hist(raw):
                return np.bincount(raw['x'].astype(int), minlength=1024) / C.n_examples(raw)
        h0 = hist(U[t['task_id']]['raw'])
        hd = hist(C.concat([U[u]['raw'] for u in dists]))
        tv.append(dict(family=fam, target=t['task_id'], tv_inputs=0.5 * float(np.abs(h0 - hd).sum()),
                       over='length' if fam == 'dfa' else 'input word'))
    return rows, tv


def token_free(c, fam, raws):
    """True iff no stripped serialization contains a task-token byte (>= task_token_first_byte)."""
    lo = c['serialization']['task_token_first_byte']
    for raw in raws:
        for b in C.serialize(c, fam, raw):
            if max(b) >= lo:
                return False
        for b in C.serialize(c, fam, raw, input_only=True):
            if max(b) >= lo:
                return False
    return True


def graded_counts(c, G, seeds):
    """Mean T_max counts per (target, mix) over the graded seeds, and the largest seed-to-seed spread."""
    D, c11 = GR.D_(), GR.c11()
    Tm = str(c11['T_checkpoints'][-1])
    out, spread = {}, 0
    for key, m in G['mixes'].items():
        lst = [D.assemble_mix(c11, GR.mix_tasks(G, key), m['target_id'], m['alpha_share'], s)['counts'][Tm]
               for s in seeds]
        ids = sorted(set().union(*lst))
        arr = np.array([[d.get(i, 0) for i in ids] for d in lst], float)
        spread = max(spread, int((arr.max(0) - arr.min(0)).max()))
        out[key] = dict(zip(ids, arr.mean(0).tolist()))
    return out, spread


def graded_refs_units(c, G, U_main):
    """Graded references: 2,048 fresh target examples (seed 20261015, stream 3). Units: the target's main-arm units;
    each distractor's 4,096 from its graded pool (fixed permutation)."""
    R, U = {}, {}
    for t in G['tasks']:
        if t['role'] == 'target':
            t11 = dict(t, task_token=t['exp11_task_token'])
            raw = MX.fresh_references(c, 'dfa', t11, c['references_per_target'], stream=3)
            R[t['task_id']] = dict(raw=raw, file_index=np.full(c['references_per_target'], -1))
            U[t['task_id']] = U_main[t['task_id']]
        else:
            z = np.load(Path(G['_out']) / 'data' / f"{t['task_id']}_train.npz")
            f = dict(bits=z['bits'].astype(np.uint32), lens=z['length'].astype(np.uint8), y=z['label'].astype(np.int64))
            perm = np.random.default_rng([c['task_sampling_seed'], 15, 2, t['task_token']]).permutation(len(f['y']))
            idx = perm[: c['scored_units_per_task']]
            U[t['task_id']] = dict(raw=C.take(f, idx), file_index=idx)
    return R, U


def stage_refs(c, out, log, graded_seeds):
    out = Path(out)
    if (out / 'references.npz').exists() and (out / 'scored_units.npz').exists():
        log('refs: already built')
        return json.loads((out / 'refs_log.json').read_text())
    t0 = time.time()
    ref_all, uni_all, info = {}, {}, dict(mix_check={}, references=[], overlap=[], tv=[])
    U_main = {}
    for fam in C.FAMILIES:
        tasks, ok, used, n = MX.reconstruct(c, fam, workers=c['workers'])
        log(f'refs: {fam} {n} mixes rebuilt, counts equal mixes.json: {ok}')
        if not ok:
            raise RuntimeError(f'{fam}: a rebuilt mix differs from mixes.json')
        info['mix_check'][fam] = dict(n_mixes=n, counts_equal=bool(ok),
                                      t_max_count_spread=MX.tmax_counts(c, fam)[1])
        rlog = []
        tasks, R, U = MX.build_refs_units(c, fam, used, rlog)
        info['references'] += rlog
        ov, tv = ref_stats(c, fam, tasks, R, U)
        info['overlap'] += ov
        info['tv'] += tv
        if fam == 'dfa':
            U_main = U
        ref, uni = J.save_refs_units(out, fam, tasks, R, U)
        ref_all.update(ref)
        uni_all.update(uni)
        info.setdefault('token_free', {})[fam] = token_free(c, fam, [r['raw'] for r in R.values()] +
                                                            [u['raw'] for u in U.values()])
        info.setdefault('disjoint', {})[fam] = all(
            not np.intersect1d(U[t]['file_index'], R[t]['file_index'][R[t]['file_index'] >= 0]).size for t in R)
    # graded arm
    gdir = out / 'graded'
    gdir.mkdir(parents=True, exist_ok=True)
    G = GR.build(c, log=log)
    G['counts_mean'], G['counts_spread'] = graded_counts(c, G, graded_seeds)
    G['mix_names'] = sorted(c['graded']['mixes'])
    G['seeds'] = list(graded_seeds)
    G['alpha_share_soft_correlation'] = GR.check_alpha_soft_spread(G)
    GR.write_data(c, G, gdir)
    G['_out'] = str(gdir)
    R, U = graded_refs_units(c, G, U_main)
    del G['_out']
    C.atomic_write_text(gdir / 'graded_tasks_full.json', json.dumps(G, indent=1))
    GR.write_contract(c, G, gdir, graded_seeds)
    gtasks = sorted(G['tasks'], key=lambda t: t['task_token'])
    ref, uni = J.save_refs_units(out, 'graded', gtasks, R, U)
    ref_all.update(ref)
    uni_all.update(uni)
    info['token_free']['graded'] = token_free(c, 'dfa', [r['raw'] for r in R.values()] + [u['raw'] for u in U.values()])
    ov = []
    for t in gtasks:
        if t['role'] != 'target':
            continue
        lay = [t['task_id']] + [x['task_id'] for x in gtasks if x.get('of_target') == t['task_id']]
        rk = set(C.content_keys('dfa', R[t['task_id']]['raw']).tolist())
        rki = set(C.content_keys('dfa', R[t['task_id']]['raw'], input_only=True).tolist())
        pool = C.concat([U[u]['raw'] for u in lay])
        ov.append(dict(family='graded', target=t['task_id'],
                       exact_content_overlap=float(np.isin(C.content_keys('dfa', pool), list(rk)).mean()),
                       reference_coverage=float(np.isin(C.content_keys('dfa', pool, True), list(rki)).mean())))
    info['overlap'] += ov
    info['graded'] = dict(construction=G['construction'], n_rejections=len(G['rejections']),
                          alpha_share_soft_correlation=G['alpha_share_soft_correlation'],
                          counts_spread=G['counts_spread'])
    C.atomic_savez(out / 'references.npz', **ref_all)
    C.atomic_savez(out / 'scored_units.npz', **uni_all)
    C.atomic_write_text(out / 'serialization_table.json', json.dumps(c['serialization'], indent=1))
    info['seconds'] = time.time() - t0
    C.atomic_write_text(out / 'refs_log.json', json.dumps(info, indent=1))
    log(f'refs: done in {info["seconds"]:.0f} s; token-free {info["token_free"]}; disjoint {info["disjoint"]}')
    return info


# ---------------------------------------------------------------- job pool

def run_jobs(c, out, jobs, budget, log):
    """Run the jobs whose part file is missing, 4 workers; no new job starts after the budget is spent."""
    out = Path(out)
    (out / 'parts').mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out / 'parts' / J.part_name(*j)).exists()]
    log(f'scores: {len(jobs) - len(todo)} of {len(jobs)} jobs done, {len(todo)} to run, budget {budget:.0f} s')
    t0 = time.time()
    done = []
    it = iter(todo)
    with cf.ProcessPoolExecutor(c['workers']) as ex:
        pending = set()
        for _ in range(c['workers']):
            j = next(it, None)
            if j is not None:
                pending.add(ex.submit(J.run_job, j))
        while pending:
            fin, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                r = f.result()
                done.append(r)
                log(f"job {r['job']} {r['wall']:.1f} s")
                if time.time() - t0 < budget:
                    j = next(it, None)
                    if j is not None:
                        pending.add(ex.submit(J.run_job, j))
    left = sum(1 for j in jobs if not (out / 'parts' / J.part_name(*j)).exists())
    return done, left


# ---------------------------------------------------------------- state, manifest

def load_state(out):
    p = Path(out) / 'run_state.json'
    return json.loads(p.read_text()) if p.exists() else dict(wall_seconds=0.0, invocations=[])


def save_state(out, st):
    C.atomic_write_text(Path(out) / 'run_state.json', json.dumps(st, indent=1))


def write_manifest(c, out, extra=None):
    import scipy
    import sklearn
    import torch
    import zstandard
    out = Path(out)
    man = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    cpu = ''
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        pass
    man.update(dict(
        sources={s: C.sha256_file(HERE / s) for s in SOURCES if (HERE / s).exists()},
        upstream=json.loads((HERE / 'upstream_sha256.json').read_text()),
        versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, zstandard=zstandard.__version__),
        host=socket.gethostname(), cpu_model=cpu, cpu_count=os.cpu_count()))
    man.update(extra or {})
    C.atomic_write_text(out / 'manifest.json', json.dumps(man, indent=1))


# ---------------------------------------------------------------- check-only

def check_only(c, scratch, log):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    C.check_upstream(c)
    info = stage_refs(c, scratch, log, c['graded_seeds'])
    G = json.loads((scratch / 'graded' / 'graded_tasks_full.json').read_text())
    gtab = [dict(task=t['task_id'], r=t['r'], K=t['K'], acceptance=round(t['acceptance_rate'], 4),
                 agree=round(t['agree'], 4), agree_exact_uniform=round(t['agree_exact_uniform'], 4),
                 rho=round(t['rho'], 4)) for t in G['tasks'] if t['role'] == 'distractor']
    mixes = {k: dict(alpha_share=m['alpha_share'], alpha_soft=round(m['alpha_soft'], 4),
                     alpha_agree=round(m['alpha_agree'], 4)) for k, m in G['mixes'].items()}
    work = dict(
        main_targets=dict(dfa=16, dsl=20), graded_targets=4,
        scored_units=dict(dfa=32 * c['scored_units_per_task'], dsl=40 * 512, graded=32 * c['scored_units_per_task']),
        probe_jobs=3 * len(c['probes']), nonprobe_jobs=16 + 20 + 4, t2v_jobs=(16 + 20 + 4) * len(c['probes']),
        graded_runs=len(c['graded_seeds']) * 20)
    rec = dict(check='PASS' if all(info['token_free'].values()) and all(info['disjoint'].values())
               and all(v['counts_equal'] for v in info['mix_check'].values())
               and info['graded']['construction'][-1]['passed'] else 'FAIL',
               mix_check=info['mix_check'], token_free=info['token_free'], disjoint=info['disjoint'],
               references=info['references'], overlap=info['overlap'], tv=info['tv'], graded=info['graded'],
               graded_distractors=gtab, graded_mixes=mixes, work=work)
    print(json.dumps(rec, indent=1))
    return rec


# ---------------------------------------------------------------- pilot

def pilot(c, log):
    out = HERE / 'results_pilot'
    out.mkdir(exist_ok=True)
    log.path = out / 'run.log'
    seed = c['pilot_seed']
    t_all = time.time()
    info = stage_refs(c, out, log, [seed])
    probe = c['probes'][0]
    tg = {}
    for fam in C.FAMILIES:
        ts = [t for t in C.load_tasks(c, fam) if t['role'] == 'target']
        lv = sorted({t['K'] for t in ts})
        tg[fam] = [next(t['task_id'] for t in ts if t['K'] == lv[0]), next(t['task_id'] for t in ts if t['K'] == lv[-1])]
    jobs = [('probe', (fam, probe, str(out), tg[fam])) for fam in C.FAMILIES]
    jobs += [('t2v', (fam, probe, t, str(out))) for fam in C.FAMILIES for t in tg[fam]]
    jobs += [('nonprobe', (fam, t, str(out))) for fam in C.FAMILIES for t in tg[fam]]
    t0 = time.time()
    done, _ = run_jobs(c, out, jobs, 1e9, log)
    t_scores = time.time() - t0
    # 5 graded runs: the first graded target, every mix, pilot seed
    G = json.loads((out / 'graded' / 'graded_tasks_full.json').read_text())
    tgt = next(t['task_id'] for t in sorted(G['tasks'], key=lambda t: t['task_token']) if t['role'] == 'target')
    runs = [(seed, f'{tgt}_{m}') for m in sorted(c['graded']['mixes'])]
    t0 = time.time()
    gdone, _ = GR.run_grid(c, out / 'graded', runs, 1e9, log)
    t_graded = time.time() - t0
    timing = {r['job']: dict(wall=r['wall'], detail=r.get('timing')) for r in done}
    per_run = float(np.mean([s for _, s in gdone]))
    # projection (4 workers): graded runs, then scoring (DFA-sized graded jobs scaled by pool size)
    W = c['workers']
    pj = {fam: timing[J.part_name('probe', (fam, probe))]['wall'] for fam in C.FAMILIES}
    tj = {fam: float(np.mean([timing[J.part_name('t2v', (fam, probe, t))]['wall'] for t in tg[fam]]))
          for fam in C.FAMILIES}
    nj = {fam: float(np.mean([timing[J.part_name('nonprobe', (fam, t))]['wall'] for t in tg[fam]]))
          for fam in C.FAMILIES}
    n_p = len(c['probes'])
    cpu = dict(
        probe=n_p * (pj['dfa'] * (1 + 36 / 32) + pj['dsl']),
        t2v=n_p * (16 * tj['dfa'] + 20 * tj['dsl'] + 4 * tj['dfa'] * 11 / 6),
        nonprobe=16 * nj['dfa'] + 20 * nj['dsl'] + 4 * nj['dfa'] * 9 / 17)
    graded_proj = {n: per_run * n / W for n in (160, 80)}
    longest_job = max(pj.values())
    proj = dict(per_graded_run_seconds=per_run, graded_wall_seconds=graded_proj,
                scoring_cpu_seconds=cpu, scoring_wall_seconds=sum(cpu.values()) / W,
                longest_single_job_seconds=longest_job,
                total_wall_seconds=sum(cpu.values()) / W + graded_proj[160],
                graded_fallback=graded_proj[160] > 60 * c['graded_fallback_minutes'])
    summary = dict(pilot_seed=seed, probe=probe, targets=tg, graded_runs=[r for r, _ in gdone],
                   refs_seconds=info['seconds'], scoring_seconds=t_scores, graded_seconds=t_graded,
                   job_timing=timing, projection=proj, wall_seconds=time.time() - t_all)
    C.atomic_write_text(out / 'pilot_summary.json', json.dumps(summary, indent=1))
    write_manifest(c, out, dict(pilot=True))
    log(f'pilot done: projection {json.dumps(proj)}')
    return summary


# ---------------------------------------------------------------- gated run

def gated(c, out, stage, max_seconds, resume, log):
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f'{out} exists and is not empty; use --resume to continue it')
    out.mkdir(parents=True, exist_ok=True)
    log.path = out / 'run.log'
    C.check_upstream(c)
    st = load_state(out)
    t0 = time.time()
    cap = c['max_wall_seconds'] - st['wall_seconds']
    budget = min(max_seconds, cap)
    if budget <= 0:
        log('wall cap reached; nothing started')
        return
    seeds = c['graded_fallback_seeds'] if c['graded_use_fallback'] else c['graded_seeds']
    status = {}
    try:
        if stage in ('refs', 'all'):
            stage_refs(c, out, log, seeds)
        if stage in ('graded', 'all'):
            G = json.loads((out / 'graded' / 'graded_tasks_full.json').read_text())
            runs = [(s, k) for s in seeds for k in sorted(G['mixes'])]
            left = budget - (time.time() - t0)
            _, n_left = GR.run_grid(c, out / 'graded', runs, left, log)
            status['graded_left'] = n_left
            if n_left == 0:
                GR.aggregate(c, out / 'graded', G, seeds)
                log('graded: metrics.csv written')
        if stage in ('scores', 'all'):
            left = budget - (time.time() - t0)
            _, n_left = run_jobs(c, out, J.all_jobs(c, out), left, log)
            status['score_jobs_left'] = n_left
            if n_left == 0:
                for arm in J.ARMS:
                    J.merge_scores(c, out, arm)
                log('scores: merged into scores/')
        if stage in ('eq2', 'all') and (out / 'scores' / 'dsl_task2vec.json').exists():
            rows = []
            for fam in C.FAMILIES:
                rows += J.eq2_check(c, out, fam, log=log)
            C.atomic_write_text(out / 'eq2_check.json', json.dumps(rows, indent=1))
            log('eq2: exact Eq. 2 on full mixes written')
    finally:
        el = time.time() - t0
        st['wall_seconds'] += el
        st['invocations'].append(dict(started=now(), stage=stage, seconds=el, status=status))
        save_state(out, st)
        write_manifest(c, out, dict(wall_seconds_total=st['wall_seconds'], graded_seeds=seeds))
        if st['wall_seconds'] >= c['max_wall_seconds']:
            log('wall cap max_wall_seconds reached; stopping (resumable)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--scratch', default=None)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--output', default=None)
    ap.add_argument('--stage', default='all', choices=['refs', 'scores', 'graded', 'eq2', 'all'])
    ap.add_argument('--max-seconds', type=float, default=3300)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    c = C.load_config()
    log = Log()
    if a.check_only:
        check_only(c, a.scratch or os.path.join(os.environ.get('TMPDIR', '/tmp'), 'e15_check'), log)
    elif a.pilot:
        pilot(c, log)
    elif a.output:
        gated(c, HERE / a.output if not os.path.isabs(a.output) else Path(a.output), a.stage, a.max_seconds,
              a.resume, log)
    else:
        ap.error('one of --check-only, --pilot, --output')


if __name__ == '__main__':
    sys.exit(main())
