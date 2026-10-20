"""Experiment 14 runner: tasks, probes, Task2Vec embeddings and diversity coefficients (PROTOCOL Sections 2-3).

  python run.py --check-only                       build tasks and run every pre-run check; trains nothing
  python run.py --pilot                            disclosed pilot (seed 14909) -> results_pilot/
  python run.py --output results [--resume] [--budget-seconds S] [--stages probes,embed,diversity]

Work is split into small jobs (one probe, or a few task embeddings, or a few diversity mixes), run on `workers`
processes with one torch thread each. Each job writes one part file atomically under results/parts/, so a stopped
run resumes with --resume and redoes nothing. --budget-seconds stops submitting new jobs after S seconds of this
invocation; the cumulative wall time over invocations is kept in results/run_state.json and the run stops cleanly
at max_wall_seconds. When every job is done the parts are merged into embeddings.npz and batch_embeddings/, and
manifest.json is written.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import scipy
import torch

import probe as P
import task2vec as V
import tasks14 as T

warnings.filterwarnings('ignore', message='.*[Ss]parse.*')
HERE = Path(__file__).resolve().parent
EMB_CHUNK = 8      # task embeddings per job
DIV_CHUNK = 6      # diversity mixes per job
SOURCES = ['config.json', 'tasks14.py', 'probe.py', 'task2vec.py', 'stats.py', 'independent_stats.py', 'run.py',
           'analyze.py', 'verify.py', 'report.py', 'requirements.txt']


# ---------------------------------------------------------------- job lists

def emb_key(fam, tid, probe, est, variant):
    return f'{fam}|{tid}|{probe}|{est}|{variant}'


def div_key(fam, probe, mix_id, kind, i):
    return f'{fam}|{probe}|{mix_id}|{kind}|{i}'


def _chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def diversity_items(c, roster, fam):
    """(mix dict, kind) for every scored mix: interleaved for Experiment 11/12 and coverage mixes, single-task for
    coverage mixes, and the two reference bounds."""
    exp, cov = T.exp_mixes(c, roster, fam), T.coverage_mixes(c, roster, fam)
    items = [(m, 'interleaved') for m in cov + exp] + [(m, 'single') for m in cov]   # Gate 4 mixes first
    items += [(dict(mix_id=f'{fam}_bound_{w}', kind='bound', which=w), 'bound') for w in ('lower', 'upper')]
    return items


def all_jobs(c, roster):
    """Every job of the full run (PROTOCOL Section 3)."""
    probes, emb, div = [], [], []
    for fam in T.FAMILIES:
        for s in c['probe_seeds']:
            probes.append(dict(kind='probe', id=f'probe_{fam}_{s}', fam=fam, seed=s, train=True))
        probes.append(dict(kind='probe', id=f'probe_{fam}_{c["random_init_seed"]}', fam=fam,
                           seed=c['random_init_seed'], train=False))
        tids = [t['task_id'] for t in T.embedded_tasks(roster, fam)]
        noise = [(t['task_id'], r) for t in T.noise_tasks(c, roster, fam) for r in c['noise_rates']]
        for s in c['probe_seeds']:
            for i, ch in enumerate(_chunks(tids, EMB_CHUNK)):
                emb.append(dict(kind='embed', id=f'emb_{fam}_{s}_main_{i:02d}', fam=fam, seed=s, variant='main',
                                items=[(t, None) for t in ch], ests=['E0', 'E1', 'E2', 'E3']))
            for i, ch in enumerate(_chunks(noise, EMB_CHUNK)):
                emb.append(dict(kind='embed', id=f'emb_{fam}_{s}_noise_{i:02d}', fam=fam, seed=s, variant='noise',
                                items=ch, ests=['E0']))
            for i, ch in enumerate(_chunks(tids, EMB_CHUNK)):
                emb.append(dict(kind='embed', id=f'emb_{fam}_{s}_rescaled_{i:02d}', fam=fam, seed=s,
                                variant='rescaled', items=[(t, None) for t in ch], ests=['E0']))
        r = c['random_init_seed']
        for i, ch in enumerate(_chunks(tids, EMB_CHUNK)):
            emb.append(dict(kind='embed', id=f'emb_{fam}_{r}_main_{i:02d}', fam=fam, seed=r, variant='main',
                            items=[(t, None) for t in ch], ests=['E0']))
        items = [(m['mix_id'], k) for m, k in diversity_items(c, roster, fam)]
        for s in c['probe_seeds']:
            for i, ch in enumerate(_chunks(items, DIV_CHUNK)):
                div.append(dict(kind='div', id=f'div_{fam}_{s}_{i:02d}', fam=fam, seed=s, items=ch,
                                n_batches=c['diversity_batches']))
    return probes, emb, div


def work_counts(c, roster):
    out = {}
    for fam in T.FAMILIES:
        nt, nn = len(T.embedded_tasks(roster, fam)), len(T.noise_tasks(c, roster, fam)) * len(c['noise_rates'])
        items = diversity_items(c, roster, fam)
        n_int = sum(1 for _, k in items if k == 'interleaved')
        n_single = sum(1 for _, k in items if k == 'single')
        nps = len(c['probe_seeds'])
        out[fam] = dict(embedded_tasks=nt, noise_embeddings_per_probe=nn,
                        all_estimator_task_embeddings=nps * nt,
                        e0_only_task_embeddings=nps * (nn + nt) + nt,
                        interleaved_mixes=n_int, single_task_mixes=n_single,
                        batch_embeddings=nps * (n_int + n_single + 2) * c['diversity_batches'],
                        probes_pretrained=nps, random_init_probes=1)
    return out


# ---------------------------------------------------------------- worker side

_W = {}


def _init_worker(out):
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', message='.*[Ss]parse.*')
    _W['c'] = T.load_config()
    _W['out'] = Path(out)
    _W['roster'] = T.load_roster(Path(out) / 'tasks14.json')[0]
    _W['spec'] = {}


def _spec(fam, seed, variant):
    key = (fam, seed, variant == 'rescaled')
    if key not in _W['spec']:
        c = _W['c']
        m = P.load_probe(c, fam, P.probe_path(_W['out'], fam, seed))
        m.eval()
        if variant == 'rescaled':
            m = P.rescaled_copy(m, fam)
        _W['spec'] = {k: v for k, v in _W['spec'].items() if k[0] == fam and k[1] == seed}
        _W['spec'][key] = V.Spec(m, fam)
    return _W['spec'][key]


def _task(fam, tid):
    return next(t for t in _W['roster'][fam] if t['task_id'] == tid)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def part_path(out, job):
    sub = dict(probe='probes', embed='parts/embed', div='parts/div')[job['kind']]
    ext = '.json' if job['kind'] == 'probe' else '.npz'
    return Path(out) / sub / f"{job['id']}{ext}"


def job_done(out, job):
    p = part_path(out, job)
    if not p.exists():
        return False
    if job['kind'] == 'probe':
        return P.probe_path(out, job['fam'], job['seed']).exists()
    try:
        with np.load(p, allow_pickle=False) as z:
            return 'f' in z.files
    except Exception:
        return False


def run_probe_job(job):
    c, out, roster = _W['c'], _W['out'], _W['roster']
    fam, seed = job['fam'], job['seed']
    t0 = time.time()
    if job['train']:
        model, rec = P.pretrain_probe(c, roster, fam, seed)
    else:
        model = P.build_probe(c, fam, seed)
        model.eval()
        rec = dict(family=fam, seed=seed, examples=0, steps=0, curve=[], random_init=True,
                   parameters=sum(p.numel() for p in model.parameters()))
    path = P.probe_path(out, fam, seed)
    P.save_probe(c, fam, model, path)
    model = P.load_probe(c, fam, path)
    model.eval()
    spec = V.Spec(model, fam)
    rec['embedding_dim'] = int(spec.layout[-1][2])
    rec['layout'] = spec.layout
    rec['evaluation'] = P.evaluate_probe(c, roster, fam, model)
    model.eval()
    if job['train']:
        ex = T.example_set(c, roster, fam, T.embedded_tasks(roster, fam)[0])
        rec['rescale_max_output_diff'] = P.max_output_diff(model, P.rescaled_copy(model, fam), fam, ex)
        assert rec['rescale_max_output_diff'] < 1e-5, rec['rescale_max_output_diff']
    rec['seconds_total'] = round(time.time() - t0, 3)
    _atomic_json(part_path(out, job), rec)
    return dict(id=job['id'], seconds=rec['seconds_total'])


def embed_one(c, roster, fam, seed, variant, tid, rate, ests):
    """One task's embeddings for one probe and variant; returns {estimator: dict(f, head_loss, head_error)}."""
    spec = _spec(fam, seed, variant)
    task = _task(fam, tid)
    if variant == 'noise':
        ex = T.noisy_example_set(c, roster, fam, task, rate)
        sp = (seed, fam, tid, 'noise', f'{rate:.2f}')
    else:
        ex = T.example_set(c, roster, fam, task)
        sp = (seed, fam, tid, 'main')       # the rescaled probe reuses the original probe's draws
    return V.embed_task(c, spec, ex, sp, tuple(ests))


def variant_name(variant, rate):
    return f'noise{rate:.2f}' if variant == 'noise' else variant


def run_embed_job(job):
    c, out, roster = _W['c'], _W['out'], _W['roster']
    keys, F_, hl, he, secs = [], [], [], [], []
    for tid, rate in job['items']:
        t0 = time.time()
        res = embed_one(c, roster, job['fam'], job['seed'], job['variant'], tid, rate, job['ests'])
        dt = time.time() - t0
        for est in job['ests']:
            keys.append(emb_key(job['fam'], tid, job['seed'], est, variant_name(job['variant'], rate)))
            F_.append(res[est]['f'])
            hl.append(res[est]['head_loss'])
            he.append(res[est]['head_error'])
            secs.append(dt / len(job['ests']))
    _atomic_npz(part_path(out, job), keys=np.array(keys), f=np.array(F_, np.float32), head_loss=np.array(hl),
                head_error=np.array(he), seconds=np.array(secs))
    return dict(id=job['id'], seconds=float(sum(secs)), units=len(job['items']))


def mix_lookup(c, roster, fam):
    if ('mixes', fam) not in _W:
        _W[('mixes', fam)] = {m['mix_id']: m for m, _ in diversity_items(c, roster, fam)}
    return _W[('mixes', fam)]


def diversity_batches(c, roster, fam, mix, kind, n_batches):
    """Yield the n_batches example batches (diversity_batch_size each) of one mix and kind. Batch sampling depends
    only on (task_sampling_seed, family, mix, kind), so every probe sees the same batches."""
    B = c['diversity_batch_size']
    if kind == 'bound':
        for i in range(n_batches):
            rng = np.random.default_rng(T.task_seed_int(c['task_sampling_seed'], fam, mix['mix_id'], i))
            yield T.bound_example_set(c, fam, mix['which'], B, rng)
        return
    exs = [T.example_set(c, roster, fam, _task(fam, tid)) for tid in mix['tasks']]
    gen = torch.Generator().manual_seed(T.task_seed_int(c['task_sampling_seed'], fam, mix['mix_id'], kind))
    for i in range(n_batches):
        yield V.sample_batch(exs, mix['shares'], B, gen, single_task=(kind == 'single'))


def batch_embedding(c, spec, fam, seed, mix_id, kind, i, batch):
    return V.embed_batch_e3(c, spec, batch, (seed, fam, mix_id, kind, i))


def run_div_job(job):
    c, out, roster = _W['c'], _W['out'], _W['roster']
    fam, seed = job['fam'], job['seed']
    spec = _spec(fam, seed, 'main')
    mixes = mix_lookup(c, roster, fam)
    keys, F_, hl, he, secs = [], [], [], [], []
    for mix_id, kind in job['items']:
        for i, batch in enumerate(diversity_batches(c, roster, fam, mixes[mix_id], kind, job['n_batches'])):
            t0 = time.time()
            r = batch_embedding(c, spec, fam, seed, mix_id, kind, i, batch)
            secs.append(time.time() - t0)
            keys.append(div_key(fam, seed, mix_id, kind, i))
            F_.append(r['f'])
            hl.append(r['head_loss'])
            he.append(r['head_error'])
    _atomic_npz(part_path(out, job), keys=np.array(keys), f=np.array(F_, np.float32), head_loss=np.array(hl),
                head_error=np.array(he), seconds=np.array(secs))
    return dict(id=job['id'], seconds=float(sum(secs)), units=len(keys))


def run_job(job):
    try:
        t0 = time.time()
        r = dict(probe=run_probe_job, embed=run_embed_job, div=run_div_job)[job['kind']](job)
        r['wall'] = time.time() - t0
        return r
    except Exception:
        return dict(id=job['id'], error=traceback.format_exc())


# ---------------------------------------------------------------- driver

class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"{time.strftime('%m-%d-%Y %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.path, 'a') as f:
            f.write(line + '\n')


def run_pool(out, jobs, workers, log, deadline):
    """Run jobs (skipping finished ones) on `workers` processes until done or `deadline` (time.time()). Returns
    (n_done_now, n_failed, n_pending)."""
    todo = [j for j in jobs if not job_done(out, j)]
    log(f'{len(jobs) - len(todo)} of {len(jobs)} jobs already done; {len(todo)} to run')
    if not todo:
        return 0, 0, 0
    ctx = mp.get_context('spawn')
    done = failed = 0
    with cf.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=(str(out),)) as ex:
        it = iter(todo)
        running = {}
        stopped = False
        while True:
            while not stopped and len(running) < workers:
                if time.time() > deadline:
                    stopped = True
                    break
                j = next(it, None)
                if j is None:
                    stopped = True
                    break
                running[ex.submit(run_job, j)] = j
            if not running:
                break
            fin, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                j = running.pop(f)
                r = f.result()
                if 'error' in r:
                    failed += 1
                    log(f"FAILED {j['id']}:\n{r['error']}")
                else:
                    done += 1
                    log(f"done {r['id']} {r['wall']:.1f}s ({done} this run)")
    pending = sum(1 for j in jobs if not job_done(out, j))
    return done, failed, pending


def source_hashes():
    return {s: T.sha256_file(HERE / s) for s in SOURCES if (HERE / s).exists()}


def versions():
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__)


def merge(c, out, roster, log, partial=False):
    """Merge part files into embeddings.npz and batch_embeddings/<family>_<probe>.npz. partial=True merges only the
    finished jobs (a run stopped at the wall cap) and returns the ids of the missing ones."""
    probes, emb, div = all_jobs(c, roster)
    missing = [j['id'] for j in emb + div if not job_done(out, j)]
    assert partial or not missing, missing
    emb = [j for j in emb if job_done(out, j)]
    div = [j for j in div if job_done(out, j)]
    arrays = {}       # one set of arrays per family: the embedding dimension differs (DFA 1,564, DSL 1,566)
    for fam in T.FAMILIES:
        keys, F_, hl, he, secs = [], [], [], [], []
        for j in emb:
            if j['fam'] != fam:
                continue
            with np.load(part_path(out, j), allow_pickle=False) as z:
                keys += z['keys'].tolist()
                F_.append(z['f'])
                hl.append(z['head_loss'])
                he.append(z['head_error'])
                secs.append(z['seconds'])
        order = np.argsort(keys, kind='stable')
        keys = np.array(keys)[order]
        assert len(set(keys.tolist())) == len(keys)
        arrays.update({f'{fam}_keys': keys, f'{fam}_f': np.concatenate(F_)[order],
                       f'{fam}_head_loss': np.concatenate(hl)[order], f'{fam}_head_error': np.concatenate(he)[order],
                       f'{fam}_seconds': np.concatenate(secs)[order]})
        log(f'merged {len(keys)} {fam} task embeddings into embeddings.npz')
    _atomic_npz(out / 'embeddings.npz', **arrays)
    by = {}
    for j in div:
        by.setdefault((j['fam'], j['seed']), []).append(j)
    for (fam, seed), js in by.items():
        keys, F_, hl, he, secs = [], [], [], [], []
        for j in js:
            with np.load(part_path(out, j), allow_pickle=False) as z:
                keys += z['keys'].tolist()
                F_.append(z['f'])
                hl.append(z['head_loss'])
                he.append(z['head_error'])
                secs.append(z['seconds'])
        _atomic_npz(out / 'batch_embeddings' / f'{fam}_{seed}.npz', keys=np.array(keys), f=np.concatenate(F_),
                    head_loss=np.concatenate(hl), head_error=np.concatenate(he), seconds=np.concatenate(secs))
        log(f'merged {len(keys)} batch embeddings into batch_embeddings/{fam}_{seed}.npz')
    return missing


def write_manifest(c, out, state, extra=None):
    mods = T.exp_modules(c)
    man = dict(experiment=c['experiment'], sources=source_hashes(), versions=versions(),
               machine=f'{os.cpu_count()}-core CPU, no GPU', workers=c['workers'], threads=c['threads'],
               diversity_batches=c['diversity_batches'],
               upstream={fam: dict(dir=str(Path(os.path.relpath(m['dir'], HERE))), source_sha256=m['hashes'],
                                   manifest_sha256=T.sha256_file(m['dir'] / 'results' / 'manifest.json'),
                                   tasks_sha256=T.sha256_file(m['dir'] / 'results' / 'tasks.json'),
                                   mixes_sha256=T.sha256_file(m['dir'] / 'results' / 'mixes.json'),
                                   metrics_sha256=T.sha256_file(m['dir'] / 'results' / 'metrics.csv'))
                         for fam, m in mods.items()},
               wall_seconds_cumulative=round(state['wall_seconds'], 1), invocations=state['invocations'])
    if extra:
        man.update(extra)
    _atomic_json(out / 'manifest.json', man)


def prepare_tasks(c, out, log):
    path = out / 'tasks14.json'
    if path.exists():
        return T.load_roster(path)[0]
    t0 = time.time()
    roster, logs = T.build_roster(c)
    T.save_roster(path, roster, logs)
    log(f'tasks14.json written ({time.time() - t0:.1f}s)')
    return roster


def check_only(c):
    t0 = time.time()
    roster, logs = T.build_roster(c)
    ok1, rep1 = T.check_roster(c, roster)
    ok2, rep2 = T.check_inputs(c, roster)
    mods = T.exp_modules(c)
    rep = dict(source_hashes_match_upstream_manifests={fam: m['hashes'] for fam, m in mods.items()},
               roster=rep1, inputs=rep2, dfa_selection_log=logs['dfa'], work_counts=work_counts(c, roster),
               ok=bool(ok1 and ok2), seconds=round(time.time() - t0, 1))
    probes, emb, div = all_jobs(c, roster)
    rep['jobs'] = dict(probe=len(probes), embed=len(emb), diversity=len(div))
    for fam in T.FAMILIES:
        m = P.build_probe(c, fam, 0)
        spec = V.Spec(m, fam)
        rep[f'{fam}_probe'] = dict(parameters=sum(p.numel() for p in m.parameters()),
                                   embedding_dim=int(spec.layout[-1][2]),
                                   vocabulary=int(spec.body[f"{V.NAMES[fam]['tok']}.weight"].shape[0]),
                                   null_token_row=2 + T.null_token(c, fam))
    print(json.dumps(rep, indent=1))
    print('CHECK', 'PASS' if rep['ok'] else 'FAIL')
    return rep['ok']


def pilot(c, out, log):
    """Disclosed pilot (PROTOCOL Section 2 end): probes with the pilot seed, 8 tasks x all estimators, E0-only and
    rescaled-probe timing, 2 diversity mixes, per family; all on the worker pool, so timings include contention."""
    out.mkdir(parents=True, exist_ok=True)
    roster = prepare_tasks(c, out, log)
    s = c['pilot_seed']
    jobs = [dict(kind='probe', id=f'probe_{fam}_{s}', fam=fam, seed=s, train=True) for fam in T.FAMILIES]
    t0 = time.time()
    run_pool(out, jobs, c['workers'], log, float('inf'))
    t_probe = time.time() - t0
    ejobs, djobs = [], []
    for fam in T.FAMILIES:
        tasks = T.embedded_tasks(roster, fam)
        pick = [tasks[i]['task_id'] for i in np.linspace(0, len(tasks) - 1, 8).round().astype(int)]
        for i, tid in enumerate(pick):
            ejobs.append(dict(kind='embed', id=f'emb_{fam}_{s}_main_{i:02d}', fam=fam, seed=s, variant='main',
                              items=[(tid, None)], ests=['E0', 'E1', 'E2', 'E3']))
        for i, tid in enumerate(pick[:4]):
            ejobs.append(dict(kind='embed', id=f'emb_{fam}_{s}_rescaled_{i:02d}', fam=fam, seed=s,
                              variant='rescaled', items=[(tid, None)], ests=['E0']))
        items = diversity_items(c, roster, fam)
        mixes = [next(m for m, k in items if m['kind'] == 'exp'), next(m for m, k in items if m['kind'] == 'coverage'
                                                                         and m['m'] == 16)]
        for i, m in enumerate(mixes):
            djobs.append(dict(kind='div', id=f'div_{fam}_{s}_{i:02d}', fam=fam, seed=s,
                              items=[(m['mix_id'], 'interleaved')], n_batches=c['diversity_batches']))
    t1 = time.time()
    run_pool(out, ejobs + djobs, c['workers'], log, float('inf'))
    t_rest = time.time() - t1
    tim = {}
    for fam in T.FAMILIES:
        pr = json.loads(part_path(out, jobs[T.FAMILIES.index(fam)]).read_text())
        e_all = [float(np.load(part_path(out, j))['seconds'].sum()) for j in ejobs
                 if j['fam'] == fam and j['variant'] == 'main']
        e0 = [float(np.load(part_path(out, j))['seconds'].sum()) for j in ejobs
              if j['fam'] == fam and j['variant'] == 'rescaled']
        bt = np.concatenate([np.load(part_path(out, j))['seconds'] for j in djobs if j['fam'] == fam])
        tim[fam] = dict(probe_pretrain_seconds=pr['seconds'], probe_job_seconds=pr['seconds_total'],
                        all_estimators_per_task_seconds=float(np.mean(e_all)),
                        e0_only_per_task_seconds=float(np.mean(e0)),
                        e3_batch_seconds=float(np.mean(bt)), n_batches_timed=int(len(bt)))
    wc = work_counts(c, roster)

    def project(n_batches):
        tot = 0.0
        parts = {}
        for fam in T.FAMILIES:
            w, t = wc[fam], tim[fam]
            nb = len(c['probe_seeds']) * (w['interleaved_mixes'] + w['single_task_mixes'] + 2) * n_batches
            parts[fam] = dict(
                probes=(len(c['probe_seeds']) * t['probe_job_seconds'] + 60.0),
                embeddings=w['all_estimator_task_embeddings'] * t['all_estimators_per_task_seconds']
                + w['e0_only_task_embeddings'] * t['e0_only_per_task_seconds'],
                diversity=nb * t['e3_batch_seconds'])
            tot += sum(parts[fam].values())
        hours = tot / c['workers'] / 3600 + 0.25      # + analysis, bootstrap and verification
        return dict(core_seconds=parts, projected_wall_hours=round(hours, 3))

    p12 = project(c['diversity_batches'])
    p8 = project(c['fallback_diversity_batches'])
    decision = ('fallback: cut diversity batches to 8' if p12['projected_wall_hours'] > c['fallback_projection_hours']
                else 'no fallback: keep 12 diversity batches')
    res = dict(seed=s, timings=tim, wall_probe_stage=round(t_probe, 1), wall_embed_and_diversity=round(t_rest, 1),
               work_counts=wc, projection_12_batches=p12, projection_8_batches=p8,
               threshold_hours=c['fallback_projection_hours'], decision=decision,
               note='timings measured on 4 worker processes running at once (projection = core seconds / 4 + '
                    '0.25 h for analysis and verification)')
    _atomic_json(out / 'pilot_timing.json', res)
    log(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--output', default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--budget-seconds', type=float, default=float('inf'))
    ap.add_argument('--stages', default='probes,embed,diversity')
    ap.add_argument('--merge-partial', action='store_true',
                    help='run no jobs: merge the finished parts of a run stopped at the wall cap')
    a = ap.parse_args()
    c = T.load_config()
    torch.set_num_threads(c['threads'])
    if a.check_only:
        sys.exit(0 if check_only(c) else 1)
    if a.pilot:
        out = HERE / (a.output or 'results_pilot')
        if out.exists() and any(out.iterdir()) and not a.resume:
            sys.exit(f'refusing to overwrite non-empty {out}')
        pilot(c, out, Log(out / 'run_log.txt'))
        return
    out = HERE / (a.output or 'results')
    if out.exists() and any(out.iterdir()) and not a.resume:
        sys.exit(f'refusing to overwrite non-empty {out} (use --resume to continue a stopped run)')
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / 'run_log.txt')
    st_path = out / 'run_state.json'
    state = json.loads(st_path.read_text()) if st_path.exists() else dict(wall_seconds=0.0, invocations=[])
    t_start = time.time()
    left = c['max_wall_seconds'] - state['wall_seconds']
    if left <= 0 and not a.merge_partial:
        log(f"wall cap max_wall_seconds={c['max_wall_seconds']} reached; stopping")
        return
    deadline = t_start + min(a.budget_seconds, left)
    log(f"invocation {len(state['invocations']) + 1}: cumulative wall {state['wall_seconds']:.0f}s, "
        f"budget {min(a.budget_seconds, left):.0f}s, sources {json.dumps(source_hashes())}")
    roster = prepare_tasks(c, out, log)
    probes, emb, div = all_jobs(c, roster)
    stages = [] if a.merge_partial else a.stages.split(',')
    pending = 0
    if 'probes' in stages:
        d, f, pending = run_pool(out, probes, c['workers'], log, deadline)
        if f:
            log(f'{f} probe jobs failed; stopping')
    if not pending and ('embed' in stages or 'diversity' in stages):
        # gate inputs first (deviation 2, CKPT): all-estimator embeddings (Gates 1-3), label noise (Gate 5),
        # diversity jobs holding interleaved coverage mixes (Gate 4); then other diversity, rescaled, random-init
        def g4(j):
            return any(k == 'interleaved' and '_cov_' in m for m, k in j['items'])
        rank = []
        if 'embed' in stages:
            rank += [(0, j) for j in emb if j['variant'] == 'main' and len(j['ests']) == 4]
            rank += [(1, j) for j in emb if j['variant'] == 'noise']
            rank += [(4, j) for j in emb if j['variant'] == 'rescaled']
            rank += [(5, j) for j in emb if j['variant'] == 'main' and len(j['ests']) == 1]
        if 'diversity' in stages:
            rank += [(2 if g4(j) else 3, j) for j in div]
        jobs = [j for _, j in sorted(rank, key=lambda r: r[0])]
        d, f, pending = run_pool(out, jobs, c['workers'], log, deadline)
    dt = time.time() - t_start
    state['wall_seconds'] += dt
    state['invocations'].append(dict(start=time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(t_start)),
                                     seconds=round(dt, 1), stages=stages, pending_after=pending))
    _atomic_json(st_path, state)
    all_pending = sum(1 for j in probes + emb + div if not job_done(out, j))
    if all_pending:
        log(f'stopped with {all_pending} jobs pending (cumulative wall {state["wall_seconds"]:.0f}s); '
            f'rerun with --resume')
        if a.merge_partial or state['wall_seconds'] >= c['max_wall_seconds']:
            missing = merge(c, out, roster, log, partial=True)
            write_manifest(c, out, state, dict(complete=False, stopped_at_wall_cap=True, missing_jobs=missing))
            log(f'wall cap reached: merged the finished jobs; {len(missing)} jobs missing (listed in manifest.json)')
        return
    merge(c, out, roster, log)
    write_manifest(c, out, state, dict(complete=True))
    log(f'all jobs done; cumulative wall {state["wall_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
