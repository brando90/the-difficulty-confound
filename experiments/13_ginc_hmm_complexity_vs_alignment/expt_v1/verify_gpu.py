"""Experiment 13 verification, arm 13-GPU (verify.py --arm gpu dispatches here). Prints PASS or the first failure.

Per shard, on a GPU node (launch scripts, MODE=verify): verify.py --arm gpu --shard I --output OUT
  1. every file listed in DONE matches its SHA-256 (checkpoints included);
  2. the shard ran the current sources and the current concepts/references (prep SHA-256s);
  3. the training stream regenerates from seeds (SHA-256 and per-checkpoint document counts);
  4. every checkpoint is reloaded and the held-out model code lengths are recomputed in float64 with deterministic
     algorithms: per-slice means and the 128-position curve to max_abs_diff < 1e-9 bits.
  It then writes OUT/shards/<%05d>/VERIFIED (JSON) next to DONE.
Merged, on the CPU box: verify.py --arm gpu [--output results/gpu] [--workers 4]
  0. merged shards ran the current sources; 1. concepts regenerate, ranks recertified, K_bits recomputed, held-out
  documents regenerate; 2. oracle and mixture code lengths agree with the independent log-space forward algorithm to
  1e-9; 3. all 360 shards carry a VERIFIED marker matching their DONE file; 4. every law is refit (stage 1 into a
  scratch copy: forecasts hash and parameters to 1e-4), stage 2 gives identical verdicts, the independent fitter agrees.
"""
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hmm as H  # noqa: E402
import run as R  # noqa: E402


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def verify_shard(c, out, i, prep, allow_cpu=False):
    import torch
    import gpu_arm as G
    import model as M
    t0 = time.time()
    d = Path(out) / 'shards' / f'{i:05d}'
    need((d / 'DONE').exists(), f'shard {i} is not DONE')
    done = json.loads((d / 'DONE').read_text())
    for f, h in done['files'].items():
        need(R.sha256_file(d / f) == h, f'{f} does not match its DONE hash')
    rec = json.loads((d / 'run.json').read_text())
    srcs = {s: R.sha256_file(HERE / s) for s in G.GPU_SOURCES}
    need(rec['runtime']['sources'] == srcs, 'sources changed since the shard ran')
    need(rec['prep_sha256'] == G.prep_hashes(Path(prep)), 'concepts or references differ from the shard\'s')
    tasks, refs, _ = G.load_prep(prep)
    job = G.shard_job(c, tasks, i)
    need(job == rec['job'], f'shard map differs: {job} vs {rec["job"]}')
    print(f'shard {i}: 1-2 files, sources and prep match ({time.time() - t0:.0f}s)', flush=True)
    # 3. training stream
    n_docs, L, Bd = rec['n_docs'], c['doc_len'], c['batch_docs']
    args = G.stream_args(c, tasks, job, n_docs)
    stream = G.Stream(c['data_workers'], args, n_docs // Bd, depth=4 * c['data_workers'])
    sha = hashlib.sha256()
    try:
        for _ in range(n_docs // Bd):
            sha.update(stream.get().tobytes())
    finally:
        stream.close()
    need(sha.hexdigest() == rec['stream_sha256'], 'training stream hash differs')
    cids, _ = H.schedule(job['seed'], args[2], args[3], job['alpha'], n_docs)
    counts = {}
    for T in rec['T_checkpoints']:
        m = T // L
        counts[str(T)] = {str(int(x)): int((cids[:m] == x).sum()) for x in np.unique(cids[:m])}
    need(counts == rec['counts'], 'per-checkpoint document counts differ')
    print(f'shard {i}: 3 training stream regenerates ({time.time() - t0:.0f}s)', flush=True)
    # 4. checkpoint replay
    device, hw = G.device_or_die(allow_cpu)
    docs = refs['docs'][job['target_id']]
    oc = refs['oracle'][job['target_id']]
    rows = {(r['T'], r['slice']): r for r in rec['rows']}
    curves = {p['T']: np.array(p['model']) for p in rec['per_position']}
    worst = 0.0
    for T in rec['T_checkpoints']:
        m = M.load_npz(M.build_model(G.model_config(c, job['size'])), d / 'checkpoints' / rec['run'] / f'T_{T}.npz')
        cl = G.code_lengths(m, docs, c['V'], device)
        for s in R.SLICES:
            sm = R.slice_mask(c, s)
            r = rows[(T, s)]
            worst = max(worst, abs(cl[:, sm].mean() - r['model_bits']),
                        abs(cl[:, sm].mean() - oc[:, sm].mean() - r['excess_bits']))
        worst = max(worst, float(np.max(np.abs(cl.mean(0) - curves[T]))))
        del m
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    need(worst < c['verify_tol_bits'], f'checkpoint replay differs by {worst} bits')
    print(f'shard {i}: 4 {len(rec["T_checkpoints"])} checkpoints replayed (max abs diff {worst:.2e} bits)', flush=True)
    R.atomic_write_text(d / 'VERIFIED', json.dumps(dict(
        result='PASS', shard=i, run=rec['run'], max_abs_diff_bits=worst, done_sha256=R.sha256_file(d / 'DONE'),
        replay_hardware=hw, trained_on=rec['hardware'], seconds=round(time.time() - t0, 1),
        date=time.strftime('%m-%d-%Y %H:%M:%S')), indent=1))
    print(f'PASS shard {i} ({time.time() - t0:.0f}s)')


def _logspace_job(args):
    """Independent log-space oracle and mixture code lengths for one target (or pilot) concept."""
    defs, cid, dist_cids, docs, alphas = args
    cm = {int(k): H.from_json(v) for k, v in defs.items()}
    own = H.forward_logspace(cm[cid], docs)
    codes = [own] + [H.forward_logspace(cm[x], docs) for x in dist_cids]
    ln2 = math.log(2.0)
    mix = {}
    for a in alphas:
        if a >= 1:
            mix[a] = own
            continue
        prior = R.prior_for(a, len(dist_cids))
        tot = np.stack([math.log(w) - np.cumsum(cd, 1) * ln2 for cd, w in zip(codes, prior)])
        m = tot.max(0)
        lt = m + np.log(np.exp(tot - m).sum(0))
        o = np.empty_like(lt)
        o[:, 0] = -lt[:, 0] / ln2
        o[:, 1:] = -(lt[:, 1:] - lt[:, :-1]) / ln2
        mix[a] = o
    return cid, own, mix


def verify_merged(c, out, workers):
    import analyze as AN
    import gpu_arm as G
    t0 = time.time()

    def note(m):
        print(f'[{time.time() - t0:7.0f}s] {m}', flush=True)
    man = json.loads((out / 'manifest.json').read_text())
    srcs = {s: R.sha256_file(HERE / s) for s in G.GPU_SOURCES}
    need(man['sources'] == srcs, 'sources changed since the shards ran')
    note('0 sources match the shards')
    tasks = json.loads((out / 'tasks.json').read_text())
    regen, _ = R.build_concepts(c, log=lambda m: None)
    need(len(regen) == len(tasks), 'concept count differs')
    for t, g in zip(tasks, regen):
        h = H.from_json(t['definition'])
        need(H.same(h, H.from_json(g['definition'])) and t['task_id'] == g['task_id'], f'{t["task_id"]} differs')
        need(H.process_rank(h)[0] == t['K'] == t['process_rank'], f'rank not certified for {t["task_id"]}')
        need(len(H.encode(h)) == t['K_bits'] == H.K_bits(h['n'], h['V'], h['q']), f'K_bits {t["task_id"]}')
        n_docs = c['eval_docs_per_target'] if t['role'] != 'distractor' else c['distractor_eval_docs']
        dd = np.load(out / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
        need(np.array_equal(dd, R.eval_docs(c, h, t['cid'], n_docs)), f'eval docs differ {t["task_id"]}')
    note(f'1 {len(tasks)} concepts regenerate; ranks certified; K_bits recomputed; eval docs regenerate')
    o = np.load(out / 'oracle.npz')
    mx = np.load(out / 'mixture.npz')
    defs = {str(t['cid']): t['definition'] for t in tasks}
    dist = [x['cid'] for x in R.distractors(tasks)]
    byc = {t['cid']: t for t in tasks}
    worst = 0.0
    jobs = []
    for t in tasks:
        dd = np.load(out / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
        if t['role'] == 'distractor':
            worst = max(worst, float(np.max(np.abs(H.forward_logspace(H.from_json(t['definition']), dd)
                                                   - o[t['task_id']]))))
        else:
            jobs.append((defs, t['cid'], dist, dd, c['alpha_grid']))
    with cf.ProcessPoolExecutor(workers) as ex:
        for cid, own, mix in ex.map(_logspace_job, jobs):
            tid = byc[cid]['task_id']
            worst = max(worst, float(np.max(np.abs(own - o[tid]))))
            for a, v in mix.items():
                worst = max(worst, float(np.max(np.abs(v - mx[f'{tid}|{a:.2f}']))))
    need(worst < c['verify_tol_bits'], f'oracle/mixture recomputation differs by {worst}')
    note(f'2 oracle and mixture code lengths agree (max abs diff {worst:.2e} bits)')
    shards = json.loads((out / 'shards.json').read_text())
    need(len(shards) == c['n_shards'], f'{len(shards)} shards merged')
    for s in shards:
        v = s['verified']
        need(v is not None and v['result'] == 'PASS', f'shard {s["shard"]} has no VERIFIED marker')
        need(v['done_sha256'] == s['done_sha256'], f'shard {s["shard"]}: VERIFIED does not match DONE')
        need(v['max_abs_diff_bits'] < c['verify_tol_bits'], f'shard {s["shard"]} replay diff {v["max_abs_diff_bits"]}')
    mixes = json.loads((out / 'mixes.json').read_text())
    need(len(mixes) == c['n_shards'], f'{len(mixes)} runs in mixes.json')
    note(f'3 {len(shards)} shards VERIFIED (stream regenerated, checkpoints replayed; max diff '
         f'{max(s["verified"]["max_abs_diff_bits"] for s in shards):.2e} bits)')
    tmp = Path(tempfile.mkdtemp(prefix='exp13_verify_gpu_'))
    for n in ('metrics.csv', 'tasks.json', 'distractor_loss.csv', 'posterior16.json'):
        shutil.copy2(out / n, tmp / n)
    ca = AN.load_config('gpu')
    import analyze_gpu as AG
    AG.stage1(ca, tmp, tasks)
    F0 = json.loads((out / 'fits.json').read_text())
    F1 = json.loads((tmp / 'fits.json').read_text())
    need(F0['meta']['forecasts_sha256'] == F1['meta']['forecasts_sha256'], 'forecast hash differs on refit')
    for k, f0 in F0['fits'].items():
        for p, v in f0['params'].items():
            w = F1['fits'][k]['params'][p]
            need(abs(v - w) <= 1e-4 * max(abs(v), abs(w)) + 1e-9, f'refit differs: {k} {p} {v} vs {w}')
    note(f'4 {len(F0["fits"])} law fits reproduce to 1e-4; forecasts hash identical')
    AG.stage2(ca, tmp, tasks)
    G0 = json.loads((out / 'gates.json').read_text())
    G1 = json.loads((tmp / 'gates.json').read_text())
    need(G0['gates'] == G1['gates'], 'gate verdicts differ on rerun')
    A1 = json.loads((tmp / 'analysis.json').read_text())
    need(A1['independent_fit']['agree'], f'independent fitter disagrees: {A1["independent_fit"]["disagreements"][:1]}')
    note('4 gate verdicts identical on rerun; independent fitter AGREE')
    shutil.rmtree(tmp, ignore_errors=True)
    return t0


def main(args):
    import os
    c = R.load_config('gpu')
    prep = Path(args.prep) if args.prep else HERE / 'prep' / 'gpu'
    try:
        if args.shard is not None:
            if not args.output:
                raise SystemExit('--shard needs --output DIR (the launch output folder)')
            verify_shard(c, args.output, args.shard, prep, os.environ.get('TDC_ALLOW_CPU') == '1')
            return
        out = Path(args.output) if args.output else HERE / 'results' / 'gpu'
        t0 = verify_merged(c, out, args.workers)
    except Fail as e:
        print(f'FAIL: {e}')
        sys.exit(1)
    (out / 'verify_log.json').write_text(json.dumps(dict(result='PASS', seconds=round(time.time() - t0, 1)), indent=1))
    print(f'PASS ({time.time() - t0:.0f}s)')
