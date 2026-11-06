"""Experiment 24 verification (PROTOCOL Section 6). Prints PASS or the first failure.

Per shard, on its GPU right after training: verify24.py --shard I --output OUT
  every file in DONE matches its SHA-256; the shard ran the current sources and prep; the training stream regenerates
  from seeds (SHA-256 and per-checkpoint document counts); every checkpoint is replayed in float64 with deterministic
  algorithms, per-slice means and the 128-position curve to < 1e-9 bits. Writes VERIFIED, then deletes the
  checkpoints not in keep_checkpoints (their SHA-256s stay in DONE and are listed in VERIFIED).
Merged, on CPU: verify24.py [--results DIR] [--workers W]
  every concept (targets, pilot concepts, all 148 pool members) regenerates from its seed with certified rank and
  recomputed K_bits; held-out documents regenerate; oracle and mixture code lengths agree with the independent
  log-space forward algorithm to 1e-9; all 918 shards carry a VERIFIED marker matching their DONE file; the analysis
  reruns to identical verdicts and point estimates (1e-9) and the independent Tobit fitter agrees.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run24 as R  # noqa: E402  (also puts Experiment 13's expt_v1 on the path)
import concepts24 as C  # noqa: E402
import hmm as H  # noqa: E402


class Fail(Exception):
    pass


def need(c, msg):
    if not c:
        raise Fail(msg)


def verify_shard(cfg, out, i, prep, allow_cpu=False, delete=True):
    import torch
    import gpu_arm as G
    import model as M
    t0 = time.time()
    d = Path(out) / 'shards' / f'{i:05d}'
    need((d / 'DONE').exists(), f'shard {i} is not DONE')
    if (d / 'VERIFIED').exists():
        print(f'shard {i} already VERIFIED')
        return
    done = json.loads((d / 'DONE').read_text())
    keep = set(cfg['grid']['keep_checkpoints'])
    for f, h in done['files'].items():
        if not (d / f).exists():  # only a checkpoint removed after an earlier replay may be missing
            need(f.startswith('checkpoints/') and int(f.rsplit('T_', 1)[1].split('.')[0]) not in keep,
                 f'{f} is missing')
            continue
        need(C.sha256_file(d / f) == h, f'{f} does not match DONE')
    rec = json.loads((d / 'run.json').read_text())
    need(rec['runtime']['sources'] == R.sources_sha(), 'sources changed since the shard ran')
    need(rec['prep_sha256'] == C.prep_hashes(prep), 'prep differs from the shard\'s')
    tasks, refs, _ = R.load_prep(prep)
    job = R.shard_job(cfg, tasks, i)
    need(job == rec['job'], f'shard map differs: {job} vs {rec["job"]}')
    target = next(t for t in tasks if t['task_id'] == job['target_id'])
    pool = C.pool_of(tasks, job['target_id'], job['similarity']) if job['alpha'] < 1 else []
    need([p['task_id'] for p in pool] == rec['pool'], 'pool differs')
    n_docs, L, Bd = rec['n_docs'], cfg['concepts']['doc_len'], cfg['model']['batch_docs']
    defs = {str(t['cid']): t['definition'] for t in [target] + pool}
    dist = [p['cid'] for p in pool]
    st = G.Stream(cfg['model']['data_workers'], (defs, job['seed'], target['cid'], dist, job['alpha'], n_docs, L, Bd),
                  n_docs // Bd, depth=4 * cfg['model']['data_workers'])
    sha = hashlib.sha256()
    try:
        for _ in range(n_docs // Bd):
            sha.update(st.get().tobytes())
    finally:
        st.close()
    need(sha.hexdigest() == rec['stream_sha256'], 'training stream hash differs')
    cids, _ = H.schedule(job['seed'], target['cid'], dist, job['alpha'], n_docs)
    counts = {str(T): {str(int(c)): int((cids[:T // L] == c).sum()) for c in np.unique(cids[:T // L])}
              for T in rec['T_checkpoints']}
    need(counts == rec['counts'], 'document counts differ')
    device, hw = G.device_or_die(allow_cpu)
    V = cfg['concepts']['V_model_tokens']
    docs, oc = refs['docs'][job['target_id']], refs['oracle'][job['target_id']]
    rows = {(r['T'], r['slice']): r for r in rec['rows']}
    curves = {p['T']: np.array(p['model']) for p in rec['per_position']}
    worst = 0.0
    replayed = []
    for T in rec['T_checkpoints']:
        ck = d / 'checkpoints' / rec['run'] / f'T_{T}.npz'
        if not ck.exists():
            continue
        replayed.append(T)
        m = M.load_npz(M.build_model(R.model_config(cfg, job['size'])), ck)
        cl = G.code_lengths(m, docs, V, device)
        for s in R.SLICES:
            lo, hi = cfg['eval']['slices'][s]
            sm = np.zeros(L, bool)
            sm[lo - 1:hi] = True
            r = rows[(T, s)]
            worst = max(worst, abs(cl[:, sm].mean() - r['model_bits']),
                        abs(cl[:, sm].mean() - oc[:, sm].mean() - r['excess_bits']))
        worst = max(worst, float(np.max(np.abs(cl.mean(0) - curves[T]))))
        del m
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    need(worst < cfg['analysis']['verify_tol_bits'], f'checkpoint replay differs by {worst} bits')
    need((keep & set(rec['T_checkpoints'])) <= set(replayed), 'a kept checkpoint is missing')
    removed = []
    if delete:
        for T in rec['T_checkpoints']:
            p = d / 'checkpoints' / rec['run'] / f'T_{T}.npz'
            if T not in keep and p.exists():
                p.unlink()
                removed.append(f'checkpoints/{rec["run"]}/T_{T}.npz')
    R.R13.atomic_write_text(d / 'VERIFIED', json.dumps(dict(
        result='PASS', shard=i, run=rec['run'], max_abs_diff_bits=worst, done_sha256=C.sha256_file(d / 'DONE'),
        replay_hardware=hw, replayed=replayed, removed_after_replay=removed, seconds=round(time.time() - t0, 1),
        date=time.strftime('%m-%d-%Y %H:%M:%S')), indent=1))
    print(f'PASS shard {i} (max abs diff {worst:.2e} bits, {time.time() - t0:.0f}s)')


def _logspace_job(args):
    defs, cid, pools, docs, alphas = args
    cm = {int(k): H.from_json(v) for k, v in defs.items()}
    own = H.forward_logspace(cm[cid], docs)
    out = {}
    ln2 = math.log(2.0)
    for sim, pc in pools.items():
        codes = [own] + [H.forward_logspace(cm[x], docs) for x in pc]
        for a in alphas:
            if a >= 1:
                continue
            prior = C.prior_for(a, len(pc))
            tot = np.stack([math.log(w) - np.cumsum(c, 1) * ln2 for c, w in zip(codes, prior)])
            mx = tot.max(0)
            lt = mx + np.log(np.exp(tot - mx).sum(0))
            o = np.empty_like(lt)
            o[:, 0] = -lt[:, 0] / ln2
            o[:, 1:] = -(lt[:, 1:] - lt[:, :-1]) / ln2
            out[f'{sim}|{a:.2f}'] = o
    return cid, own, out


def verify_merged(cfg, res, workers):
    import analyze24 as A
    t0 = time.time()

    def note(m):
        print(f'[{time.time() - t0:7.0f}s] {m}', flush=True)
    man = json.loads((res / 'manifest.json').read_text())
    need(man['sources'] == R.sources_sha(), 'sources changed since the shards ran')
    note('0 sources match the shards')
    tasks = json.loads((res / 'tasks.json').read_text())
    regen, _ = C.build_all(cfg, log=lambda m: None)
    need(len(regen) == len(tasks), 'concept count differs')
    for t, g in zip(tasks, regen):
        h = H.from_json(t['definition'])
        need(H.same(h, H.from_json(g['definition'])) and t['task_id'] == g['task_id'], f'{t["task_id"]} differs')
        need(H.process_rank(h)[0] == t['K'] == t['process_rank'], f'rank not certified: {t["task_id"]}')
        need(len(H.encode(h)) == t['K_bits'] == H.K_bits(h['n'], h['V'], h['q']), f'K_bits {t["task_id"]}')
        nd = cfg['eval']['distractor_eval_docs'] if t['role'] == 'distractor' else cfg['eval']['eval_docs_per_target']
        dd = np.load(res / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
        need(np.array_equal(dd, C.eval_docs(cfg, h, t['cid'], nd)), f'eval docs differ {t["task_id"]}')
    note(f'1 {len(tasks)} concepts regenerate; ranks certified; K_bits recomputed; eval docs regenerate')
    o, mx = np.load(res / 'oracle.npz'), np.load(res / 'mixture.npz')
    defs = {str(t['cid']): t['definition'] for t in tasks}
    worst = 0.0
    jobs = []
    for t in tasks:
        dd = np.load(res / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
        if t['role'] == 'distractor':
            worst = max(worst, float(np.max(np.abs(H.forward_logspace(H.from_json(t['definition']), dd)
                                                   - o[t['task_id']]))))
            continue
        pools = {s: [p['cid'] for p in C.pool_of(tasks, t['task_id'], s)] for s in cfg['concepts']['similarity_levels']}
        jobs.append((defs, t['cid'], {s: v for s, v in pools.items() if v}, dd, cfg['grid']['alphas']))
    byc = {t['cid']: t for t in tasks}
    with cf.ProcessPoolExecutor(workers) as ex:
        for cid, own, mix in ex.map(_logspace_job, jobs):
            tid = byc[cid]['task_id']
            worst = max(worst, float(np.max(np.abs(own - o[tid]))))
            for k, v in mix.items():
                worst = max(worst, float(np.max(np.abs(v - mx[f'{tid}|{k}']))))
    need(worst < cfg['analysis']['verify_tol_bits'], f'oracle/mixture recomputation differs by {worst}')
    note(f'2 oracle and mixture agree with the log-space algorithm (max abs diff {worst:.2e} bits)')
    sh = json.loads((res / 'shards.json').read_text())
    need(len(sh) == R.n_shards(cfg), f'{len(sh)} shards merged')
    for s in sh:
        v = s['verified']
        need(v is not None and v['result'] == 'PASS' and v['done_sha256'] == s['done_sha256'],
             f'shard {s["shard"]} lacks a matching VERIFIED marker')
    note(f'3 {len(sh)} shards VERIFIED (max replay diff {max(s["verified"]["max_abs_diff_bits"] for s in sh):.2e})')
    tmp = Path(tempfile.mkdtemp(prefix='exp24_verify_'))
    for n in ('metrics.csv', 'tasks.json', 'distractor_loss.csv', 'posterior16.json', 'oracle.npz', 'mixture.npz'):
        shutil.copy2(res / n, tmp / n)
    if (res / 'task2vec.json').exists():
        shutil.copy2(res / 'task2vec.json', tmp / 'task2vec.json')
    A0 = json.loads((res / 'analysis.json').read_text())
    runs = A.read_runs(tmp)
    tab = A.tstar_table(runs, cfg['analysis']['eps_star'])
    point, fits = A.stats(tab)
    for k, v in A0['point'].items():
        need(abs(point[k] - v) <= 1e-6 * max(1.0, abs(v)) or (math.isinf(v) and math.isinf(point[k])),
             f'point estimate {k} differs on rerun: {v} vs {point[k]}')
    G0 = json.loads((res / 'gates.json').read_text())
    point2, fits2, per_seed, boot, g = A.full_analysis(cfg, tab, cfg['analysis']['bootstrap_draws'], workers,
                                                       cfg['analysis']['analysis_seed'])
    v1 = {k: v['verdict'] for k, v in g.items() if isinstance(v, dict) and 'verdict' in v}
    need(v1 == G0['verdicts'], f'gate verdicts differ on rerun: {v1} vs {G0["verdicts"]}')
    ind = A.independent_check(tab, fits, cfg['analysis']['independent_fit_tol'])
    need(ind['agree'], f'independent Tobit fitter disagrees: {ind["disagreements"][:2]}')
    note('4 point estimates and gate verdicts identical on rerun; independent fitter AGREE')
    shutil.rmtree(tmp, ignore_errors=True)
    return t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int)
    ap.add_argument('--output')
    ap.add_argument('--prep')
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    cfg = C.load_config()
    prep = Path(a.prep) if a.prep else HERE / 'prep'
    try:
        if a.shard is not None:
            verify_shard(cfg, a.output, a.shard, prep, os.environ.get('TDC_ALLOW_CPU') == '1')
            return
        res = Path(a.results)
        t0 = verify_merged(cfg, res, a.workers)
    except Fail as e:
        print(f'FAIL: {e}')
        sys.exit(1)
    (res / 'verify_log.json').write_text(json.dumps(dict(result='PASS', seconds=round(time.time() - t0, 1)), indent=1))
    print(f'PASS ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
