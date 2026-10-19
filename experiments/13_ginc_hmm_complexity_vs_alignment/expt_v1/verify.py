"""Experiment 13 verification (PROTOCOL Section 8). Prints PASS or the first failure.

Usage (from expt_v1/): verify.py --arm small [--output results/small] [--workers 4]
Checks:
  0. training sources match manifest.json;
  1. every concept regenerates identically from the task seed; every process rank is recertified (= n) and every
     K_bits recomputed from the stored definition (len(encode)); held-out documents regenerate identically;
  2. oracle and mixture code lengths recomputed with the independent log-space forward algorithm to 1e-9;
  3. every run's training-stream SHA-256 and per-checkpoint document counts regenerate from seeds;
  4. every checkpoint is reloaded and the held-out model code lengths are recomputed in float64: per-slice means
     (metrics.csv) and the 128-position curve (per_position.csv) to max_abs_diff < 1e-9 bits;
  5. every law is refit (analyze.py stage 1 into a scratch copy): forecasts hash and fitted parameters agree with
     fits.json to 1e-4 relative; stage 2 rerun gives identical gate verdicts; the independent fitter agrees.
"""
import argparse
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


def _ckpt_job(args):
    import torch
    import model as M
    torch.set_num_threads(1)
    out, run_name, target_id, Ts, mc, V = args
    out = Path(out)
    docs = np.load(out / 'data' / f'{target_id}_test.npz')['docs'].astype(np.int64)
    res = []
    for T in Ts:
        m = M.load_npz(M.build_model(mc), out / 'checkpoints' / run_name / f'T_{T}.npz')
        res.append((T, M.code_lengths(m, docs, V)))
    return run_name, res


def _stream_job(args):
    defs, seed, target_cid, dist_cids, alpha, n_docs, L, B, T_list = args
    cm = {int(k): H.from_json(v) for k, v in defs.items()}
    cids, js = H.schedule(seed, target_cid, dist_cids, alpha, n_docs)
    h = hashlib.sha256()
    for b in H.stream_batches(cm, seed, cids, js, L, B):
        h.update(b.astype(np.int16).tobytes())
    counts = {}
    for T in T_list:
        m = T // L
        counts[str(T)] = {str(int(c)): int((cids[:m] == c).sum()) for c in np.unique(cids[:m])}
    return h.hexdigest(), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='small')
    ap.add_argument('--output', default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--skip-checkpoints', action='store_true', help='debug only; never for the PASS of record')
    a = ap.parse_args()
    c = R.load_config(a.arm)
    out = Path(a.output) if a.output else HERE / 'results' / a.arm
    t0 = time.time()
    log = []

    def note(m):
        print(f'[{time.time() - t0:7.0f}s] {m}', flush=True)
        log.append(m)
    try:
        man = json.loads((out / 'manifest.json').read_text())
        for s in R.SOURCES:
            need(man['sources'].get(s) == R.sha256_file(HERE / s), f'source changed since the grid: {s}')
        note('0 sources match manifest')
        tasks = json.loads((out / 'tasks.json').read_text())
        regen, _ = R.build_concepts(c, log=lambda m: None)
        need(len(regen) == len(tasks), 'concept count differs')
        for t, g in zip(tasks, regen):
            h = H.from_json(t['definition'])
            need(H.same(h, H.from_json(g['definition'])) and t['task_id'] == g['task_id'], f'{t["task_id"]} differs')
            need(H.process_rank(h)[0] == t['K'] == t['process_rank'], f'rank not certified for {t["task_id"]}')
            need(len(H.encode(h)) == t['K_bits'] == H.K_bits(h['n'], h['V'], h['q']), f'K_bits {t["task_id"]}')
            n_docs = c['eval_docs_per_target'] if t['role'] != 'distractor' else c['distractor_eval_docs']
            d = np.load(out / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
            need(np.array_equal(d, R.eval_docs(c, h, t['cid'], n_docs)), f'eval docs differ {t["task_id"]}')
        note(f'1 {len(tasks)} concepts regenerate; ranks certified; K_bits recomputed; eval docs regenerate')
        cm = R.concept_map(tasks)
        dist = R.distractors(tasks)
        o = np.load(out / 'oracle.npz')
        mx = np.load(out / 'mixture.npz')
        worst = 0.0
        for t in tasks:
            d = np.load(out / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)
            ol = H.forward_logspace(cm[t['cid']], d)
            worst = max(worst, float(np.max(np.abs(ol - o[t['task_id']]))))
            if t['role'] == 'distractor':
                continue
            for al in c['alpha_grid']:
                key = f"{t['task_id']}|{al:.2f}"
                if key not in mx.files:
                    continue
                concepts = [cm[t['cid']]] + ([cm[x['cid']] for x in dist] if al < 1 else [])
                ml = H.mixture_logspace(concepts, R.prior_for(al, len(dist)), d)
                worst = max(worst, float(np.max(np.abs(ml - mx[key]))))
        need(worst < c['verify_tol_bits'], f'oracle/mixture recomputation differs by {worst}')
        note(f'2 oracle and mixture code lengths agree (max abs diff {worst:.2e} bits)')
        mixes = json.loads((out / 'mixes.json').read_text())
        byid = {t['task_id']: t for t in tasks}
        defs = {str(t['cid']): t['definition'] for t in tasks}
        Ts = c['T_checkpoints']
        jobs, names = [], []
        for name, m in sorted(mixes.items()):
            jobs.append((defs, m['seed'], byid[m['target_id']]['cid'], [x['cid'] for x in dist], m['alpha'],
                         m['n_docs'], c['doc_len'], c['batch_docs'], Ts))
            names.append(name)
        need(len(names) == len(R.grid_runs(c, tasks)), f'{len(names)} runs in mixes.json')
        with cf.ProcessPoolExecutor(a.workers) as ex:
            for name, (hx, counts) in zip(names, ex.map(_stream_job, jobs)):
                need(hx == mixes[name]['stream_sha256'], f'stream hash differs for {name}')
                need(counts == mixes[name]['counts'], f'document counts differ for {name}')
        note(f'3 {len(names)} training streams regenerate (hash and counts)')
        metrics = {}
        with open(out / 'metrics.csv') as f:
            for r in csv.DictReader(f):
                metrics[(R.run_name(int(r['seed']), r['target_id'], float(r['alpha'])), int(r['T']), r['slice'])] = r
        perpos = {}
        with open(out / 'per_position.csv') as f:
            rd = csv.reader(f)
            next(rd)
            for row in rd:
                perpos[(R.run_name(int(row[0]), row[1], float(row[2])), int(row[3]))] = np.array(row[4:], float)
        if not a.skip_checkpoints:
            ck_jobs = [(str(out), name, mixes[name]['target_id'], Ts, c['model'], c['V']) for name in names]
            worst = 0.0
            done = 0
            with cf.ProcessPoolExecutor(a.workers) as ex:
                for name, res in ex.map(_ckpt_job, ck_jobs):
                    tid = mixes[name]['target_id']
                    oc = o[tid]
                    for T, cl in res:
                        for s in R.SLICES:
                            sm = R.slice_mask(c, s)
                            r = metrics[(name, T, s)]
                            worst = max(worst, abs(cl[:, sm].mean() - float(r['model_bits'])),
                                        abs(cl[:, sm].mean() - oc[:, sm].mean() - float(r['excess_bits'])))
                        worst = max(worst, float(np.max(np.abs(cl.mean(0) - perpos[(name, T)]))))
                    done += 1
                    if done % 16 == 0:
                        note(f'4 ... {done}/{len(ck_jobs)} runs replayed, max diff so far {worst:.2e}')
            need(worst < c['verify_tol_bits'], f'checkpoint replay differs by {worst} bits')
            note(f'4 {len(ck_jobs) * len(Ts)} checkpoints replayed (max abs diff {worst:.2e} bits)')
        import analyze as AN
        tmp = Path(tempfile.mkdtemp(prefix='exp13_verify_'))
        for n in ('metrics.csv', 'tasks.json', 'distractor_loss.csv', 'posterior16.json'):
            shutil.copy2(out / n, tmp / n)
        ca = AN.load_config(a.arm)
        AN.stage1(ca, tmp, tasks)
        F0 = json.loads((out / 'fits.json').read_text())
        F1 = json.loads((tmp / 'fits.json').read_text())
        need(F0['meta']['forecasts_sha256'] == F1['meta']['forecasts_sha256'], 'forecast hash differs on refit')
        for k, f0 in F0['fits'].items():
            f1 = F1['fits'][k]
            for p, v in f0['params'].items():
                w = f1['params'][p]
                need(abs(v - w) <= 1e-4 * max(abs(v), abs(w)) + 1e-9, f'refit differs: {k} {p} {v} vs {w}')
        note(f'5 {len(F0["fits"])} law fits reproduce to 1e-4; forecasts hash identical')
        AN.stage2(ca, tmp, tasks)
        G0 = json.loads((out / 'gates.json').read_text())
        G1 = json.loads((tmp / 'gates.json').read_text())
        need(G0['gates'] == G1['gates'], 'gate verdicts differ on rerun')
        A1 = json.loads((tmp / 'analysis.json').read_text())
        need(A1['independent_fit']['agree'], f'independent fitter disagrees: {A1["independent_fit"]["disagreements"][:1]}')
        note('5 gate verdicts identical on rerun; independent fitter AGREE')
        shutil.rmtree(tmp, ignore_errors=True)
    except Fail as e:
        print(f'FAIL: {e}')
        sys.exit(1)
    (out / 'verify_log.json').write_text(json.dumps(dict(result='PASS', seconds=round(time.time() - t0, 1),
                                                         steps=log), indent=1))
    print(f'PASS ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
