"""Experiment 23 runner: Experiment 13-small and Experiment 17b's arms trained to convergence, 8 seeds (PROTOCOL.md).

Same concepts, eval documents, exact references, 354,240-parameter transformer, optimizer, batch and alpha schedule as
Experiments 13-small and 17b (their code is imported unchanged: Experiment 17's hmm.py, model.py and run.py; its hmm.py
is Experiment 13's plus two region helpers). What changes: every run trains in stages to 2^20, 2^22, 2^24 and at most
2^26 tokens, and a pre-registered grid-level stopping rule (PROTOCOL Section 3) ends the grid when it has converged.
Training state (weights, AdamW moments, scheduler) is saved at every stage end, so stage k+1 resumes exactly.

Usage (venv python, from this folder; OUT is one output folder per experiment, e.g. ~/e23/out/e13):
  run23.py build-prep --prep prep                         (repo side) copy the frozen Exp 13 / 17b references into
                                                          prep/<exp>/ and add the validation set (val.npz)
  run23.py init  --exp e13 --prep prep --output OUT       check prep sha256s and copy it into OUT
  run23.py stage --exp e13 --stage K --task I --output OUT [--workers W]
                                                          one array task: runs I*R..I*R+R-1 of the grid through stage
                                                          K (resume-safe; exits at once if the grid already stopped)
  run23.py decide --exp e13 --stage K --output OUT        stopping rule after stage K: writes decisions/ and STOP
  run23.py aggregate --exp e13 --output OUT               metrics.csv etc. over the stages up to STOP
  run23.py smoke --output DIR                             2 cells (one per experiment) through stage 0, throughput,
                                                          compute estimate
  run23.py estimate [--tokens-per-second X]               compute estimate for the full grid
  run23.py count --exp e13                                runs and array tasks per stage
"""
import argparse
import concurrent.futures as cf
import copy
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
SRC = {k: (HERE / v).resolve() for k, v in CFG['sources'].items()}
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(SRC['exp17']))

import torch  # noqa: E402

import hmm as H  # noqa: E402  Experiment 17's copy: Experiment 13's hmm.py + region helpers
import model as M  # noqa: E402  identical to Experiment 13's model.py
import run as R17  # noqa: E402  Experiment 17's runner (stream_ids, mix_key, make_optimizer, ...)

EXPS = ('e13', 'e17b')
SLICES = R17.SLICES
SOURCES = ['config.json', 'run23.py', 'analyze23.py', 'report23.py', 'verify23.py']
E13_COLS = ['seed', 'target_id', 'K', 'K_bits', 'alpha', 'N', 'T', 'step', 'slice', 'excess_bits', 'ident_bits',
            'learn_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'n_tokens', 'excess_se']
E17_COLS = R17.ROW_COLS
COOL_EXTRA = ['phase', 'T_branch', 'T_total']


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(text)
    os.replace(tmp, path)


def sha256_file(path):
    return R17.sha256_file(path)


def base_config(exp):
    """The frozen training/eval config of the source experiment (Experiment 13 'small', or Experiment 17)."""
    if exp == 'e13':
        raw = json.loads((SRC['exp13'] / 'config.json').read_text())
        c = dict(raw['shared'])
        c.update(raw['small'])
    else:
        c = json.loads((SRC['exp17'] / 'config.json').read_text())
    c['threads'] = CFG['threads_per_run']
    return c


def source_prep(exp):
    """Where the source experiment's frozen references live in a full checkout (built by its own --check-only)."""
    if exp == 'e13':
        return SRC['exp13'] / 'results' / 'small'
    return SRC['exp17b'] / 'prep'


# ---------------------------------------------------------------- the grid (PROTOCOL Section 2)

def targets_of(tasks):
    return [t for t in tasks if t['role'] == 'target']


def grid(exp, tasks):
    """Every run as (seed, pool, target_id, alpha), in a fixed order (array tasks take consecutive blocks)."""
    e = CFG[exp]
    runs = []
    for s in e['seeds']:
        if exp == 'e13':
            for t in targets_of(tasks):
                for a in e['alphas']:
                    runs.append((s, 'D', t['task_id'], float(a)))
        else:
            for pool in e['pools']:
                runs.append((s, pool, '-', 0.0))
                for t in targets_of(tasks):
                    runs.append((s, pool, t['task_id'], float(e['alpha_H'])))
    return runs


def run_name(exp, run):
    s, pool, tid, a = run
    return f'{s}_{tid}_{a:.2f}' if exp == 'e13' else R17.run_name(s, pool, tid, a)


def n_tasks(exp, tasks):
    return math.ceil(len(grid(exp, tasks)) / CFG['runs_per_task'])


def task_runs(exp, tasks, i):
    R = CFG['runs_per_task']
    return grid(exp, tasks)[i * R:(i + 1) * R]


def stream(exp, tasks, run, n_docs):
    """(concept ids, per-concept document indices) of the run's training stream: the source experiment's exact rule."""
    s, pool, tid, a = run
    if exp == 'e13':
        tcid = next(t['cid'] for t in tasks if t['task_id'] == tid)
        dcids = [d['cid'] for d in tasks if d['role'] == 'distractor']
        return H.schedule(s, tcid, dcids, a, n_docs)
    return R17.stream_ids(tasks, s, pool, tid, a, n_docs)


def scored_targets(tasks, run):
    tid = run[2]
    return [t for t in tasks if t['task_id'] == tid] if tid != '-' else targets_of(tasks)


def side_concepts(exp, tasks, run):
    """Concepts whose loss is logged (not gated): Exp 13's distractors, or the run's training pool."""
    if exp == 'e13':
        return [t for t in tasks if t['role'] == 'distractor']
    return R17.by_role(tasks, run[1])


def mixture_codes(exp, refs, tid, pool, alpha):
    if exp == 'e13':
        return refs['mixture'][f'{tid}|{alpha:.2f}']
    if alpha <= 0:
        return refs['oracle'][tid]  # alpha = 0: target unseen, no identification term (Experiment 17's rule)
    return refs['mixture'][R17.mix_key(tid, pool, alpha)]


# ---------------------------------------------------------------- references and the validation set

def load_refs(out, tasks):
    o = np.load(out / 'oracle.npz')
    m = np.load(out / 'mixture.npz')
    docs = {t['task_id']: np.load(out / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64) for t in tasks}
    v = np.load(out / 'val.npz')
    val = {k: v[k] for k in v.files}
    return dict(oracle={k: o[k] for k in o.files}, mixture={k: m[k] for k in m.files}, docs=docs, val=val)


def val_concepts(exp, tasks):
    """Concepts with a validation set: every target; for Exp 17b also every concept of the four training pools."""
    out = targets_of(tasks)
    if exp == 'e17b':
        for pool in CFG['e17b']['pools']:
            out += R17.by_role(tasks, pool)
    return out


def build_val(exp, c, tasks):
    """Validation documents (stream seed eval_seed + offset, never the eval or any training stream) and their exact
    oracle code lengths. Used only by the stopping rule."""
    v = CFG['validation']
    seed = c['eval_seed'] + v['seed_offset']
    arrays = {}
    for t in val_concepts(exp, tasks):
        h = H.from_json(t['definition'])
        n = v['docs_per_target'] if t['role'] == 'target' else v['docs_per_pool_concept']
        docs = H.sample_docs(h, seed, t['cid'], range(n), c['doc_len'])
        arrays[f"docs|{t['task_id']}"] = docs.astype(np.int16)
        arrays[f"oracle|{t['task_id']}"] = H.forward_logprobs(h, docs)
    return arrays


def build_prep(prep_root, sources=None):
    """Copy each source experiment's frozen references (tasks, eval documents, oracle and mixture codes) and add val.npz;
    write prep_manifest.json with every file's sha256. sources: {exp: folder} overriding source_prep (the oracle and
    mixture npz files are not in git; they come from the source experiments' own --check-only builds)."""
    man = {}
    for exp in EXPS:
        src = Path((sources or {}).get(exp) or source_prep(exp))
        dst = Path(prep_root) / exp
        dst.mkdir(parents=True, exist_ok=True)
        for name in CFG[exp]['prep_files']:
            shutil.copy2(src / name, dst / name)
        tasks = json.loads((dst / 'tasks.json').read_text())
        (dst / 'data').mkdir(exist_ok=True)
        for t in tasks:
            shutil.copy2(src / 'data' / f"{t['task_id']}_test.npz", dst / 'data' / f"{t['task_id']}_test.npz")
        with open(dst / 'val.npz', 'wb') as f:
            np.savez(f, **build_val(exp, base_config(exp), tasks))
        man[exp] = {str(p.relative_to(dst)): sha256_file(p) for p in sorted(dst.rglob('*')) if p.is_file()}
    atomic_write_text(Path(prep_root) / 'prep_manifest.json', json.dumps(man, indent=1, sort_keys=True))
    return man


def init_output(exp, prep_root, out):
    """Verify the shipped prep against prep_manifest.json and copy it into the experiment's output folder."""
    man = json.loads((Path(prep_root) / 'prep_manifest.json').read_text())[exp]
    src = Path(prep_root) / exp
    bad = [k for k, h in man.items() if sha256_file(src / k) != h]
    if bad:
        raise SystemExit(f'prep sha256 mismatch for {exp}: {bad[:5]}')
    out.mkdir(parents=True, exist_ok=True)
    for k in man:
        (out / k).parent.mkdir(parents=True, exist_ok=True)
        if not (out / k).exists() or sha256_file(out / k) != man[k]:
            shutil.copy2(src / k, out / k)
    atomic_write_text(out / 'prep_manifest.json', json.dumps(man, indent=1, sort_keys=True))
    print(f'{exp}: {len(man)} prep files verified and in {out}')


# ---------------------------------------------------------------- training (PROTOCOL Sections 2-3)

def make_model(c, seed):
    torch.manual_seed(seed)
    model = M.build_model(c['model'])
    opt = R17.make_optimizer(c, model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c['warmup_steps']))
    return model, opt, sched


def train_step(c, model, opt, batch):
    x, y = M.inputs_targets(batch, c['V'])
    loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, c['V'] + 1), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
    opt.step()


def evaluate(exp, c, tasks, refs, run, model, cids, T, phase, T_branch, T_total):
    """Exact float64 code lengths on each scored target's 512 eval documents (Exp 13/17 measurement), the validation
    monitor for the stopping rule, and the side-concept losses (float32, not gated)."""
    s, pool, run_tid, a = run
    L = c['doc_len']
    m_doc = T_total // L
    rows, perpos = [], []
    for tg in scored_targets(tasks, run):
        tid = tg['task_id']
        docs, oc = refs['docs'][tid], refs['oracle'][tid]
        mc = mixture_codes(exp, refs, tid, pool, a)
        m_codes = M.code_lengths(model, docs, c['V'])
        for sl in SLICES:
            sm = R17.slice_mask(c, sl)
            mb, ob, xb = m_codes[:, sm].mean(), oc[:, sm].mean(), mc[:, sm].mean()
            per_doc = (m_codes[:, sm] - oc[:, sm]).mean(1)
            rows.append(dict(seed=s, pool=pool, run_target=run_tid, target_id=tid, K=tg['K'], K_bits=tg['K_bits'],
                             alpha=a, N=M.nonembedding_count(model), T=T, step=T // (L * c['batch_docs']), slice=sl,
                             excess_bits=float(mb - ob), ident_bits=float(xb - ob), learn_bits=float(mb - xb),
                             model_bits=float(mb), oracle_bits=float(ob), mixture_bits=float(xb),
                             n_tokens=int(sm.sum() * len(docs)),
                             target_docs_seen=int((cids[:m_doc] == tg['cid']).sum()),
                             excess_se=float(per_doc.std(ddof=1) / math.sqrt(len(per_doc))),
                             phase=phase, T_branch=T_branch, T_total=T_total))
        perpos.append(dict(target_id=tid, T=T, phase=phase, model=m_codes.mean(0).tolist()))
    # validation monitor: target excess (alpha > 0) or mean training-pool excess (alpha = 0), all positions
    mon_ids = [t['task_id'] for t in scored_targets(tasks, run)] if a > 0 else \
        [t['task_id'] for t in side_concepts(exp, tasks, run)]
    per = {}
    for tid in mon_ids:
        vd = refs['val'][f'docs|{tid}'].astype(np.int64)
        per[tid] = float((M.code_lengths(model, vd, c['V']) - refs['val'][f'oracle|{tid}']).mean())
    val = dict(T=T, phase=phase, T_total=T_total, monitor_bits=float(np.mean(list(per.values()))), per_concept=per,
               monitor='target' if a > 0 else 'pool')
    side = []
    with torch.no_grad():
        model.eval()
        for d in side_concepts(exp, tasks, run):
            dd = refs['docs'][d['task_id']]
            x, y = M.inputs_targets(dd, c['V'])
            lp = torch.nn.functional.log_softmax(model(x), -1)
            nll = float((-lp.gather(-1, y[..., None])).mean()) / math.log(2.0)
            side.append(dict(T=T, phase=phase, concept_id=d['task_id'], model_bits=nll,
                             oracle_bits=float(refs['oracle'][d['task_id']].mean())))
        model.train()
    return rows, perpos, val, side


def counts_at(cids, m_doc):
    u, n = np.unique(cids[:m_doc], return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u, n)}


def _save_partial(sdir, k, T, model, opt, sched, acc):
    """Training state and records at a checkpoint inside stage k, so an interrupted task resumes there (the .json is
    written last and marks the pair complete)."""
    tmp = sdir / f'partial_{k}.pt.tmp'
    torch.save(dict(T=T, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict()), tmp)
    os.replace(tmp, sdir / f'partial_{k}.pt')
    atomic_write_text(sdir / f'partial_{k}.json', json.dumps(dict(acc, T=T)))


def _load_partial(sdir, k, T0, T1):
    pj, pt = sdir / f'partial_{k}.json', sdir / f'partial_{k}.pt'
    if not (pj.exists() and pt.exists()):
        return None
    acc = json.loads(pj.read_text())
    st = torch.load(pt, weights_only=True)
    if st['T'] != acc['T'] or not (T0 < st['T'] < T1):
        return None
    return st, acc


class _Interrupt(Exception):
    """Test hook: stop a stage right after the partial state at a given T was saved."""


def train_stage(exp, c, tasks, refs, run, k, out, stages=None, ckpts=None, cool_frac=None, save_weights=True,
                _interrupt_at=None):
    """Run `run` through stage k: resume from stage k-1's saved state (or from this stage's last checkpoint if an
    earlier attempt was interrupted), train with the constant learning rate to stages[k] tokens, evaluating at every
    checkpoint, save the state, then the cooldown branch (reported only)."""
    stages = CFG['stages_T'] if stages is None else stages
    ckpts = CFG['T_checkpoints'] if ckpts is None else ckpts
    cool_frac = CFG['cooldown_fraction'] if cool_frac is None else cool_frac
    torch.set_num_threads(c['threads'])
    t0 = time.time()
    s, pool, tid, a = run
    name = run_name(exp, run)
    L, Bd = c['doc_len'], c['batch_docs']
    T0, T1 = (0 if k == 0 else stages[k - 1]), stages[k]
    cd_docs = int(round(T1 * cool_frac)) // L
    assert cd_docs % Bd == 0 and T0 % (L * Bd) == 0 and T1 % (L * Bd) == 0
    cids, js = stream(exp, tasks, run, T1 // L + cd_docs)
    cm = R17.concept_map(tasks)
    model, opt, sched = make_model(c, s)
    sdir = out / 'state' / name
    sdir.mkdir(parents=True, exist_ok=True)
    acc = dict(rows=[], perpos=[], vals=[], side=[], counts={}, segments=[], t_train=0.0, t_eval=0.0, resumed_at=[])
    Tstart = T0
    part = _load_partial(sdir, k, T0, T1)
    if part is not None:
        st, acc = part
        Tstart = st['T']
        acc['resumed_at'] = acc.get('resumed_at', []) + [Tstart]
    elif k > 0:
        st = torch.load(sdir / f'stage_{k - 1}.pt', weights_only=True)
        assert st['T'] == T0, (st['T'], T0)
    if part is not None or k > 0:
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
    t_train_at_start = acc['t_train']
    h = hashlib.sha256()
    model.train()
    gen = H.stream_batches(cm, s, cids[Tstart // L:T1 // L], js[Tstart // L:T1 // L], L, Bd)
    for step in range(Tstart // (L * Bd), T1 // (L * Bd)):
        batch = next(gen)
        h.update(batch.astype(np.int16).tobytes())
        ts = time.time()
        train_step(c, model, opt, batch)
        sched.step()
        acc['t_train'] += time.time() - ts
        T = (step + 1) * Bd * L
        if T in ckpts:
            ts = time.time()
            r, p, v, sd = evaluate(exp, c, tasks, refs, run, model, cids, T, 'const', T, T)
            acc['rows'] += r
            acc['perpos'] += p
            acc['vals'].append(v)
            acc['side'] += sd
            acc['counts'][str(T)] = counts_at(cids, T // L)
            acc['segments'].append([T, h.hexdigest()])  # stream hash per checkpoint segment
            h = hashlib.sha256()
            if save_weights:
                (out / 'checkpoints' / name).mkdir(parents=True, exist_ok=True)
                tmp = out / 'checkpoints' / name / f'T_{T}.npz.tmp'
                M.save_npz(model, tmp)
                os.replace(tmp, out / 'checkpoints' / name / f'T_{T}.npz')
            acc['t_eval'] += time.time() - ts
            if T < T1:
                _save_partial(sdir, k, T, model, opt, sched, acc)
                if _interrupt_at == T:
                    raise _Interrupt(T)
    if T1 not in ckpts:
        acc['segments'].append([T1, h.hexdigest()])
    rows, perpos, vals, side, counts = acc['rows'], acc['perpos'], acc['vals'], acc['side'], acc['counts']
    t_train, t_eval = acc['t_train'], acc['t_eval']
    stream_sha = hashlib.sha256('|'.join(f'{T}:{x}' for T, x in acc['segments']).encode()).hexdigest()
    tmp = sdir / f'stage_{k}.pt.tmp'
    torch.save(dict(T=T1, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict()), tmp)
    os.replace(tmp, sdir / f'stage_{k}.pt')
    for f in (sdir / f'partial_{k}.json', sdir / f'partial_{k}.pt'):
        f.unlink(missing_ok=True)
    # cooldown branch: the same stream continues, learning rate linear from its constant value to 0; discarded after
    ts = time.time()
    cmodel = M.build_model(c['model'])
    cmodel.load_state_dict(model.state_dict())
    copt = R17.make_optimizer(c, cmodel)
    copt.load_state_dict(copy.deepcopy(opt.state_dict()))  # never share moment tensors with the saved state
    n_cd = cd_docs // Bd
    hc = hashlib.sha256()
    cmodel.train()
    gen = H.stream_batches(cm, s, cids[T1 // L:], js[T1 // L:], L, Bd)
    lr0 = sched.get_last_lr()[0]
    for i in range(n_cd):
        for g in copt.param_groups:
            g['lr'] = lr0 * (1.0 - (i + 1) / n_cd)
        batch = next(gen)
        hc.update(batch.astype(np.int16).tobytes())
        train_step(c, cmodel, copt, batch)
    t_cool = time.time() - ts
    ts = time.time()
    crow, cpp, cval, cside = evaluate(exp, c, tasks, refs, run, cmodel, cids, T1, 'cooldown', T1, T1 + cd_docs * L)
    t_eval += time.time() - ts
    rec = dict(run=name, exp=exp, seed=s, pool=pool, target_id=tid, alpha=a, stage=k, T_start=T0, T_end=T1,
               rows=rows, cooldown_rows=crow, per_position=perpos + cpp, val=vals + [cval], side_rows=side + cside,
               counts=counts, stream_sha256=stream_sha, stream_segments=acc['segments'], resumed_at=acc['resumed_at'],
               cooldown_stream_sha256=hc.hexdigest(),
               cooldown_tokens=cd_docs * L, lr_const=lr0,
               train_seconds=round(t_train, 3), cooldown_seconds=round(t_cool, 3), eval_seconds=round(t_eval, 3),
               tokens_per_second_train=(T1 - Tstart) / max(t_train - t_train_at_start, 1e-9),
               wall_seconds=round(time.time() - t0, 3),
               host=socket.gethostname(), torch=torch.__version__,
               finite=bool(all(math.isfinite(r['model_bits']) for r in rows + crow)))
    atomic_write_text(out / 'runs' / name / f'stage_{k}.json', json.dumps(rec))
    return rec


# ---------------------------------------------------------------- array tasks

_W = {}


def _pool(n, **kw):
    """Worker processes started fresh (spawn): a fork after torch has run in the parent can deadlock."""
    import multiprocessing as mp
    return cf.ProcessPoolExecutor(n, mp_context=mp.get_context('spawn'), **kw)


def _init_worker(exp, out, cfg=None):
    global CFG
    if cfg is not None:
        CFG = cfg
    torch.set_num_threads(CFG['threads_per_run'])
    out = Path(out)
    _W.update(exp=exp, out=out, c=base_config(exp), tasks=json.loads((out / 'tasks.json').read_text()))
    _W['refs'] = load_refs(out, _W['tasks'])


def _work(args):
    run, k = args
    rec = train_stage(_W['exp'], _W['c'], _W['tasks'], _W['refs'], tuple(run), k, _W['out'])
    return rec['run'], rec['wall_seconds'], rec['tokens_per_second_train']


def read_stop(out):
    p = out / 'STOP'
    return json.loads(p.read_text()) if p.exists() else None


def cmd_stage(exp, k, i, out, workers):
    stop = read_stop(out)
    if stop is not None and stop['T_stop'] < CFG['stages_T'][k]:
        print(f'{exp} stage {k} task {i}: grid stopped at T = {stop["T_stop"]} ({stop["reason"]}); nothing to do')
        return 0
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = task_runs(exp, tasks, i)
    todo = [r for r in runs if not (out / 'runs' / run_name(exp, r) / f'stage_{k}.json').exists()]
    if k > 0:
        lack = [run_name(exp, r) for r in todo if not (out / 'state' / run_name(exp, r) / f'stage_{k - 1}.pt').exists()]
        if lack:
            print(f'missing stage {k - 1} state for {lack}', file=sys.stderr)
            return 2
    print(f'{exp} stage {k} task {i}: {len(runs) - len(todo)} done, {len(todo)} to run on {socket.gethostname()}',
          flush=True)
    if not todo:
        return 0
    fails = 0
    with _pool(min(workers, len(todo)), initializer=_init_worker, initargs=(exp, str(out), CFG)) as ex:
        futs = {ex.submit(_work, (r, k)): r for r in todo}
        for f in cf.as_completed(futs):
            try:
                name, wall, tps = f.result()
                print(f'done {name} stage {k} {wall:.0f}s train {tps:.0f} tok/s', flush=True)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f'FAILED {run_name(exp, futs[f])} stage {k}: {e!r}', file=sys.stderr, flush=True)
    return 1 if fails else 0


# ---------------------------------------------------------------- the stopping rule (PROTOCOL Section 3)

def monitor_cell(exp, tasks, run):
    s, pool, tid, a = run
    if exp == 'e13':
        K = next(t['K'] for t in tasks if t['task_id'] == tid)
        return f'n={K}|alpha={a:.2f}'
    if a <= 0:
        return f'{pool}|alpha=0.00|pool'
    K = next(t['K'] for t in tasks if t['task_id'] == tid)
    return f'{pool}|alpha={a:.2f}|n={K}'


def monitor_values(out, exp, run, k):
    """{T: monitor_bits} of the constant-learning-rate checkpoints of stages 0..k."""
    vals = {}
    for j in range(k + 1):
        rec = json.loads((out / 'runs' / run_name(exp, run) / f'stage_{j}.json').read_text())
        for v in rec['val']:
            if v['phase'] == 'const':
                vals[v['T']] = v['monitor_bits']
    return vals


def stop_rule(cells, rule):
    """cells: {cell: (before, after)} cell-mean validation excess at T_k / 2^window and at T_k.
    A cell has converged when before - after <= max(abs_bits, rel * after). The grid stops when every cell has."""
    table, ok = {}, True
    for cell, (before, after) in sorted(cells.items()):
        thr = max(rule['abs_bits'], rule['rel'] * after)
        conv = (before - after) <= thr
        ok &= conv
        table[cell] = dict(before=before, after=after, improvement=before - after, threshold=thr, converged=conv)
    return ok, table


def cmd_decide(exp, k, out):
    stop = read_stop(out)
    if stop is not None:
        print(f'{exp}: grid already stopped at T = {stop["T_stop"]} ({stop["reason"]})')
        return 0
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = grid(exp, tasks)
    missing = [run_name(exp, r) for r in runs if not (out / 'runs' / run_name(exp, r) / f'stage_{k}.json').exists()]
    dec = dict(exp=exp, stage=k, T=CFG['stages_T'][k], utc=datetime.now(timezone.utc).isoformat(), missing=missing)
    if missing:
        dec['decision'] = 'HALT: runs missing; resubmit this stage (submit_full.sh --from-stage)'
        atomic_write_text(out / 'decisions' / f'stage_{k}.json', json.dumps(dec, indent=1))
        print(f'{exp} stage {k}: {len(missing)} runs missing, e.g. {missing[:3]}', file=sys.stderr)
        return 1
    rule = CFG['stop_rule']
    Tk = CFG['stages_T'][k]
    Tb = Tk // (2 ** rule['window_doublings'])
    acc = {}
    nonfinite = []
    for r in runs:
        v = monitor_values(out, exp, r, k)
        if not (math.isfinite(v[Tk]) and math.isfinite(v[Tb])):
            nonfinite.append(run_name(exp, r))
        acc.setdefault(monitor_cell(exp, tasks, r), []).append((v[Tb], v[Tk]))
    cells = {cell: (float(np.mean([x[0] for x in xs])), float(np.mean([x[1] for x in xs]))) for cell, xs in acc.items()}
    ok, table = stop_rule(cells, rule)
    final = k == len(CFG['stages_T']) - 1
    dec.update(T_before=Tb, cells=table, all_converged=ok, nonfinite_runs=nonfinite,
               decision='STOP' if (ok or final) else 'CONTINUE', reason='converged' if ok else ('cap' if final else ''))
    atomic_write_text(out / 'decisions' / f'stage_{k}.json', json.dumps(dec, indent=1))
    if ok or final:
        atomic_write_text(out / 'STOP', json.dumps(dict(T_stop=Tk, stage=k, reason=dec['reason'], utc=dec['utc'])))
    n_conv = sum(v['converged'] for v in table.values())
    print(f'{exp} stage {k} (T = {Tk}): {n_conv}/{len(table)} cells converged -> {dec["decision"]} {dec["reason"]}')
    return 0


# ---------------------------------------------------------------- aggregation

def _w(row, cols):
    return {k: (repr(row[k]) if isinstance(row[k], float) else row[k]) for k in cols}


def cmd_aggregate(exp, out):
    stop = read_stop(out)
    if stop is None:
        raise SystemExit(f'{exp}: no STOP yet; the grid has not finished')
    K = stop['stage']
    tasks = json.loads((out / 'tasks.json').read_text())
    runs = grid(exp, tasks)
    recs = {}
    for r in runs:
        n = run_name(exp, r)
        recs[n] = [json.loads((out / 'runs' / n / f'stage_{j}.json').read_text()) for j in range(K + 1)]
    cols = E13_COLS if exp == 'e13' else E17_COLS
    order = sorted(recs, key=lambda n: (recs[n][0]['seed'], recs[n][0]['pool'], recs[n][0]['target_id'],
                                        recs[n][0]['alpha']))
    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for n in order:
            for rec in recs[n]:
                for row in rec['rows']:
                    w.writerow(_w(row, cols))
    with open(out / 'cooldown.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols + COOL_EXTRA)
        w.writeheader()
        for n in order:
            for rec in recs[n]:
                for row in rec['cooldown_rows']:
                    w.writerow(_w(row, cols + COOL_EXTRA))
    with open(out / 'val.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'seed', 'pool', 'run_target', 'alpha', 'T', 'phase', 'T_total', 'monitor', 'monitor_bits'])
        for n in order:
            for rec in recs[n]:
                for v in rec['val']:
                    w.writerow([n, rec['seed'], rec['pool'], rec['target_id'], rec['alpha'], v['T'], v['phase'],
                                v['T_total'], v['monitor'], repr(v['monitor_bits'])])
    with open(out / 'per_position.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'pool', 'run_target', 'target_id', 'alpha', 'T', 'phase'] +
                   [f'p{i + 1}' for i in range(128)])
        for n in order:
            for rec in recs[n]:
                for pp in rec['per_position']:
                    w.writerow([rec['seed'], rec['pool'], rec['target_id'], pp['target_id'], rec['alpha'], pp['T'],
                                pp['phase']] + [repr(x) for x in pp['model']])
    side_name = 'distractor_loss.csv' if exp == 'e13' else 'pool_loss_all.csv'
    with open(out / side_name, 'w', newline='') as f:
        if exp == 'e13':  # Experiment 13's columns (its analysis reads them at Tmax)
            w = csv.DictWriter(f, ['seed', 'target_id', 'alpha', 'T', 'distractor_id', 'model_bits', 'oracle_bits'])
            w.writeheader()
            for n in order:
                for rec in recs[n]:
                    for sd in rec['side_rows']:
                        if sd['phase'] == 'const':
                            w.writerow(dict(seed=rec['seed'], target_id=rec['target_id'], alpha=rec['alpha'], T=sd['T'],
                                            distractor_id=sd['concept_id'], model_bits=sd['model_bits'],
                                            oracle_bits=sd['oracle_bits']))
        else:
            w = csv.writer(f)
            w.writerow(['run', 'T', 'phase', 'pool_id', 'model_bits', 'oracle_bits'])
            for n in order:
                for rec in recs[n]:
                    for sd in rec['side_rows']:
                        w.writerow([n, sd['T'], sd['phase'], sd['concept_id'], repr(sd['model_bits']),
                                    repr(sd['oracle_bits'])])
    if exp == 'e17b':  # Experiment 17's pool_loss.csv: training-pool loss at T_max
        with open(out / 'pool_loss.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run', 'pool_id', 'model_bits', 'oracle_bits'])
            for n in order:
                for sd in recs[n][-1]['side_rows']:
                    if sd['phase'] == 'const' and sd['T'] == stop['T_stop']:
                        w.writerow([n, sd['concept_id'], repr(sd['model_bits']), repr(sd['oracle_bits'])])
    mixes = {n: dict(stage_stream_sha256=[rec['stream_sha256'] for rec in recs[n]],
                     cooldown_stream_sha256=[rec['cooldown_stream_sha256'] for rec in recs[n]],
                     counts={T: cnt for rec in recs[n] for T, cnt in rec['counts'].items()},
                     data_order_seed=[int(recs[n][0]['seed']), 7777]) for n in order}
    atomic_write_text(out / 'mixes.json', json.dumps(mixes, indent=0, sort_keys=True))
    timing = dict(runs=len(order), stages=K + 1, T_stop=stop['T_stop'], reason=stop['reason'],
                  train_hours=sum(rec['train_seconds'] for n in order for rec in recs[n]) / 3600,
                  wall_hours=sum(rec['wall_seconds'] for n in order for rec in recs[n]) / 3600,
                  median_tokens_per_second=float(np.median([rec['tokens_per_second_train'] for n in order
                                                            for rec in recs[n]])),
                  hosts=sorted({rec['host'] for n in order for rec in recs[n]}),
                  nonfinite=[n for n in order if not all(rec['finite'] for rec in recs[n])])
    man = dict(sources={s: sha256_file(HERE / s) for s in SOURCES if (HERE / s).exists()},
               source_experiments={k: {p.name: sha256_file(p) for p in sorted(v.glob('*.py'))} for k, v in SRC.items()},
               versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
               timing=timing, stop=stop)
    atomic_write_text(out / 'manifest.json', json.dumps(man, indent=1))
    print(json.dumps(timing, indent=1))
    return 0


# ---------------------------------------------------------------- smoke and compute estimate

SMOKE_RUNS = {'e13': (13101, 'D', 'hmm_n032_t1', 0.4), 'e17b': (17101, 'cov32', '-', 0.0)}


def _smoke_one(args):
    exp, out = args
    out = Path(out)
    c = base_config(exp)
    tasks = json.loads((out / 'tasks.json').read_text())
    refs = load_refs(out, tasks)
    rec = train_stage(exp, c, tasks, refs, SMOKE_RUNS[exp], 0, out)
    return exp, rec


def cmd_smoke(out_root, prep_root):
    out_root = Path(out_root)
    for exp in EXPS:
        if not (out_root / exp / 'tasks.json').exists():
            init_output(exp, prep_root, out_root / exp)
    summ = {}
    with _pool(2) as ex:
        for exp, rec in ex.map(_smoke_one, [(e, str(out_root / e)) for e in EXPS]):
            all_rows = rec['rows'] + rec['cooldown_rows']
            assert rec['finite'], f'{exp}: non-finite loss'
            last = [r for r in rec['rows'] if r['slice'] == 'all' and r['T'] == rec['T_end']]
            cool = [r for r in rec['cooldown_rows'] if r['slice'] == 'all']
            n_eval = len({(r['target_id'], r['T'], r['phase']) for r in all_rows})
            summ[exp] = dict(run=rec['run'], tokens_per_second_train=rec['tokens_per_second_train'],
                             train_seconds=rec['train_seconds'], cooldown_seconds=rec['cooldown_seconds'],
                             eval_seconds=rec['eval_seconds'], wall_seconds=rec['wall_seconds'],
                             eval_seconds_per_target_checkpoint=rec['eval_seconds'] / n_eval,
                             excess_at_T_end={r['target_id']: r['excess_bits'] for r in last},
                             excess_after_cooldown={r['target_id']: r['excess_bits'] for r in cool},
                             monitor=[(v['T'], v['phase'], round(v['monitor_bits'], 4)) for v in rec['val']],
                             host=rec['host'])
            print(f"{exp} {rec['run']}: {rec['tokens_per_second_train']:.0f} train tok/s, wall {rec['wall_seconds']:.0f}s,"
                  f" excess at 2^20 {summ[exp]['excess_at_T_end']}, after cooldown {summ[exp]['excess_after_cooldown']}")
    tps = float(np.mean([summ[e]['tokens_per_second_train'] for e in EXPS]))
    ev = float(np.mean([summ[e]['eval_seconds_per_target_checkpoint'] for e in EXPS]))
    summ['estimate'] = estimate(tps, ev)
    atomic_write_text(out_root / 'smoke_summary.json', json.dumps(summ, indent=1))
    print(json.dumps(summ['estimate'], indent=1))
    return 0


def estimate(tps, eval_s, data_overhead=1.2):
    """Core-hours and wall time per stage if the grid runs to each stage (constant-LR tokens + cooldowns + evals).
    tps: training tokens/s of one single-thread run; eval_s: seconds per (target, checkpoint) evaluation incl. the
    validation monitor; data_overhead: on-the-fly sampling cost on top of training (measured ~1.2x)."""
    st = CFG['stages_T']
    ck = CFG['T_checkpoints']
    fr = CFG['cooldown_fraction']
    n_runs = {'e13': 256, 'e17b': 160}
    n_targets = {'e13': [1] * 256, 'e17b': ([4] + [1] * 4) * 32}
    res = {}
    for k in range(len(st)):
        T0 = 0 if k == 0 else st[k - 1]
        toks = (st[k] - T0) * (1 + fr)
        n_ck = sum(1 for T in ck if T0 < T <= st[k]) + 1
        per = {}
        for exp in EXPS:
            run_s = [toks / tps * data_overhead + n_ck * nt * eval_s for nt in n_targets[exp]]
            R = CFG['runs_per_task']
            tasks = math.ceil(n_runs[exp] / R)
            task_s = max(run_s)  # runs of a task go in parallel, one core each
            thr = CFG['farmshare']['throttle'][exp]
            per[exp] = dict(core_hours=sum(run_s) / 3600, task_hours=task_s / 3600, array_tasks=tasks,
                            wall_hours=math.ceil(tasks / thr) * task_s / 3600)
        res[f'stage_{k}_to_T_{st[k]}'] = dict(per, wall_hours_both=max(per['e13']['wall_hours'],
                                                                         per['e17b']['wall_hours']),
                                              core_hours_both=per['e13']['core_hours'] + per['e17b']['core_hours'])
    cum_core = np.cumsum([v['core_hours_both'] for v in res.values()]).tolist()
    cum_wall = np.cumsum([v['wall_hours_both'] for v in res.values()]).tolist()
    return dict(assumed_train_tokens_per_second=tps, assumed_eval_seconds=eval_s, per_stage=res,
                cumulative_core_hours_if_stop_after_stage=cum_core, cumulative_wall_hours_if_stop_after_stage=cum_wall)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build-prep', 'init', 'stage', 'decide', 'aggregate', 'smoke', 'estimate', 'count'])
    ap.add_argument('--exp', choices=EXPS)
    ap.add_argument('--stage', type=int)
    ap.add_argument('--task', type=int)
    ap.add_argument('--output')
    ap.add_argument('--prep', default=str(HERE / 'prep'))
    ap.add_argument('--src-e13', help='build-prep: folder with Experiment 13-small references (default results/small)')
    ap.add_argument('--src-e17b', help='build-prep: folder with Experiment 17b references (default expt_v2_coverage/prep)')
    ap.add_argument('--workers', type=int, default=CFG['runs_per_task'])
    ap.add_argument('--tokens-per-second', type=float, default=5000.0)
    ap.add_argument('--eval-seconds', type=float, default=20.0)
    a = ap.parse_args()
    out = Path(a.output).expanduser() if a.output else None
    if a.cmd == 'build-prep':
        print(json.dumps({k: len(v) for k, v in build_prep(a.prep, {'e13': a.src_e13, 'e17b': a.src_e17b}).items()}))
        return 0
    if a.cmd == 'init':
        init_output(a.exp, a.prep, out)
        return 0
    if a.cmd == 'stage':
        if a.task is None:
            a.task = int(os.environ['SLURM_ARRAY_TASK_ID'])
        return cmd_stage(a.exp, a.stage, a.task, out, a.workers)
    if a.cmd == 'decide':
        return cmd_decide(a.exp, a.stage, out)
    if a.cmd == 'aggregate':
        return cmd_aggregate(a.exp, out)
    if a.cmd == 'smoke':
        return cmd_smoke(out, a.prep)
    if a.cmd == 'estimate':
        print(json.dumps(estimate(a.tokens_per_second, a.eval_seconds), indent=1))
        return 0
    if a.cmd == 'count':
        tasks = json.loads((Path(a.prep) / a.exp / 'tasks.json').read_text())
        print(len(grid(a.exp, tasks)), n_tasks(a.exp, tasks))
        return 0


if __name__ == '__main__':
    sys.exit(main())
