"""Experiment 19 diagnostic (AT-494): is the n=16/32 stall a constant-learning-rate failure? PREREG.md is the plan.

  python diag_train.py --out results --budget-minutes 18 --mask-cache <exp19_testmask.npz>   the GPU run
  python diag_train.py --out /tmp/smoke --device cpu --max-log2T 11 --mask-cache ...          a CPU smoke test

The 64 alpha=1 cells of Experiment 19's first four seeds (11101-11404 x 16 targets), each from the same initialization
as in Experiment 19 (torch.manual_seed(seed), then Experiment 11's model.build_model), trained twice side by side on
one shared data stream: arm "constant" is Experiment 19's schedule (8 warmup steps, then lr 1e-3 flat) and arm
"cosine" is the same warmup followed by cosine decay from 1e-3 to 0 at the last step (2^20 examples = 16384 steps).
Nothing else differs between the arms: same init, same strings in the same order, same AdamW, clipping and batch.
Writes metrics.csv (Experiment 19's columns plus arm and lr), predictions/, checkpoints/ (final T only),
run_state.json and manifest.json.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
X19 = HERE.parent / 'expt_v1'
sys.path.insert(0, str(X19))
import common as C  # noqa: E402
import ensemble as E  # noqa: E402
import gpu_train as GT  # noqa: E402

COLS = ['arm'] + GT.COLS + ['lr']
SOURCES = {'diag_train.py': HERE, 'diag_config.json': HERE, 'common.py': X19, 'ensemble.py': X19,
           'gpu_train.py': X19}
ARMS = ['constant', 'cosine']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_cfg():
    return json.loads((HERE / 'diag_config.json').read_text())


def lr_factor(arm, s, warmup, n_steps):
    """Multiplier on the peak lr for optimizer step s (0-based). Both arms share the warmup."""
    if s < warmup:
        return min(1.0, (s + 1) / warmup)
    if arm == 'constant':
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / (n_steps - warmup)))


def init_hash(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(np.ascontiguousarray(state_dict[k].detach().float().cpu().numpy()).tobytes())
    return h.hexdigest()


class Feed:
    """The shared data stream: the attributes gpu_train.Group.batch reads, for 64 cells at alpha = 1."""

    def __init__(self, cells, distractors, c11, n_steps, seed, dev):
        R = len(cells)
        self.cells = cells
        rng = np.random.default_rng(seed)
        Ls = list(range(c11['length_min'], c11['length_max'] + 1))
        self.lengths = C.D11.length_schedule(rng, n_steps, Ls)
        self.alpha = np.ones(R)
        self.ttok = torch.tensor([t['task_token'] for _, t in cells], device=dev)
        dtok = np.array([d['task_token'] for d in distractors])  # never drawn at alpha = 1; batch() still indexes it
        self.dperm = torch.from_numpy(np.stack([dtok] * R)).to(dev)
        self.dcount = torch.zeros(R, dtype=torch.long, device=dev)
        self.tcount = np.zeros(R, np.int64)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.B = c11['batch_size']
        self.dev = dev


class Arm:
    """One stacked copy of the 64 cells with its own optimizer and schedule (the attributes GT.evaluate reads)."""

    def __init__(self, name, cells, inits, ttok, c11, n_steps, dev):
        self.name, self.cells, self.ttok, self.dev = name, cells, ttok, dev
        self.P = E.init_stack(inits, dev)
        fused = dict(fused=True) if dev.type == 'cuda' else {}
        self.opt = torch.optim.AdamW(list(self.P.values()), lr=c11['lr'], betas=tuple(c11['betas']),
                                     weight_decay=c11['weight_decay'], **fused)
        w = c11['warmup_steps']
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_factor(name, s, w, n_steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget-minutes', type=float, required=True, help='wall-clock cap for this process')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-log2T', type=int, default=None, help='override for smoke tests only')
    ap.add_argument('--mask-cache', default=None)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + 60 * args.budget_minutes
    c11, d = C.load_c11(), load_cfg()
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    assert not (out / 'metrics.csv').exists(), f'{out} already holds a run; use a fresh folder'
    tasks = C.load_tasks()
    targets = sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: t['task_token'])
    distractors = sorted((t for t in tasks if t['role'] == 'distractor'), key=lambda t: t['task_token'])
    log2max = args.max_log2T or d['T_log2_max']
    Ts = [2 ** k for k in range(d['T_log2_min'], log2max + 1)]
    B = c11['batch_size']
    n_steps = Ts[-1] // B
    seeds = d['seeds']
    assert all(a == 1.0 for a in [d['alpha']])

    masks = C.test_masks(c11, args.mask_cache, args.workers)
    offs, flat, o = {}, [], 0
    for L in sorted(masks):
        offs[L] = o
        flat.append(~masks[L])
        o += len(masks[L])
    valid = torch.from_numpy(np.concatenate(flat)).to(dev)
    del masks, flat
    tables = C.TaskTables(tasks, dev)
    tb, tl = C.D11.build_test_set(c11)
    test_bits = torch.from_numpy(tb.astype(np.int64)).to(dev)
    tlab = torch.stack([torch.from_numpy(C.D11.labels_for(t, tb, tl).astype(np.int64)) for t in targets]).to(dev)
    t_prep = time.time() - t0

    cells, inits, ihash = [], [], {}
    for s in seeds:
        torch.manual_seed(s)
        m = C.M11.build_model(c11)
        ihash[str(s)] = init_hash(m.state_dict())
        for t in targets:
            cells.append((s, t))
            inits.append(m)
    feed = Feed(cells, distractors, c11, n_steps, d['data_seed'], dev)
    arms = [Arm(a, cells, inits, feed.ttok, c11, n_steps, dev) for a in ARMS]
    tpos = {t['task_id']: i for i, t in enumerate(targets)}
    test = (test_bits, tl, tlab[[tpos[t['task_id']] for _, t in cells]])
    nl, nh = c11['n_layers'], c11['n_heads']
    state = dict(experiment='19_dfa_transformer_gpu_converged/diag_cosine_lr', device=str(dev),
                 budget_minutes=args.budget_minutes, n_steps=n_steps, arms_planned=ARMS, arms_dropped=[],
                 T_planned=Ts, T_done=[], stop_reason=None, prep_seconds=round(t_prep, 1), octaves=[],
                 init_sha256=ihash)
    if dev.type == 'cuda':
        state['gpu'] = torch.cuda.get_device_name(0)
    fcsv = open(out / 'metrics.csv', 'w', newline='')
    w = csv.DictWriter(fcsv, fieldnames=COLS)
    w.writeheader()

    def save_state():
        (out / 'run_state.json').write_text(json.dumps(state, indent=1))

    step, oct_start, oct_t0 = 0, 0, time.time()
    for T in Ts:
        s_end = T // B
        for s in range(step, s_end):
            x, y = GT.Group.batch(feed, s, tables, valid, offs)
            for a in arms:
                logits = E.forward(a.P, x, nl, nh)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction='none')
                loss.reshape(len(cells), -1).mean(1).sum().backward()
                E.clip_per_model_(a.P, c11['grad_clip'])
                a.opt.step()
                a.sched.step()
                a.opt.zero_grad(set_to_none=True)
        step = s_end
        GT.sync(dev)
        t_train = time.time() - oct_t0
        te = time.time()
        wall = round(time.time() - t0, 3)
        preds, names, ckpt = [], [], {}
        for a in arms:
            lr_last = lr_factor(a.name, s_end - 1, c11['warmup_steps'], n_steps) * c11['lr']
            per, cor = GT.evaluate(a, c11, test, nl, nh)
            for r, (sd, t) in enumerate(cells):
                pr = per[r].astype(np.float64)
                by_len = [float(pr[tl == L].mean()) for L in np.unique(tl)]
                assert int(feed.tcount[r]) == T
                w.writerow(dict(arm=a.name, seed=sd, target_id=t['task_id'], K=t['K'], K_bits=t['K_bits'],
                                alpha=1.0, T=T, step=s_end, excess_bits=float(pr.mean()),
                                accuracy=float(cor[r].mean()), n_test=len(pr), target_examples_seen=T,
                                distractor_examples_seen=0, excess_bits_by_length=json.dumps(by_len),
                                wall_seconds=wall, lr=lr_last))
                names.append(f"{a.name}_{sd}_{t['task_id']}")
            preds.append(per)
            if T == Ts[-1]:
                for k, v in a.P.items():
                    ckpt.setdefault(k, []).append(v.detach().float().cpu().numpy())
        fcsv.flush()
        np.savez(out / 'predictions' / f'T_{T}.npz', bits=np.concatenate(preds), cells=np.array(names))
        if ckpt:
            np.savez(out / 'checkpoints' / f'T_{T}.npz', cells=np.array(names),
                     **{k: np.concatenate(v) for k, v in ckpt.items()})
        GT.sync(dev)
        t_eval = time.time() - te
        n_oct = s_end - oct_start
        state['T_done'].append(T)
        state['octaves'].append(dict(T=T, steps=n_oct, arms=[a.name for a in arms], train_seconds=round(t_train, 2),
                                     eval_seconds=round(t_eval, 2), seconds_per_step=round(t_train / max(1, n_oct), 5),
                                     elapsed=round(time.time() - t0, 1)))
        save_state()
        print(json.dumps(state['octaves'][-1]), flush=True)
        oct_start, oct_t0 = s_end, time.time()
        if T == Ts[-1]:
            state['stop_reason'] = 'reached T_max'
            break
        per_step = t_train / max(1, n_oct)
        n_left_evals = len(Ts) - len(state['T_done'])
        # Control-drop rule (PREREG Section 4), decided once at T = 2^decision: the cosine schedule must finish.
        if T == 2 ** d['control_drop_decision_log2T'] and len(arms) == 2:
            need = (n_steps - s_end) * per_step * 1.05 + n_left_evals * t_eval + 60
            if time.time() + need > deadline:
                state['arms_dropped'] = ['constant']
                state['control_dropped_at_T'] = T
                arms = [a for a in arms if a.name == 'cosine']
                per_step /= 2
                save_state()
        if time.time() + (2 * T - T) // B * per_step * 1.05 + t_eval + 60 > deadline:
            state['stop_reason'] = f'budget: next octave to {2 * T} projected past the deadline (cosine incomplete)'
            break
    fcsv.close()
    state['wall_seconds'] = round(time.time() - t0, 1)
    state['T_top'] = state['T_done'][-1]
    save_state()
    man = dict(sources={f: sha(p / f) for f, p in SOURCES.items()},
               e11_sources={f: sha(C.E11 / f) for f in GT.E11_SOURCES},
               torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
               host=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'),
               cuda=torch.version.cuda, run_state=state)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(dict(T_top=state['T_top'], stop=state['stop_reason'], wall=state['wall_seconds'],
                          dropped=state['arms_dropped'])), flush=True)


if __name__ == '__main__':
    main()
