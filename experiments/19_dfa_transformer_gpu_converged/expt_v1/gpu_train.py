"""Experiment 19: Experiment 11's transformer grid trained far longer, all runs side by side on one GPU.

  python gpu_train.py --out results --budget-minutes 95            the gated run (PROTOCOL Section 3)
  python gpu_train.py --out /tmp/smoke --device cpu --max-log2T 11  a CPU smoke test of the whole pipeline

One model per (seed, target, alpha) cell, 512 in all. A seed's 64 cells share Experiment 11's initialization for that
seed (torch.manual_seed(seed), then model.build_model) and one length schedule, so each seed is one stacked group
with no padding. Data are fresh uniform strings from the train partition of the hash split (never a test string),
labelled by the cell's tasks; the target share is exact (round(alpha * T) target examples after every step).
Checkpoints at every power of two from 2^9; after each one the run projects the next octave and stops cleanly if it
would pass the budget (PROTOCOL Section 3.3). Writes metrics.csv in Experiment 11's format, predictions/, checkpoints/,
run_state.json and manifest.json.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

import common as C
import ensemble as E

COLS = ['seed', 'target_id', 'K', 'K_bits', 'alpha', 'T', 'step', 'excess_bits', 'accuracy', 'n_test',
        'target_examples_seen', 'distractor_examples_seen', 'excess_bits_by_length', 'wall_seconds']
SOURCES = ['common.py', 'ensemble.py', 'gpu_train.py', 'config.json']
E11_SOURCES = ['dfa.py', 'data.py', 'model.py', 'config.json', 'results/tasks.json']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


class Group:
    """The 64 cells of one seed: stacked parameters, optimizer, schedule state and per-cell metadata."""

    def __init__(self, seed, targets, distractors, alphas, c11, n_steps, dev):
        self.seed = seed
        self.cells = [(t, a) for t in targets for a in alphas]
        R = len(self.cells)
        torch.manual_seed(seed)
        init = C.M11.build_model(c11)
        self.P = E.init_stack([init] * R, dev)
        fused = dict(fused=True) if dev.type == 'cuda' else {}
        self.opt = torch.optim.AdamW(list(self.P.values()), lr=c11['lr'], betas=tuple(c11['betas']),
                                     weight_decay=c11['weight_decay'], **fused)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt,
                                                       lambda s: min(1.0, (s + 1) / c11['warmup_steps']))
        rng = np.random.default_rng([seed, 19])
        Ls = list(range(c11['length_min'], c11['length_max'] + 1))
        self.lengths = C.D11.length_schedule(rng, n_steps, Ls)
        self.alpha = np.array([a for _, a in self.cells])
        self.ttok = torch.tensor([t['task_token'] for t, _ in self.cells], device=dev)
        dtok = np.array([d['task_token'] for d in distractors])
        self.dperm = torch.from_numpy(np.stack([dtok[rng.permutation(len(dtok))] for _ in range(R)])).to(dev)
        self.dcount = torch.zeros(R, dtype=torch.long, device=dev)
        self.tcount = np.zeros(R, np.int64)
        self.gen = torch.Generator(device=dev).manual_seed(int(seed) * 1000 + 19)
        self.B = c11['batch_size']
        self.dev = dev

    def batch(self, s, tables, valid, offs):
        """Tokens and labels for step s (0-based) for every cell of the group."""
        R, B, dev, g = len(self.cells), self.B, self.dev, self.gen
        L = int(self.lengths[s])
        k = np.floor(self.alpha * B * (s + 1) + 0.5).astype(np.int64) - np.floor(self.alpha * B * s + 0.5).astype(np.int64)
        self.tcount += k
        kt = torch.from_numpy(k).to(dev)
        rank = torch.rand(R, B, generator=g, device=dev).argsort(1).argsort(1)
        is_t = rank < kt[:, None]
        nt_rank = torch.cumsum((~is_t).long(), 1) - 1
        d_idx = (self.dcount[:, None] + nt_rank) % self.dperm.shape[1]
        tok = torch.where(is_t, self.ttok[:, None], self.dperm.gather(1, d_idx))
        self.dcount += B - kt
        cand = torch.randint(0, 2 ** L, (R, 2 * B), generator=g, device=dev)
        ok = valid[offs[L] + cand]
        assert bool((ok.sum(1) >= B).all()), 'not enough train-partition strings drawn'
        order = torch.argsort((~ok).to(torch.int8), dim=1, stable=True)[:, :B]
        bits = cand.gather(1, order)
        return C.tokens(tok, bits, L), tables.labels(tok, bits, L)


@torch.no_grad()
def evaluate(group, c11, test, nl, nh):
    """Per-string held-out bits (R, 4096) float32 and correctness, test set ordered as Experiment 11's."""
    bits_t, lens, labels = test
    R = len(group.cells)
    per = torch.empty(R, len(lens), device=group.dev)
    cor = torch.empty(R, len(lens), dtype=torch.bool, device=group.dev)
    for L in np.unique(lens):
        idx = np.flatnonzero(lens == L)
        it = torch.from_numpy(idx).to(group.dev)
        b = bits_t[it][None, :].expand(R, -1)
        x = C.tokens(group.ttok[:, None].expand(-1, len(idx)), b, int(L))
        y = labels[:, it]
        logits = E.forward(group.P, x, nl, nh)
        per[:, it] = E.bits_from_logits(logits, y)
        cor[:, it] = logits.argmax(-1) == y
    return per.cpu().numpy().astype(np.float32), cor.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget-minutes', type=float, required=True, help='wall-clock cap for this process')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-log2T', type=int, default=None, help='override for smoke tests only')
    ap.add_argument('--mask-cache', default=None)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    ap.add_argument('--seeds', default='all')
    args = ap.parse_args()
    t0 = time.time()
    deadline = t0 + 60 * args.budget_minutes
    c11, c = C.load_c11(), C.load_config()
    x19 = c['exp19']
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False  # float32 as on the CPU grid
        torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    assert not (out / 'metrics.csv').exists(), f'{out} already holds a run; use a fresh folder'
    tasks = C.load_tasks()
    targets = sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: t['task_token'])
    distractors = sorted((t for t in tasks if t['role'] == 'distractor'), key=lambda t: t['task_token'])
    log2max = args.max_log2T or x19['T_log2_max']
    Ts = [2 ** k for k in range(x19['T_log2_min'], log2max + 1)]
    B = c11['batch_size']
    n_steps = Ts[-1] // B
    seeds = c11['seeds'] if args.seeds == 'all' else [int(s) for s in args.seeds.split(',')]

    masks = C.test_masks(c11, args.mask_cache, args.workers)
    Ls = sorted(masks)
    offs, flat, o = {}, [], 0
    for L in Ls:
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

    groups = [Group(s, targets, distractors, c11['alphas'], c11, n_steps, dev) for s in seeds]
    tpos = {t['task_id']: i for i, t in enumerate(targets)}
    for g in groups:
        g.test = (test_bits, tl, tlab[[tpos[t['task_id']] for t, _ in g.cells]])
    nl, nh = c11['n_layers'], c11['n_heads']
    state = dict(experiment='19_dfa_transformer_gpu_converged', device=str(dev), budget_minutes=args.budget_minutes,
                 seeds_planned=seeds, seeds_dropped=[], T_planned=Ts, T_done=[], stop_reason=None,
                 prep_seconds=round(t_prep, 1), octaves=[])
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
            for g in groups:
                x, y = g.batch(s, tables, valid, offs)
                logits = E.forward(g.P, x, nl, nh)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction='none')
                loss.reshape(len(g.cells), -1).mean(1).sum().backward()
                E.clip_per_model_(g.P, c11['grad_clip'])
                g.opt.step()
                g.sched.step()
                g.opt.zero_grad(set_to_none=True)
        step = s_end
        sync(dev)
        t_train = time.time() - oct_t0
        te = time.time()
        wall = round(time.time() - t0, 3)
        preds, cells, ckpt = [], [], {}
        for g in groups:
            per, cor = evaluate(g, c11, g.test, nl, nh)
            for r, (t, a) in enumerate(g.cells):
                pr = per[r].astype(np.float64)
                by_len = [float(pr[tl == L].mean()) for L in np.unique(tl)]
                tseen = int(np.floor(a * T + 0.5))
                assert int(g.tcount[r]) == tseen
                w.writerow(dict(seed=g.seed, target_id=t['task_id'], K=t['K'], K_bits=t['K_bits'], alpha=a, T=T,
                                step=s_end, excess_bits=float(pr.mean()), accuracy=float(cor[r].mean()),
                                n_test=len(pr), target_examples_seen=tseen, distractor_examples_seen=T - tseen,
                                excess_bits_by_length=json.dumps(by_len), wall_seconds=wall))
                cells.append(f"{g.seed}_{t['task_id']}_{a:.2f}")
            preds.append(per)
            for k, v in g.P.items():
                ckpt.setdefault(k, []).append(v.detach().float().cpu().numpy())
        fcsv.flush()
        np.savez(out / 'predictions' / f'T_{T}.npz', bits=np.concatenate(preds), cells=np.array(cells))
        np.savez(out / 'checkpoints' / f'T_{T}.npz', cells=np.array(cells),
                 **{k: np.concatenate(v) for k, v in ckpt.items()})
        sync(dev)
        t_eval = time.time() - te
        n_oct_steps = s_end - oct_start
        state['T_done'].append(T)
        state['octaves'].append(dict(T=T, steps=n_oct_steps, train_seconds=round(t_train, 2),
                                     eval_seconds=round(t_eval, 2), groups=len(groups),
                                     seconds_per_step=round(t_train / max(1, n_oct_steps), 5),
                                     elapsed=round(time.time() - t0, 1)))
        save_state()
        print(json.dumps(state['octaves'][-1]), flush=True)
        oct_start, oct_t0 = s_end, time.time()
        if T == Ts[-1]:
            state['stop_reason'] = 'reached T_max'
            break
        per_step = t_train / max(1, n_oct_steps)
        # Subset rule (PROTOCOL 3.3): decided once, at T = 2^12, from the measured speed of the full grid.
        if T == 2 ** x19['subset_decision_log2T'] and len(groups) == len(c11['seeds']):
            need = (2 ** x19['min_log2T_full_grid'] - T) // B * per_step + t_eval * (x19['min_log2T_full_grid'] - x19['subset_decision_log2T'])
            if time.time() + need > deadline:
                keep = set(x19['subset_seeds'])
                state['seeds_dropped'] = [g.seed for g in groups if g.seed not in keep]
                groups = [g for g in groups if g.seed in keep]
                per_step *= len(groups) / len(c11['seeds'])
                save_state()
        nxt = 2 * T
        if time.time() + (nxt - T) // B * per_step * 1.05 + t_eval + 60 > deadline:
            state['stop_reason'] = f'budget: next octave to {nxt} projected past the deadline'
            break
    fcsv.close()
    state['wall_seconds'] = round(time.time() - t0, 1)
    state['T_top'] = state['T_done'][-1]
    save_state()
    here = Path(__file__).resolve().parent
    man = dict(sources={f: sha(here / f) for f in SOURCES},
               e11_sources={f: sha(C.E11 / f) for f in E11_SOURCES},
               torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
               host=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'),
               cuda=torch.version.cuda, run_state=state)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps(dict(T_top=state['T_top'], stop=state['stop_reason'], wall=state['wall_seconds'],
                          dropped=state['seeds_dropped'])), flush=True)


if __name__ == '__main__':
    main()
