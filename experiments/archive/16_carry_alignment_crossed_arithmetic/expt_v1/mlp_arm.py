"""Arm 16a: answer-matched 3-digit addition with Experiment 06's MLP, target stratum share alpha (PROTOCOL 2.1).

Experiment 08's dataset() and Experiment 06's model are imported read-only and hash-checked (carry.py).
"""
import math
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import carry

if str(carry.EXP08) not in sys.path:
    sys.path.append(str(carry.EXP08))  # Experiment 08's run.py imports its sibling addition.py
exp08 = carry._load('exp08_run', carry.EXP08 / 'run.py', carry.IMPORT_SHA256['exp08_run.py'])

TARGETS = ['nocarry', 'isolated', 'cascade']


def data(cm, split_seed):
    keys = ['minimum_pairs_per_answer_cell', 'test_per_answer_cell', 'calibration_per_answer_cell', 'probe_per_answer_cell']
    a, b, cell, _, answers, pools, splits = exp08.dataset({**{k: cm[k] for k in keys}, 'split_seed': split_seed})
    return dict(a=a, b=b, cell=cell, answers=answers, pools=pools, splits=splits)


def cell_totals(cm, target, alpha):
    """Exact per-cell totals: the target stratum gets n*alpha, each distractor stratum n*(1-alpha)/2 (split evenly over cells)."""
    n = cm['train_size']
    totals = {}
    for stratum, cells in cm['strata'].items():
        share = alpha if stratum == target else (1 - alpha) / 2
        for cid in cells:
            v = n * share / len(cells)
            assert abs(v - round(v)) < 1e-9, (target, alpha, cid, v)
            totals[cid] = int(round(v))
    assert sum(totals.values()) == n
    return totals


def training_ids(cm, d, seed, target, alpha):
    """Exactly train_per_answer pairs per answer and exact cell totals; nested across alpha (common random numbers).

    Each answer gets floor(per_answer * total/n) pairs per cell. The remaining units (equal for every answer) are
    dealt round-robin: unit j of the flat list [cell k repeated R_k times] goes to answer j mod |answers|, which is a
    deterministic feasible flow with every answer receiving exactly the remainder.
    """
    totals = cell_totals(cm, target, alpha)
    per, answers = cm['train_per_answer'], d['answers']
    quota = {k: per * v // cm['train_size'] for k, v in totals.items()}
    extra_units = [k for k in sorted(totals) for _ in range(totals[k] - len(answers) * quota[k])]
    r = per - sum(quota.values())
    assert len(extra_units) == r * len(answers)
    take = {(int(ans), k): quota[k] for ans in answers for k in totals}
    for j, k in enumerate(extra_units):
        take[int(answers[j % len(answers)]), k] += 1
    ids = []
    for (ans, k), m in sorted(take.items()):
        pool = d['pools'][ans, k]
        assert m <= len(pool), (ans, k, m, len(pool))
        perm = np.random.default_rng([seed, ans, k]).permutation(pool)  # one permutation per (seed, answer, cell)
        ids.extend(perm[:m])
    return np.asarray(ids, dtype=np.int64)


def carry_rates(d, ids):
    da, db = carry.digits_msb(d['a'][ids], 3)[:, ::-1], carry.digits_msb(d['b'][ids], 3)[:, ::-1]
    c0 = (da[:, 0] + db[:, 0]) >= 10
    c1 = (da[:, 1] + db[:, 1] + c0) >= 10
    return dict(units=float(c0.mean()), tens=float(c1.mean()))


def tensors(d):
    xd = np.concatenate([exp08.digits_of(d['a'], 3), exp08.digits_of(d['b'], 3)], 1)
    x = F.one_hot(torch.from_numpy(xd).long(), 10).float().reshape(-1, 60)
    y = torch.from_numpy(exp08.digits_of(d['a'] + d['b'], 4)).long()
    return x, y


def stratum_of_cells(cm):
    return {cid: s for s, cells in cm['strata'].items() for cid in cells}


@torch.no_grad()
def evaluate(state, width, x, y, ids):
    """float64 teacher-forced bits per answer (sum over 4 digits), per-digit bits, and exact-match correctness."""
    net = carry.addition06.model(dict(hidden_width=width)).double()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, np.float64)) for k, v in state.items()})
    z = net(x[ids].double()).reshape(-1, 4, 10)
    bits = F.cross_entropy(z.reshape(-1, 10), y[ids].reshape(-1), reduction='none').reshape(-1, 4) / math.log(2)
    correct = (z.argmax(-1) == y[ids]).all(-1)
    return bits.numpy(), correct.numpy()


def state_arrays(net):
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def run_one(cm, d, seed, target, alpha, width, ckpt_dir=None):
    """Train one 16a run; return metric rows for every checkpoint and evaluation stratum, plus its training ids."""
    torch.set_num_threads(cm['threads'])
    torch.use_deterministic_algorithms(True)
    ids = training_ids(cm, d, seed, target, alpha)
    x, y = tensors(d)
    torch.manual_seed(seed)  # same initialization per (seed, width) across targets and alpha
    net = carry.addition06.model(dict(hidden_width=width))
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cm['lr'], weight_decay=cm['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / cm['warmup']))
    gen = torch.Generator().manual_seed(seed * 1000 + width)  # same batch-index stream across targets and alpha
    tx, ty = x[ids], y[ids]
    test = d['splits']['test']
    strat = stratum_of_cells(cm)
    test_stratum = np.array([strat[int(k)] for k in d['cell'][test]])
    rows, started = [], time.monotonic()
    for step in range(1, max(cm['checkpoints_updates']) + 1):
        batch = torch.randint(len(tx), (cm['batch_size'],), generator=gen)
        loss = F.cross_entropy(net(tx[batch]).reshape(-1, 10), ty[batch].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step in cm['checkpoints_updates']:
            state = state_arrays(net)
            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                np.savez(ckpt_dir / f'T_{step * cm["batch_size"]}.npz', **state)
            bits, correct = evaluate(state, width, x, y, test)
            for stratum in TARGETS:
                m = test_stratum == stratum
                rows.append(metric_row(cm, seed, target, alpha, width, n_params, step, stratum, bits[m], correct[m]))
    return rows, ids, dict(seconds=time.monotonic() - started, n_params=n_params)


def metric_row(cm, seed, target, alpha, width, n_params, step, stratum, bits, correct):
    c = cm['chain'][target]
    row = dict(seed=seed, target_id=target, K=c + 1, K_bits='', alpha=alpha, T=step * cm['batch_size'], step=step,
               excess_bits=float(bits.sum(1).mean()), accuracy=float(correct.mean()), n_test=int(len(bits)),
               arm='mlp', format='mlp', N=n_params, width=width, c=c, s=c + 1, eval_stratum=stratum,
               eval_c=cm['chain'][stratum])
    for j in range(9):
        row[f'digit_bits_{j}'] = float(bits[:, j].mean()) if j < bits.shape[1] else ''
    return row


def grid(cm, seeds):
    return [(s, t, a, w) for s in seeds for t in TARGETS for a in cm['alpha_grid'] for w in cm['widths']]
