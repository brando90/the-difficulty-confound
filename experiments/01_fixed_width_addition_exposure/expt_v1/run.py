"""A fixed-task, paired-seed intervention on addition training exposure.

Run only on the designated execution host. No model-provider API is used.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digits_of(values, width):
    return (np.asarray(values)[:, None] // (10 ** np.arange(width))) % 10


def carry_chain(a, b, digits):
    """Maximum consecutive carry-out chain; independent of model predictions."""
    carry = np.zeros_like(a)
    run = np.zeros_like(a)
    longest = np.zeros_like(a)
    for place in range(digits):
        carry = ((a // 10**place) % 10 + (b // 10**place) % 10 + carry) // 10
        run = (run + 1) * carry
        longest = np.maximum(longest, run)
    return longest


def column_addition(a, b, digits):
    """Independent fixed-width oracle, checked against Python integer addition."""
    carry = 0
    answer = 0
    for place in range(digits):
        column = a // 10**place % 10 + b // 10**place % 10 + carry
        answer += (column % 10) * 10**place
        carry = column // 10
    return answer + carry * 10**digits


def dataset(config):
    limit = 10 ** config['digits']
    a, b = np.triu_indices(limit)
    strata = carry_chain(a, b, config['digits'])
    rng = np.random.default_rng(config['split_seed'])
    pools, val_indices, test_indices = [], [], []
    for k in range(config['digits'] + 1):
        indices = np.flatnonzero(strata == k)
        rng.shuffle(indices)
        nt, nv = config['test_per_stratum'], config['validation_per_stratum']
        test_indices.extend(indices[:nt])
        val_indices.extend(indices[nt:nt + nv])
        pools.append(indices[nt + nv:])
    x_digits = np.concatenate([digits_of(a, config['digits']), digits_of(b, config['digits'])], 1)
    y = digits_of(a + b, config['digits'] + 1)
    return a, b, strata, x_digits, y, pools, np.array(val_indices), np.array(test_indices)


def checks(config):
    a, b, strata, xd, y, pools, val, test = dataset(config)
    assert len(a) == 500500
    assert len(set(test)) == len(test) == 4000
    assert len(set(val)) == len(val) == 1000
    assert not set(test) & set(val)
    all_train = np.concatenate(pools)
    assert not set(all_train) & (set(test) | set(val))
    assert np.all(a <= b)  # swapped pairs never cross splits
    assert np.all((y * 10 ** np.arange(4)).sum(1) == a + b)
    # Exhaustive independent oracle check, including all boundary/carry cases.
    assert all(column_addition(int(x), int(z), 3) == x + z for x, z in zip(a, b))
    assert carry_chain(np.array([123, 109, 199, 999]), np.array([234, 1, 1, 1]), 3).tolist() == [0, 1, 2, 3]
    for weights in config['conditions'].values():
        counts = np.rint(np.array(weights) * config['train_size']).astype(int)
        assert counts.sum() == config['train_size']
        assert all(n <= len(pool) for n, pool in zip(counts, pools))
    return {'oracle_pairs_checked': len(a), 'test_pairs': len(test), 'validation_pairs': len(val),
            'split_overlap': 0, 'population_per_stratum': np.bincount(strata).tolist()}


def model(config):
    width = config['hidden_width']
    return nn.Sequential(nn.Linear(6 * 10, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 4 * 10))


@torch.no_grad()
def evaluate(net, x, labels, indices, a, b, strata, condition, seed, split):
    rows = []
    for start in range(0, len(indices), 1024):
        ix = indices[start:start + 1024]
        logits = net(x[ix]).reshape(-1, 4, 10)
        pred_digits = logits.argmax(-1)
        predicted = (pred_digits * (10 ** torch.arange(4))).sum(-1).numpy()
        bits = F.cross_entropy(logits.reshape(-1, 10), labels[ix].reshape(-1), reduction='none').reshape(-1, 4).sum(1).numpy() / math.log(2)
        for j, item in enumerate(ix):
            rows.append({'seed': seed, 'condition': condition, 'split': split,
                         'item_id': int(item), 'a': int(a[item]), 'b': int(b[item]),
                         'carry_chain': int(strata[item]), 'target': int(a[item] + b[item]),
                         'prediction': int(predicted[j]), 'correct': int(predicted[j] == a[item] + b[item]),
                         'conditional_bits': float(bits[j])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    check_results = checks(config)
    print(json.dumps({'checks': check_results}), flush=True)
    if args.check_only:
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'runtime.json').exists():
        raise RuntimeError('Existing run directory: preserve it and use a new output path.')
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    runtime = {'hostname': socket.gethostname(), 'python': platform.python_version(),
               'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
               'threads': config['threads'], 'git_revision': revision,
               'config_sha256': sha(ROOT / 'config.json'), 'code_sha256': sha(__file__),
               'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'pid': os.getpid(), 'checks': check_results,
               'expected_cells': len(config['seeds']) * len(config['conditions'])}
    save_json(out / 'runtime.json', runtime)
    a, b, strata, xd, y, pools, val, test = dataset(config)
    x = F.one_hot(torch.from_numpy(xd).long(), 10).float().reshape(-1, 60)
    labels = torch.from_numpy(y).long()
    write_csv(out / 'heldout_items.csv', ({'split': split, 'item_id': int(i), 'a': int(a[i]), 'b': int(b[i]), 'carry_chain': int(strata[i]), 'target': int(a[i] + b[i])} for split, indices in [('validation', val), ('test', test)] for i in indices))
    cells = [{'seed': seed, 'condition': cond, 'status': 'pending'} for seed in config['seeds'] for cond in config['conditions']]
    save_json(out / 'cells.json', cells)
    all_metrics = []
    for cell in cells:
        if time.monotonic() - start > config['max_wall_seconds']:
            raise TimeoutError('Prespecified whole-run wall budget reached; pending cells retained.')
        seed, condition = cell['seed'], cell['condition']
        cell.update(status='running', started_elapsed_seconds=time.monotonic() - start)
        save_json(out / 'cells.json', cells)
        torch.manual_seed(seed)
        net = model(config)
        if 'parameter_count' not in runtime:
            runtime['parameter_count'] = sum(p.numel() for p in net.parameters())
            save_json(out / 'runtime.json', runtime)
        generator = torch.Generator().manual_seed(seed + 10000)
        rng = np.random.default_rng(seed)
        shuffled_pools = [rng.permutation(pool) for pool in pools]
        counts = np.rint(np.array(config['conditions'][condition]) * config['train_size']).astype(int)
        train_indices = np.concatenate([pool[:n] for pool, n in zip(shuffled_pools, counts)])
        train_indices = train_indices[np.random.default_rng(seed + 1000).permutation(len(train_indices))]
        assert len(set(train_indices)) == config['train_size']
        assert not set(train_indices) & (set(test) | set(val))
        train_x, train_y = x[train_indices], labels[train_indices].clone()
        if condition == 'shuffled_labels':
            permutation = np.random.default_rng(seed + 2000).permutation(len(train_y))
            train_y = train_y[permutation]
        write_csv(out / f'train_{seed}_{condition}.csv', ({'item_id': int(i), 'training_target': int((train_y[j] * (10 ** torch.arange(4))).sum())} for j, i in enumerate(train_indices)))
        optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        history = []
        for step in range(1, config['steps'] + 1):
            batch = torch.randint(len(train_x), (config['batch_size'],), generator=generator)
            logits = net(train_x[batch]).reshape(-1, 10)
            loss = F.cross_entropy(logits, train_y[batch].reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {cell}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                history.append({'step': step, 'batch_loss_nats_per_digit': float(loss.detach())})
                save_json(out / 'heartbeat.json', {'seed': seed, 'condition': condition, 'step': step, 'elapsed_seconds': time.monotonic() - start})
                if time.monotonic() - start > config['max_wall_seconds']:
                    raise TimeoutError('Prespecified whole-run wall budget reached.')
        prediction_rows = evaluate(net, x, labels, test, a, b, strata, condition, seed, 'test')
        prediction_rows += evaluate(net, x, labels, val, a, b, strata, condition, seed, 'validation')
        write_csv(out / f'predictions_{seed}_{condition}.csv', prediction_rows)
        for split in ['test', 'validation']:
            for k in range(4):
                selected = [row for row in prediction_rows if row['split'] == split and row['carry_chain'] == k]
                correct = sum(row['correct'] for row in selected)
                all_metrics.append({'seed': seed, 'condition': condition, 'split': split, 'carry_chain': k,
                                    'correct': correct, 'n': len(selected), 'accuracy': correct / len(selected),
                                    'mean_conditional_bits': float(np.mean([row['conditional_bits'] for row in selected]))})
        write_csv(out / 'metrics.csv', all_metrics)
        save_json(out / f'learning_{seed}_{condition}.json', history)
        torch.save(net.state_dict(), out / f'model_{seed}_{condition}.pt')
        cell.update(status='complete', steps=config['steps'], train_counts=counts.tolist(), elapsed_seconds=time.monotonic() - start - cell['started_elapsed_seconds'])
        save_json(out / 'cells.json', cells)
        print(json.dumps(cell), flush=True)
    runtime.update(completed_cells=len(cells), elapsed_seconds=time.monotonic() - start,
                   finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    save_json(out / 'runtime.json', runtime)
    print(json.dumps({'status': 'complete', 'elapsed_seconds': runtime['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
