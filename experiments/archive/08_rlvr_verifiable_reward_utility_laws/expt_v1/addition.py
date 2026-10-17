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
