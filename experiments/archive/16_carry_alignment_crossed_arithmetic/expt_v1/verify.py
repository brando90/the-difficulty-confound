"""PROTOCOL Section 8 verification; prints PASS or the first failure. Never retrains.

    python verify.py --arm mlp --output results/mlp
    python verify.py --arm transformer --output results/transformer
    python verify.py --arm transformer --shard I --output DIR      # one 16b shard; writes VERIFIED next to DONE
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import carry

ROOT = Path(__file__).resolve().parent


def fail(msg):
    print(f'FAIL: {msg}', flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_common(c):
    if sum(carry.stratum_counts(8)) != 10 ** 16:
        fail('8-digit dynamic program does not total 10**16')
    if carry.stratum_counts(3) != carry.brute_counts(3):
        fail('3-digit dynamic program differs from brute force')
    for name, h in carry.IMPORT_SHA256.items():
        if carry.import_hashes()[name] != h:
            fail(f'read-only import {name} changed')


def verify_mlp(c, out):
    import mlp_arm as M
    cm, tol = c['mlp'], c['shared']['verify_tol_bits']
    d = M.data(cm, c['shared']['split_seed'])
    x, y = M.tensors(d)
    test = d['splits']['test']
    strat = M.stratum_of_cells(cm)
    test_stratum = np.array([strat[int(k)] for k in d['cell'][test]])
    # (2) independent oracle on every test pair, chain length per stratum
    for i in test:
        if carry.column_addition(int(d['a'][i]), int(d['b'][i]), 3) != d['a'][i] + d['b'][i]:
            fail(f'oracle mismatch on item {i}')
    chain = carry.carry_chain(d['a'][test], d['b'][test], 3)
    for s, cc in cm['chain'].items():
        if not np.all(chain[test_stratum == s] == cc):
            fail(f'chain length of stratum {s}')
    rows = list(csv.DictReader((out / 'metrics.csv').open()))
    mixes = {m['run']: m for m in json.loads((out / 'mixes.json').read_text())}
    held = set(np.concatenate(list(d['splits'].values())).tolist())
    max_diff, n_ck = 0.0, 0
    for (seed, target, alpha, width) in M.grid(cm, cm['seeds']):
        rid = f'{seed}_{target}_{alpha}_w{width}'
        ids = M.training_ids(cm, d, seed, target, alpha)
        # (4) training sets regenerate exactly; (5) disjoint from evaluation splits
        if hashlib.sha256(np.sort(ids).tobytes()).hexdigest() != mixes[rid]['ids_sha256']:
            fail(f'training set of {rid} does not regenerate')
        if held & set(ids.tolist()):
            fail(f'training set of {rid} overlaps evaluation splits')
        for step in cm['checkpoints_updates']:
            T = step * cm['batch_size']
            state = dict(np.load(out / 'checkpoints' / rid / f'T_{T}.npz'))
            bits, correct = M.evaluate(state, width, x, y, test)
            for stratum in M.TARGETS:
                r = next(r for r in rows if r['seed'] == str(seed) and r['target_id'] == target and float(r['alpha']) == alpha
                         and int(float(r['width'])) == width and int(float(r['step'])) == step and r['eval_stratum'] == stratum)
                m = test_stratum == stratum
                diff = abs(float(bits[m].sum(1).mean()) - float(r['excess_bits']))
                max_diff = max(max_diff, diff)
                # (1) replay reproduces every stored eps to verify_tol_bits
                if diff > tol or abs(float(correct[m].mean()) - float(r['accuracy'])) > 0:
                    fail(f'replay of {rid} T={T} {stratum}: diff {diff:.3e}')
            n_ck += 1
    return dict(checkpoints_replayed=n_ck, max_abs_diff_bits=max_diff)


def verify_shard(c, out, i):
    import torch
    import transformer_arm as TA
    t, tol = c['transformer'], c['shared']['verify_tol_bits']
    d = Path(out) / 'shards' / f'{i:05d}'
    done = json.loads((d / 'DONE').read_text())
    for name, h in done['files'].items():
        if sha(d / name) != h:
            fail(f'shard {i}: {name} hash differs from DONE')
    job = done['job']
    pairs, reject = TA.test_set(t, c['shared']['split_seed'])
    rows = list(csv.DictReader((d / 'metrics.csv').open()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_diff = 0.0
    for ck in sorted((d / 'checkpoints').glob('T_*.npz')):
        T = int(ck.stem[2:])
        net = TA.build(t, job['size']).to(device)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in np.load(ck).items()})
        for cc, (a, b) in pairs.items():
            if not np.all(carry.carry_chain(a, b, 8) == cc):
                fail(f'test stratum c{cc} chain length')
            bits, correct = TA.evaluate(net, a, b, job['format'], device, t['eval_batch'])
            r = next(r for r in rows if int(float(r['T'])) == T and r['eval_stratum'] == f'c{cc}')
            diff = abs(float(bits.sum(1).mean()) - float(r['excess_bits']))
            max_diff = max(max_diff, diff)
            if diff > tol:
                fail(f'shard {i} T={T} c{cc}: diff {diff:.3e}')
    # (4) the training stream regenerates: first block of every level, and (5) no test pair or twin in it
    for lv in t['train_levels']:
        a, b = TA.StratumStream(job['seed'], lv, t['stream_block'], reject).take(0, t['stream_block'])
        if np.isin(a * 10 ** 8 + b, reject).any() or not np.all(carry.carry_chain(a, b, 8) == lv):
            fail(f'shard {i}: stream of level {lv}')
    (d / 'VERIFIED').write_text(json.dumps(dict(max_abs_diff_bits=max_diff)) + '\n')
    return dict(shard=i, max_abs_diff_bits=max_diff)


def verify_fits(c, arm, out):
    """(6) The independent fitter agrees (recorded by analyze.py)."""
    g = json.loads((out / 'gates.json').read_text())
    if g['independent_fit']['status'] != 'AGREE':
        fail(f"independent fit: {g['independent_fit']['first']}")
    digest = sha(out / 'forecasts.csv')
    if digest != (out / 'forecasts.sha256').read_text().strip():
        fail('forecasts.csv changed after stage 1')
    return dict(independent_fit='AGREE', forecasts_sha256=digest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['mlp', 'transformer'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', type=int)
    args = ap.parse_args()
    c = json.loads((ROOT / 'config.json').read_text())
    check_common(c)
    if args.arm == 'transformer' and args.shard is not None:
        res = verify_shard(c, args.output, args.shard)
    elif args.arm == 'mlp':
        res = dict(verify_mlp(c, args.output), **verify_fits(c, 'mlp', args.output))
    else:
        # Each shard is replayed on the cluster (MODE=verify writes VERIFIED); merged results carry no checkpoints.
        mixes = json.loads((args.output / 'mixes.json').read_text())
        unverified = [m['shard'] for m in mixes if not m.get('verified')]
        if len(mixes) != c['transformer']['n_shards'] or unverified:
            fail(f'shards not VERIFIED on the cluster: {unverified[:10]}')
        res = dict(shards_verified=len(mixes), **verify_fits(c, 'transformer', args.output))
    print('PASS ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
