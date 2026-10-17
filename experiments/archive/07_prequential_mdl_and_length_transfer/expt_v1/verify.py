"""Recompute evidence from saved rows and replay trained weights; never retrain."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from addition import digits_of, model, sha, save_json
from run import check, dataset, training_ids, evaluate
from independent_fit import reconstruct

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    c = json.loads((ROOT / 'config.json').read_text())
    o = ROOT / 'results'
    receipt = check(c)
    old = ROOT.parents[1] / 'archive/02_heldout_exposure_difficulty_laws/expt_v1/addition.py'
    assert sha(ROOT / 'addition.py') == sha(old)
    runtime = json.loads((o / 'runtime.json').read_text())
    for name, h in runtime['source_sha256'].items():
        assert sha(ROOT / name) == h, name
    assert sha(ROOT / 'config.json') == runtime['config_sha256']
    cells = json.loads((o / 'cells.json').read_text())
    assert len(cells) == 200 and all(r['status'] == 'complete' for r in cells)
    a, b, cell_arr, k, answers, pools, splits = dataset(c)
    saved = np.load(o / 'heldout_items.npz')
    for name, v in dict(a=a, b=b, cell=cell_arr, carry=k, answers=answers, **splits).items():
        assert np.array_equal(saved[name], v)
    for s in c['seeds']:
        st = np.load(o / f'train_{s}.npz')
        for q in c['train_mixtures'] + c['heldout_mixtures']:
            assert np.array_equal(st[f'q{q:.2f}'], training_ids(c, pools, s, q))
    aggregate_count = 0
    item_count = 0
    table = {}
    pre_table = {}
    for split in ('test', 'calibration'):
        rows = read(o / f'{split}_metrics.csv')
        assert len(rows) == 1200
        for r in rows:
            s, w, q, t, carry = (
                int(r['seed']),
                int(r['width']),
                float(r['mixture']),
                int(r['steps']),
                int(r['carry_chain']),
            )
            z = np.load(o / f'{split}_{s}_{w}_{q:.2f}_{t}.npz')
            assert np.array_equal(z['item_id'], splits[split])
            mask = k[z['item_id']] == carry
            assert int(r['n']) == int(mask.sum())
            assert int(r['correct']) == int(np.sum(z['prediction'][mask] == (a + b)[z['item_id'][mask]]))
            assert np.isclose(float(r['mean_bits']), float(z['bits'][mask].mean()), atol=1e-12, rtol=0)
            aggregate_count += 1
            item_count += int(mask.sum())
            if split == 'test':
                table[s, w, q, t, carry] = r
        pre_rows = read(o / f'pretrain_{split}_metrics.csv')
        assert len(pre_rows) == 400
        for r in pre_rows:
            s, w, q, t, carry = (
                int(r['seed']),
                int(r['width']),
                float(r['mixture']),
                int(r['steps']),
                int(r['carry_chain']),
            )
            assert t == 0
            z = np.load(o / f'{split}_{s}_{w}_{q:.2f}_0.npz')
            assert np.array_equal(z['item_id'], splits[split])
            mask = k[z['item_id']] == carry
            assert int(r['n']) == int(mask.sum())
            assert int(r['correct']) == int(np.sum(z['prediction'][mask] == (a + b)[z['item_id'][mask]]))
            assert np.isclose(float(r['mean_bits']), float(z['bits'][mask].mean()), atol=1e-12, rtol=0)
            aggregate_count += 1
            item_count += int(mask.sum())
            if split == 'test':
                pre_table[s, w, q, carry] = r
    torch.set_num_threads(c['threads'])
    torch.set_num_interop_threads(1)
    xd = np.concatenate([digits_of(a, 3), digits_of(b, 3)], 1)
    x = F.one_hot(torch.from_numpy(xd).long(), 10).float().reshape(-1, 60)
    y = torch.from_numpy(digits_of(a + b, 4)).long()
    replay_count = 0
    probe_count = 0
    max_difference = 0.0
    for cell in cells:
        seed, w, q = cell['seed'], cell['width'], cell['mixture']
        net = model(dict(hidden_width=w))
        net.load_state_dict(torch.load(o / f'model_{seed}_{w}_{q:.2f}.pt', weights_only=True))
        for split in ('test', 'calibration'):
            pred, bits = evaluate(net, x, y, splits[split])
            z = np.load(o / f'{split}_{seed}_{w}_{q:.2f}_{max(c["checkpoints"])}.npz')
            assert np.array_equal(pred, z['prediction'])
            d = float(np.max(abs(bits - z['bits'])))
            max_difference = max(max_difference, d)
            assert d <= 1e-6
        replay_count += 1
        net.load_state_dict(torch.load(o / f'warm_{seed}_{w}_{q:.2f}.pt', weights_only=True))
        pred_p, bits_p = evaluate(net, x, y, splits['probe'])
        zp = np.load(o / f'probe_{seed}_{w}_{q:.2f}.npz')
        assert np.array_equal(pred_p, zp['prediction'])
        assert np.max(abs(bits_p - zp['bits'])) <= 1e-6
        probe_count += 1
    receipt.update(
        status='PASS',
        source_hashes_match=True,
        unchanged_reference_sha256=sha(ROOT / 'addition.py'),
        completed_trajectories=200,
        aggregate_rows=aggregate_count,
        item_rows=item_count,
        final_models_replayed=replay_count,
        warm_models_replayed=probe_count,
        max_replay_bits_difference=max_difference,
    )
    save_json(o / 'verification.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
