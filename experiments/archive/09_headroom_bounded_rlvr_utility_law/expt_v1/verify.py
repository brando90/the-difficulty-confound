"""Recompute evidence from saved rows, replay trained weights and refit laws independently; never retrain."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from addition import digits_of, model, sha, save_json
from analyze import freeze, read as read_float
from independent_fit import objective, refit
from laws import bounded_inputs
from run import check, dataset, group_sizes, training_ids, evaluate

ROOT = Path(__file__).resolve().parent
# Works before and after this experiment moves under experiments/archive/.
REFERENCE = next(p for p in ROOT.parents if (p / 'experiments').is_dir()) / 'experiments/archive/02_heldout_exposure_difficulty_laws/expt_v1/addition.py'


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def check_rows(o, split, rows, splits, k, a, b):
    """Every aggregate row must equal a recomputation from its saved per-item predictions."""
    items = 0
    for r in rows:
        s, w, q, t, carry = int(r['seed']), int(r['width']), float(r['mixture']), int(r['steps']), int(r['carry_chain'])
        tag = '0' if t == 0 else f"g{int(r['group_size'])}_{t}"
        z = np.load(o / f'{split}_{s}_{w}_{q:.2f}_{tag}.npz')
        assert np.array_equal(z['item_id'], splits[split])
        mask = k[z['item_id']] == carry
        assert int(r['n']) == int(mask.sum())
        assert int(r['correct']) == int(np.sum(z['prediction'][mask] == (a + b)[z['item_id'][mask]]))
        assert np.isclose(float(r['mean_bits']), float(z['bits'][mask].mean()), atol=1e-12, rtol=0)
        items += int(mask.sum())
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    o = parser.parse_args().output
    c = json.loads((o / 'config.json').read_text())
    runtime = json.loads((o / 'runtime.json').read_text())
    receipt = check(c)
    assert sha(ROOT / 'addition.py') == sha(REFERENCE)
    for name, h in runtime['source_sha256'].items():
        assert sha(ROOT / name) == h, name
    assert sha(ROOT / 'config.json') == runtime['config_sha256']
    if not runtime['smoke']:
        assert json.loads((ROOT / 'config.json').read_text()) == c
    cells = json.loads((o / 'cells.json').read_text())
    assert len(cells) == len(c['seeds']) * 25 and all(r['status'] == 'complete' for r in cells)
    a, b, cell_arr, k, answers, pools, splits = dataset(c)
    saved = np.load(o / 'heldout_items.npz')
    for name, v in dict(a=a, b=b, cell=cell_arr, carry=k, answers=answers, **splits).items():
        assert np.array_equal(saved[name], v)
    for s in c['seeds']:
        st = np.load(o / f'train_{s}.npz')
        for q in c['train_mixtures'] + c['heldout_mixtures']:
            assert np.array_equal(st[f'q{q:.2f}'], training_ids(c, pools, s, q))

    # Aggregate rows reconcile with item rows.
    aggregate_count, item_count = 0, 0
    for split in ('test', 'calibration'):
        for prefix in ('', 'pretrain_'):
            rows = read(o / f'{prefix}{split}_metrics.csv')
            item_count += check_rows(o, split, rows, splits, k, a, b)
            aggregate_count += len(rows)

    # Replay every final (per G) and warm checkpoint.
    torch.set_num_threads(c['threads'])
    torch.set_num_interop_threads(1)
    xd = np.concatenate([digits_of(a, 3), digits_of(b, 3)], 1)
    x = F.one_hot(torch.from_numpy(xd).long(), 10).float().reshape(-1, 60)
    y = torch.from_numpy(digits_of(a + b, 4)).long()
    replay_count, warm_count, max_difference = 0, 0, 0.0
    for cell in cells:
        seed, w, q = cell['seed'], cell['width'], cell['mixture']
        net = model(dict(hidden_width=w))
        for G in group_sizes(c):
            net.load_state_dict(torch.load(o / f'model_{seed}_{w}_{q:.2f}_g{G}.pt', weights_only=True))
            for split in ('test', 'calibration'):
                pred, bits = evaluate(net, x, y, splits[split])
                z = np.load(o / f'{split}_{seed}_{w}_{q:.2f}_g{G}_{max(c["checkpoints"])}.npz')
                assert np.array_equal(pred, z['prediction'])
                max_difference = max(max_difference, float(np.max(abs(bits - z['bits']))))
            replay_count += 1
        net.load_state_dict(torch.load(o / f'warm_{seed}_{w}_{q:.2f}.pt', weights_only=True))
        pred_p, bits_p = evaluate(net, x, y, splits['probe'])
        zp = np.load(o / f'probe_{seed}_{w}_{q:.2f}.npz')
        assert np.array_equal(pred_p, zp['prediction'])
        max_difference = max(max_difference, float(np.max(abs(bits_p - zp['bits']))))
        warm_count += 1
    assert max_difference <= 1e-6, max_difference  # same software environment as training

    # Frozen forecasts are reproducible, and an independent optimizer finds no better bounded-law optimum.
    frozen = json.loads((o / 'frozen_forecasts.json').read_text())
    freeze(o, c)  # asserts the refit equals the saved frozen forecasts
    cal = read_float(o / 'calibration_metrics.csv')
    parity_gap = 0.0
    for fit in frozen['fits']:
        if fit['law'].startswith('additive'):
            continue
        rows = [r for r in cal if tuple(r[kk] for kk in ('seed', 'width', 'mixture', 'group_size', 'steps', 'carry_chain')) in set(map(tuple, fit['training_keys']))]
        form, z, s, acc0, n, _, _ = bounded_inputs(rows, fit['law'], c)
        yv = np.array([r['rl_utility'] for r in rows])
        _, independent = refit(form, z, s, acc0, n, yv, c['ridge_lambda'], c['fit_starts'])
        ours = objective(np.array(fit['coefficients']), form, z, s, acc0, n, yv, c['ridge_lambda'])
        parity_gap = max(parity_gap, ours - independent)
    assert parity_gap <= 1e-8, parity_gap

    receipt.update(
        status='PASS',
        source_hashes_match=True,
        unchanged_reference_sha256=sha(ROOT / 'addition.py'),
        completed_cells=len(cells),
        aggregate_rows=aggregate_count,
        item_rows=item_count,
        final_models_replayed=replay_count,
        warm_models_replayed=warm_count,
        max_replay_bits_difference=max_difference,
        frozen_forecasts_reproduced=True,
        independent_fit_max_objective_gap=parity_gap,
    )
    save_json(o / 'verification.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
