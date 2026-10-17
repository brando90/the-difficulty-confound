"""Recompute evidence from saved rows and replay trained weights; never retrain."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from addition import digits_of, model, sha, save_json
from run import check, dataset, training_ids, evaluate, previous_paths
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
    assert [sha(p) for p in previous_paths(c)] == runtime['previous_split_sha256']
    cells = json.loads((o / 'cells.json').read_text())
    assert len(cells) == 200 and all(r['status'] == 'complete' for r in cells)
    a, b, k, answers, pools, splits = dataset(c)
    saved = np.load(o / 'heldout_items.npz')
    for name, v in dict(a=a, b=b, carry=k, answers=answers, **splits).items():
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
        # Replay warm start on probe and test step=0
        net.load_state_dict(torch.load(o / f'warm_{seed}_{w}_{q:.2f}.pt', weights_only=True))
        pred_p, bits_p = evaluate(net, x, y, splits['probe'])
        zp = np.load(o / f'probe_{seed}_{w}_{q:.2f}.npz')
        assert np.array_equal(pred_p, zp['prediction'])
        assert np.max(abs(bits_p - zp['bits'])) <= 1e-6
        pred_t0, bits_t0 = evaluate(net, x, y, splits['test'])
        zt0 = np.load(o / f'test_{seed}_{w}_{q:.2f}_0.npz')
        assert np.array_equal(pred_t0, zt0['prediction'])
        assert np.max(abs(bits_t0 - zt0['bits'])) <= 1e-6
        probe_count += 1
    for r in read(o / 'probe_features.csv'):
        seed, w, q, cc = int(r['seed']), int(r['width']), float(r['mixture']), int(r['carry_chain'])
        z = np.load(o / f'probe_{seed}_{w}_{q:.2f}.npz')
        mask = k[z['item_id']] == cc
        assert int(r['n']) == int(mask.sum()) and np.isclose(
            float(r['starting_bits']), float(z['bits'][mask].mean()), atol=1e-12, rtol=0
        )
    # Independent primary-contrast calculation on pre-intervention step=0
    widths = c['train_widths'] + c['heldout_widths']
    vl, vh = [], []
    for seed in c['seeds']:
        def acc0(w, q, cc):
            r = pre_table[seed, w, q, cc]
            return int(r['correct']) / int(r['n'])

        vl.append(sum(acc0(w, 0.1, 0) - acc0(w, 0.1, 2) for w in widths) / 5)
        vh.append(sum(acc0(w, 0.9, 2) - acc0(w, 0.9, 0) for w in widths) / 5)
    draws = np.random.default_rng(c['analysis_seed']).integers(0, 8, (c['bootstrap_draws'], 8))
    inter = json.loads((o / 'intervention_summary.json').read_text())
    for name, values in [
        ('e_low', np.array(vl)),
        ('e_high', np.array(vh)),
        ('alignment', (np.array(vl) + vh) / 2),
        ('structure', (np.array(vl) - vh) / 2),
    ]:
        target = inter['statistics'][name]
        assert abs(target['estimate'] - values.mean()) < 1e-12
        ci = np.quantile(values[draws].mean(1), [0.025, 0.975])
        assert np.max(abs(ci - [target['lo'], target['hi']])) < 1e-12
    # Independent forecast reconstruction
    frozen = json.loads((o / 'frozen_forecasts.json').read_text())
    assert frozen['analysis_sha256'] == sha(ROOT / 'analyze.py')
    assert frozen['calibration_sha256'] == sha(o / 'calibration_metrics.csv')
    assert frozen['laws_sha256'] == sha(ROOT / 'laws.py')
    cal = read(o / 'calibration_metrics.csv')
    preds = read(o / 'forecasts.csv')
    scored = read(o / 'scored_forecasts.csv')
    assert len(preds) == len(scored) == 4608

    def keys(r):
        return (
            int(float(r['seed'])),
            int(float(r['width'])),
            float(r['mixture']),
            int(float(r['steps'])),
            int(float(r['carry_chain'])),
        )

    lookup = {keys(r): r for r in cal}

    def matrix(rows, names):
        mapping = {
            'log_parameters': 'parameter_count',
            'log_updates': 'steps',
            'log_starting_bits': 'starting_bits',
            'log_pretrain_exposure': 'pretrain_stratum_fraction',
        }
        return np.array(
            [
                [
                    1.0
                    if name == 'intercept'
                    else float(float(r['carry_chain']) == 2)
                    if name == 'two_carry'
                    else np.log(float(r[mapping[name]]))
                    for name in names
                ]
                for r in rows
            ],
            float,
        )

    max_pred_diff = 0.0
    for fit in frozen['fits']:
        fr = [lookup[tuple(t)] for t in fit['training_keys']]
        assert len(fr) == 54
        assert all(int(float(r['width'])) in c['train_widths'] and float(r['mixture']) in c['train_mixtures'] for r in fr)
        coef, names = reconstruct(fr, fit['law'], c)
        assert names == fit['columns']
        assert np.max(abs(coef - fit['coefficients'])) < 1e-8
        if fit['law'].startswith('stable'):
            for n in ['log_parameters', 'log_updates', 'log_pretrain_exposure']:
                if n in names:
                    assert coef[names.index(n)] <= 1e-10
            if fit['law'].endswith('difficulty'):
                assert -1e-10 <= coef[-1] <= 1 + 1e-10
        rr = [r for r in preds if int(float(r['seed'])) == fit['seed'] and r['law'] == fit['law']]
        computed = np.exp(matrix([lookup[keys(r)] for r in rr], fit['columns']) @ coef)
        stored = np.array([float(r['predicted_bits']) for r in rr])
        d = float(np.max(abs(computed - stored)))
        max_pred_diff = max(max_pred_diff, d)
        assert np.allclose(computed, stored, atol=1e-8, rtol=1e-8, equal_nan=True)
    for r in scored:
        observed = float(table[keys(r)]['mean_bits'])
        assert observed == float(r['observed_bits'])
        assert np.isclose(float(r['error']), float(r['predicted_bits']) - observed, atol=1e-12, rtol=0, equal_nan=True)
    for r in read(o / 'prediction_comparisons.csv'):
        group = r['axis']
        selected = [
            z
            for z in scored
            if (
                z['axis'] == group
                if group in ['joint', 'model_only', 'mixture_only']
                else z['axis'] == 'joint' and int(float(z['width'])) == int(group.rsplit('_', 1)[1])
            )
        ]

        def mses(law):
            return np.array(
                [np.mean([float(z['error']) ** 2 for z in selected if z['law'] == law and int(float(z['seed'])) == s]) for s in c['seeds']]
            )

        base, ext = mses(r['baseline']), mses(r['extended'])
        gain = 1 - np.sqrt(ext.mean() / base.mean())
        ci = np.quantile(1 - np.sqrt(ext[draws].mean(1) / base[draws].mean(1)), [0.025, 0.975])
        assert np.isclose(gain, float(r['relative_rmse_reduction']), atol=max(1e-12, 8 * abs(np.spacing(gain))), rtol=0, equal_nan=True)
        assert np.allclose(ci, [float(r['lo']), float(r['hi'])], atol=1e-12, rtol=0, equal_nan=True)
    for r in read(o / 'prediction_metrics.csv'):
        group = r['axis']
        rows = [
            z
            for z in scored
            if z['law'] == r['law']
            and (
                z['axis'] == group
                if group in ['joint', 'model_only', 'mixture_only']
                else z['axis'] == 'joint' and int(float(z['width'])) == int(group.rsplit('_', 1)[1])
            )
        ]
        errors = np.array([float(z['error']) for z in rows])
        pred = np.array([float(z['predicted_bits']) for z in rows])
        obs = np.array([float(z['observed_bits']) for z in rows])
        assert len(rows) == int(r['n_targets'])
        vals = {
            'rmse': np.sqrt(np.mean(errors**2)),
            'mae': np.mean(abs(errors)),
            'bias': errors.mean(),
            'prediction_mean': pred.mean(),
            'outcome_mean': obs.mean(),
        }
        for name, val in vals.items():
            assert np.isclose(float(r[name]), val, atol=max(1e-12, 8 * abs(np.spacing(val))), rtol=0, equal_nan=True)
        seed_mse = np.array([np.mean([float(z['error']) ** 2 for z in rows if int(float(z['seed'])) == seed]) for seed in c['seeds']])
        ci = np.quantile(np.sqrt(seed_mse[draws].mean(1)), [0.025, 0.975])
        assert np.allclose(ci, [float(r['rmse_lo']), float(r['rmse_hi'])], atol=max(1e-12, 8 * abs(np.spacing(vals['rmse']))), rtol=0, equal_nan=True)
        assert int(r['invalid_predictions']) == int((~np.isfinite(pred) | (pred <= 0)).sum())
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
        independent_bootstrap_match=True,
        forecasts_reconstructed=4608,
        independent_active_set_fit_match=True,
        coefficient_bounds_verified=True,
        max_forecast_difference=max_pred_diff,
    )
    save_json(o / 'verification.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
