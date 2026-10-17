"""Prespecified paired compositional contrasts and constrained held-out forecasts."""
import csv
import json
from pathlib import Path
import numpy as np
from addition import save_json, sha, write_csv

ROOT = Path(__file__).resolve().parent
from laws import LAWS, design, fit_law

KEYS = ['seed', 'width', 'mixture', 'steps', 'carry_chain']


def read(path):
    with Path(path).open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def key(r):
    return tuple(r[k] for k in KEYS)


def axis(r, c):
    w = r['width'] in c['heldout_widths']
    q = r['mixture'] in c['heldout_mixtures']
    return 'joint' if w and q else 'model_only' if w else 'mixture_only' if q else 'fit'


def boot_draws(c):
    return np.random.default_rng(c['analysis_seed']).integers(0, len(c['seeds']), (c['bootstrap_draws'], len(c['seeds'])))


def interval(value, samples):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(estimate=float(value), lo=float(lo), hi=float(hi))


def freeze(out, c):
    cal = read(out / 'calibration_metrics.csv')
    assert len(cal) == 1200 and len(set(map(key, cal))) == 1200
    features = [{k: v for k, v in r.items() if k not in ('mean_bits', 'correct', 'n')} for r in cal]
    assert all(not {'mean_bits', 'correct', 'n'} & r.keys() for r in features)
    fits = []
    predictions = []
    for seed in c['seeds']:
        training = [r for r in cal if r['seed'] == seed and axis(r, c) == 'fit']
        targets = [r for r in features if r['seed'] == seed and axis(r, c) != 'fit']
        assert len(training) == 54 and len(targets) == 96
        for law in LAWS:
            fit = fit_law(training, law, c)
            z, _ = design(targets, law)
            with np.errstate(over='ignore', under='ignore'):
                pred = np.exp(z @ np.array(fit['coefficients']))
            fits.append(dict(seed=seed, law=law, training_keys=[key(r) for r in training], **fit))
            predictions.extend(
                {**{k: r[k] for k in KEYS}, 'axis': axis(r, c), 'law': law, 'predicted_bits': float(v)}
                for r, v in zip(targets, pred)
            )
    frozen = dict(
        config_sha256=sha(ROOT / 'config.json'),
        analysis_sha256=sha(__file__),
        laws_sha256=sha(ROOT / 'laws.py'),
        calibration_sha256=sha(out / 'calibration_metrics.csv'),
        fits=fits,
    )
    target = out / 'frozen_forecasts.json'
    if target.exists():
        assert json.loads(target.read_text()) == frozen, 'Frozen forecasts changed'
    else:
        save_json(target, frozen)
    write_csv(out / 'forecasts.csv', predictions)
    return predictions


def intervention(out, c, test, cell_test, draws):
    table = {key(r): r['correct'] / r['n'] for r in test}
    ctable = {(r['seed'], r['width'], r['mixture'], r['steps'], int(r['cell_id'])): r['correct'] / r['n'] for r in cell_test}
    rows = []
    widths = c['train_widths'] + c['heldout_widths']
    step = max(c['checkpoints'])
    for seed in c['seeds']:
        for w in widths:
            def acc(q, s):
                return table[seed, w, q, step, s]

            def cacc(q, cid):
                return ctable[seed, w, q, step, cid]

            # Stratum contrast: stratum 1 (isolated single-carry (1,0)+(0,1)) vs stratum 2 (paired (0,0)+(1,1))
            low = acc(0.1, 1) - acc(0.1, 2)
            high = acc(0.9, 2) - acc(0.9, 1)
            # Pure compositional contrast: isolated single-carry (1,0)+(0,1) vs composed two-carry chain (1,1)
            pure_low = acc(0.1, 1) - cacc(0.1, 3)
            pure_high = cacc(0.9, 3) - acc(0.9, 1)
            rows.append(
                dict(
                    seed=seed,
                    width=w,
                    e_low=low,
                    e_high=high,
                    alignment=(low + high) / 2,
                    structure=(low - high) / 2,
                    balanced_gap=acc(0.5, 1) - acc(0.5, 2),
                    pure_single_vs_double_low=pure_low,
                    pure_double_vs_single_high=pure_high,
                    pure_alignment=(pure_low + pure_high) / 2,
                    pure_structure=(pure_low - pure_high) / 2,
                )
            )
    write_csv(out / 'intervention_seed_width.csv', rows)
    metric_names = [
        'e_low',
        'e_high',
        'alignment',
        'structure',
        'balanced_gap',
        'pure_single_vs_double_low',
        'pure_double_vs_single_high',
        'pure_alignment',
        'pure_structure',
    ]
    seedrows = [{name: float(np.mean([r[name] for r in rows if r['seed'] == s])) for name in metric_names} for s in c['seeds']]
    summaries = {}
    for name in metric_names:
        values = np.array([r[name] for r in seedrows])
        summaries[name] = interval(values.mean(), values[draws].mean(1))
    aa = np.array([r['alignment'] for r in seedrows])
    ss = np.array([r['structure'] for r in seedrows])
    summaries['alignment_minus_abs_structure'] = interval(
        aa.mean() - abs(ss.mean()), aa[draws].mean(1) - abs(ss[draws].mean(1))
    )
    bywidth = {str(w): {n: float(np.mean([r[n] for r in rows if r['width'] == w])) for n in metric_names} for w in widths}
    supported = all(
        summaries[n]['estimate'] >= c['crossover_min_gap'] and summaries[n]['lo'] > 0 for n in ['e_low', 'e_high']
    ) and all(bywidth[str(w)][n] > 0 for w in widths for n in ['e_low', 'e_high'])
    pure_supported = all(
        summaries[n]['estimate'] >= c['pure_carry_crossover_min_gap'] and summaries[n]['lo'] > 0
        for n in ['pure_single_vs_double_low', 'pure_double_vs_single_high']
    ) and all(bywidth[str(w)][n] > 0 for w in widths for n in ['pure_single_vs_double_low', 'pure_double_vs_single_high'])
    obj = dict(
        status='SUPPORTED' if supported else 'NOT_ESTABLISHED',
        pure_compositional_status='SUPPORTED' if pure_supported else 'NOT_ESTABLISHED',
        n_seed_blocks=8,
        n_widths=5,
        test_items_per_stratum=1440,
        test_items_per_cell=720,
        interval='95% percentile bootstrap of complete seed blocks; fixed test set',
        p_value=None,
        threshold=c['crossover_min_gap'],
        pure_threshold=c['pure_carry_crossover_min_gap'],
        statistics=summaries,
        by_width=bywidth,
    )
    save_json(out / 'intervention_summary.json', obj)
    return obj


def prediction(out, c, test, forecasts, draws):
    target = {key(r): r for r in test}
    scored = [
        {**r, 'observed_bits': target[key(r)]['mean_bits'], 'error': r['predicted_bits'] - target[key(r)]['mean_bits']}
        for r in forecasts
    ]
    write_csv(out / 'scored_forecasts.csv', scored)
    metrics = []
    comparisons = []
    for group in ['joint', 'model_only', 'mixture_only'] + [f'joint_width_{w}' for w in c['heldout_widths']]:
        rr = (
            [r for r in scored if r['axis'] == group]
            if group in ['joint', 'model_only', 'mixture_only']
            else [r for r in scored if r['axis'] == 'joint' and r['width'] == int(group.rsplit('_', 1)[1])]
        )
        mses = {}
        for law in LAWS:
            rows = [r for r in rr if r['law'] == law]
            err = np.array([r['error'] for r in rows])
            obs = np.array([r['observed_bits'] for r in rows])
            pred = np.array([r['predicted_bits'] for r in rows])
            mses[law] = np.array([np.mean([r['error'] ** 2 for r in rows if r['seed'] == s]) for s in c['seeds']])
            ci = interval(np.sqrt(mses[law].mean()), np.sqrt(mses[law][draws].mean(1)))
            metrics.append(
                dict(
                    axis=group,
                    law=law,
                    n_targets=len(rows),
                    n_seeds=8,
                    rmse=ci['estimate'],
                    rmse_lo=ci['lo'],
                    rmse_hi=ci['hi'],
                    mae=float(abs(err).mean()),
                    bias=float(err.mean()),
                    prediction_mean=float(pred.mean()),
                    outcome_mean=float(obs.mean()),
                    invalid_predictions=int((~np.isfinite(pred) | (pred <= 0)).sum()),
                )
            )
        for base, extra in [
            ('stable_base', 'stable_difficulty'),
            ('legacy_difficulty', 'stable_difficulty'),
            ('legacy_difficulty', 'ridge_difficulty'),
            ('legacy_base', 'legacy_difficulty'),
            ('legacy_base', 'stable_base'),
            ('ridge_difficulty', 'stable_difficulty'),
        ]:
            point = 1 - np.sqrt(mses[extra].mean() / mses[base].mean())
            boot = 1 - np.sqrt(mses[extra][draws].mean(1) / mses[base][draws].mean(1))
            ci = interval(point, boot)
            comparisons.append(
                dict(
                    axis=group,
                    baseline=base,
                    extended=extra,
                    n_targets=len(rr) // len(LAWS),
                    relative_rmse_reduction=ci['estimate'],
                    lo=ci['lo'],
                    hi=ci['hi'],
                )
            )
    write_csv(out / 'prediction_metrics.csv', metrics)
    write_csv(out / 'prediction_comparisons.csv', comparisons)

    def gate_for(base, extended):
        chosen = [r for r in comparisons if r['baseline'] == base and r['extended'] == extended]
        main = next(r for r in chosen if r['axis'] == 'joint')
        finite = all(r['invalid_predictions'] == 0 for r in metrics if r['law'] in (base, extended))
        gate = (
            finite
            and main['relative_rmse_reduction'] >= c['predictive_min_relative_rmse_gain']
            and main['lo'] > 0
            and all(r['relative_rmse_reduction'] > 0 for r in chosen if r['axis'].startswith('joint_width_'))
        )
        return ('SUPPORTED' if gate else 'NOT_ESTABLISHED'), main

    status, main = gate_for('stable_base', 'stable_difficulty')
    repair, repair_main = gate_for('legacy_difficulty', 'stable_difficulty')
    obj = dict(
        status=status,
        repair_status=repair,
        primary_secondary_comparison=main,
        repair_comparison=repair_main,
        criterion='>=10% joint gain, lower interval >0, both held-out widths improve; no invalid forecasts',
        n_seed_blocks=8,
        p_value=None,
        all_forecasts_positive=all(r['invalid_predictions'] == 0 for r in metrics),
        metrics=metrics,
        comparisons=comparisons,
    )
    save_json(out / 'prediction_summary.json', obj)
    return obj


def main():
    c = json.loads((ROOT / 'config.json').read_text())
    out = ROOT / 'results'
    forecasts = freeze(out, c)
    test = read(out / 'test_metrics.csv')
    cell_test = read(out / 'cell_test_metrics.csv')
    assert len(test) == 1200 and len(cell_test) == 2400
    draws = boot_draws(c)
    a = intervention(out, c, test, cell_test, draws)
    b = prediction(out, c, test, forecasts, draws)
    print(
        json.dumps(
            dict(
                stratum_reversal=a['status'],
                pure_compositional_reversal=a['pure_compositional_status'],
                prediction=b['status'],
                repair=b['repair_status'],
            )
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
