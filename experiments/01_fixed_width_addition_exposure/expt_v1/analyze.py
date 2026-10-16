"""Recompute every reported estimate from the committed per-item tables."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def interval(values, draws):
    values = np.asarray(values, dtype=float)
    boot = values[draws].mean(1)
    return {'mean': float(values.mean()), 'lo': float(np.quantile(boot, .025)),
            'hi': float(np.quantile(boot, .975)), 'seed_sd': float(values.std(ddof=1))}


def run(out):
    config = json.loads((ROOT / 'config.json').read_text())
    cells = json.loads((out / 'cells.json').read_text())
    expected = {(s, c) for s in config['seeds'] for c in config['conditions']}
    complete = {(c['seed'], c['condition']) for c in cells if c['status'] == 'complete'}
    if complete != expected or len(cells) != len(expected):
        raise RuntimeError(f'Full matrix not complete: {len(complete)}/{len(expected)}')
    # Independent recomputation from predictions, not the training script's summaries.
    measured = {}
    all_rows = []
    test_identity = None
    for seed, condition in sorted(expected):
        with (out / f'predictions_{seed}_{condition}.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5000
        test_rows = [r for r in rows if r['split'] == 'test']
        identity = [(r['item_id'], r['a'], r['b'], r['target'], r['carry_chain']) for r in test_rows]
        if test_identity is None:
            test_identity = identity
        assert identity == test_identity
        for r in rows:
            assert int(r['target']) == int(r['a']) + int(r['b'])
            assert int(r['correct']) == int(int(r['prediction']) == int(r['target']))
        train_path = out / f'train_{seed}_{condition}.csv'
        with train_path.open() as f:
            train = list(csv.DictReader(f))
        assert len(train) == config['train_size']
        train_ids = {r['item_id'] for r in train}
        assert len(train_ids) == len(train)
        assert not train_ids & {r['item_id'] for r in rows}
        for k in range(4):
            subset = [r for r in test_rows if int(r['carry_chain']) == k]
            assert len(subset) == config['test_per_stratum']
            measured[seed, condition, k] = {
                'accuracy': np.mean([int(r['correct']) for r in subset]),
                'bits': np.mean([float(r['conditional_bits']) for r in subset]),
                'correct': sum(int(r['correct']) for r in subset)}
        all_rows.extend(rows)
    rng = np.random.default_rng(config['analysis_seed'])
    seeds = config['seeds']
    draws = rng.integers(len(seeds), size=(config['bootstrap_draws'], len(seeds)))

    def vals(cond, k, metric='accuracy'):
        return np.array([measured[s, cond, k][metric] for s in seeds])

    table = []
    for condition in config['conditions']:
        for k in range(4):
            row = {'condition': condition, 'carry_chain': k,
                   **interval(vals(condition, k), draws),
                   'correct': sum(measured[s, condition, k]['correct'] for s in seeds),
                   'n_predictions': len(seeds) * config['test_per_stratum'],
                   'unique_test_items': config['test_per_stratum'],
                   'bits': interval(vals(condition, k, 'bits'), draws)}
            table.append(row)
    baseline_easy = vals('scarce_long_carry', 0)
    baseline_hard = vals('scarce_long_carry', 3)
    balanced_easy = vals('balanced', 0)
    balanced_hard = vals('balanced', 3)
    gap = baseline_easy - baseline_hard
    gain = balanced_hard - baseline_hard
    denominator = float(gap.mean())
    gap_boot = gap[draws].mean(1)
    if denominator > 0 and np.all(gap_boot > 0):
        ratios = gain[draws].mean(1) / gap_boot
        recovery = {'mean': float(gain.mean() / denominator),
                    'lo': float(np.quantile(ratios, .025)), 'hi': float(np.quantile(ratios, .975))}
    else:
        recovery = {'mean': None, 'lo': None, 'hi': None, 'reason': 'Nonpositive baseline-gap mean or bootstrap denominator.'}
    retention = balanced_easy - baseline_easy
    contrast = gain - .5 * gap
    signs = np.array(list(itertools.product([-1, 1], repeat=len(seeds))))
    p = float(np.mean((signs * contrast).mean(1) >= contrast.mean() - 1e-12))
    gates = {
        'baseline_gap_at_least_20_percentage_points': denominator >= .2,
        'baseline_no_carry_accuracy_at_least_90_percent': float(baseline_easy.mean()) >= .9,
        'balanced_long_carry_accuracy_at_least_90_percent': float(balanced_hard.mean()) >= .9,
        'no_carry_accuracy_drop_at_most_5_percentage_points': float(retention.mean()) >= -.05,
        'recovery_95_percent_lower_bound_above_half': recovery['lo'] is not None and recovery['lo'] > .5,
        'shuffled_label_overall_accuracy_below_5_percent': float(np.mean([vals('shuffled_labels', k).mean() for k in range(4)])) < .05,
    }
    summary = {'complete_cells': len(complete), 'expected_cells': len(expected),
               'test_predictions': sum(r['split'] == 'test' for r in all_rows),
               'unique_test_items': 4000, 'independent_training_seeds': len(seeds),
               'uncertainty': 'Paired percentile bootstrap over eight training/data-sampling seeds; 20000 draws. Conditional on this fixed test set, architecture, split and optimizer.',
               'table': table, 'baseline_gap': interval(gap, draws),
               'long_carry_gain': interval(gain, draws), 'no_carry_change': interval(retention, draws),
               'recovered_fraction': recovery,
               'primary_half_gap_contrast': interval(contrast, draws),
               'primary_p_value': p,
               'primary_test': 'One-sided exhaustive paired sign-flip test of gain minus 0.5*baseline-gap; null mean <= 0 at boundary, assumes symmetric paired differences. 256 sign assignments. Supporting diagnostic; prespecified verdict uses bootstrap interval.',
               'gates': gates, 'verdict': 'SUPPORTED_IN_THIS_SETUP' if all(gates.values()) else 'NOT_ESTABLISHED_IN_THIS_SETUP'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Addition exposure results', '', '**TLDR:** ' + summary['verdict'] + '. All numbers below are from the fixed test set.', '',
             '| Training condition | Maximum carry chain | Exact sums / predictions | Accuracy, 95% seed interval | Conditional bits / answer, 95% seed interval |',
             '|---|---:|---:|---:|---:|']
    for row in table:
        bits = row['bits']
        lines.append(f"| {row['condition']} | {row['carry_chain']} | {row['correct']}/{row['n_predictions']} | {100*row['mean']:.2f}% [{100*row['lo']:.2f}, {100*row['hi']:.2f}] | {bits['mean']:.3f} [{bits['lo']:.3f}, {bits['hi']:.3f}] |")
    lines += ['', f"Recovered fraction of baseline gap: {recovery}", f"Half-gap sign-flip p-value: {p}", '', summary['uncertainty'], '', 'Accuracy and code-length table entries are descriptive: p-val=n/a. Code lengths exclude the model and training data and are not Kolmogorov complexity or a complete minimum-description-length code.', '', '## Prespecified criteria', '']
    lines += [f'- {key}: {value}' for key, value in gates.items()]
    (out / 'table.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'table'}, indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        labels = {'scarce_long_carry': '97% no carry', 'balanced': 'Balanced exposure',
                  'rich_long_carry': '97% length-three carry', 'shuffled_labels': 'Shuffled-label control'}
        colors = ['#b45309', '#1d4ed8', '#047857', '#64748b']
        for condition, color in zip(config['conditions'], colors):
            subset = [r for r in table if r['condition'] == condition]
            means = np.array([r['mean'] for r in subset])
            low = np.array([r['lo'] for r in subset]); high = np.array([r['hi'] for r in subset])
            axes[0].errorbar(range(4), means * 100, yerr=np.array([means-low, high-means])*100,
                             marker='o', capsize=3, label=labels[condition], color=color)
            bits = [r['bits']['mean'] for r in subset]
            axes[1].plot(range(4), bits, marker='o', color=color)
        axes[0].set(ylabel='Exact-sum accuracy (%)', ylim=(-2, 102))
        axes[1].set(ylabel='Conditional predictive bits per answer')
        for ax in axes:
            ax.set(xlabel='Longest carry chain (same three-digit task)', xticks=range(4))
            ax.spines[['right', 'top']].set_visible(False)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Same test questions and training budget; different data exposure')
        fig.tight_layout()
        fig.savefig(out / 'exposure_effect.png', dpi=180)
        plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; numerical results are complete, figure omitted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    run(args.output)
