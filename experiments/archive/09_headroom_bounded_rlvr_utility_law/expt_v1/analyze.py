"""Frozen utility-law forecasts, law-comparison Gates 1-3, and Experiment 08 replication Gates 4-5 on the G=16 slice."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from addition import save_json, sha, write_csv
from laws import fit_law, predict_law

ROOT = Path(__file__).resolve().parent
KEYS = ['seed', 'width', 'mixture', 'group_size', 'steps', 'carry_chain']
OUTCOMES = {'mean_bits', 'correct', 'online_active_group_fraction', 'rl_utility'}
PRIMARY_LAWS = [
    'additive_base',
    'additive_signal_difficulty',
    'bounded_base',
    'bounded_signal_difficulty',
    'logit_base',
    'logit_signal_difficulty',
]
GROUP_LAWS = ['bounded_base_g', 'bounded_signal_g']
COMPARISONS = [
    ('additive_base', 'bounded_base'),  # Gate 1: structural repair
    ('bounded_base', 'bounded_signal_difficulty'),  # Gate 2: probe features inside the bound
    ('additive_base', 'additive_signal_difficulty'),  # Experiment 08's Gate 2, for reference
    ('logit_base', 'logit_signal_difficulty'),
    ('bounded_base_g', 'bounded_signal_g'),  # Gate 3: rho(G) beyond G
]


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
    """Fit every law on calibration rows and forecast every target before any post-RL test outcome is read."""
    cal = read(out / 'calibration_metrics.csv')
    # Pre-RL test accuracy is the bound's starting point for test targets; it is measured before any RL update.
    start = {(r['seed'], r['width'], r['mixture'], r['carry_chain']): r['correct'] / r['n'] for r in read(out / 'pretrain_test_metrics.csv')}
    features = [
        {**{k: v for k, v in r.items() if k not in OUTCOMES}, 'start_accuracy': start[r['seed'], r['width'], r['mixture'], r['carry_chain']]}
        for r in cal
    ]
    primary, fit_g, held_g = c['primary_group_size'], c['train_group_sizes'], c['heldout_group_sizes']
    families = [
        # (family, laws, fit-row filter, target filter)
        ('primary', PRIMARY_LAWS, lambda r: r['group_size'] == primary and axis(r, c) == 'fit', lambda r: r['group_size'] == primary and axis(r, c) != 'fit'),
        ('group', GROUP_LAWS, lambda r: r['group_size'] in fit_g and axis(r, c) == 'fit', lambda r: r['group_size'] in held_g),
    ]
    fits, predictions = [], []
    for seed in c['seeds']:
        for family, laws, fit_rows, target_rows in families:
            training = [r for r in cal if r['seed'] == seed and fit_rows(r)]
            targets = [r for r in features if r['seed'] == seed and target_rows(r)]
            assert len(training) == (54 if family == 'primary' else 108), (family, len(training))
            for law in laws:
                fit = fit_law(training, law, c)
                pred = predict_law(fit, targets, law, c)
                fits.append(dict(seed=seed, family=family, law=law, training_keys=[key(r) for r in training], **fit))
                predictions.extend(
                    {**{k: r[k] for k in KEYS}, 'axis': axis(r, c), 'family': family, 'law': law,
                     'start_accuracy': r['start_accuracy'], 'predicted_utility': float(v)}
                    for r, v in zip(targets, pred)
                )
    frozen = dict(
        config_sha256=sha(out / 'config.json'),
        analysis_sha256=sha(__file__),
        laws_sha256=sha(ROOT / 'laws.py'),
        calibration_sha256=sha(out / 'calibration_metrics.csv'),
        fits=fits,
    )
    target = out / 'frozen_forecasts.json'
    if target.exists():
        assert json.loads(target.read_text()) == json.loads(json.dumps(frozen)), 'Frozen forecasts changed'
    else:
        save_json(target, frozen)
    write_csv(out / 'forecasts.csv', predictions)
    return predictions


def target_groups(c):
    """Held-out target sets scored per family: name -> (family, filter)."""
    groups = {
        'joint': ('primary', lambda r: r['axis'] == 'joint'),
        'model_only': ('primary', lambda r: r['axis'] == 'model_only'),
        'mixture_only': ('primary', lambda r: r['axis'] == 'mixture_only'),
        'joint_negative_utility': ('primary', lambda r: r['axis'] == 'joint' and r['observed_utility'] < 0),
        # Group-size axis: Gate 3 holds out only G (fit widths and fit histories).
        'group': ('group', lambda r: r['axis'] == 'fit'),
        'group_triple_heldout': ('group', lambda r: r['axis'] == 'joint'),
    }
    for w in c['heldout_widths']:
        groups[f'joint_width_{w}'] = ('primary', lambda r, w=w: r['axis'] == 'joint' and r['width'] == w)
    for g in c['heldout_group_sizes']:
        groups[f'group_g{g}'] = ('group', lambda r, g=g: r['axis'] == 'fit' and r['group_size'] == g)
    for t in c['checkpoints']:
        groups[f'joint_step_{t}'] = ('primary', lambda r, t=t: r['axis'] == 'joint' and r['steps'] == t)
    return groups


def score(out, c, test, forecasts, draws):
    observed = {key(r): r for r in test}
    scored = []
    for r in forecasts:
        o = observed[key(r)]
        acc_t, acc0, u = o['correct'] / o['n'], r['start_accuracy'], r['predicted_utility']
        impossible = not np.isfinite(u) or u > 1 - acc0 + 1e-12 or u < -acc0 - 1e-12
        scored.append({**r, 'observed_utility': o['rl_utility'], 'error': u - o['rl_utility'], 'impossible': int(impossible),
                       'noise_var': (acc_t * (1 - acc_t) + acc0 * (1 - acc0)) / o['n']})
    write_csv(out / 'scored_forecasts.csv', scored)
    metrics, comparisons = {}, {}
    for name, (family, keep) in target_groups(c).items():
        rows = [r for r in scored if r['family'] == family and keep(r)]
        laws = sorted({r['law'] for r in rows})
        if not laws:
            continue
        mses = {}
        metrics[name] = {}
        for law in laws:
            rr = [r for r in rows if r['law'] == law]
            err = np.array([r['error'] for r in rr])
            # Seeds without targets in a sparse group (e.g. negative utility) count as zero-weight via nanmean.
            mses[law] = np.array([np.mean([r['error'] ** 2 for r in rr if r['seed'] == s]) if any(r['seed'] == s for r in rr) else np.nan for s in c['seeds']])
            ci = interval(np.sqrt(np.nanmean(mses[law])), np.sqrt(np.nanmean(mses[law][draws], 1)))
            metrics[name][law] = dict(
                n_targets=len(rr), rmse=ci['estimate'], rmse_lo=ci['lo'], rmse_hi=ci['hi'],
                mae=float(abs(err).mean()), bias=float(err.mean()), impossible=int(sum(r['impossible'] for r in rr)),
            )
        # Binomial noise in observed U for this target set; the same targets for every law.
        sigma = float(np.sqrt(np.mean([r['noise_var'] for r in rows if r['law'] == laws[0]])))
        metrics[name]['noise_sigma'] = sigma
        comparisons[name] = {}
        for base, ext in COMPARISONS:
            if base in mses and ext in mses:
                point = 1 - np.sqrt(np.nanmean(mses[ext]) / np.nanmean(mses[base]))
                boot = 1 - np.sqrt(np.nanmean(mses[ext][draws], 1) / np.nanmean(mses[base][draws], 1))
                comparisons[name][f'{base}->{ext}'] = interval(point, boot)
    save_json(out / 'law_metrics.json', dict(metrics=metrics, comparisons=comparisons))
    return metrics, comparisons


def law_gate(c, metrics, comparisons, base, ext, main, subgroups, rule):
    """>=10% joint gain, lower bound > 0, zero impossible forecasts, and each subgroup improves or is non-inferior."""
    gain = comparisons[main][f'{base}->{ext}']
    sub = {}
    for g in subgroups:
        rb, re_, sigma = metrics[g][base]['rmse'], metrics[g][ext]['rmse'], metrics[g]['noise_sigma']
        sub[g] = dict(base_rmse=rb, ext_rmse=re_, noise_sigma=sigma, ok=bool(re_ < rb if rule == 'improve' else re_ <= rb + sigma))
    impossible = metrics[main][ext]['impossible'] + (metrics[main][base]['impossible'] if rule == 'noninferior' else 0)
    passed = (gain['estimate'] >= c['predictive_min_relative_rmse_gain'] and gain['lo'] > 0
              and impossible == 0 and all(v['ok'] for v in sub.values()))
    return dict(status='SUPPORTED' if passed else 'NOT_ESTABLISHED', baseline=base, extended=ext, targets=main,
                relative_rmse_reduction=gain, subgroup_rule=rule, subgroups=sub, impossible_forecasts=impossible)


def g16(rows, c):
    """Experiment 08 replications use only the primary group size; step-0 rows (group_size 0) are shared."""
    return [r for r in rows if r.get('group_size', 0) in (0, c['primary_group_size'])]


def seed_summaries(rows, c, names, draws):
    seedrows = [{n: float(np.mean([r[n] for r in rows if r['seed'] == s])) for n in names} for s in c['seeds']]
    return {n: interval(np.mean([r[n] for r in seedrows]), np.array([r[n] for r in seedrows])[draws].mean(1)) for n in names}


def utility_reversal(out, c, post_test, draws):
    """Gate 4 (Experiment 08 Gate 3): RL utility favors whichever regime was scarce in pretraining."""
    step = max(c['checkpoints'])
    u = {(r['seed'], r['width'], r['mixture'], r['carry_chain']): r['rl_utility'] for r in g16(post_test, c) if r['steps'] == step}
    widths = c['train_widths'] + c['heldout_widths']
    rows = [
        dict(seed=s, width=w, u_low=u[s, w, 0.1, 2] - u[s, w, 0.1, 1], u_high=u[s, w, 0.9, 1] - u[s, w, 0.9, 2])
        for s in c['seeds'] for w in widths
    ]
    write_csv(out / 'rl_utility_seed_width.csv', rows)
    stats = seed_summaries(rows, c, ['u_low', 'u_high'], draws)
    by_width = {str(w): {n: float(np.mean([r[n] for r in rows if r['width'] == w])) for n in ['u_low', 'u_high']} for w in widths}
    passed = all(stats[n]['estimate'] >= c['utility_reversal_min_gap'] and stats[n]['lo'] > 0 for n in stats) and all(
        v > 0 for bw in by_width.values() for v in bw.values()
    )
    return dict(status='SUPPORTED' if passed else 'NOT_ESTABLISHED', statistics=stats, by_width=by_width)


def retention(out, c, pre_test, post_test, cell_test, draws):
    """Gate 5 (Experiment 08 Gate 4): q_pre=0.1 remediation of the pure (1,1) gap without hurting (1,0)+(0,1)."""
    step = max(c['checkpoints'])
    cells = {(r['seed'], r['width'], r['mixture'], int(r['cell_id']), r['steps']): r['correct'] / r['n'] for r in g16(cell_test, c)}
    widths = c['train_widths'] + c['heldout_widths']
    rows = []
    for s in c['seeds']:
        for w in widths:
            control = {t: 0.5 * (cells[s, w, 0.1, 1, t] + cells[s, w, 0.1, 2, t]) for t in (0, step)}
            target = {t: cells[s, w, 0.1, 3, t] for t in (0, step)}
            rows.append(dict(seed=s, width=w, pure_control_pre=control[0], pure_target_pre=target[0], pure_control_post=control[step],
                             pure_target_post=target[step], pure_initial_gap=control[0] - target[0],
                             pure_target_gain=target[step] - target[0], pure_control_change=control[step] - control[0]))
    write_csv(out / 'retention_seed_width.csv', rows)
    stats = seed_summaries(rows, c, [k for k in rows[0] if k not in ('seed', 'width')], draws)
    seed_gap = np.array([np.mean([r['pure_initial_gap'] for r in rows if r['seed'] == s]) for s in c['seeds']])
    seed_gain = np.array([np.mean([r['pure_target_gain'] for r in rows if r['seed'] == s]) for s in c['seeds']])
    stats['pure_recovery_fraction'] = interval(seed_gain.mean() / seed_gap.mean(), seed_gain[draws].mean(1) / seed_gap[draws].mean(1))
    passed = (
        stats['pure_initial_gap']['lo'] >= c['retention_min_initial_gap']
        and stats['pure_recovery_fraction']['lo'] >= c['retention_min_recovery_fraction']
        and stats['pure_control_change']['estimate'] >= -c['retention_max_control_drop']
        and stats['pure_control_change']['lo'] >= -c['retention_max_control_drop']
    )
    return dict(status='SUPPORTED' if passed else 'NOT_ESTABLISHED', statistics=stats)


def group_size_effect(out, c, post_test):
    """Descriptive: seed-mean utility, probe rho(G) and online mixed-reward fraction for every G (no gate)."""
    cells = sorted({(r['width'], r['mixture'], r['carry_chain'], r['steps'], r['group_size']) for r in post_test})
    rows = []
    for w, q, k, t, g in cells:
        rr = [r for r in post_test if (r['width'], r['mixture'], r['carry_chain'], r['steps'], r['group_size']) == (w, q, k, t, g)]
        rows.append(dict(width=w, mixture=q, carry_chain=k, steps=t, group_size=g,
                         **{n: float(np.mean([r[n] for r in rr])) for n in ('rl_utility', 'start_accuracy', 'probe_rho', 'online_active_group_fraction')}))
    write_csv(out / 'group_size_effect.csv', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    out = parser.parse_args().output
    c = json.loads((out / 'config.json').read_text())
    forecasts = freeze(out, c)
    pre_test = read(out / 'pretrain_test_metrics.csv')
    test = read(out / 'test_metrics.csv')
    cell_test = read(out / 'cell_test_metrics.csv')
    n_cells = len(c['seeds']) * 25
    n_groups = len(c['train_group_sizes'] + c['heldout_group_sizes'])
    assert len(pre_test) == 2 * n_cells and len(test) == 6 * n_cells * n_groups and len(set(map(key, test))) == len(test)
    draws = boot_draws(c)
    metrics, comparisons = score(out, c, test, forecasts, draws)
    widths = [f'joint_width_{w}' for w in c['heldout_widths']]
    gates = dict(
        gate1_structural_repair=law_gate(c, metrics, comparisons, 'additive_base', 'bounded_base', 'joint', widths, 'improve'),
        gate2_probe_features=law_gate(c, metrics, comparisons, 'bounded_base', 'bounded_signal_difficulty', 'joint', widths, 'noninferior'),
        gate3_group_size=law_gate(c, metrics, comparisons, 'bounded_base_g', 'bounded_signal_g', 'group',
                                  [f'group_g{g}' for g in c['heldout_group_sizes']], 'noninferior'),
        gate4_utility_reversal=utility_reversal(out, c, test, draws),
        gate5_remediation_retention=retention(out, c, pre_test, test, cell_test, draws),
    )
    group_size_effect(out, c, test)
    save_json(out / 'gates.json', dict(gates=gates, n_seed_blocks=len(c['seeds']), p_value=None,
                                       interval='95% percentile bootstrap of complete seed blocks; fixed test set'))
    print(json.dumps({k: v['status'] for k, v in gates.items()}), flush=True)


if __name__ == '__main__':
    main()
