"""Post-processing only: describe every frozen outcome for Experiment 05, with no selection/refits."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'expt_v1/results'


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def write(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(x):
    return f'{float(x):.4f}'


def ci(r, base='rmse'):
    return f"{fmt(r[base])} [{fmt(r[base + '_lo'])}, {fmt(r[base + '_hi'])}]"


def main():
    c = json.loads((ROOT / 'expt_v1/config.json').read_text())
    rt = json.loads((OUT / 'runtime.json').read_text())
    v = json.loads((OUT / 'verification.json').read_text())
    assert v['status'] == 'PASS'
    p = json.loads((OUT / 'prediction_summary.json').read_text())
    a = json.loads((OUT / 'intervention_summary.json').read_text())
    ret = json.loads((OUT / 'retention_summary.json').read_text())
    scored = read(OUT / 'scored_forecasts.csv')
    fits = json.loads((OUT / 'frozen_forecasts.json').read_text())['fits']
    seedstats = []
    for law in sorted({r['law'] for r in scored}):
        for seed in c['seeds']:
            rr = [r for r in scored if r['axis'] == 'joint' and r['law'] == law and float(r['seed']) == seed]
            err = np.array([float(r['error']) for r in rr])
            worst = max(rr, key=lambda r: abs(float(r['error'])))
            seedstats.append(
                dict(
                    law=law,
                    seed=seed,
                    n=len(rr),
                    rmse=float(np.sqrt(np.mean(err**2))),
                    mae=float(np.mean(abs(err))),
                    bias=float(err.mean()),
                    worst_prediction=float(worst['predicted_bits']),
                    worst_observation=float(worst['observed_bits']),
                    worst_width=float(worst['width']),
                    worst_mixture=float(worst['mixture']),
                    worst_steps=float(worst['steps']),
                    worst_carry=float(worst['carry_chain']),
                )
            )
    write(OUT / 'seed_prediction_metrics.csv', seedstats)
    coefficient_rows = []
    for fit in fits:
        row = {'seed': fit['seed'], 'law': fit['law'], **dict(zip(fit['columns'], fit['coefficients']))}
        row.setdefault('log_starting_bits', '')
        row.setdefault('log_pretrain_exposure', '')
        coefficient_rows.append(row)
    write(OUT / 'coefficient_table.csv', coefficient_rows)
    joint = [r for r in p['metrics'] if r['axis'] == 'joint']
    rs = ret['statistics']
    lines = [
        '# Historical exposure, starting-difficulty forecasts, and no-harm retention',
        '',
        '**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/05_pretraining_history_and_retention/RESULTS_history_retention.md>',
        '',
        f"**TLDR:** Starting-difficulty feature-value criterion: **{p['status']}**. Targeted gap recovery & strict no-harm retention criterion: **{ret['status']}**. Pre-intervention ranking-reversal criterion: **{a['status']}**. Extrapolation repair criterion: **{p['repair_status']}**. All 200 trajectories and 4,608 forecasts are retained.",
        '',
        '## Setup and why training history was varied',
        '',
        'Canonical experiment: `experiments/archive/05_pretraining_history_and_retention/`. In Experiment 04, constrained monotone ridge (`stable_difficulty`) repaired extrapolation blow-up across held-out widths, but improved over `stable_base` by only 3.48% [1.12, 6.24] because all starting checkpoints shared the identical balanced (`q=0.50`) warmup history. Meanwhile, Experiment 01 recovered 94.34% of the carry gap under balanced training, but missed its strict no-harm retention gate (-6.08 percentage-point drop on no-carry vs the allowed -5.00 points).',
        '',
        'This study directly tests the core mechanism of *Difficulty needs a training history*: starting checkpoints $m_0(s, w, q_{\\text{pre}})$ are pretrained for 1,500 updates under heterogeneous carry-exposure mixtures ($q_{\\text{pre}} \\in \\{.1, .5, .9\\}$ fit; $\\{.2, .8\\}$ held-out) across five widths ($32, 64, 128$ fit; $96, 384$ held-out) and eight fresh seeds (`4101`–`4808`). New evaluation splits (`split_seed=20261025`) exclude all evaluation items from both Experiment 03 and Experiment 04 while matching all 117 answer frequencies exactly (4,680 training pairs per mixture; 1,872 test, 936 calibration, and 936 probe questions).',
        '',
        'Before post-training begins, each starting checkpoint $m_0$ is evaluated on the 936 independent probe items to measure learner-relative starting difficulty $D_{m_0}$ (`starting_bits`). In Stage 2, every checkpoint undergoes balanced post-training remediation ($q_{\\text{post}} = 0.50$) evaluated at 100, 300, and 1,000 updates. A standard post-training law (`stable_base`) knows model scale $N$, post-training updates $T_{\\text{post}}$, and carry stratum $c$, but does not know the checkpoint\'s prior training history $q_{\\text{pre}}$. `stable_difficulty` adds $\\log D_{m_0}$ under the exact constrained monotone ridge specification frozen in Experiment 04 ($\\lambda = 0.01$, $\\beta_N, \\beta_T \\le 0$, $\\beta_D \\in [0, 1]$). `stable_oracle_history` replaces $\\log D_{m_0}$ with the true pretraining exposure $\\log q_{\\text{pre}, c}$.',
        '',
        '## Joint held-out prediction (unseen widths 96/384 × unseen histories .2/.8)',
        '',
        '| Law | Root mean squared error [95% interval] | Mean absolute error | Mean bias | Forecast mean | Observed mean | Invalid / targets |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in joint:
        lines.append(
            f"| {r['law']} | {ci(r)} | {fmt(r['mae'])} | {float(r['bias']):+.4f} | {fmt(r['prediction_mean'])} | {fmt(r['outcome_mean'])} | {r['invalid_predictions']}/{r['n_targets']} |"
        )
    lines += [
        '',
        'Relative error reduction is 100 × (1 − extended error / baseline error); positive is better. Intervals use 20,000 paired bootstrap draws over eight complete seed blocks; p-val=n/a.',
        '',
        '| Baseline → extended | Reduction %, 95% interval |',
        '|---|---:|',
    ]
    for r in p['comparisons']:
        if r['axis'] == 'joint':
            lines.append(
                f"| {r['baseline']} → {r['extended']} | {100 * r['relative_rmse_reduction']:.2f} [{100 * r['lo']:.2f}, {100 * r['hi']:.2f}] |"
            )
    lines += [
        '',
        f"Black-box starting difficulty (`stable_difficulty`) captures **{100 * p['oracle_gain_share_captured_by_probe']:.1f}%** of the joint RMSE reduction achieved by the oracle pretraining-history law (`stable_oracle_history`).",
        '',
        '## Every held-out split',
        '',
        '| Split | Law | Targets | Root mean squared error [95% interval] |',
        '|---|---|---:|---:|',
    ]
    for r in p['metrics']:
        lines.append(f"| {r['axis']} | {r['law']} | {r['n_targets']} | {ci(r)} |")
    lines += [
        '',
        '## Targeted gap recovery and strict no-harm retention',
        '',
        f"Frozen retention verdict: **{ret['status']}**. Requires initial gap lower bound ≥ 20 percentage points, gap-recovery fraction lower bound ≥ 50%, post-remediation target accuracy ≥ 90%, and control accuracy drop at most 5.0 percentage points (estimate and 95% lower bound ≥ -5.00 points).",
        '',
        '| Remediation metric (1,000 balanced post-training updates) | Scarce two-carry history ($q_{\\text{pre}}=0.1 \\to q_{\\text{post}}=0.5$) | Symmetric abundant history ($q_{\\text{pre}}=0.9 \\to q_{\\text{post}}=0.5$) |',
        '|---|---:|---:|',
        f"| Pre-intervention control accuracy % | {100*rs['control_pre']['estimate']:.2f} [{100*rs['control_pre']['lo']:.2f}, {100*rs['control_pre']['hi']:.2f}] | {100*rs['sym_control_pre']['estimate']:.2f} [{100*rs['sym_control_pre']['lo']:.2f}, {100*rs['sym_control_pre']['hi']:.2f}] |",
        f"| Pre-intervention target accuracy % | {100*rs['target_pre']['estimate']:.2f} [{100*rs['target_pre']['lo']:.2f}, {100*rs['target_pre']['hi']:.2f}] | {100*rs['sym_target_pre']['estimate']:.2f} [{100*rs['sym_target_pre']['lo']:.2f}, {100*rs['sym_target_pre']['hi']:.2f}] |",
        f"| Initial difficulty gap (percentage points) | {100*rs['initial_gap']['estimate']:.2f} [{100*rs['initial_gap']['lo']:.2f}, {100*rs['initial_gap']['hi']:.2f}] | {100*rs['sym_initial_gap']['estimate']:.2f} [{100*rs['sym_initial_gap']['lo']:.2f}, {100*rs['sym_initial_gap']['hi']:.2f}] |",
        f"| Post-remediation target accuracy % | {100*rs['target_post']['estimate']:.2f} [{100*rs['target_post']['lo']:.2f}, {100*rs['target_post']['hi']:.2f}] | {100*rs['sym_target_post']['estimate']:.2f} [{100*rs['sym_target_post']['lo']:.2f}, {100*rs['sym_target_post']['hi']:.2f}] |",
        f"| Target accuracy gain (percentage points) | +{100*rs['target_gain']['estimate']:.2f} [+{100*rs['target_gain']['lo']:.2f}, +{100*rs['target_gain']['hi']:.2f}] | +{100*rs['sym_target_gain']['estimate']:.2f} [+{100*rs['sym_target_gain']['lo']:.2f}, +{100*rs['sym_target_gain']['hi']:.2f}] |",
        f"| Fraction of initial gap recovered % | {100*rs['recovery_fraction']['estimate']:.2f} [{100*rs['recovery_fraction']['lo']:.2f}, {100*rs['recovery_fraction']['hi']:.2f}] | {100*rs['sym_recovery_fraction']['estimate']:.2f} [{100*rs['sym_recovery_fraction']['lo']:.2f}, {100*rs['sym_recovery_fraction']['hi']:.2f}] |",
        f"| Post-remediation control accuracy % | {100*rs['control_post']['estimate']:.2f} [{100*rs['control_post']['lo']:.2f}, {100*rs['control_post']['hi']:.2f}] | {100*rs['sym_control_post']['estimate']:.2f} [{100*rs['sym_control_post']['lo']:.2f}, {100*rs['sym_control_post']['hi']:.2f}] |",
        f"| Control accuracy change (percentage points; gate ≥ -5.00) | {100*rs['control_change']['estimate']:+.2f} [{100*rs['control_change']['lo']:+.2f}, {100*rs['control_change']['hi']:+.2f}] | {100*rs['sym_control_change']['estimate']:+.2f} [{100*rs['sym_control_change']['lo']:+.2f}, {100*rs['sym_control_change']['hi']:+.2f}] |",
        '',
        'Per-width remediation breakdown ($q_{\\text{pre}}=0.1 \\to q_{\\text{post}}=0.5$):',
        '',
        '| Width | Pre control % | Pre target % | Post control % | Post target % | Gap recovered % | Control change (pp) |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for w in sorted(map(int, ret['by_width'])):
        rw = ret['by_width'][str(w)]
        lines.append(
            f"| {w} | {100*rw['control_pre']:.2f} | {100*rw['target_pre']:.2f} | {100*rw['control_post']:.2f} | {100*rw['target_post']:.2f} | {100*rw['recovery_fraction']:.2f} | {100*rw['control_change']:+.2f} |"
        )
    lines += [
        '',
        '## Pre-intervention difficulty ranking reversal (step = 0)',
        '',
        f"Frozen ranking-reversal verdict: **{a['status']}**.",
        '',
        '| Contrast | Percentage points [95% interval] |',
        '|---|---:|',
    ]
    for name, r in a['statistics'].items():
        lines.append(f"| {name} | {100 * r['estimate']:.2f} [{100 * r['lo']:.2f}, {100 * r['hi']:.2f}] |")
    lines += [
        '',
        '## Verification and provenance',
        '',
        f"Completed 200 trajectories and 200 pretraining checkpoints in {rt['elapsed_seconds'] / 60:.2f} minutes on {rt['host']} (`{rt['git_revision']}`). Independent verification (`PASS`): {v['oracle_pairs']:,} oracle pairs, {v['aggregate_rows']:,} aggregate rows reconstructed from {v['item_rows']:,} item rows, {v['final_models_replayed']} final and {v['warm_models_replayed']} warm models replayed (`max_replay_bits_difference = {v['max_replay_bits_difference']}`), and {v['forecasts_reconstructed']:,} forecasts reconstructed (`max_forecast_difference = {v['max_forecast_difference']:.3g}`). [Verification receipt](expt_v1/results/verification.json).",
        '',
        '![Historical starting difficulty prediction and no-harm retention](history_retention_results.png)',
    ]
    (ROOT / 'RESULTS_history_retention.md').write_text('\n'.join(lines) + '\n')
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    order = ['legacy_base', 'stable_base', 'stable_difficulty', 'stable_oracle_history', 'ridge_difficulty', 'legacy_difficulty']
    jmap = {r['law']: r for r in joint}
    names = [k.replace('_', '\n') for k in order]
    val = np.array([jmap[k]['rmse'] for k in order])
    lo = np.array([jmap[k]['rmse_lo'] for k in order])
    hi = np.array([jmap[k]['rmse_hi'] for k in order])
    ax[0].errorbar(range(len(order)), val, yerr=[val - lo, hi - val], fmt='o', capsize=4, color='#1b5e20')
    ax[0].set_xticks(range(len(order)), names, fontsize=7.5)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Joint held-out RMSE (bits / answer)')
    ax[0].set_title('Heterogeneous training history: 192 joint targets / law')
    widths = sorted(map(int, ret['by_width']))
    ax[1].plot(widths, [100 * ret['by_width'][str(w)]['target_pre'] for w in widths], 'o--', color='#c62828', label='Two-carry before remediation (q_pre=0.1)')
    ax[1].plot(widths, [100 * ret['by_width'][str(w)]['target_post'] for w in widths], 'o-', color='#2e7d32', label='Two-carry after remediation (q_post=0.5)')
    ax[1].plot(widths, [100 * ret['by_width'][str(w)]['control_pre'] for w in widths], 's--', color='#1565c0', label='No-carry before remediation (q_pre=0.1)')
    ax[1].plot(widths, [100 * ret['by_width'][str(w)]['control_post'] for w in widths], 's-', color='#0d47a1', label='No-carry after remediation (q_post=0.5)')
    ax[1].set_xlabel('Hidden width')
    ax[1].set_ylabel('Exact-sum accuracy (%)')
    ax[1].set_title('Gap recovery with strict no-harm retention')
    ax[1].legend(fontsize=7)
    fig.savefig(ROOT / 'history_retention_results.png', dpi=170)
    fig.savefig(ROOT / 'history_retention_results.pdf')
    plt.close(fig)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()}
    (ROOT / 'ARTIFACT_SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(
        json.dumps(
            dict(
                feature_value=p['status'],
                repair=p['repair_status'],
                oracle=p['oracle_status'],
                retention=ret['status'],
                reversal=a['status'],
                artifact_files=len(manifest),
            )
        )
    )


if __name__ == '__main__':
    main()
