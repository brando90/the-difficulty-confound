"""Post-processing only: describe every frozen outcome for Experiment 06, with no selection/refits."""
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
    scored = read(OUT / 'scored_forecasts.csv')
    fits = json.loads((OUT / 'frozen_forecasts.json').read_text())['fits']
    marginals = read(OUT / 'input_digit_marginals.csv')
    tv = []
    for seed in c['seeds']:
        for col in range(6):
            def hist(q):
                return np.array(
                    [int(r['count']) / int(r['n']) for r in marginals if int(r['seed']) == seed and int(r['column']) == col and float(r['mixture']) == q]
                )

            tv.append(dict(seed=seed, column=col, total_variation=float(abs(hist(0.1) - hist(0.9)).sum() / 2)))
    write(OUT / 'input_marginal_total_variation.csv', tv)
    mean_tv = float(np.mean([r['total_variation'] for r in tv]))
    max_tv = float(np.max([r['total_variation'] for r in tv]))
    min_tv = float(np.min([r['total_variation'] for r in tv]))
    joint = [r for r in p['metrics'] if r['axis'] == 'joint']
    s = a['statistics']
    lines = [
        '# Compositional carry exposure with invariant per-column carry rates and matched answers',
        '',
        '**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/06_compositional_carry_marginal_matched/RESULTS_compositional.md>',
        '',
        f"**TLDR:** Pure compositional carry ranking reversal (isolated single carries vs cascading two-carry chains): **{a['pure_compositional_status']}**. Balanced regime reversal ($S_{{\\text{{iso}}}}$ vs $S_{{\\text{{comp}}}}$): **{a['status']}**. Extrapolation repair criterion: **{p['repair_status']}**. All 200 trajectories and 3,840 forecasts completed and verified.",
        '',
        '## Setup and why per-column carry marginals were held invariant',
        '',
        'Canonical experiment: `experiments/archive/06_compositional_carry_marginal_matched/`. In Experiments 03 and 04, matching exact answer histograms across no-carry `(0,0)` and two-carry `(1,1)` still left substantial input-digit marginal differences (total variation distance up to 0.656, mean 0.443) because `(0,0)` has 0% carries at ones and tens whereas `(1,1)` has 100% carries at ones and tens.',
        '',
        'This study tests whether difficulty rankings reverse even when **both** exact answer histograms **and** per-column carry rates are held strictly invariant across every training mixture:',
        '- **Isolated Single-Carry Regime ($S_{\\text{iso}}$, stratum 1):** 50% ones-only carry `(1,0)` + 50% tens-only carry `(0,1)`.',
        '- **Paired Zero/Double-Carry Regime ($S_{\\text{comp}}$, stratum 2):** 50% no-carry `(0,0)` + 50% cascading two-carry chain `(1,1)`.',
        '',
        f'Across all 180 eligible answers (<1000) and all five mixtures ($q \\in \\{{.1, .2, .5, .8, .9\\}}$), every training pool has 7,200 pairs (40 per answer), **exactly 50.00% (3,600/7,200) carries at the ones column**, **exactly 50.00% (3,600/7,200) carries at the tens column**, and **0.00% carries at the hundreds column**. Across all six input-digit columns and eight seeds (`6101`–`6808`), endpoint ($q=.1$ vs $q=.9$) input-digit total variation distance drops from mean 0.443 (max 0.656) in Experiments 03–04 to **mean {mean_tv:.4f} (range {min_tv:.4f}–{max_tv:.4f})** ([all values](expt_v1/results/input_marginal_total_variation.csv)).',
        '',
        '## Compositional difficulty ranking reversal',
        '',
        f"Pure single-carry vs double-carry reversal gate: **{a['pure_compositional_status']}** (10-point threshold). Regime-level $S_{{\\text{{iso}}}}$ vs $S_{{\\text{{comp}}}}$ gate: **{a['status']}** (5-point threshold). Intervals use 20,000 paired bootstrap draws over eight complete seed blocks; p-val=n/a.",
        '',
        '| Contrast at 3,000 updates | Percentage points [95% interval] |',
        '|---|---:|',
        f"| Pure single-carry `(1,0)+(0,1)` minus double-carry `(1,1)` at $q=0.1$ | {100*s['pure_single_vs_double_low']['estimate']:.2f} [{100*s['pure_single_vs_double_low']['lo']:.2f}, {100*s['pure_single_vs_double_low']['hi']:.2f}] |",
        f"| Pure double-carry `(1,1)` minus single-carry `(1,0)+(0,1)` at $q=0.9$ | {100*s['pure_double_vs_single_high']['estimate']:.2f} [{100*s['pure_double_vs_single_high']['lo']:.2f}, {100*s['pure_double_vs_single_high']['hi']:.2f}] |",
        f"| Pure compositional alignment contrast | {100*s['pure_alignment']['estimate']:.2f} [{100*s['pure_alignment']['lo']:.2f}, {100*s['pure_alignment']['hi']:.2f}] |",
        f"| Pure residual structure contrast | {100*s['pure_structure']['estimate']:.2f} [{100*s['pure_structure']['lo']:.2f}, {100*s['pure_structure']['hi']:.2f}] |",
        f"| Regime $S_{{\\text{{iso}}}}$ minus $S_{{\\text{{comp}}}}$ at $q=0.1$ (`e_low`) | {100*s['e_low']['estimate']:.2f} [{100*s['e_low']['lo']:.2f}, {100*s['e_low']['hi']:.2f}] |",
        f"| Regime $S_{{\\text{{comp}}}}$ minus $S_{{\\text{{iso}}}}$ at $q=0.9$ (`e_high`) | {100*s['e_high']['estimate']:.2f} [{100*s['e_high']['lo']:.2f}, {100*s['e_high']['hi']:.2f}] |",
        f"| Regime alignment contrast | {100*s['alignment']['estimate']:.2f} [{100*s['alignment']['lo']:.2f}, {100*s['alignment']['hi']:.2f}] |",
        '',
        '| Width | Single-carry lead at $q=0.1$ (pp) | Double-carry lead at $q=0.9$ (pp) | $S_{\\text{iso}}$ lead at $q=0.1$ (pp) | $S_{\\text{comp}}$ lead at $q=0.9$ (pp) |',
        '|---:|---:|---:|---:|---:|',
    ]
    for w in sorted(map(int, a['by_width'])):
        bw = a['by_width'][str(w)]
        lines.append(
            f"| {w} | {100*bw['pure_single_vs_double_low']:.2f} | {100*bw['pure_double_vs_single_high']:.2f} | {100*bw['e_low']:.2f} | {100*bw['e_high']:.2f} |"
        )
    cell_test = read(OUT / 'cell_test_metrics.csv')
    draws = np.random.default_rng(c['analysis_seed']).integers(0, 8, (c['bootstrap_draws'], 8))
    cell_names = {0: 'No carry `(0,0)`', 1: 'Ones carry `(1,0)`', 2: 'Tens carry `(0,1)`', 3: 'Composed two-carry `(1,1)`'}
    lines += [
        '',
        'Final exact-answer accuracy across all four carry cells (pooled equally across widths within seed, 720 fixed test items per cell):',
        '',
        '| Mixture $q$ ($S_{\\text{comp}}$ fraction) | Carry cell | Accuracy %, 95% interval |',
        '|---:|---|---:|',
    ]
    for q in [0.1, 0.5, 0.9]:
        for cid in range(4):
            vals = np.array(
                [
                    np.mean(
                        [
                            int(r['correct']) / int(r['n'])
                            for r in cell_test
                            if int(r['seed']) == seed and float(r['mixture']) == q and int(r['steps']) == 3000 and int(r['cell_id']) == cid
                        ]
                    )
                    for seed in c['seeds']
                ]
            )
            lo, hi = np.quantile(vals[draws].mean(1), [0.025, 0.975])
            lines.append(f'| {q} | {cell_names[cid]} | {100*vals.mean():.2f} [{100*lo:.2f}, {100*hi:.2f}] |')
    lines += [
        '',
        '## Joint held-out prediction',
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
        '## Verification and provenance',
        '',
        f"Completed 200 trajectories and 40 warm starts in {rt['elapsed_seconds'] / 60:.2f} minutes on {rt['host']} (`{rt['git_revision']}`). Verification **PASS**: {v['oracle_pairs']:,} oracle pairs, {v['aggregate_rows']:,} aggregates, {v['final_models_replayed']} final and {v['warm_models_replayed']} warm models replayed (`max_replay_bits_difference = {v['max_replay_bits_difference']}`), and {v['forecasts_reconstructed']:,} forecasts reconstructed. [Receipt](expt_v1/results/verification.json).",
        '',
        '![Compositional carry reversal and input marginal matching](compositional_results.png)',
    ]
    (ROOT / 'RESULTS_compositional.md').write_text('\n'.join(lines) + '\n')
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    widths = sorted(map(int, a['by_width']))
    ax[0].plot(widths, [100 * a['by_width'][str(w)]['pure_single_vs_double_low'] for w in widths], 'o-', color='#1565c0', label='Single-carry lead at q=0.1 (90% isolated)')
    ax[0].plot(widths, [100 * a['by_width'][str(w)]['pure_double_vs_single_high'] for w in widths], 's-', color='#c62828', label='Double-carry lead at q=0.9 (90% composed)')
    ax[0].axhline(0, color='black', lw=0.7)
    ax[0].axhline(10, color='gray', ls=':', lw=0.8, label='10-point pooled gate')
    ax[0].set_xlabel('Hidden width')
    ax[0].set_ylabel('Accuracy advantage (percentage points)')
    ax[0].set_title('Single vs composed double-carry (50% carry/col fixed)')
    ax[0].legend(fontsize=7)
    cols = range(6)
    col_tv = [np.mean([r['total_variation'] for r in tv if r['column'] == col]) for col in cols]
    old_tv = [0.6459, 0.5786, 0.0668, 0.6469, 0.5853, 0.1336]
    x_pos = np.arange(6)
    ax[1].bar(x_pos - 0.18, old_tv, width=0.35, color='#90a4ae', label='Expt 03/04 (answer-matched only)')
    ax[1].bar(x_pos + 0.18, col_tv, width=0.35, color='#2e7d32', label='Expt 06 (answer + carry-rate matched)')
    ax[1].set_xticks(x_pos, ['a0 (1s)', 'a1 (10s)', 'a2 (100s)', 'b0 (1s)', 'b1 (10s)', 'b2 (100s)'], fontsize=8)
    ax[1].set_ylabel('Endpoint input-digit TV distance')
    ax[1].set_title('Input-digit marginal mismatch reduction')
    ax[1].legend(fontsize=7)
    fig.savefig(ROOT / 'compositional_results.png', dpi=170)
    fig.savefig(ROOT / 'compositional_results.pdf')
    plt.close(fig)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()}
    (ROOT / 'ARTIFACT_SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(
        json.dumps(
            dict(
                pure_compositional=a['pure_compositional_status'],
                regime_reversal=a['status'],
                mean_input_tv=mean_tv,
                repair=p['repair_status'],
                artifact_files=len(manifest),
            )
        )
    )


if __name__ == '__main__':
    main()
