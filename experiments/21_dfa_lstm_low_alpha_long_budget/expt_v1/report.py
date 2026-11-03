"""Write RESULTS_dfa_lstm_long_budget.md and the four-panel figure from results/analysis.json (formatting only).

  report.py [--results results] [--notes notes.md]
`--notes` is a markdown file with the hand-written sections (takeaway, deviations); it is inserted verbatim after the
title. Written after the grid ran (PROTOCOL Section 3.2 item 6); it computes nothing that decides a verdict.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

LEVEL_COLORS = {4: '#86b6ef', 8: '#3987e5', 16: '#1c5cab', 32: '#0d366b'}  # ordinal blue ramp, as in Exp 20
SERIES = ['#2a78d6', '#eb6834']
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dc'

# PROTOCOL Section 5, verbatim as sets of acceptable verdicts
PREDICTED = {'n4_a0.15': {'LATE_LEARNING'}, 'n8_a0.15': {'LATE_LEARNING'},
             'n16_a0.15': {'DESCENDING', 'LATE_LEARNING'},
             'n4_a0.05': {'DESCENDING', 'LATE_LEARNING'},
             'n8_a0.05': {'FLOOR_HOLDS', 'DESCENDING'}, 'n16_a0.05': {'FLOOR_HOLDS', 'DESCENDING'},
             'n32_a0.15': {'FLOOR_HOLDS'}, 'n32_a0.05': {'FLOOR_HOLDS'}}


def f(x, d=3):
    return 'n/a' if x is None else f'{x:.{d}f}'


def fci(c, d=3):
    return f'[{c[0]:.{d}f}, {c[1]:.{d}f}]'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def per_run_curves(res):
    out = {}
    for r in csv.DictReader(open(res / 'metrics.csv')):
        key = (int(r['K']), round(float(r['alpha']), 2))
        out.setdefault(key, {}).setdefault((r['seed'], r['target_id']), {})[int(r['T'])] = float(r['excess_bits'])
    return out


def figure(c, res, A):
    Ts = c['T_checkpoints']
    curves = per_run_curves(res)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, a, tag in ((axs[0, 0], 0.15, '(a)'), (axs[0, 1], 0.05, '(b)')):
        for n in c['n_levels']:
            runs = curves.get((n, a), {})
            if not runs:
                continue
            m = np.array([[d[T] for T in Ts] for d in runs.values()])
            ax.plot(Ts, m.mean(0), color=LEVEL_COLORS[n], lw=2, marker='o', ms=4, label=f'n = {n}')
            ax.fill_between(Ts, np.percentile(m, 10, 0), np.percentile(m, 90, 0), color=LEVEL_COLORS[n],
                            alpha=0.15, lw=0)
        ax.axvline(c['verdict']['T_ref'], color=MUTED, lw=1, ls=':')
        ax.axhline(c['verdict']['cross_eps'], color=MUTED, lw=1, ls='--')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('T (examples seen, all tasks)', color=INK)
        ax.set_ylabel('target held-out excess loss ε (bits)', color=INK)
        ax.set_title(f'{tag} α = {a}: mean and 10–90% band of 32 runs per n', color=INK, fontsize=10, loc='left')
        ax.legend(fontsize=8, frameon=False)
        style(ax)
    cells = A['cells']
    keys = [f'n{n}_a{a:.2f}' for a in sorted(c['alphas'], reverse=True) for n in c['n_levels']]
    keys = [k for k in keys if cells[k].get('verdict') != 'INCOMPLETE']
    x = np.arange(len(keys))
    labels = [f"n={cells[k]['n']}\nα={cells[k]['alpha']}" for k in keys]
    for ax, name, tag, title, lines in (
            (axs[1, 0], 'drop', '(c)', 'drop ε(2^16) − ε(2^18) within run, seed 95% CI',
             [(c['verdict']['late_min_drop_lb'], '--'), (c['verdict']['floor_max_drop_ub'], ':'), (0, '-')]),
            (axs[1, 1], 'interference_end', '(d)', 'interference I at 2^18 vs Exp 20 α = 1 at matched target data',
             [(0, '-')])):
        for i, k in enumerate(keys):
            m, lo, hi = cells[k][name], *cells[k][name + '_ci']
            col = SERIES[0] if cells[k]['alpha'] > 0.1 else SERIES[1]
            ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt='o', color=col, ms=5, capsize=3)
        for y, ls in lines:
            ax.axhline(y, color=MUTED, lw=1, ls=ls)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('bits' if name == 'drop' else 'log2 ratio', color=INK)
        ax.set_title(f'{tag} {title}', color=INK, fontsize=10, loc='left')
        style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(EXP / f'dfa_lstm_long_budget_results.{ext}', dpi=150)


def markdown(c, res, A, notes):
    cells = A['cells']
    v = c['verdict']
    keys = [f'n{n}_a{a:.2f}' for a in sorted(c['alphas'], reverse=True) for n in c['n_levels']]
    L = ['# Results: floor or late learning? Experiment 20\'s low-share LSTM cells run 4× longer '
         '(`21_dfa_lstm_low_alpha_long_budget`)', '']
    if notes:
        L += [notes.rstrip(), '']
    L += [f"**Overall verdict (PROTOCOL Section 4.3, decided on n ∈ {{4, 8, 16}}): {A['overall_verdict']}.** "
          f"Runs analysed: {A['n_runs']} of 256; missing: {len(A['missing'])}.", '',
          '![Figure](dfa_lstm_long_budget_results.png)', '',
          '## Cell verdicts (PROTOCOL Section 4.2)', '',
          'ε in bits, means over the 32 runs of a cell; intervals are 95% seed-block bootstrap (8 seeds, 20,000 '
          'draws). Crossed = ε(2^18) ≤ 0.5 bits; moved = drop ≥ 0.1 bits; plateau = ε(2^18) ≥ 0.8 ε(2^16) and '
          '> 0.1 bits.', '',
          '| n | α | ε(2^16) | ε(2^17) | ε(2^18) | drop [95% CI] | crossed 2^16 → 2^18 | moved | plateau | '
          'verdict | predicted | right? |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    right = []
    for k in keys:
        x = cells[k]
        if x['verdict'] == 'INCOMPLETE':
            L.append(f"| {x['n']} | {x['alpha']} | | | | | | | | INCOMPLETE ({x['runs']}/32) | | |")
            continue
        ok = x['verdict'] in PREDICTED[k]
        right.append((k, ok))
        L.append(f"| {x['n']} | {x['alpha']} | {f(x['e_ref'])} | {f(x['e_mid'])} | {f(x['e_end'])} | "
                 f"{f(x['drop'])} {fci(x['drop_ci'])} | {x['crossers_ref']} → {x['crossers_end']} | "
                 f"{x['moved']}/32 | {x['plateau_end']}/32 | {x['verdict']} | "
                 f"{' or '.join(sorted(PREDICTED[k]))} | {'yes' if ok else 'no'} |")
    L += ['', f"Cell predictions right: {sum(ok for _, ok in right)} of {len(right)}. Overall prediction "
          f"LATE_LEARNING (about 60% confidence): {'right' if A['overall_verdict'] == 'LATE_LEARNING' else 'wrong'}.",
          '', 'Verdict rules (frozen): LATE_LEARNING if the drop CI lower bound > '
          f"{v['late_min_drop_lb']} bits and ≥ {v['late_min_crossers']} of 32 crossed; FLOOR_HOLDS if the upper bound "
          f"< {v['floor_max_drop_ub']} bits and ≤ {v['floor_max_crossers']} crossed; DESCENDING if the lower bound > "
          f"{v['descend_min_drop_lb']} bits; otherwise UNRESOLVED.", '',
          '## Reported, not part of the verdict (PROTOCOL Section 4.4)', '',
          '| n | α | interference I at 2^18 [CI] (median) | I at 2^16 [CI] | replicate: ε(2^16) − Exp 20 [CI] | '
          'ε(2^18) − marginals-only floor [CI] | marginals-only floor | GM T*(0.5) in target examples (censored) |',
          '|---|---|---|---|---|---|---|---|']
    for k in keys:
        x = cells[k]
        if x['verdict'] == 'INCOMPLETE':
            continue
        L.append(f"| {x['n']} | {x['alpha']} | {f(x['interference_end'], 2)} {fci(x['interference_end_ci'], 2)} "
                 f"({f(x['interference_end_median'], 2)}) | {f(x['interference_ref'], 2)} "
                 f"{fci(x['interference_ref_ci'], 2)} | {f(x['replicate_diff_ref'])} "
                 f"{fci(x['replicate_diff_ref_ci'])} | {f(x['end_minus_marginal'])} {fci(x['end_minus_marginal_ci'])} "
                 f"| {f(x['marginal_floor'])} | {x['tstar_target_examples_gm']:,.0f} ({x['tstar_censored']}/32) |")
    rd = [cells[k]['replicate_diff_ref'] for k in keys if cells[k]['verdict'] != 'INCOMPLETE']
    L += ['', f"Replicate check over all cells: mean difference {np.mean(rd):+.3f} bits (prediction: within ±0.05).",
          '', 'Interference I = log2[max(ε(2^18), 10^-3) / ε_α=1(α·2^18)], the reference being Experiment 20\'s '
          'α = 1 run of the same target and seed at the same number of target examples. I > 0: the mixed run is '
          'worse than a pure run with equal target data.', '']
    D = A.get('distractors')
    if D:
        L += ['### Distractor losses (mean held-out excess loss over the 16 distractors, from saved checkpoints)', '',
              '| α | T | mean over runs | min run | max run | runs |', '|---|---|---|---|---|---|']
        for key in sorted(D):
            a, T = key.split('_')
            d = D[key]
            L.append(f"| {a[1:]} | 2^{int(np.log2(int(T[1:])))} | {f(d['mean_distractor_bits'])} | {f(d['min'])} | "
                     f"{f(d['max'])} | {d['cells']} |")
        L.append('')
    L += ['## Mean learning curves by cell (bits)', '',
          '| n | α | ' + ' | '.join(f'2^{int(np.log2(T))}' for T in c['T_checkpoints']) + ' |',
          '|---|---|' + '---|' * len(c['T_checkpoints'])]
    for k in keys:
        x = cells[k]
        if x['verdict'] == 'INCOMPLETE':
            continue
        L.append(f"| {x['n']} | {x['alpha']} | " + ' | '.join(f(x['curve_mean'][str(T)]) for T in c['T_checkpoints'])
                 + ' |')
    ref = A['exp20_reference_curves']
    L += ['', 'Experiment 20 reference curves (seen data), mean ε at 2^16 by n: ' + '; '.join(
        f"α = {a}: " + ', '.join(f"n={n} {f(ref[f'n{n}_a{a:.2f}']['65536'])}" for n in c['n_levels'])
        for a in (0.4, 1.0)) + '.', '']
    if A['missing']:
        L += ['Missing runs: ' + ', '.join(A['missing']), '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--notes')
    args = ap.parse_args()
    c = json.loads((HERE / 'config.json').read_text())
    res = Path(args.results).resolve()
    A = json.loads((res / 'analysis.json').read_text())
    notes = Path(args.notes).read_text() if args.notes else ''
    figure(c, res, A)
    (EXP / 'RESULTS_dfa_lstm_long_budget.md').write_text(markdown(c, res, A, notes))
    print('wrote', EXP / 'RESULTS_dfa_lstm_long_budget.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
