"""Write RESULTS_carry_crossed.md (gate table, fitted laws, multipliers, mechanism tables) and figures for every arm
that has results. Figures need matplotlib; without it the report is written without them.

    python report.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
ARMS = {'transformer': '16b (primary, GPU transformers, 8-digit)', 'mlp': '16a (secondary, CPU MLP, 3-digit)'}
GATES = [('G1', 'Chain length is a real data cost (model-free M_s at top alpha, largest model)'),
         ('G2', 'Chain length adds held-out value beyond alignment'),
         ('G3a', 'Coefficient picture'), ('G3b', 'Exponent picture'),
         ('G4', 'Alignment dominates in units of data (E and E_mf)'),
         ('G5', 'Same C3 verdict as exact complexity (Experiments 11, 12)'),
         ('G6', 'Bigger models shrink the chain multiplier'), ('G7', 'Chain cost needs a sequential output format')]


def fmt_i(d, pct=False):
    if not isinstance(d, dict) or 'estimate' not in d:
        return 'n/a'
    f = (lambda v: f'{100 * v:.2f}%') if pct else (lambda v: f'{v:.3g}' if isinstance(v, (int, float)) else str(v))
    return f"{f(d['estimate'])} [{f(d['lo'])}, {f(d['hi'])}]"


def gate_rows(g):
    rows = []
    for name, desc in GATES:
        if name in ('G3a', 'G3b'):
            est = f"kappa {fmt_i(g['G3']['kappa'])}; gamma1 {fmt_i(g['G3']['gamma1'])}; exp vs coef {fmt_i(g['G3']['exp_vs_coef'], True)}"
            rows.append((name, desc, est, g['G3'][name]))
        elif name in g:
            v = g[name]
            est = {'G1': lambda: f"M_s {fmt_i(v)}", 'G2': lambda: f"{v['k_law']}: {fmt_i(v['reduction'], True)}; held-out level {100 * v['reduction_heldout_level']:.2f}%; invalid {v['invalid']}",
                   'G4': lambda: f"E {fmt_i(v['E'])}; E_mf {v['E_mf']['estimate']:.3g} at eps* {v['E_mf_eps_star']}",
                   'G5': lambda: f"this: {v['this']}; others: {v['others']}; reason: {v['reason']}",
                   'G6': lambda: f"M_s(25M)/M_s(1M) {fmt_i(v['ratio'])}; monotone {v['monotone']}",
                   'G7': lambda: f"log M_s(plain)/log M_s(reversed) {fmt_i(v['log_ratio'])}"}[name]()
            rows.append((name, desc, est, v['verdict']))
    return rows


def figures(arm, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = list(csv.DictReader((out / 'metrics.csv').open()))
    tr = [r for r in rows if r['eval_stratum'] == r['target_id']]
    largest = max(tr, key=lambda r: float(r['N']))['N']
    tr = [r for r in tr if r['N'] == largest and r['format'] in ('mlp', 'plain')]
    alphas = sorted({float(r['alpha']) for r in tr})
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 3.4), sharey=True)
    for ax, a in zip(np.atleast_1d(axes), alphas):
        for s in sorted({float(r['s']) for r in tr}):
            pts = sorted((float(r['T']), np.mean([float(x['excess_bits']) for x in tr if x['s'] == r['s'] and x['alpha'] == r['alpha'] and x['T'] == r['T']]))
                         for r in tr if float(r['s']) == s and float(r['alpha']) == a)
            pts = sorted(set(pts))
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], marker='o', label=f's={s:g}')
        ax.set_title(f'alpha = {a:g}')
        ax.set_xlabel('training examples T')
    np.atleast_1d(axes)[0].set_ylabel('excess bits per answer (seed mean)')
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    path = EXP / f'carry_crossed_{arm}.png'
    fig.savefig(path, dpi=150)
    return path.name


def main():
    lines = ['# Results: carry-chain length crossed with alignment (`16_carry_alignment_crossed_arithmetic`)', '',
             'Carry-chain length c is a structural demand, not Kolmogorov complexity; the law variable is s = c + 1. '
             'Intervals are 95% seed-block bootstrap intervals (20,000 draws); p-val=n/a.', '']
    for arm, title in ARMS.items():
        out = ROOT / 'results' / arm
        if not (out / 'gates.json').exists():
            lines += [f'## {title}', '', f'{arm}: pending, not run.', '']
            continue
        g = json.loads((out / 'gates.json').read_text())
        fits = json.loads((out / 'fits.json').read_text())
        lines += [f'## {title}', '', f"eps* = {g['eps_star']} bits per answer; {g['n_seeds']} seeds "
                  f"({g['distinct_resamples']} distinct bootstrap resamples).", '',
                  '| Gate | Question | Estimate | Verdict |', '|---|---|---|---|']
        lines += [f'| {a} | {b} | {c} | **{d}** |' for a, b, c, d in gate_rows(g)]
        lines += ['', '**Held-out RMSE (log2 bits), all held-out rows:**', '', '| Law | RMSE | Invalid forecasts |', '|---|---|---|']
        lines += [f"| `{law}` | {fmt_i(v)} | {g['heldout_invalid'][law]} |" for law, v in g['heldout_rmse'].items()]
        lines += ['', '**Fitted `M_K_both` per seed (fit rows):**', '', '| Seed | kappa | gamma0 | gamma1 | eta | beta |', '|---|---|---|---|---|---|']
        for seed, f in fits['fits'].items():
            p = f['M_K_both']['params']
            lines.append(f"| {seed} | {p['kappa']:.3f} | {p['gamma0']:.3f} | {p['gamma1']:.3f} | {p['eta']:.3f} | {p['beta']:.3f} |")
        d = g['draft']
        lines += ['', '**Draft gates of the protocol (reported, not gated):** '
                  f"law-based M_s {fmt_i(d['law_M_s'])}; M_alpha {fmt_i(d['M_alpha'])}; range ratio R {fmt_i(d['range_ratio_R'])}; "
                  f"rho {fmt_i(d['rho'])}; gap ratio Delta({d['gap_T'][1]})/Delta({d['gap_T'][0]}) {fmt_i(d['gap_ratio_Delta'])}.", '',
                  '**Multiplier curves (law-based):** ' + '; '.join(f"eps {k}: M_s {v['M_s']:.3g}, E {v['E']}" for k, v in g['multiplier_curves'].items()), '',
                  f"**Sensitivity:** {json.dumps(g['sensitivity'])}", '',
                  f"**Per-digit excess at the last checkpoint (top alpha):** {json.dumps(g['mechanism']['per_digit'])}", '',
                  f"**Independent fitter:** {g['independent_fit']['status']} (max relative difference {g['independent_fit']['max_relative_difference']:.2e}).", '']
        if (out / 'mixes.json').exists() and arm == 'mlp':
            mixes = json.loads((out / 'mixes.json').read_text())
            rates = {}
            for m in mixes:
                _, target, alpha, _ = m['run'].split('_')
                rates.setdefault((target, alpha), m['carry_rates'])
            lines += ['**Column-carry rates per mix (confound, logged):**', '', '| Target | alpha | units | tens |', '|---|---|---|---|']
            lines += [f'| {t} | {a} | {r["units"]:.4f} | {r["tens"]:.4f} |' for (t, a), r in sorted(rates.items())]
            lines.append('')
        fig = figures(arm, out)
        if fig:
            lines += [f'![{arm}]({fig})', '']
    (EXP / 'RESULTS_carry_crossed.md').write_text('\n'.join(lines) + '\n')
    print('wrote', EXP / 'RESULTS_carry_crossed.md')


if __name__ == '__main__':
    main()
