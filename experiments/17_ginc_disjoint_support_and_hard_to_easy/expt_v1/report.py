"""Experiment 17 report: gate tables (verify.py checks them against RESULTS_exp17.md) and the results figure.

Usage: report.py [--output results]   prints the tables and writes ../exp17_results.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

COL = {'easy': '#0072B2', 'hard': '#D55E00', 'near': '#7F7F7F', 'far': '#CC79A7', 'none': '#000000'}


def f3(x):
    return f'{x:.3f}'


def ci(v):
    return f'[{v[0]:.3f}, {v[1]:.3f}]'


def teq(lst):
    vals = sorted(lst, key=lambda x: x[1])
    med = vals[len(vals) // 2]
    return f"{med[0] if med[0] != '=' else ''}{med[1] / 1024:.0f}k".replace('=', '')


def tables(a, c, tasks):
    H, D = a['H'], a['D']
    Tm = H['Tmax']
    L = []
    L.append(f'### Arm H (primary): hard -> easy, same family (T_max = {Tm:,} tokens)\n')
    L.append('Δ = excess(mismatched pool) − excess(matched pool) at T_max, bits/token; positive = the matched pool '
             'wins. Intervals: two-stage bootstrap over seeds and targets.\n')
    L.append('| Targets | α | mismatched − matched | 95% interval | pairs > 0 | log2 ratio | excess: easy pool | '
             'hard pool | mixed pool | verdict |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for which in ('easy_targets', 'hard_targets'):
        for al in [0.0] + list(c['alpha_H']):
            x = H['contrasts'][f'{which}|{al:.2f}']
            m = x['mean_excess_at_Tmax_by_pool']
            L.append(f"| {which.split('_')[0]} (n = {4 if which == 'easy_targets' else 32}) | {al:g} | "
                     f"{f3(x['mean'])} | {ci(x['ci95'])} | {x['n_positive']}/{x['n']} | {f3(x['log2_ratio_mean'])} | "
                     f"{f3(m['easy'])} | {f3(m['hard'])} | {f3(m['near'])} | {x['verdict']} |")
    L.append('')
    L.append(f"**H1 hard does not subsume easy: {H['H1_hard_does_not_subsume_easy']}. "
             f"H2 crossover: {H['H2_crossover']}.**\n")
    L.append('Worth of each pool in target-only tokens (median over seeds of the α = 1 tokens reaching the same '
             'excess; "<" below the first checkpoint, ">" beyond the last):\n')
    L.append('| Target | α | easy pool | hard pool | mixed pool |')
    L.append('|---|---|---|---|---|')
    for t in H['easy_targets'] + H['hard_targets']:
        for al in [0.0] + list(c['alpha_H']):
            L.append(f'| {t} | {al:g} | ' + ' | '.join(teq(H['t_equiv'][f'{p}|{t}|{al:.2f}'])
                                                     for p in ('easy', 'hard', 'near')) + ' |')
    L.append('')
    L.append('Δ at every checkpoint (mean over pairs, bits/token):\n')
    Ts = a['T_checkpoints']
    L.append('| Targets | α | ' + ' | '.join(f'2^{int(np.log2(T))}' for T in Ts) + ' |')
    L.append('|---|---|' + '---|' * len(Ts))
    for which in ('easy_targets', 'hard_targets'):
        for al in [0.0] + list(c['alpha_H']):
            x = H['contrasts'][f'{which}|{al:.2f}']
            L.append(f"| {which.split('_')[0]} | {al:g} | " + ' | '.join(f3(v) for v in x['per_T']) + ' |')
    L.append('')
    es = D['eps_star']
    L.append(f'### Arm D (secondary): disjoint support (ε* = {es:g} bits/token)\n')
    d1 = D['D1_zero_shot']
    L.append('| Quantity | Estimate | 95% interval | Verdict |')
    L.append('|---|---|---|---|')
    L.append(f"| D1 zero-shot excess, far − near (far {f3(d1['far_mean'])}, near {f3(d1['near_mean'])}) | "
             f"{f3(d1['mean'])} | {ci(d1['ci95'])} ({d1['n_positive']}/{d1['n']} > 0) | {d1['verdict']} |")
    for al in c['alpha_D']:
        x = D[f'far_minus_near|{al:.2f}']
        L.append(f"| excess at T_max, far − near, α = {al:g} | {f3(x['mean'])} | {ci(x['ci95'])} "
                 f"({x['n_positive']}/{x['n']} > 0) | reported |")
    for pool in ('near', 'far'):
        e = D[pool]['E'][f'{es:g}']
        L.append(f"| E_{pool} = e_α/e_K (M_K_both, K in bits) | {f3(D[pool]['E_star'])} | "
                 f"seed {ci(D[pool]['E_star_ci95_seed'])}; per seed {', '.join(f3(x) for x in e['E_per_seed'])} | "
                 f"{D['D2_E_far_above_1'] if pool == 'far' else 'reported'} |")
    d3 = D['D3_E_far_minus_near']
    L.append(f"| D3 E_far − E_near | {f3(d3['diff'])} | per seed {', '.join(f3(x) for x in d3['per_seed'])} | "
             f"{d3['verdict']} |")
    L.append(f"| η (M_align_eff), near / far | {f3(D['eta_near'])} / {f3(D['eta_far'])} | — | reported |")
    for pool in ('near', 'far'):
        mf = D['model_free'][pool]
        v = f3(mf['E_mf']) if mf['E_mf'] is not None else 'n/a'
        L.append(f"| model-free E_{pool} (T* interpolated; {mf['n_alpha']} α pairs, {mf['n_K']} K pairs, "
                 f"{mf['censored_alpha']} censored) | {v} | — | reported |")
    L.append('')
    L.append('E across the error target:\n')
    eps = sorted(D['near']['E'], key=float)
    L.append('| ε | ' + ' | '.join(eps) + ' |')
    L.append('|---|' + '---|' * len(eps))
    for pool in ('near', 'far'):
        L.append(f'| E_{pool} | ' + ' | '.join(f3(D[pool]['E'][e]['E']) for e in eps) + ' |')
    L.append('')
    L.append('Cell means of excess at T_max (bits/token):\n')
    L.append('| Pool | n | α | excess |')
    L.append('|---|---|---|---|')
    for k, v in a['cell_means'].items():
        p, n, al, T = k.split('|')
        if int(T) == Tm:
            L.append(f'| {p} | {n} | {al} | {f3(v)} |')
    L.append('')
    pe = a['pool_excess_at_Tmax']
    L.append(f"Training-pool excess at T_max (sanity: every run learned its own pool): mean {f3(pe['mean'])}, "
             f"max {f3(pe['max'])} bits/token over {pe['n']} pool-concept evaluations.\n")
    return '\n'.join(L)


def plot(a, c, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H, D = a['H'], a['D']
    Ts = np.array(a['T_checkpoints'])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    seeds = [str(s) for s in a['seeds']]
    # 1: zero-shot and alpha_H excess at T_max by pool, easy vs hard targets
    xs, labels = 0, []
    for which, tl in (('easy_targets', H['easy_targets']), ('hard_targets', H['hard_targets'])):
        for al in [0.0] + list(c['alpha_H']):
            for j, p in enumerate(('easy', 'hard', 'near')):
                vals = [H['cells'][f'{p}|{t}|{al:.2f}'][s][-1] for t in tl for s in seeds]
                ax[0].bar(xs + j * 0.27, np.mean(vals), 0.25, color=COL[p],
                          label={'near': 'mixed'}.get(p, p) + ' pool' if xs == 0 else None)
                ax[0].scatter([xs + j * 0.27] * len(vals), vals, s=8, color='k', zorder=3)
            labels.append((xs + 0.27, f"{which.split('_')[0]}\nα={al:g}"))
            xs += 1
    ax[0].set_xticks([x for x, _ in labels], [l for _, l in labels])
    ax[0].set_yscale('log')
    ax[0].set_ylabel('excess at T_max (bits/token)')
    ax[0].set_title('H: matched vs mismatched pool')
    ax[0].legend(frameon=False, fontsize=8)
    # 2: easy targets alpha_H curves
    al = c['alpha_H'][0]
    for p in ('easy', 'hard', 'near'):
        m = np.mean([[H['cells'][f'{p}|{t}|{al:.2f}'][s] for s in seeds] for t in H['easy_targets']], axis=(0, 1))
        ax[1].plot(Ts, m, 'o-', color=COL[p], label={'near': 'mixed'}.get(p, p) + f' pool, α={al:g}')
        m2 = np.mean([[H['cells'][f'{p}|{t}|0.00'][s] for s in seeds] for t in H['easy_targets']], axis=(0, 1))
        ax[1].plot(Ts, m2, 's--', color=COL[p], alpha=0.6, label={'near': 'mixed'}.get(p, p) + ' pool, α=0')
    ax[1].set_xscale('log', base=2)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('training tokens T')
    ax[1].set_title('H: easy (n=4) targets')
    ax[1].legend(frameon=False, fontsize=7)
    # 3: hard targets
    for p in ('easy', 'hard', 'near'):
        m = np.mean([[H['cells'][f'{p}|{t}|{al:.2f}'][s] for s in seeds] for t in H['hard_targets']], axis=(0, 1))
        ax[2].plot(Ts, m, 'o-', color=COL[p], label={'near': 'mixed'}.get(p, p) + f' pool, α={al:g}')
    ax[2].set_xscale('log', base=2)
    ax[2].set_yscale('log')
    ax[2].set_xlabel('training tokens T')
    ax[2].set_title('H: hard (n=32) targets')
    ax[2].legend(frameon=False, fontsize=7)
    # 4: E across eps, near vs far
    eps = sorted(D['near']['E'], key=float)
    for pool in ('near', 'far'):
        ax[3].plot([float(e) for e in eps], [D[pool]['E'][e]['E'] for e in eps], 'o-', color=COL[pool],
                   label=f'E_{pool}')
    ax[3].axhline(1.0, color='k', lw=0.8, ls=':')
    ax[3].axvline(D['eps_star'], color='k', lw=0.8, ls='--')
    ax[3].set_xscale('log')
    ax[3].set_xlabel('error target ε (bits/token)')
    ax[3].set_ylabel('E = e_α / e_K')
    ax[3].set_title('D: alignment vs complexity')
    ax[3].legend(frameon=False, fontsize=8)
    for x in ax:
        x.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(HERE / 'results'))
    args = ap.parse_args()
    out = Path(args.output)
    c = json.loads((HERE / 'config.json').read_text())
    a = json.loads((out / 'analysis.json').read_text())
    tasks = json.loads((out / 'tasks.json').read_text())
    print(tables(a, c, tasks))
    plot(a, c, HERE.parent / 'exp17_results.png')


if __name__ == '__main__':
    main()
