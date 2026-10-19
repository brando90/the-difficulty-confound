"""Experiment 17b tables (and figure) from analysis_v2.json. verify_v2.py checks that tables() output appears verbatim
in RESULTS_exp17b.md.

Usage: report_v2.py [--output results] [--figure ../exp17b_results.png]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'expt_v1'
sys.path.insert(0, str(V1))
import report as R1  # noqa: E402

f3, ci, teq = R1.f3, R1.ci, R1.teq
NAMES = {'easy': 'easy (Exp 17)', 'cov16': 'cov16', 'cov32': 'cov32', 'hard': 'hard (Exp 17)', 'near': 'mixed (Exp 17)'}


def tables(a, c, c2):
    L = []
    pools = c2['pool_order']
    alphas = [0.0] + list(c['alpha_H'])
    Ts = c['T_checkpoints']
    L.append('### Gates')
    L.append('')
    L.append('| Pool | Q: hard-only still beats it zero-shot on easy targets? | H1′ hard does not subsume easy | H2′ crossover |')
    L.append('|---|---|---|---|')
    for P in pools:
        g = a['gates'][P]
        L.append(f"| {P}{' (primary)' if g['primary'] else ''} | {g['Q_hard_still_beats_easy_zero_shot']} | "
                 f"{g['H1_prime']} | {g['H2_prime']} |")
    L.append('')
    L.append('### Contrasts at T_max = 1,048,576 tokens (positive = the matched pool wins)')
    L.append('')
    L.append('| Pool | Targets | α | matched − mismatched advantage | 95% interval | pairs > 0 | log2 ratio | verdict |')
    L.append('|---|---|---|---|---|---|---|---|')
    for P in pools:
        for which, lab in (('easy_targets', 'easy (n = 4)'), ('hard_targets', 'hard (n = 32)')):
            for al in alphas:
                x = a['contrasts'][f'{P}|{which}|{al:.2f}']
                L.append(f"| {P} | {lab} | {al:g} | {f3(x['mean'])} | {ci(x['ci95'])} | {x['n_positive']}/{x['n']} | "
                         f"{f3(x['log2_ratio_mean'])} | {x['verdict']} |")
    L.append('')
    L.append('### Mean excess at T_max by training pool (bits/token; 3 seeds × 2 targets)')
    L.append('')
    pl = ['easy'] + pools + ['hard', 'near']
    L.append('| Targets | α | ' + ' | '.join(NAMES[p] for p in pl) + ' |')
    L.append('|---|---|' + '---|' * len(pl))
    for which, lab in (('easy_targets', 'easy'), ('hard_targets', 'hard')):
        for al in alphas:
            m = a['contrasts'][f'{pools[0]}|{which}|{al:.2f}']['mean_excess_at_Tmax_by_pool']
            L.append(f'| {lab} | {al:g} | ' + ' | '.join(f3(m[p]) for p in pl) + ' |')
    L.append('')
    L.append('Zero-shot (α = 0) excess per target at T_max, mean over seeds (`hmm_n004_t1` uses token 18, which the '
             'Exp 17 easy pool never emits):')
    L.append('')
    L.append('| Target | ' + ' | '.join(NAMES[p] for p in pl) + ' |')
    L.append('|---|' + '---|' * len(pl))
    for t in a['easy_targets'] + a['hard_targets']:
        L.append(f'| {t} | ' + ' | '.join(f3(a["per_target_alpha0"][f"{p}|{t}"]) for p in pl) + ' |')
    L.append('')
    L.append('Share of the Exp 17 zero-shot gap closed on easy targets, (ε_easy − ε_P)/(ε_easy − ε_hard), and '
             'ε_easy − ε_P:')
    L.append('')
    L.append('| Pool | gap closed (α = 0) | 95% interval | easy − P, α = 0 | easy − P, α = 0.05 |')
    L.append('|---|---|---|---|---|')
    for P in pools:
        g = a['gap_closed'][P]
        v0, v5 = a['vs_easy'][f'{P}|0.00'], a['vs_easy'][f'{P}|{alphas[1]:.2f}']
        L.append(f"| {P} | {f3(g['estimate'])} | {ci(g['ci95'])} | {f3(v0['mean'])} {ci(v0['ci95'])} | "
                 f"{f3(v5['mean'])} {ci(v5['ci95'])} |")
    L.append('')
    L.append('Advantage at every checkpoint (mean over pairs, bits/token; positive = matched pool wins):')
    L.append('')
    L.append('| Pool | Targets | α | ' + ' | '.join(f'2^{int(T).bit_length() - 1}' for T in Ts) + ' |')
    L.append('|---|---|---|' + '---|' * len(Ts))
    for P in pools:
        for which, lab in (('easy_targets', 'easy'), ('hard_targets', 'hard')):
            for al in alphas:
                x = a['contrasts'][f'{P}|{which}|{al:.2f}']
                L.append(f'| {P} | {lab} | {al:g} | ' + ' | '.join(f3(v) for v in x['per_T']) + ' |')
    L.append('')
    L.append('Worth of each new pool in target-only tokens (median over seeds; "<" below 2^16, ">" beyond 2^20):')
    L.append('')
    L.append('| Target | α | ' + ' | '.join(pools) + ' |')
    L.append('|---|---|' + '---|' * len(pools))
    for t in a['easy_targets'] + a['hard_targets']:
        for al in alphas:
            L.append(f'| {t} | {al:g} | ' + ' | '.join(teq(a['t_equiv'][f'{P}|{t}|{al:.2f}']) for P in pools) + ' |')
    L.append('')
    pe = a['pool_excess_at_Tmax']
    L.append(f"Training-pool excess at T_max (sanity: every new run learned its own pool): mean {f3(pe['mean'])}, "
             f"max {f3(pe['max'])} bits/token over {pe['n']} pool-concept evaluations.")
    L.append('')
    return '\n'.join(L)


def plot(a, c, c2, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Ts = c['T_checkpoints']
    pl = ['easy'] + c2['pool_order'] + ['hard']
    colors = {'easy': '#9aa5b1', 'cov16': '#5b8def', 'cov32': '#1f4fbf', 'hard': '#d9822b'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, (which, targets) in zip(axes, (('easy targets (n = 4)', a['easy_targets']),
                                           ('hard targets (n = 32)', a['hard_targets']))):
        for p in pl:
            ys = [sum(a['cells'][f'{p}|{t}|0.00'][str(s)][i] for t in targets for s in c['seeds']) /
                  (len(targets) * len(c['seeds'])) for i in range(len(Ts))]
            ax.plot(Ts, ys, marker='o', ms=4, lw=2, color=colors[p], label=NAMES[p])
        ax.set_xscale('log', base=2)
        ax.set_title(f'Zero-shot excess on {which}', fontsize=10)
        ax.set_xlabel('training tokens')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('excess, bits/token')
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(HERE / 'results'))
    ap.add_argument('--figure', default=str(HERE.parent / 'exp17b_results.png'))
    args = ap.parse_args()
    out = Path(args.output)
    c = json.loads((V1 / 'config.json').read_text())
    c2 = json.loads((HERE / 'config_v2.json').read_text())
    a = json.loads((out / 'analysis_v2.json').read_text())
    print(tables(a, c, c2))
    if args.figure:
        plot(a, c, c2, args.figure)


if __name__ == '__main__':
    main()
