"""Write RESULTS_dfa_complexity_alignment.md and the four-panel figure from results/analysis.json.

  report.py [--results results] [--notes notes.md]
`--notes` is a markdown file with the hand-written sections (takeaway, pilot disclosure, deviations); it is inserted
verbatim after the title.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

import analyze as AN  # noqa: E402

LEVEL_COLORS = {4: '#86b6ef', 8: '#3987e5', 16: '#1c5cab', 32: '#0d366b'}  # ordinal blue ramp
SERIES = ['#2a78d6', '#eb6834']
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dc'


def fmt(x, d=3):
    if x is None:
        return 'n/a'
    if isinstance(x, (list, tuple)):
        return '[' + ', '.join(fmt(v, d) for v in x) + ']'
    if isinstance(x, str):
        return x
    if not np.isfinite(x) or abs(x) >= 1e299:
        return '∞'
    return f'{x:.{d}g}'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def figure(c, res, A, tasks):
    import csv
    rows = list(csv.DictReader(open(res / 'metrics.csv')))
    Ts = c['T_checkpoints']
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    # (a) alpha = 1 curves by level: mean over targets and seeds, band = min..max of seed means
    ax = axs[0, 0]
    for n in c['n_levels']:
        per_seed = {}
        for r in rows:
            if int(r['K']) == n and float(r['alpha']) == 1.0:
                per_seed.setdefault(int(r['seed']), {}).setdefault(int(r['T']), []).append(float(r['excess_bits']))
        m = np.array([[np.mean(per_seed[s][T]) for T in Ts] for s in sorted(per_seed)])
        ax.plot(Ts, m.mean(0), color=LEVEL_COLORS[n], lw=2, marker='o', ms=4, label=f'n = {n}')
        ax.fill_between(Ts, m.min(0), m.max(0), color=LEVEL_COLORS[n], alpha=0.18, lw=0)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('T (examples seen)', color=INK)
    ax.set_ylabel('held-out excess loss ε (bits)', color=INK)
    ax.set_title('(a) α = 1 learning curves by minimal-DFA size', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    # (b) T* heatmap n x alpha at the primary eps*
    ax = axs[0, 1]
    es = str(c['eps_star_primary'])
    H = np.zeros((len(c['n_levels']), len(c['alphas'])))
    Cn = np.zeros_like(H)
    for i, n in enumerate(c['n_levels']):
        tids = [t['task_id'] for t in tasks if t['role'] == 'target' and t['K'] == n]
        for j, a in enumerate(c['alphas']):
            H[i, j] = np.mean([np.log2(A['tstar_summary'][t][es][str(a)]['gm']) for t in tids])
            Cn[i, j] = sum(A['tstar_summary'][t][es][str(a)]['censored'] for t in tids)
    im = ax.imshow(H, cmap=SEQ, aspect='auto', origin='lower')
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f'2^{H[i, j]:.1f}\n{int(Cn[i, j])}/32 cens.', ha='center', va='center', fontsize=7,
                    color='white' if H[i, j] > (H.min() + H.max()) / 2 else INK)
    ax.set_xticks(range(len(c['alphas'])), [str(a) for a in c['alphas']])
    ax.set_yticks(range(len(c['n_levels'])), [str(n) for n in c['n_levels']])
    ax.set_xlabel('target share α', color=INK)
    ax.set_ylabel('minimal DFA states n', color=INK)
    ax.set_title(f'(b) GM T* at ε* = {es} bits (log2; censored runs at T bounds)', color=INK, fontsize=10, loc='left')
    fig.colorbar(im, ax=ax, label='log2 T*')
    ax.tick_params(colors=MUTED, labelsize=8)
    # (c) E versus eps*
    ax = axs[1, 0]
    cur = A['E']['curve']
    E = np.array([v if np.isfinite(v) else np.nan for v in cur['E_bits']])
    ax.plot(cur['eps_star'], E, color=SERIES[0], lw=2, label='E (K in bits)')
    ax.plot(cur['eps_star'], E / A['E']['log_Kbits_on_log_n_slope'], color=SERIES[1], lw=2, label='E (K in states)')
    ax.axhline(1, color=MUTED, lw=1, ls='--')
    if A['E']['eps_flip']:
        ax.axvline(A['E']['eps_flip'], color=MUTED, lw=1, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('ε* (bits)', color=INK)
    ax.set_ylabel('E = e_α / e_K', color=INK)
    ax.set_title('(c) alignment-to-complexity elasticity ratio (M_K_both, all points)', color=INK, fontsize=10,
                 loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    # (d) held-out forecast versus observed
    ax = axs[1, 1]
    fr = json.loads((res / 'forecasts_frozen.json').read_text())
    obs = {(int(r['seed']), r['target_id'], float(r['alpha']), int(r['T'])): float(r['excess_bits']) for r in rows}
    k_law = A['g2_k_law']
    for law, col in ((k_law, SERIES[0]), ('M_align_eff', SERIES[1])):
        xs, ys = [], []
        for s, f in fr['forecasts']['K_bits'].items():
            for p, v in zip(fr['points'], f[law]):
                xs.append(obs[(int(s), p['target_id'], p['alpha'], p['T'])])
                ys.append(v)
        ax.scatter(xs, ys, s=9, color=col, alpha=0.5, lw=0, label=law)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('observed ε (bits), held-out points', color=INK)
    ax.set_ylabel('frozen forecast (bits)', color=INK)
    ax.set_title('(d) held-out forecasts: G2 K law vs M_align_eff', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(EXP / f'dfa_complexity_alignment_results.{ext}', dpi=150)


def markdown(c, res, A, tasks, notes):
    V = json.loads((res / 'verification.json').read_text()) if (res / 'verification.json').exists() else {}
    g = A
    L = []
    L.append('# Results: exact DFA complexity versus data alignment with an LSTM learner (`20_dfa_lstm_complexity_vs_alignment`)\n')
    L.append(notes.strip() + '\n')
    L.append('## Gate table\n')
    L.append('All gates are reported with equal prominence. Intervals are 95% seed-block bootstrap percentile '
             'intervals (20,000 draws over the 8 seeds); "two-stage" also resamples the 4 targets within each level.\n')
    L.append('| Gate | Statistic | Estimate | 95% interval(s) | Threshold | Verdict |')
    L.append('|---|---|---|---|---|---|')
    L.append(f"| G1 complexity is real | {g['G1']['statistic']} (ε* = {g['G1']['eps_star']} bits) | "
             f"{fmt(g['G1']['estimate'])} | seed {fmt(g['G1']['ci_seed'])}; two-stage {fmt(g['G1']['ci_two_stage'])} | "
             f"{g['G1']['threshold']} | **{g['G1']['verdict']}** |")
    L.append(f"| G2 K adds value beyond alignment | {g['G2']['statistic']} | {fmt(g['G2']['estimate'])} | "
             f"{fmt(g['G2']['ci_seed'])} | {g['G2']['threshold']} | **{g['G2']['verdict']}** |")
    L.append(f"| G3a coefficient (MDL) | κ in M_K_both; γ1 interval contains 0; RMSE coef ≤ exp | κ = {fmt(g['G3']['kappa'])}, "
             f"γ1 = {fmt(g['G3']['gamma1'])} | κ {fmt(g['G3']['kappa_ci'])}; γ1 {fmt(g['G3']['gamma1_ci'])} | "
             f"κ LB > 0 | **{g['G3']['G3a']}** |")
    L.append(f"| G3b exponent (quanta) | γ1 in M_K_both; M_K_exp gain over M_K_coef | γ1 = {fmt(g['G3']['gamma1'])}, "
             f"gain = {fmt(g['G3']['exp_over_coef_gain'])} | γ1 {fmt(g['G3']['gamma1_ci'])}; gain "
             f"{fmt(g['G3']['exp_over_coef_gain_ci'])} | γ1 UB < 0, gain ≥ 5%, LB > 0 | **{g['G3']['G3b']}** |")
    L.append(f"| G4 alignment dominates | {g['G4']['statistic']} at ε* = {g['G4']['eps_star']} | bits {fmt(g['G4']['estimate_bits'])}; "
             f"states {fmt(g['G4']['estimate_states'])}; E_mf {fmt(g['G4']['E_mf'])} (ε* = {g['G4']['E_mf_eps_star']}) | "
             f"bits {fmt(g['G4']['ci_bits'])}; states {fmt(g['G4']['ci_states'])} | {g['G4']['threshold']} | "
             f"**{g['G4']['verdict']}** |")
    L.append(f"| G5 gap shrinks | {g['G5']['statistic']} | {fmt(g['G5']['estimate'])} (gap {fmt(g['G5']['g_ref'])} → "
             f"{fmt(g['G5']['g_max'])} bits) | {fmt(g['G5']['ci_seed'])} | {g['G5']['threshold']} | **{g['G5']['verdict']}** |")
    L.append(f"| G6 alignment floor | {g['G6']['statistic']} | a1 = {fmt(g['G6']['a1'])}; gain = {fmt(g['G6']['estimate'])} | "
             f"a1 {fmt(g['G6']['a1_ci'])}; gain {fmt(g['G6']['ci_seed'])} | {g['G6']['threshold']} | **{g['G6']['verdict']}** |")
    L.append(f"| G7 quanta mechanism | {g['G7']['statistic']} | {fmt(g['G7']['estimate'])} (partial given log K: "
             f"{fmt(g['G7']['partial_given_logK'])}) | seed {fmt(g['G7']['ci_seed'])}; two-stage {fmt(g['G7']['ci_two_stage'])} | "
             f"{g['G7']['threshold']} | **{g['G7']['verdict']}** |")
    L.append(f"\nC3 reading: **{g['G3']['C3']}**.\n")
    L.append('## Target table\n')
    es_levels = c['eps_star_levels']
    L.append('| id | n | K_bits | acceptance | sensitivity | τ | γ_j (α=1) | ' +
             ' | '.join(f'GM T*({e}) at α=1 [censored/8]' for e in es_levels) + ' |')
    L.append('|---|---|---|---|---|---|---|' + '---|' * len(es_levels))
    for t in sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: t['task_token']):
        ts = A['tstar_summary'][t['task_id']]
        L.append(f"| {t['task_id']} | {t['K']} | {t['K_bits']} | {t['acceptance_rate']:.3f} | {t['sensitivity']:.3f} | "
                 f"{t['visit_tail_steepness']:.2f} | {A['G7']['gamma_j'][t['task_id']]:.3f} | " +
                 ' | '.join(f"{ts[str(e)]['1.0']['gm']:.0f} [{ts[str(e)]['1.0']['censored']}]" for e in es_levels) + ' |')
    L.append('\n## Held-out RMSE (log2 bits, censored residual) by slice\n')
    L.append('| law | all (K bits) | K = 16 | α = 0.15 | T = T_max | all (K in states) | invalid forecasts |')
    L.append('|---|---|---|---|---|---|---|')
    for law, v in A['heldout_rmse_log2']['K_bits'].items():
        L.append(f"| {law} | {v['all']:.4f} | {v['K16']:.4f} | {v['alpha015']:.4f} | {v['Tmax']:.4f} | "
                 f"{A['heldout_rmse_log2']['K'][law]['all']:.4f} | {A['heldout_invalid_forecasts']['K_bits'][law]} |")
    L.append(f"\nHeld-out points: {A['heldout_points_per_seed']} per seed. G2 K law (chosen on fit cells before held-out "
             f"rows were read): `{A['g2_k_law']}`. Frozen forecasts SHA-256 `{A['forecasts_frozen_sha256']}`. Fitter "
             f"agreement (laws.py vs independent_fit.py): {A['fitter_agreement']['all_agree']}, max relative forecast "
             f"difference {A['fitter_agreement']['max_rel_forecast_diff']:.2e}.\n")
    al = A['alignment']
    reading = ('pure effective data' if al['eta_k_law_ci'][0] <= 1 <= al['eta_k_law_ci'][1] else
               'transfer (distractors help)' if al['eta_k_law_ci'][1] < 1 else 'interference (distractors hurt)')
    L.append('## Alignment: η and the two alignment laws\n')
    L.append(f"- η from M_align_eff: {fmt(al['eta_align_eff'])} {fmt(al['eta_align_eff_ci'])}; from `{A['g2_k_law']}`: "
             f"{fmt(al['eta_k_law'])} {fmt(al['eta_k_law_ci'])}; from the all-points M_K_both refit: "
             f"{fmt(al['eta_refit_both'])} {fmt(al['eta_refit_both_ci'])}. Reading (G2 K law interval versus 1): **{reading}**.")
    L.append(f"- M_align_eff versus M_align_add held-out RMSE: {al['rmse_align_eff']:.4f} versus {al['rmse_align_add']:.4f}; "
             f"relative reduction {fmt(al['eff_over_add_gain'])} {fmt(al['eff_over_add_gain_ci'])}; invalid forecasts "
             f"{al['invalid_align_eff']} versus {al['invalid_align_add']}.\n")
    L.append('## Elasticity ratio E and range ratio R\n')
    L.append('| ε* (bits) | E (K in bits) | 95% interval | E (K in states) | e_α | e_K (bits) |')
    L.append('|---|---|---|---|---|---|')
    for es, v in A['E']['levels'].items():
        L.append(f"| {es} | {fmt(v['E_bits'])} | {fmt(v['ci_bits'])} | {fmt(v['E_states'])} | {fmt(v['e_alpha'])} | "
                 f"{fmt(v['e_K_bits'])} |")
    L.append(f"\nFlip point ε_flip (E = 1) in [1e-3, 1] bits: {fmt(A['E']['eps_flip'])}. Slope of log K_bits on log n: "
             f"{A['E']['log_Kbits_on_log_n_slope']:.3f}. E from a refit with K = n: {fmt(A['E']['E_refit_states'])} "
             f"{fmt(A['E']['E_refit_states_ci'])}. Model-free E_mf = {fmt(A['E']['E_mf'])} at ε* = {A['E']['E_mf_eps_star']} "
             f"(censored fraction per seed {fmt(A['E']['E_mf_censored_fraction_per_seed'], 2)}).\n")
    R = A['range_ratio_R']
    L.append(f"Range ratio R = {fmt(R['estimate'])} {fmt(R['ci_seed'])}: alignment multiplier T*(α=0.05)/T*(α=1) = "
             f"{fmt(R['alignment_multiplier'])} versus complexity multiplier M_K = {fmt(R['complexity_multiplier'])}.\n")
    kv = A['K_vs_sensitivity']
    L.append('## K versus sensitivity\n')
    L.append(f"Regression of seed-mean log2 T*_j(α = 1, {kv['eps_star']} bits) on standardized log K_bits and "
             f"standardized sensitivity (16 targets): log K coefficient {fmt(kv['coef_logK'])} (seed {fmt(kv['coef_logK_ci_seed'])}, "
             f"two-stage {fmt(kv['coef_logK_ci_two_stage'])}); sensitivity coefficient {fmt(kv['coef_sensitivity'])} "
             f"(seed {fmt(kv['coef_sensitivity_ci_seed'])}, two-stage {fmt(kv['coef_sensitivity_ci_two_stage'])}); "
             f"correlation {fmt(kv['corr_logK_sensitivity'])}, VIF {fmt(kv['vif'])}. Censored T* at α = 1: "
             f"{kv['alpha1_censored']}.\n")
    L.append('## Plateau census and per-length loss\n')
    pc = A['plateau_census']
    L.append(f"Runs flagged as plateaued (ε(T_max) ≥ 0.8 ε(2^14) and ε(T_max) > 0.1 bits), out of "
             f"{pc['runs_per_cell']} runs per (level, α) cell:\n")
    L.append('| n | ' + ' | '.join(f'α = {a}' for a in c['alphas']) + ' |')
    L.append('|---|' + '---|' * len(c['alphas']))
    for n, v in pc['counts'].items():
        L.append(f'| {n} | ' + ' | '.join(str(v[str(a)]) for a in c['alphas']) + ' |')
    pl = A['per_length_loss_alpha1_Tmax']
    L.append('\nPer-length loss at α = 1, T = T_max (bits), by level:\n')
    L.append('| n | ' + ' | '.join(str(l) for l in pl['lengths']) + ' |')
    L.append('|---|' + '---|' * len(pl['lengths']))
    for n, v in pl['by_level'].items():
        L.append(f'| {n} | ' + ' | '.join(f'{x:.2f}' for x in v) + ' |')
    L.append('## Verification\n')
    L.append(f"`verify.py`: **{V.get('status', 'not run')}**. " +
             ', '.join(f'{k} = {v}' for k, v in V.items() if k[0].isdigit()) + '.\n')
    L.append('![results](dfa_complexity_alignment_results.png)\n')
    L.append('**TLDR-end:** [tdc: Expt 20 results] ' + A.get('tldr', '') + '\n')
    (EXP / 'RESULTS_dfa_complexity_alignment.md').write_text('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--notes', required=True)
    ap.add_argument('--tldr', required=True)
    args = ap.parse_args()
    c = AN.load_config()
    res = Path(args.results)
    A = json.loads((res / 'analysis.json').read_text())
    A['tldr'] = args.tldr
    tasks = json.loads((res / 'tasks.json').read_text())
    figure(c, res, A, tasks)
    markdown(c, res, A, tasks, Path(args.notes).read_text())


if __name__ == '__main__':
    main()
