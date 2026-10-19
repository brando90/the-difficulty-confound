"""Write RESULTS_dsl_complexity_alignment.md and the figures from results/analysis.json.

  report.py --notes notes.md --tldr "..." [--results results]
`--notes` is a markdown file with the hand-written sections (takeaway, pilot disclosure, deviations, limitations,
consumer note, artifact location); it is inserted verbatim after the title. Layout follows Experiment 11's report.py.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

import analyze as AN  # noqa: E402
import dsl  # noqa: E402

LEVEL_COLORS = {2: '#9ec5f4', 3: '#5f9ee9', 4: '#2f7bd6', 5: '#1c5cab', 6: '#0d366b'}  # ordinal blue ramp
SERIES = ['#2a78d6', '#eb6834', '#3a9a5b']
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dc'


def fmt(x, d=3):
    if x is None:
        return 'n/a'
    if isinstance(x, (list, tuple)):
        return '[' + ', '.join(fmt(v, d) for v in x) + ']'
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return str(x)
    if np.isnan(x):
        return 'n/a'
    if not np.isfinite(x) or abs(x) >= 1e299:
        return '∞' if x > 0 else '−∞'
    return f'{x:.{d}g}'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def targets(tasks):
    return sorted((t for t in tasks if t['role'] == 'target'), key=lambda t: t['task_token'])


def program(t):
    return ' ∘ '.join(reversed(t['definition']['primitives']))


def figure(c, res, A, tasks):
    rows = [r for r in csv.DictReader(open(res / 'metrics.csv')) if r['arm'] == 'main']
    Ts = c['T_checkpoints']
    tg = targets(tasks)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    # (a) alpha = 1 curves by level and type: mean over targets and seeds, band = min..max of seed means
    ax = axs[0, 0]
    for k in c['K_levels']:
        for aff in (False, True):
            ids = {t['task_id'] for t in tg if t['K'] == k and t['is_affine'] == aff}
            per_seed = {}
            for r in rows:
                if r['target_id'] in ids and float(r['alpha']) == 1.0:
                    per_seed.setdefault(int(r['seed']), {}).setdefault(int(r['T']), []).append(float(r['excess_bits']))
            m = np.array([[np.mean(per_seed[s][T]) for T in Ts] for s in sorted(per_seed)])
            ax.plot(Ts, m.mean(0), color=LEVEL_COLORS[k], lw=2 if not aff else 1.4, ls='--' if aff else '-',
                    marker='o' if not aff else None, ms=3, label=f'K = {k}' if not aff else None)
            if not aff:
                ax.fill_between(Ts, m.min(0), m.max(0), color=LEVEL_COLORS[k], alpha=0.15, lw=0)
    ax.plot([], [], color=MUTED, ls='--', label='affine targets')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('T (examples seen)', color=INK)
    ax.set_ylabel('held-out excess loss ε (bits per example)', color=INK)
    ax.set_title('(a) α = 1 learning curves by program length K', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    # (b) T* heatmap K x alpha at the primary eps*
    ax = axs[0, 1]
    es = str(c['eps_star_primary'])
    H = np.zeros((len(c['K_levels']), len(c['alphas'])))
    Cn = np.zeros_like(H)
    for i, k in enumerate(c['K_levels']):
        ids = [t['task_id'] for t in tg if t['K'] == k]
        for j, a in enumerate(c['alphas']):
            H[i, j] = np.mean([np.log2(A['tstar_summary'][t][es][str(a)]['gm']) for t in ids])
            Cn[i, j] = sum(A['tstar_summary'][t][es][str(a)]['censored'] for t in ids)
    im = ax.imshow(H, cmap=SEQ, aspect='auto', origin='lower')
    n_runs = len(c['seeds']) * c['targets_per_level']
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f'2^{H[i, j]:.1f}\n{int(Cn[i, j])}/{n_runs} cens.', ha='center', va='center', fontsize=6.5,
                    color='white' if H[i, j] > (H.min() + H.max()) / 2 else INK)
    ax.set_xticks(range(len(c['alphas'])), [str(a) for a in c['alphas']])
    ax.set_yticks(range(len(c['K_levels'])), [f"{k} ({b} bits)" for k, b in zip(c['K_levels'], c['K_bits'])])
    ax.set_xlabel('target share α', color=INK)
    ax.set_ylabel('shortest-program length K', color=INK)
    ax.set_title(f'(b) GM T* at ε* = {es} bits (log2; censored runs at T bounds)', color=INK, fontsize=10, loc='left')
    fig.colorbar(im, ax=ax, label='log2 T*')
    ax.tick_params(colors=MUTED, labelsize=8)
    # (c) E versus eps*
    ax = axs[1, 0]
    cur = A['E']['curve']
    E = np.array([v if np.isfinite(v) and abs(v) < 1e299 else np.nan for v in cur['E_bits']])
    ax.plot(cur['eps_star'], E, color=SERIES[0], lw=2, label='E (K in bits)')
    ax.plot(cur['eps_star'], E / A['E']['log_Kbits_on_log_len_slope'], color=SERIES[1], lw=2,
            label='E (K as program length)')
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
                xs.append(max(obs[(int(s), p['target_id'], p['alpha'], p['T'])], c['eps_res']))
                ys.append(v)
        ax.scatter(xs, ys, s=9, color=col, alpha=0.5, lw=0, label=law)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('observed ε (bits), held-out points (floored at ε_res)', color=INK)
    ax.set_ylabel('frozen forecast (bits)', color=INK)
    ax.set_title('(d) held-out forecasts: G2 K law vs M_align_eff', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(EXP / f'dsl_complexity_alignment_results.{ext}', dpi=150)
    plt.close(fig)

    # second figure: per-target T* against K, BDM and zlib
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    y = np.array([np.log2(A['tstar_summary'][t['task_id']][es]['1.0']['gm']) for t in tg])
    cens = np.array([A['tstar_summary'][t['task_id']][es]['1.0']['censored'] > 0 for t in tg])
    sb = A['simplicity_bias']['spearman_Tstar']
    for ax, key, lab, rho in ((axs[0], 'K', 'shortest-program length K', sb['K']),
                              (axs[1], 'bdm_rowmajor', 'BDM (row-major, bits)', sb['bdm_rowmajor']),
                              (axs[2], 'zlib_bytes', 'zlib size (bytes)', sb['zlib'])):
        x = np.array([t[key] for t in tg], float)
        for aff, mk in ((False, 'o'), (True, 's')):
            m = np.array([t['is_affine'] == aff for t in tg])
            ax.scatter(x[m & ~cens], y[m & ~cens], marker=mk, s=28, color=SERIES[0], lw=0,
                       label=('affine' if aff else 'non-affine'))
            ax.scatter(x[m & cens], y[m & cens], marker=mk, s=28, facecolor='none', edgecolor=SERIES[0],
                       label=('affine' if aff else 'non-affine') + ', some seeds censored')
        ax.set_xlabel(lab, color=INK)
        ax.set_title(f'Spearman ρ = {fmt(rho, 2)}', color=INK, fontsize=10, loc='left')
        style(ax)
    axs[0].set_ylabel(f'log2 GM T* at α = 1, ε* = {es} bits', color=INK)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(EXP / f'dsl_complexity_alignment_tstar_vs_proxies.{ext}', dpi=150)
    plt.close(fig)


def markdown(c, res, A, tasks, notes, tldr):
    V = json.loads((res / 'verification.json').read_text()) if (res / 'verification.json').exists() else {}
    g = A
    L = []
    L.append('# Results: exact DSL program complexity versus data alignment '
             '(`12_dsl_shortest_program_complexity_vs_alignment`)\n')
    L.append(notes.strip() + '\n')
    L.append('## Gate table\n')
    L.append('All gates are reported with equal prominence. Intervals are 95% seed-block bootstrap percentile intervals '
             '(20,000 draws over the 8 seeds); "two-stage" also resamples the 4 targets within each level. Seed-only '
             'intervals are conditional on these 20 target functions. K is exact for this 12-primitive DSL.\n')
    L.append('| Gate | Statistic | Estimate | 95% interval(s) | Threshold | Verdict |')
    L.append('|---|---|---|---|---|---|')
    L.append(f"| G1 complexity is real | {g['G1']['statistic']} (ε* = {g['G1']['eps_star']} bits) | "
             f"{fmt(g['G1']['estimate'])} | seed {fmt(g['G1']['ci_seed'])}; two-stage {fmt(g['G1']['ci_two_stage'])} | "
             f"{g['G1']['threshold']} | **{g['G1']['verdict']}** |")
    L.append(f"| G2 K adds value beyond alignment | {g['G2']['statistic']} | {fmt(g['G2']['estimate'])} (K = 4 slice: "
             f"{fmt(g['G2']['k4_rmse_k_law'])} vs {fmt(g['G2']['k4_rmse_align_eff'])}) | {fmt(g['G2']['ci_seed'])} | "
             f"{g['G2']['threshold']} | **{g['G2']['verdict']}** |")
    L.append(f"| G3a coefficient (MDL) | κ in M_K_both; γ1 interval contains 0; RMSE coef ≤ exp | κ = "
             f"{fmt(g['G3']['kappa'])}, γ1 = {fmt(g['G3']['gamma1'])}; RMSE {fmt(g['G3']['rmse_coef'])} vs "
             f"{fmt(g['G3']['rmse_exp'])} | κ {fmt(g['G3']['kappa_ci'])}; γ1 {fmt(g['G3']['gamma1_ci'])} | "
             f"κ LB > 0 | **{g['G3']['G3a']}** |")
    L.append(f"| G3b exponent (quanta) | γ1 in M_K_both; M_K_exp gain over M_K_coef | γ1 = {fmt(g['G3']['gamma1'])}, "
             f"gain = {fmt(g['G3']['exp_over_coef_gain'])} | γ1 {fmt(g['G3']['gamma1_ci'])}; gain "
             f"{fmt(g['G3']['exp_over_coef_gain_ci'])} | γ1 UB < 0, gain ≥ 5%, LB > 0 | **{g['G3']['G3b']}** |")
    L.append(f"| G4 alignment dominates | {g['G4']['statistic']} at ε* = {g['G4']['eps_star']} | bits "
             f"{fmt(g['G4']['estimate_bits'])}; length {fmt(g['G4']['estimate_len'])}; E_mf {fmt(g['G4']['E_mf'])} "
             f"(ε* = {g['G4']['E_mf_eps_star']}) | bits {fmt(g['G4']['ci_bits'])}; length {fmt(g['G4']['ci_len'])} | "
             f"{g['G4']['threshold']} | **{g['G4']['verdict']}** |")
    L.append(f"| G5 gap shrinks | {g['G5']['statistic']} | {fmt(g['G5']['estimate'])} (gap {fmt(g['G5']['g_ref'])} → "
             f"{fmt(g['G5']['g_max'])} bits) | {fmt(g['G5']['ci_seed'])} | {g['G5']['threshold']} | "
             f"**{g['G5']['verdict']}** |")
    L.append(f"| G6 alignment floor | {g['G6']['statistic']} | a1 = {fmt(g['G6']['a1'])}; gain = {fmt(g['G6']['estimate'])} | "
             f"a1 {fmt(g['G6']['a1_ci'])}; gain {fmt(g['G6']['ci_seed'])} | {g['G6']['threshold']} | "
             f"**{g['G6']['verdict']}** |")
    L.append(f"| G7 quanta mechanism | {g['G7']['statistic']} | {fmt(g['G7']['estimate'])} (non-affine "
             f"{fmt(g['G7']['nonaffine_estimate'])}; partial given log K_bits {fmt(g['G7']['partial_given_logK'])}) | "
             f"seed {fmt(g['G7']['ci_seed'])}; two-stage {fmt(g['G7']['ci_two_stage'])} | {g['G7']['threshold']} | "
             f"**{g['G7']['verdict']}** |")
    L.append(f"| G8a BDM tracks exact K | {g['G8a']['statistic']} | {fmt(g['G8a']['estimate'])} | task bootstrap "
             f"{fmt(g['G8a']['ci_task'])} | {g['G8a']['threshold']} | **{g['G8a']['verdict']}** |")
    L.append(f"| G8b BDM tracks learning cost | {g['G8b']['statistic']} | {fmt(g['G8b']['estimate'])} | seed "
             f"{fmt(g['G8b']['ci_seed'])} | {g['G8b']['threshold']} | **{g['G8b']['verdict']}** |")
    g9 = g['G9']
    g9_est = (f"{g9['parameter']}: {fmt(g9['this'])} here vs {fmt(g9['exp11'])} in Experiment 11 (ratio "
              f"{fmt(g9['ratio'])})" if 'ratio' in g9 else f"Experiment 11 G3: {g9.get('exp11_g3', 'n/a')}; here: "
              f"{g9.get('this_g3', 'n/a')}")
    L.append(f"| G9 agreement with Experiment 11 | same decided G3 verdict, same sign, ratio in [0.5, 2] | {g9_est} | — | "
             f"decided and equal G3; ratio in range | **{g9['verdict']}**{' (' + g9['reason'] + ')' if g9.get('reason') else ''} |")
    L.append(f"\nC3 reading: **{g['G3']['C3']}**. Experiment 11 `analysis.json` read for G9: "
             f"`{g9.get('exp11_analysis_path')}` (SHA-256 `{g9.get('exp11_analysis_sha256')}`).\n")

    L.append('## Target table\n')
    es_levels = c['eps_star_levels']
    L.append('| id | K | K_bits | canonical program (applied right to left) | sensitivity | stratum | affine | H_rule | '
             'BDM | zlib | γ_j | ' + ' | '.join(f'GM T*({e}) α=1 [passes; censored/8]' for e in es_levels) + ' |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|' + '---|' * len(es_levels))
    for t in targets(tasks):
        ts = A['tstar_summary'][t['task_id']]
        L.append(f"| {t['task_id']} | {t['K']} | {t['K_bits']} | {program(t)} | {t['sensitivity']:.4f} | "
                 f"{t['sensitivity_stratum']} | {'yes' if t['is_affine'] else 'no'} | {t['local_rule_entropy']:.3f} | "
                 f"{t['bdm_rowmajor']:.0f} | {t['zlib_bytes']} | {A['G7']['gamma_j'][t['task_id']]:.3f} | " +
                 ' | '.join(f"{ts[str(e)]['1.0']['gm']:.0f} [{ts[str(e)]['1.0']['gm_passes']:.1f}; "
                            f"{ts[str(e)]['1.0']['censored']}]" for e in es_levels) + ' |')
    L.append('\nT* at every α (examples and target passes αT*/512, with censor flags) is in `expt_v1/results/tstar.csv`.\n')

    L.append('## Held-out RMSE (log2 bits, censored residual) by slice\n')
    L.append('| law | all (K_bits) | K = 4 | α = 0.15 | T = 2^16 | all (K as length) | invalid forecasts |')
    L.append('|---|---|---|---|---|---|---|')
    for law, v in A['heldout_rmse_log2']['K_bits'].items():
        L.append(f"| {law} | {v['all']:.4f} | {v['K4']:.4f} | {v['alpha015']:.4f} | {v['Tmax']:.4f} | "
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
             f"{fmt(al['eta_refit_both'])} {fmt(al['eta_refit_both_ci'])}. Reading (G2 K law interval versus 1): "
             f"**{reading}**.")
    L.append(f"- M_align_eff versus M_align_add held-out RMSE: {al['rmse_align_eff']:.4f} versus "
             f"{al['rmse_align_add']:.4f}; relative reduction {fmt(al['eff_over_add_gain'])} "
             f"{fmt(al['eff_over_add_gain_ci'])}; invalid forecasts {al['invalid_align_eff']} versus "
             f"{al['invalid_align_add']}.")
    L.append(f"- Per-target η_j: Spearman with K {fmt(al['spearman_eta_j_K'])}; with primitive overlap with the "
             f"distractor pool {fmt(al['spearman_eta_j_overlap'])}.\n")

    L.append('## Elasticity ratio E and range ratio R\n')
    L.append('| ε* (bits) | E (K in bits) | 95% interval | E (K as length) | 95% interval | e_α | e_K (bits) |')
    L.append('|---|---|---|---|---|---|---|')
    for es, v in A['E']['levels'].items():
        L.append(f"| {es} | {fmt(v['E_bits'])} | {fmt(v['ci_bits'])} | {fmt(v['E_len'])} | {fmt(v['ci_len'])} | "
                 f"{fmt(v['e_alpha'])} | {fmt(v['e_K_bits'])} |")
    L.append(f"\nFlip point ε_flip (E = 1) in [1e-2, 10] bits: {fmt(A['E']['eps_flip'])}. Slope of log K_bits on log K: "
             f"{A['E']['log_Kbits_on_log_len_slope']:.3f}. E from a refit with K = program length: "
             f"{fmt(A['E']['E_refit_len'])} {fmt(A['E']['E_refit_len_ci'])}. Model-free E_mf = {fmt(A['E']['E_mf'])} at "
             f"ε* = {A['E']['E_mf_eps_star']} (censored fraction per seed at the primary ε*: "
             f"{fmt(A['E']['E_mf_censored_fraction_per_seed'], 2)}; uncensored pairs per seed used: "
             f"{A['E']['E_mf_uncensored_pairs_per_seed']}).\n")
    R = A['range_ratio_R']
    L.append(f"Range ratio R = {fmt(R['estimate'])} {fmt(R['ci_seed'])}: alignment multiplier T*(α=0.05)/T*(α=1) = "
             f"{fmt(R['alignment_multiplier'])} versus complexity multiplier M_K = {fmt(R['complexity_multiplier'])}.\n")

    L.append('## K versus sensitivity versus type\n')
    kv = A['K_vs_sensitivity_vs_type']
    L.append(f"Regression of seed-mean log2 T*_j(α = 1, {kv['eps_star']} bits) on standardized log K_bits, standardized "
             f"sensitivity and the affine indicator (20 targets; {kv['censored_alpha1']} censored runs at α = 1):\n")
    L.append('| coefficient | estimate | seed interval | two-stage interval | VIF |')
    L.append('|---|---|---|---|---|')
    for n, v in kv['coefficients'].items():
        L.append(f"| {n} | {fmt(v['estimate'])} | {fmt(v['ci_seed'])} | {fmt(v['ci_two_stage'])} | {fmt(kv['vif'][n])} |")
    mt = A['M_K_by_type']
    L.append(f"\nM_K by type: affine {fmt(mt['affine']['estimate'])} {fmt(mt['affine']['ci_seed'])}; non-affine "
             f"{fmt(mt['nonaffine']['estimate'])} {fmt(mt['nonaffine']['ci_seed'])}.\n")

    L.append('## Simplicity bias versus program length (CTM/BDM and compressors)\n')
    sb = A['simplicity_bias']
    L.append(f"Spearman of per-target T* (α = 1, {sb['eps_star']} bits) with: " +
             ', '.join(f'{k} {fmt(v)}' for k, v in sb['spearman_Tstar'].items()) +
             f". Partial Spearman of T* with K given zlib: {fmt(sb['partial_Tstar_K_given_zlib'])}; with zlib given K: "
             f"{fmt(sb['partial_Tstar_zlib_given_K'])}.\n")
    L.append('| proxy | Spearman with K (200 tasks) | Spearman with T* (20 targets) |')
    L.append('|---|---|---|')
    L.append(f"| bdm_rowmajor | {fmt(A['G8a']['estimate'])} | {fmt(A['G8b']['estimate'])} |")
    for k, v in A['proxies_other'].items():
        L.append(f"| {k} | {fmt(v['spearman_with_K_200'])} | {fmt(v['spearman_with_Tstar_20'])} |")
    L.append('')

    L.append('## Train-fit census and per-output-bit loss\n')
    L.append('Runs flagged as plateaued (ε(2^16) ≥ 0.8 ε(2^14) and ε(2^16) > 1.0 bits), split into memorizing (train '
             'loss ≤ 0.1 bits) and not fitting (train loss > 1.0 bits), as plateau / memorizing / not fitting out of '
             f"{len(c['seeds'])} runs:\n")
    L.append('| level, type | ' + ' | '.join(f'α = {a}' for a in c['alphas']) + ' |')
    L.append('|---|' + '---|' * len(c['alphas']))
    for key in sorted(A['train_fit_census']):
        v = A['train_fit_census'][key]
        L.append(f'| {key} | ' + ' | '.join(f"{v[str(a)]['plateau']}/{v[str(a)]['memorizing']}/{v[str(a)]['not_fitting']}"
                                            for a in c['alphas']) + ' |')
    tt = A['train_fit_census_total']
    L.append(f"\nTotal: {tt['plateau']} plateaued of {tt['runs']} runs ({tt['memorizing']} memorizing, "
             f"{tt['not_fitting']} not fitting).\n")
    L.append('Per-output-bit excess loss at α = 1, T = 2^16 (bits; bit 0 is the least significant):\n')
    L.append('| K | ' + ' | '.join(f'bit {i}' for i in range(10)) + ' |')
    L.append('|---|' + '---|' * 10)
    for k, v in A['per_output_bit_loss_alpha1_Tmax'].items():
        L.append(f'| {k} | ' + ' | '.join(f'{x:.3f}' for x in v) + ' |')

    L.append('\n## Secondary arms\n')
    r7 = A['rep768']
    if r7.get('ran'):
        L.append(f"**Repetition (R768)**, {r7['cells']} cells (seeds {r7['seeds']}): median ratio ε_768 / ε_512 on the "
                 f"shared {r7['n_shared_test_inputs']} test inputs at T = 2^16, α = 1: "
                 f"{fmt(r7['median_ratio_alpha1_Tmax'])}. M_K with 768 training inputs {fmt(r7['M_K_768']['estimate'])} "
                 f"{fmt(r7['M_K_768']['ci_seed'])} versus {fmt(r7['M_K_512_shared']['estimate'])} "
                 f"{fmt(r7['M_K_512_shared']['ci_seed'])} with 512 (same seeds, shared inputs); η (two α levels) "
                 f"{fmt(r7['eta_two_alpha']['768']['estimate'])} {fmt(r7['eta_two_alpha']['768']['ci_seed'])} versus "
                 f"{fmt(r7['eta_two_alpha']['512_shared']['estimate'])} {fmt(r7['eta_two_alpha']['512_shared']['ci_seed'])}."
                 + (' **The median ratio is below 0.5, so the main results partly measure generalization from a finite '
                    'input set.**' if r7['note_partly_finite_input_generalization'] else ''))
        L.append('\n| α | ' + ' | '.join(f'T = {T}' for T in c['T_checkpoints']) + ' |')
        L.append('|---|' + '---|' * len(c['T_checkpoints']))
        for a, v in r7['median_ratio_768_over_512'].items():
            L.append(f'| {a} | ' + ' | '.join(f"{v[str(T)]:.3f}" for T in c['T_checkpoints']) + ' |')
        L.append('')
    else:
        L.append('The R768 arm did not run (see the notes above).\n')
    ml = A['mlp']
    if ml.get('ran'):
        tc = ml['train_fit_census']
        L.append(f"**MLP arm** ({len(ml['seeds'])} seeds, α = 1): M_K = {fmt(ml['M_K']['estimate'])} "
                 f"{fmt(ml['M_K']['ci_seed'])} (transformer: {fmt(A['G1']['estimate'])}); plateaued {tc['plateau']} of "
                 f"{tc['runs']} runs ({tc['memorizing']} memorizing, {tc['not_fitting']} not fitting); mean test bits "
                 f"at 2^16 by K: " + ', '.join(f'K = {k}: {v:.3f}' for k, v in ml['mean_test_bits_Tmax_by_level'].items())
                 + '.\n')
    else:
        L.append('The MLP arm did not run (see the notes above).\n')

    L.append('## Verification\n')
    if V:
        L.append(f"`verify.py`: **{V.get('status', 'not run')}** in {V.get('seconds')} s. Checks: " +
                 ', '.join(f'{k} = {v}' for k, v in V.get('checks', {}).items()) +
                 '. Counts: ' + ', '.join(f'{k} = {v}' for k, v in V.items() if k[0].isdigit() and
                                         not isinstance(v, (list, dict))) + '.\n')
    else:
        L.append('`verify.py` has not run.\n')
    L.append('![results](dsl_complexity_alignment_results.png)\n')
    L.append('![T* versus proxies](dsl_complexity_alignment_tstar_vs_proxies.png)\n')
    L.append('**TLDR-end:** [tdc: Expt 12 results] ' + tldr + '\n')
    (EXP / 'RESULTS_dsl_complexity_alignment.md').write_text('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--notes', required=True)
    ap.add_argument('--tldr', required=True)
    args = ap.parse_args()
    c = AN.load_config()
    res = Path(args.results)
    A = json.loads((res / 'analysis.json').read_text())
    tasks = json.loads((res / 'tasks.json').read_text())
    figure(c, res, A, tasks)
    markdown(c, res, A, tasks, Path(args.notes).read_text(), args.tldr)


if __name__ == '__main__':
    main()
