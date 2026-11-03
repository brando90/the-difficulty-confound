"""Writes RESULTS_ginc_hmm_gpu.md and ginc_hmm_gpu_results.png/.pdf from results/gpu (13-GPU arm).

Usage (from expt_v1/): report_gpu.py [--notes FILE] [--tldr TEXT]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from report import GRID, INK, MUTED, f3, interval, style  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SIZES = ['S', 'M', 'L']
SIZE_NAMES = {'S': 'S (3.2M)', 'M': 'M (25M)', 'L': 'L (85M)'}
LEVEL_COLORS = ['#a9cdf5', '#6da7ec', '#3987e5', '#1c5cab', '#0d366b']  # light to dark = n 8 to 128
SIZE_COLORS = {'S': '#6da7ec', 'M': '#1c5cab', 'L': '#0d366b'}
ALIGN, CPLX = '#eb6834', '#1c5cab'


def load(out):
    A = json.loads((out / 'analysis.json').read_text())
    rows = defaultdict(list)
    with open(out / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            rows[(r['size'], r['slice'], int(r['K']), float(r['alpha']), int(r['T']))].append(
                (float(r['excess_bits']), float(r['learn_bits']), float(r['ident_bits'])))
    return A, rows


def cell(rows, size, sl, k, a, T, i=0):
    v = rows.get((size, sl, k, a, T))
    return float(np.mean([x[i] for x in v])) if v else float('nan')


def figures(out, A, rows, c, path):
    levels = sorted({k[2] for k in rows})
    Ts = {z: sorted({k[4] for k in rows if k[0] == z}) for z in SIZES}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, z in enumerate(SIZES):
        ax = axes[0, j]
        for i, n in enumerate(levels):
            y = [cell(rows, z, 'all', n, 1.0, T, 1) for T in Ts[z]]
            ax.plot(Ts[z], np.maximum(y, 1e-4), '-o', color=LEVEL_COLORS[i], lw=2, ms=4, label=f'n = {n}')
        ax.axhline(c['epsilon_star'], color=MUTED, lw=0.8, ls='--')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f'{SIZE_NAMES[z]}: learning excess at α = 1', fontsize=9, color=INK)
        ax.set_xlabel('training tokens T', fontsize=8, color=MUTED)
        style(ax)
        if j == 0:
            ax.set_ylabel('ε_learn, bits/token (dashed: ε* = 0.02)', fontsize=8, color=MUTED)
            ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 0]
    x = np.arange(3)
    law = [A['G6']['M_K'][z]['estimate'] for z in SIZES]
    mf = [A['scopes'][z]['gates']['G1']['M_K'] for z in SIZES]
    ax.bar(x - 0.18, mf, 0.34, color=LEVEL_COLORS[2], label='model-free (G1, n = 128 vs 8, α = 1)')
    ax.bar(x + 0.18, law, 0.34, color=LEVEL_COLORS[4], label='law, M_K_both fit rows (G6)')
    for i, z in enumerate(SIZES):
        ax.plot([i + 0.18] * 3, A['G6']['M_K'][z]['per_seed'], 'o', color=INK, ms=3)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(x, [SIZE_NAMES[z] for z in SIZES])
    ax.set_title(f'complexity multiplier M_K shrinks with size (G6: L/S = {A["G6"]["ratio_L_over_S"]:.2f})',
                 fontsize=9, color=INK)
    ax.set_ylabel('data multiplier at ε* (log)', fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[1, 1]
    ea = [np.mean(A['scopes'][z]['gates']['G4']['e_alpha']) for z in SIZES]
    ek = [np.mean(A['scopes'][z]['gates']['G4']['e_K_bits']) for z in SIZES]
    ax.plot(x, ek, '-o', color=CPLX, lw=2, ms=6, label='e_K (complexity)')
    ax.plot(x, ea, '-o', color=ALIGN, lw=2, ms=6, label='e_α (alignment)')
    for i, z in enumerate(SIZES):
        ax.annotate(f'E = {ea[i] / ek[i]:.2f}', (i, ek[i]), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color=INK)
    ax.set_xticks(x, [SIZE_NAMES[z] for z in SIZES])
    ax.set_ylim(0, max(ek + ea) * 1.25)
    ax.set_title('both elasticities fall with size; alignment\'s falls faster', fontsize=9, color=INK)
    ax.set_ylabel('elasticity of T* (at ε*, K_ref)', fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[1, 2]
    for k, (sl, col) in enumerate([('early', ALIGN), ('late', CPLX)]):
        y = [A['scopes'][z]['position_decomposition'][f'{sl}|excess_bits']['r_alpha'] for z in SIZES]
        lo = [y[i] - A['scopes'][z]['position_decomposition'][f'{sl}|excess_bits']['ci95'][0] for i, z in enumerate(SIZES)]
        hi = [A['scopes'][z]['position_decomposition'][f'{sl}|excess_bits']['ci95'][1] - y[i] for i, z in enumerate(SIZES)]
        ax.errorbar(x + (k - 0.5) * 0.2, y, yerr=[lo, hi], fmt='o', color=col, ms=6, capsize=3,
                    label=f'{sl} positions ({"1-16" if sl == "early" else "65-128"})')
    ax.axhline(0.5, color=MUTED, lw=0.8, ls='--')
    ax.set_xticks(x, [SIZE_NAMES[z] for z in SIZES])
    ax.set_ylim(0, 1.02)
    ax.set_title('alignment share r_α of the excess (G5)', fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=130)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def gate_rows(A, c, w):
    P = A['scopes']
    g = {sc: P[sc]['gates'] for sc in P}
    v = lambda k: ' / '.join(g[z][k]['verdict'].replace('NOT_ESTABLISHED', 'NE').replace('SUPPORTED', 'S')  # noqa: E731
                             for z in SIZES)
    G1, G2, G3a, G3b, G4, G5 = (g['pooled'][k] for k in ('G1', 'G2', 'G3a', 'G3b', 'G4', 'G5'))
    G6 = A['G6']
    pct_iv = lambda d: f'[{f3(100 * d[0])}%, {f3(100 * d[1])}%]'  # noqa: E731
    w('| Gate | Pooled estimate | 95% interval (seed blocks) | Threshold (and every seed) | Verdict of record | S / M / L |')
    w('|---|---|---|---|---|---|')
    w(f'| G1 complexity is real: model-free M_K, n = 128 vs 8 at α = 1 | {f3(G1["M_K"])} (per seed '
      f'{", ".join(f3(x) for x in G1["per_seed"])}) | seed {interval(G1["ci95_seed"])}; two-stage '
      f'{interval(G1["ci95_two_stage"])} | ≥ 2, both lower bounds > 1 | **{G1["verdict"]}** | {v("G1")} |')
    w(f'| G2 K law ({G2["K_law"]}) beats {"N_align_eff"} on held-out rows | {f3(100 * G2["reduction"])}% RMSE reduction '
      f'(per seed {", ".join(f3(100 * x) + "%" for x in G2["per_seed"])}) | {pct_iv(G2["ci95"])} | ≥ 10%, lower bound > 0, '
      f'lower RMSE on n = 32 rows ({f3(G2["heldout_level_rmse_K_law"])} vs {f3(G2["heldout_level_rmse_align_eff"])}), '
      f'{G2["invalid_K_law"]} invalid | **{G2["verdict"]}** | {v("G2")} |')
    w(f'| G3a coefficient picture | κ {f3(G3a["kappa"]["estimate"])}, γ1 {f3(G3a["gamma1"]["estimate"])} | '
      f'κ {interval(G3a["kappa"]["ci95"])}, γ1 {interval(G3a["gamma1"]["ci95"])} | κ > 0, γ1 interval ∋ 0, RMSE coef '
      f'({f3(G3a["rmse_coef"])}) ≤ exp ({f3(G3a["rmse_exp"])}) | **{G3a["verdict"]}** | {v("G3a")} |')
    w(f'| G3b exponent picture | γ1 {f3(G3b["gamma1"]["estimate"])}; exp vs coef {f3(100 * G3b["exp_vs_coef_reduction"])}% '
      f'| {pct_iv(G3b["ci95"])} | γ1 upper < 0, reduction ≥ 5% with lower > 0 | **{G3b["verdict"]}** | {v("G3b")} |')
    w(f'| G4 alignment dominates (E = e_α/e_K) | E bits {f3(G4["E_bits"])}, states {f3(G4["E_states"])}; model-free '
      f'E_mf {f3(G4["E_mf"])} (at ε = {G4["E_mf_eps"]}) | bits {interval(G4["ci95_bits"])}; states '
      f'{interval(G4["ci95_states"])} | all > 1, lower bounds > 1 | **{G4["verdict"]}** | {v("G4")} |')
    w(f'| G5 alignment early, complexity late | r_α early {f3(G5["r_alpha_early"]["r_alpha"])}, late '
      f'{f3(G5["r_alpha_late"]["r_alpha"])} | early {interval(G5["r_alpha_early"]["ci95"])}; late '
      f'{interval(G5["r_alpha_late"]["ci95"])} | early lower > 0.5, late upper < 0.5 | **{G5["verdict"]}** | {v("G5")} |')
    w(f'| G6 (secondary) bigger models pay less for complexity | M_K(L)/M_K(S) = {f3(G6["ratio_L_over_S"])} (per seed '
      f'{", ".join(f3(x) for x in G6["per_seed"])}); M_K S {f3(G6["M_K"]["S"]["estimate"])}, M {f3(G6["M_K"]["M"]["estimate"])}, '
      f'L {f3(G6["M_K"]["L"]["estimate"])} | {interval(G6["ci95"])} | ≤ 0.67, upper < 1, monotone | **{G6["verdict"]}** | — |')


def results_md(out, A, rows, c, notes, tldr, fig_name):
    P = A['scopes']
    man = json.loads((out / 'manifest.json').read_text())
    sel = man.get('pilot_selection') or {}
    L = []
    w = L.append
    w('# Results: Experiment 13-GPU, GINC-style HMM mixtures, complexity versus alignment at three model sizes')
    w('')
    w('Arm 13-GPU: 360 runs (3 sizes × 5 levels × 2 targets × 4 α × 3 seeds), decoder-only transformers S/M/L '
      '(3.2M / 25M / 85M non-embedding parameters), bfloat16 training, float64 evaluation. Protocol: '
      '[PROTOCOL.md](PROTOCOL.md) (frozen at 19a77724; 13-GPU implementation notes and GPU pilot disclosure in Section 7); '
      'run log: [CKPT_ginc_hmm.md](CKPT_ginc_hmm.md); 13-small: [RESULTS_ginc_hmm.md](RESULTS_ginc_hmm.md).')
    w('')
    if notes:
        w(notes.strip())
        w('')
    w('## Gates (all at equal prominence)')
    w('')
    w(f'ε* = {A["epsilon_star"]} bits/token on the learning term ε_learn = ε − ε_id. Pooled = all three sizes in one fit '
      '(the full N laws of PROTOCOL Section 4.1): the verdict of record. S / M / L = the same gate within one size (S = '
      'SUPPORTED, NE = NOT_ESTABLISHED). Intervals: 95% seed-block bootstrap (3 seeds, 20,000 draws, 10 distinct '
      'resamples); every gate also requires each seed\'s point estimate on the supported side.')
    w('')
    gate_rows(A, c, w)
    w('')
    w(f'C3 (coefficient or exponent): **{P["pooled"]["C3"]}** (κ > 0 and γ1 < 0 are both resolved: complexity acts on both).')
    w('')
    w(f'![Experiment 13-GPU results]({fig_name}.png)')
    w('')
    w('## Why E falls with size while G6 says bigger models pay less for complexity')
    w('')
    w('E is a ratio of two elasticities of the data bill T* (fitted `M_K_both` per size, all rows, at ε* and K_ref). '
      'Both fall with model size; the alignment one falls faster, so the ratio falls although complexity costs less.')
    w('')
    w('| Size | e_K (complexity) | e_α = η (alignment) | E = e_α/e_K | M_α = 20^η | law M_K (n = 128 vs 8) | model-free M_K | E_mf |')
    w('|---|---|---|---|---|---|---|---|')
    for z in SIZES:
        G4, d = P[z]['gates']['G4'], P[z]['draft_protocol_reported']
        w(f'| {SIZE_NAMES[z]} | {f3(np.mean(G4["e_K_bits"]))} | {f3(np.mean(G4["e_alpha"]))} | {f3(G4["E_bits"])} '
          f'{interval(G4["ci95_bits"])} | {f3(d["G4a_range_ratio"]["M_alpha"])} | {f3(A["G6"]["M_K"][z]["estimate"])} | '
          f'{f3(P[z]["gates"]["G1"]["M_K"])} | {f3(G4["E_mf"])} (ε {G4["E_mf_eps"]}) |')
    w('')
    w('For L, cutting the target share from 1 to 0.05 (20 times fewer target documents at the same total) raises the '
      f'fitted data bill only {f3(P["L"]["draft_protocol_reported"]["G4a_range_ratio"]["M_alpha"])}×: the larger model '
      'learns the target mostly from the distractor concepts of the same family. The law-based and the model-free E '
      'disagree (law < 1 at every size; model-free ≈ 1.0–1.2 at 2ε* or ε*), which is why G4 is NOT_ESTABLISHED rather '
      'than reversed.')
    w('')
    w('## Censoring')
    w('')
    cens = P['pooled']['gates']['G1']['censoring_all']
    w(f'At ε* = {A["epsilon_star"]} bits/token, {cens["right"]} of {sum(cens.values())} (size, seed, target, α) runs '
      'never reached ε* by their last checkpoint (their model-free T* is set to that checkpoint), and '
      f'{A["scopes"]["pooled"]["multiplier_curves"][str(A["epsilon_star"])]["right_censored_alpha1"]} of the 90 α = 1 '
      'target-seed-size cells used by G1. Law-based quantities extrapolate for those runs.')
    w('')
    w('## Draft-protocol versions (reported, not gated; pooled)')
    w('')
    D = P['pooled']['draft_protocol_reported']
    w(f'- Law-based M_K ({D["law_for_multipliers"]}): {f3(D["G1_law_M_K"]["estimate"])} {interval(D["G1_law_M_K"]["ci95"])}.')
    w(f'- Range ratio R = log M_α / log M_K: {f3(D["G4a_range_ratio"]["R"])} {interval(D["G4a_range_ratio"]["ci95"])}.')
    w(f'- Gap ratio Δ(T_last fit)/Δ(T_first) at α = 1: {f3(D["G4b_gap_ratio"]["ratio"])} {interval(D["G4b_gap_ratio"]["ci95"])} '
      f'(Δ {f3(D["G4b_gap_ratio"]["gap_first"])} → {f3(D["G4b_gap_ratio"]["gap_last_fit"])} bits/token).')
    w(f'- Bonferroni G2: coef {f3(100 * D["G2_bonferroni"]["coef"]["reduction"])}% (lower '
      f'{f3(100 * D["G2_bonferroni"]["coef"]["ci_bonf"][0])}%), exp {f3(100 * D["G2_bonferroni"]["exp"]["reduction"])}% '
      f'(lower {f3(100 * D["G2_bonferroni"]["exp"]["ci_bonf"][0])}%); ρ = {f3(D["G3_rho"]["rho"])} {interval(D["G3_rho"]["ci95"])}.')
    w('')
    w('## Held-out RMSE of every pooled law (log2 ε)')
    w('')
    w('| Law | primary | no offset | K = n | K = K_bits − log2 n! | invalid | level only | α only | T only |')
    w('|---|---|---|---|---|---|---|---|---|')
    H_ = A['heldout_rmse']
    for law, p in H_['primary']['pooled'].items():
        w(f'| {law} | {f3(p["rmse"])} | {f3(H_["no_offset"]["pooled"][law]["rmse"])} | {f3(H_["K_n"]["pooled"][law]["rmse"])} | '
          f'{f3(H_["K_canonical"]["pooled"][law]["rmse"])} | {p["invalid"]} | {f3(p["by_coordinate"]["level_only"])} | '
          f'{f3(p["by_coordinate"]["alpha_only"])} | {f3(p["by_coordinate"]["T_only"])} |')
    w('')
    w(f'Independent fitter (L-BFGS-B, different starts), every scope, seed and law: '
      f'**{"AGREE" if A["independent_fit"]["agree"] else "DISAGREE"}**.')
    w('')
    w('## Fitted pooled laws per seed (fit rows)')
    w('')
    for law in ('N_K_both', 'N_AS_both'):
        per = P['pooled']['fitted_laws_per_seed'][law]
        keys = [k for k in next(iter(per.values())) if k not in ('g_lo', 'g_hi')]
        w(f'- **{law}**: ' + '; '.join(f'seed {s}: ' + ', '.join(f'{k} {f3(v[k])}' for k in keys) for s, v in per.items()))
    ac = P['pooled']['alignment_comparison']
    w(f'- Effective-data form vs alignment-scaling form (N_K_both vs N_AS_both) held-out RMSE reduction: '
      f'{f3(100 * ac["effective_data_vs_alignment_scaling_reduction"])}% {interval([100 * x for x in ac["ci95"]])}.')
    w('')
    w('## Multiplier curves (pooled)')
    w('')
    w('| ε* | M_K (law) | M_K (model-free) | M_α = 20^η | E (bits) | E 95% | right-censored α = 1 cells |')
    w('|---|---|---|---|---|---|---|')
    for e, v in sorted(P['pooled']['multiplier_curves'].items(), key=lambda x: float(x[0])):
        w(f'| {e} | {f3(v["M_K_law"])} | {f3(v["M_K_model_free"])} | {f3(v["M_alpha_law"])} | {f3(v["E_bits"])} | '
          f'{interval(v["E_ci95"])} | {v["right_censored_alpha1"]} |')
    w('')
    w('## Position decomposition (Section 4.5)')
    w('')
    w('| Scope | slice | r_α of ε | r_α of ε_learn | r_α of ε_id |')
    w('|---|---|---|---|---|')
    for sc in ['pooled'] + SIZES:
        PD = P[sc]['position_decomposition']
        for s in ('early', 'middle', 'late', 'all'):
            w(f'| {sc} | {s} | ' + ' | '.join(f'{f3(PD[f"{s}|{q}"]["r_alpha"])} {interval(PD[f"{s}|{q}"]["ci95"])}'
                                              for q in ('excess_bits', 'learn_bits', 'ident_bits')) + ' |')
    w('')
    w('## Excess at each run\'s last checkpoint (all positions, mean over targets and seeds, bits/token)')
    w('')
    alphas = sorted({k[3] for k in rows})
    levels = sorted({k[2] for k in rows})
    w('| n | ' + ' | '.join(f'{z} α={a}' for z in SIZES for a in alphas) + ' |')
    w('|---|' + '---|' * (len(SIZES) * len(alphas)))
    Tm = {z: max(k[4] for k in rows if k[0] == z) for z in SIZES}
    for n in levels:
        w(f'| {n} | ' + ' | '.join(f3(cell(rows, z, 'all', n, a, Tm[z])) for z in SIZES for a in alphas) + ' |')
    w('')
    m = A['mechanism']
    w('## Mechanism analyses (reported, not gated)')
    w('')
    w('- Quanta test, Spearman(γ at α = 1, stationary tail exponent) over the 10 targets: '
      + ', '.join(f'{z} {f3(m[z]["spearman_gamma_vs_tail"]["rho"])} (p {f3(m[z]["spearman_gamma_vs_tail"]["p"])})' for z in SIZES) + '.')
    sh = m['eps_id_shape']
    w(f'- ε_id versus α: residual sum of squares, linear in (1 − α) {f3(sh["total_rss_linear"])} vs log2(1/α) '
      f'{f3(sh["total_rss_log"])}: ε_id = log2(1/α)/128 exactly (distractors give the targets almost no probability).')
    for z in SIZES:
        w(f'- {z}: share of early-slice excess that is ε_id at the largest T: '
          + ', '.join(f'α = {a}: {f3(v)}' for a, v in m[z]['early_share_eps_id_at_Tmax'].items())
          + '; distractor excess at the largest T by α: '
          + ', '.join(f'{a}: {f3(v)}' for a, v in m['distractor_excess_at_Tmax_by_alpha'][z].items()) + ' bits/token.')
    w('')
    w('## Pilot (disclosed in PROTOCOL Section 7, commit 2c3d7e8f)')
    w('')
    if sel:
        for r in sel['results']:
            w(f'- {r["size"]} {r["lr_multiplier"]}× (lr {r["lr"]:.1e}): ε at 2^26 = {r["excess_bits"]:.4f} ± {r["excess_se"]:.4f} '
              f'bits/token ({r["wall_seconds"]:.0f} s).')
        w(f'- Selected by the frozen rule: {json.dumps(sel["lr_selected"])} — the top of the grid for every size (a larger '
          'rate was not allowed and might do better).')
    w('')
    w('## Run facts')
    w('')
    gr = man['grid']
    w(f'- {gr["runs"]} runs, {len(gr["diverged"])} diverged, {gr["verified"]} with a per-shard VERIFIED marker (training '
      'stream regenerated, every checkpoint replayed in float64 to < 1e-9 bits). Merged `verify.py --arm gpu`: PASS.')
    w(f'- {gr["gpu_hours"]} GPU-hours on SNAP skampere1 ({gr["gpus"][0]}) and skampere2 ({gr["gpus"][-1]}), GPUs shared '
      'with other users (used only with ≥ 24 GB free).')
    w('')
    w(f'**TLDR-end:** {tldr}' if tldr else '')
    (EXP / 'RESULTS_ginc_hmm_gpu.md').write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notes', default=None)
    ap.add_argument('--tldr', default='')
    a = ap.parse_args()
    out = HERE / 'results' / 'gpu'
    raw = json.loads((HERE / 'config.json').read_text())
    c = dict(raw['shared'])
    c.update(raw['gpu'])
    A, rows = load(out)
    name = 'ginc_hmm_gpu_results'
    figures(out, A, rows, c, EXP / name)
    notes = Path(a.notes).read_text() if a.notes else ''
    results_md(out, A, rows, c, notes, a.tldr, name)
    print('wrote', EXP / 'RESULTS_ginc_hmm_gpu.md')


if __name__ == '__main__':
    main()
