"""Writes RESULTS_ginc_hmm.md and the figures from results/<arm>/analysis.json, metrics.csv and per_position.csv.

Usage (from expt_v1/): report.py --arm small [--notes FILE] [--tldr TEXT]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LEVEL_COLORS = ['#6da7ec', '#3987e5', '#1c5cab', '#0d366b']  # one-hue ordinal ramp, light to dark = n 4 to 32
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def f3(x):
    return 'n/a' if x is None else (f'{x:.3g}' if isinstance(x, (int, float)) and math.isfinite(x) else str(x))


def interval(v):
    return f'[{f3(v[0])}, {f3(v[1])}]'


def figures(out, A, c, path):
    cm = A['cell_means']
    levels = sorted({int(k.split('|')[1]) for k in cm})
    alphas = sorted({float(k.split('|')[2]) for k in cm})
    Ts = sorted({int(k.split('|')[3]) for k in cm})
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for j, a in enumerate(alphas):
        ax = axes[0, j]
        for i, n in enumerate(levels):
            ex = [cm[f'all|{n}|{a}|{T}']['excess'] for T in Ts]
            ei = [cm[f'all|{n}|{a}|{T}']['ident'] for T in Ts]
            ax.plot(Ts, np.maximum(ex, 1e-4), '-o', color=LEVEL_COLORS[i], lw=2, ms=4, label=f'n = {n}')
            if a < 1:
                ax.plot(Ts, ei, '--', color=LEVEL_COLORS[i], lw=1)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f'excess ε vs T, α = {a}', fontsize=9, color=INK)
        ax.set_xlabel('training tokens T', fontsize=8, color=MUTED)
        style(ax)
        if j == 0:
            ax.set_ylabel('bits/token (dashed: ε_id floor)', fontsize=8, color=MUTED)
            ax.legend(fontsize=7, frameon=False)
    pp = {}
    with open(out / 'per_position.csv') as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            pp.setdefault((row[1], float(row[2]), int(row[3])), []).append(np.array(row[4:], float))
    tasks = {t['task_id']: t for t in json.loads((out / 'tasks.json').read_text())}
    o = np.load(out / 'oracle.npz')
    Tm = Ts[-1]
    for j, a in enumerate(alphas):
        ax = axes[1, j]
        for i, n in enumerate(levels):
            curves = []
            for (tid, al, T), v in pp.items():
                if al == a and T == Tm and tasks[tid]['K'] == n:
                    curves.append(np.mean(v, 0) - o[tid].mean(0))
            ax.plot(np.arange(1, 129), np.mean(curves, 0), color=LEVEL_COLORS[i], lw=1.5, label=f'n = {n}')
        ax.axvspan(1, 16, color=GRID, alpha=0.5, lw=0)
        ax.axvspan(65, 128, color=GRID, alpha=0.25, lw=0)
        ax.set_title(f'excess by position at T = 2^{int(math.log2(Tm))}, α = {a}', fontsize=9, color=INK)
        ax.set_xlabel('position (shaded: early 1-16, late 65-128)', fontsize=8, color=MUTED)
        style(ax)
        if j == 0:
            ax.set_ylabel('bits/token', fontsize=8, color=MUTED)
    cur = A['multiplier_curves']
    eps = sorted(float(k) for k in cur)
    ax = axes[2, 0]
    ax.plot(eps, [cur[str(e)]['M_K_law'] for e in eps], '-o', color=LEVEL_COLORS[3], lw=2, ms=4, label='M_K (law)')
    ax.plot(eps, [cur[str(e)]['M_K_model_free'] for e in eps], '-o', color=LEVEL_COLORS[1], lw=2, ms=4,
            label='M_K (model-free, censored)')
    ax.plot(eps, [cur[str(e)]['M_alpha_law'] for e in eps], '-o', color='#eb6834', lw=2, ms=4, label='M_α = 20^η')
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('data multipliers vs target excess ε*', fontsize=9, color=INK)
    ax.set_xlabel('ε* (bits/token, learning term)', fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axes[2, 1]
    Ev = [cur[str(e)]['E_bits'] for e in eps]
    ax.plot(eps, [min(v, 1e3) for v in Ev], '-o', color=LEVEL_COLORS[2], lw=2, ms=4)
    g4 = A['gates']['G4']
    ax.errorbar([c['epsilon_star']], [g4['E_bits']], yerr=[[g4['E_bits'] - g4['ci95_bits'][0]],
                [g4['ci95_bits'][1] - g4['E_bits']]], fmt='D', color=INK, ms=5, capsize=3, label='gated ε*')
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xscale('log')
    ax.set_ylim(0, max(1.2, max(min(v, 1e3) for v in Ev + [g4['E_bits']]) * 1.1))
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('elasticity ratio E = e_α / e_K (bits)', fontsize=9, color=INK)
    ax.set_xlabel('ε* (bits/token)', fontsize=8, color=MUTED)
    style(ax)
    ax = axes[2, 2]
    pos = A['position_decomposition']
    names = ['early', 'middle', 'late', 'all']
    for k, (q, col) in enumerate([('excess_bits', LEVEL_COLORS[3]), ('learn_bits', LEVEL_COLORS[1]),
                                  ('ident_bits', '#eb6834')]):
        x = np.arange(4) + (k - 1) * 0.22
        y = [pos[f'{s}|{q}']['r_alpha'] for s in names]
        lo = [y[i] - pos[f'{s}|{q}']['ci95'][0] for i, s in enumerate(names)]
        hi = [pos[f'{s}|{q}']['ci95'][1] - y[i] for i, s in enumerate(names)]
        ax.errorbar(x, y, yerr=[lo, hi], fmt='o', color=col, ms=6, capsize=3, label=q.replace('_bits', ''))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls='--')
    ax.set_xticks(range(4), names)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('alignment share r_α = SS_α/(SS_α+SS_K), 95% CI', fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    axes[2, 3].axis('off')
    g = A['gates']
    txt = '\n'.join(f'{k}: {v["verdict"]}' for k, v in g.items())
    axes[2, 3].text(0, 0.9, 'Gates (13-small)\n\n' + txt + '\n\n13-GPU: pending, not run', va='top', fontsize=10,
                    color=INK, family='monospace')
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=130)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def results_md(out, A, c, notes, tldr, fig_name):
    g = A['gates']
    G1, G2, G3a, G3b, G4, G5 = (g[k] for k in ('G1', 'G2', 'G3a', 'G3b', 'G4', 'G5'))
    pilot = json.loads((HERE / 'pilot_summary.json').read_text())
    D = A['draft_protocol_reported']
    L = []
    w = L.append
    w('# Results: Experiment 13-small, GINC-style HMM mixtures, complexity versus alignment')
    w('')
    w('Arm 13-small (CPU, 354,240-parameter transformer). **13-GPU: pending, not run.** Protocol: '
      '[PROTOCOL.md](PROTOCOL.md) as frozen; run log: [CKPT_ginc_hmm.md](CKPT_ginc_hmm.md).')
    w('')
    if notes:
        w(notes.strip())
        w('')
    w('## Gates (all at equal prominence)')
    w('')
    w('ε* = {} bits/token on the learning term ε_learn = ε − ε_id. Intervals are 95% seed-block bootstrap '
      '(4 seeds, 20,000 draws, 35 distinct resamples), unless marked two-stage.'.format(A['epsilon_star']))
    w('')
    w('| Gate | Estimate | 95% interval | Threshold | Verdict |')
    w('|---|---|---|---|---|')
    w(f'| G1 complexity is real (model-free M_K at α = 1) | {f3(G1["M_K"])} | seed {interval(G1["ci95_seed"])}; '
      f'two-stage {interval(G1["ci95_two_stage"])} | M_K ≥ {G1["threshold"]}, both lower bounds > 1 | **{G1["verdict"]}** |')
    w(f'| G2 K law ({G2["K_law"]}) beats M_align_eff on held-out rows | {f3(100 * G2["reduction"])}% RMSE reduction '
      f'| [{f3(100 * G2["ci95"][0])}%, {f3(100 * G2["ci95"][1])}%] | ≥ 10%, lower bound > 0, lower RMSE on n = 16 '
      f'({f3(G2["heldout_level_rmse_K_law"])} vs {f3(G2["heldout_level_rmse_align_eff"])}), {G2["invalid_K_law"]} invalid '
      f'| **{G2["verdict"]}** |')
    w(f'| G3a coefficient picture | κ = {f3(G3a["kappa"]["estimate"])}, γ1 = {f3(G3a["gamma1"]["estimate"])} '
      f'| κ {interval(G3a["kappa"]["ci95"])}, γ1 {interval(G3a["gamma1"]["ci95"])} | κ > 0, γ1 interval ∋ 0, '
      f'RMSE coef ({f3(G3a["rmse_coef"])}) ≤ exp ({f3(G3a["rmse_exp"])}) | **{G3a["verdict"]}** |')
    w(f'| G3b exponent picture | γ1 = {f3(G3b["gamma1"]["estimate"])}; exp vs coef {f3(100 * G3b["exp_vs_coef_reduction"])}% '
      f'| γ1 {interval(G3b["gamma1"]["ci95"])}; [{f3(100 * G3b["ci95"][0])}%, {f3(100 * G3b["ci95"][1])}%] '
      f'| γ1 upper < 0, reduction ≥ 5% with lower > 0 | **{G3b["verdict"]}** |')
    w(f'| G4 alignment dominates (E = e_α/e_K) | E_bits {f3(G4["E_bits"])}, E_states {f3(G4["E_states"])}, '
      f'E_mf {f3(G4["E_mf"])} (at ε = {G4["E_mf_eps"]}) | bits {interval(G4["ci95_bits"])}; states '
      f'{interval(G4["ci95_states"])} | all > 1, lower bounds > 1 | **{G4["verdict"]}** |')
    w(f'| G5 alignment early, complexity late | r_α early {f3(G5["r_alpha_early"]["r_alpha"])}, late '
      f'{f3(G5["r_alpha_late"]["r_alpha"])} | early {interval(G5["r_alpha_early"]["ci95"])}; late '
      f'{interval(G5["r_alpha_late"]["ci95"])} | early lower > 0.5, late upper < 0.5 | **{G5["verdict"]}**'
      + (f' ({G5["named_failure"]})' if G5.get('named_failure') else '') + ' |')
    w('| G6 (13-GPU only) | — | — | — | pending, not run |')
    w('')
    w(f'C3 (coefficient or exponent): **{A["C3"]}**.')
    w('')
    w(f'![Experiment 13-small results]({fig_name}.png)')
    w('')
    w('## Draft-protocol versions (reported, not gated)')
    w('')
    w('Before the freeze, G1–G4 were aligned with Experiment 11\'s frozen gates (PROMPT step 6). The draft '
      'definitions are reported here so nothing pre-registered disappears.')
    w('')
    w(f'- Law-based M_K ({D["law_for_multipliers"]}, fit rows, n = 32 vs 4): {f3(D["G1_law_M_K"]["estimate"])} '
      f'{interval(D["G1_law_M_K"]["ci95"])}.')
    w(f'- Range ratio R = log M_α / log M_K: {f3(D["G4a_range_ratio"]["R"])} {interval(D["G4a_range_ratio"]["ci95"])} '
      f'(M_α = {f3(D["G4a_range_ratio"]["M_alpha"])}, M_K = {f3(D["G4a_range_ratio"]["M_K"])}).')
    w(f'- Complexity gap at α = 1, Δ(T_last fit)/Δ(T_first): {f3(D["G4b_gap_ratio"]["ratio"])} '
      f'{interval(D["G4b_gap_ratio"]["ci95"])} (Δ = {f3(D["G4b_gap_ratio"]["gap_first"])} → '
      f'{f3(D["G4b_gap_ratio"]["gap_last_fit"])} bits/token).')
    w(f'- Bonferroni G2: M_K_coef {f3(100 * D["G2_bonferroni"]["coef"]["reduction"])}% '
      f'[{f3(100 * D["G2_bonferroni"]["coef"]["ci_bonf"][0])}%, …]; M_K_exp '
      f'{f3(100 * D["G2_bonferroni"]["exp"]["reduction"])}% [{f3(100 * D["G2_bonferroni"]["exp"]["ci_bonf"][0])}%, …].')
    w(f'- ρ = RMSE(M_K_exp)/RMSE(M_K_coef) = {f3(D["G3_rho"]["rho"])} {interval(D["G3_rho"]["ci95"])}.')
    w('')
    w('## Held-out RMSE of every law (log2 ε)')
    w('')
    w('| Law | primary | no offset | K = n | K = K_bits − log2 n! | invalid (primary) | level only | α only | T only |')
    w('|---|---|---|---|---|---|---|---|---|')
    H_ = A['heldout_rmse']
    for law in H_['primary']:
        p = H_['primary'][law]
        w(f'| {law} | {f3(p["rmse"])} | {f3(H_["no_offset"][law]["rmse"])} | {f3(H_["K_n"][law]["rmse"])} | '
          f'{f3(H_["K_canonical"][law]["rmse"])} | {p["invalid"]} | {f3(p["by_coordinate"]["level_only"])} | '
          f'{f3(p["by_coordinate"]["alpha_only"])} | {f3(p["by_coordinate"]["T_only"])} |')
    w('')
    w(f'Independent fitter (L-BFGS-B, different starts): **{"AGREE" if A["independent_fit"]["agree"] else "DISAGREE"}**'
      + ('' if A['independent_fit']['agree'] else f' ({len(A["independent_fit"]["disagreements"])} seed-law pairs)') + '.')
    w('')
    w('## Fitted laws per seed (fit rows)')
    w('')
    for law, per in A['fitted_laws_per_seed'].items():
        keys = [k for k in next(iter(per.values())) if k not in ('g_lo', 'g_hi')]
        w(f'- **{law}**: ' + '; '.join(f'seed {s}: ' + ', '.join(f'{k} {f3(v[k])}' for k in keys) for s, v in per.items()))
    w('')
    ac = A['alignment_comparison']
    w(f'η from M_align_eff: {f3(ac["eta_align_eff"]["estimate"])} {interval(ac["eta_align_eff"]["ci95"])}; from the G2 K law: '
      f'{f3(ac["eta_K_law"]["estimate"])} {interval(ac["eta_K_law"]["ci95"])}. M_align_eff vs M_align_add held-out '
      f'RMSE reduction: {f3(100 * ac["align_eff_vs_add_reduction"])}% [{f3(100 * ac["ci95"][0])}%, {f3(100 * ac["ci95"][1])}%].')
    w('')
    w('## Multiplier curves')
    w('')
    w('| ε* | M_K (law) | M_K (model-free) | M_α = 20^η | E (bits) | E 95% | right-censored α = 1 target-seeds |')
    w('|---|---|---|---|---|---|---|')
    for e, v in sorted(A['multiplier_curves'].items(), key=lambda x: float(x[0])):
        w(f'| {e} | {f3(v["M_K_law"])} | {f3(v["M_K_model_free"])} | {f3(v["M_alpha_law"])} | {f3(v["E_bits"])} | '
          f'{interval(v["E_ci95"])} | {v["right_censored_alpha1"]} |')
    w('')
    w('## Position decomposition (Section 4.5)')
    w('')
    w('| Slice | r_α of ε | r_α of ε_learn | r_α of ε_id |')
    w('|---|---|---|---|')
    P = A['position_decomposition']
    for s in ('early', 'middle', 'late', 'all'):
        w(f'| {s} | ' + ' | '.join(f'{f3(P[f"{s}|{q}"]["r_alpha"])} {interval(P[f"{s}|{q}"]["ci95"])}'
                                    for q in ('excess_bits', 'learn_bits', 'ident_bits')) + ' |')
    w('')
    w('Cell means (all positions, averaged over targets and seeds), bits/token:')
    w('')
    cm = A['cell_means']
    Ts = sorted({int(k.split('|')[3]) for k in cm})
    w('| n | α | ' + ' | '.join(f'ε at 2^{int(math.log2(T))}' for T in Ts) + ' | ε_id |')
    w('|---|---|' + '---|' * (len(Ts) + 1))
    for k in sorted({int(x.split('|')[1]) for x in cm}):
        for a in sorted({float(x.split('|')[2]) for x in cm}):
            w(f'| {k} | {a} | ' + ' | '.join(f3(cm[f"all|{k}|{a}|{T}"]["excess"]) for T in Ts)
              + f' | {f3(cm[f"all|{k}|{a}|{Ts[0]}"]["ident"])} |')
    w('')
    m = A['mechanism']
    w('## Mechanism analyses (reported, not gated)')
    w('')
    w(f'- Quanta test: Spearman(γ at α = 1, stationary tail exponent) over the 8 targets = '
      f'{f3(m["spearman_gamma_vs_tail"]["rho"])} (p = {f3(m["spearman_gamma_vs_tail"]["p"])}).')
    sh = m['eps_id_shape']
    w(f'- ε_id versus α: total residual sum of squares, linear in (1 − α) {f3(sh["total_rss_linear"])} vs '
      f'log2(1/α) {f3(sh["total_rss_log"])} (one parameter per target).')
    w(f'- Share of early-slice excess that is ε_id at the largest T: '
      + ', '.join(f'α = {a}: {f3(v)}' for a, v in m['early_share_eps_id_at_Tmax'].items()) + '.')
    w(f'- Distractor excess (model − distractor oracle) at the largest T by α: '
      + ', '.join(f'{a}: {f3(v)}' for a, v in m['distractor_excess_at_Tmax_by_alpha'].items()) + ' bits/token.')
    w('')
    w('## Diversity link')
    w('')
    w(A['diversity'] + '.')
    w('')
    w('## Pilot decisions (disclosed before the freeze)')
    w('')
    w(f'- Seed {pilot["seed"]}, 4 pilot concepts, α ∈ {{0.05, 0.4}}, T to 2^21: {pilot["runs"]} runs, '
      f'{pilot["train_tokens_per_second_per_worker"]["median"]:.0f} training tokens/s per worker under 4-way load, '
      f'{pilot["eval_seconds_per_checkpoint"]:.1f} s evaluation per checkpoint.')
    w(f'- Projected grid: {pilot["projected_hours_primary"]:.2f} h at T_max = 2^21, {pilot["projected_hours_fallback"]:.2f} h '
      f'at 2^20 → T range {"fallback 2^16..2^20" if pilot["use_fallback"] else "primary 2^17..2^21"}.')
    w(f'- ε* rule: n = 32 pilot concept at α = 0.4 had ε_learn = {pilot["hard_learn_bits_at_largest_fit_T_alpha0_4"]:.4f} '
      f'at T = {pilot["largest_fit_T"]} → ε* = {pilot["epsilon_star"]} '
      f'({"raised" if pilot["eps_star_rule_fires"] else "unchanged"}). Eval SE of ε (median) '
      f'{pilot["eval_se_excess_all"]["median"]:.4f} bits/token.')
    w('')
    w('## Which PROTOCOL Section 6 row matches')
    w('')
    w(section6(A))
    w('')
    w(f'**TLDR-end:** {tldr}' if tldr else '')
    (EXP / 'RESULTS_ginc_hmm.md').write_text('\n'.join(L) + '\n')


def section6(A):
    g = {k: v['verdict'] == 'SUPPORTED' for k, v in A['gates'].items()}
    rows = []
    if g['G1'] and g['G2'] and g['G4']:
        rows.append('"G1, G2, G4 SUPPORTED": Brando\'s claim holds on GINC-style data.')
    if g['G1'] and not g['G4']:
        rows.append('"G1 SUPPORTED, G4 fails": complexity rivals or beats alignment here.')
    if g['G1'] and not g['G2']:
        rows.append('"G2 NOT_ESTABLISHED, G1 SUPPORTED": complexity matters but the K term does not generalize to the '
                    'held-out level.')
    if g['G3a']:
        rows.append('"G3 coefficient": description length acts like an MDL parameter count.')
    if g['G3b']:
        rows.append('"G3 exponent": complexity changes the learning rate itself.')
    if g['G5']:
        rows.append('"G5 SUPPORTED": alignment pays for identification, complexity for modeling.')
    elif A['gates']['G5'].get('named_failure') == 'K dominates everywhere':
        rows.append('"G5 K dominates everywhere": complex concepts are also harder to identify.')
    if not rows:
        rows.append('No row of Section 6 applies cleanly: no primary gate was SUPPORTED, so the run neither confirms '
                    'nor refutes the claims at this model size and budget.')
    return '\n'.join(f'- {r}' for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='small')
    ap.add_argument('--notes', default=None)
    ap.add_argument('--tldr', default='')
    a = ap.parse_args()
    out = HERE / 'results' / a.arm
    A = json.loads((out / 'analysis.json').read_text())
    raw = json.loads((HERE / 'config.json').read_text())
    c = dict(raw['shared'])
    c.update(raw[a.arm])
    name = 'ginc_hmm_results'
    figures(out, A, c, EXP / name)
    notes = Path(a.notes).read_text() if a.notes else ''
    results_md(out, A, c, notes, a.tldr, name)
    print('wrote', EXP / 'RESULTS_ginc_hmm.md')


if __name__ == '__main__':
    main()
