"""Write RESULTS_task2vec_complexity.md and results/task2vec_vs_K.png from results/gates.json and analysis.json.

  report.py --notes notes.md --tldr "..." [--results results]
`--notes` is a markdown file with the hand-written sections (takeaway, pilot disclosure, deviations, limitations,
artifact location); it is inserted verbatim after the title. Layout follows Experiment 12's report.py.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

import tasks14 as T  # noqa: E402

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, PROBE = '#1a1a19', '#6b6a63', '#e4e3dc', '#b9b8b0'
FAM_NAME = dict(dfa='DFA (Experiment 11)', dsl='DSL (Experiment 12)')
GATE_TEXT = dict(
    G1=('Spearman(K_hat, K) on embedded tasks', 'rho', 'ci', 'g1_min_rho', 'g1_lb_above'),
    G2=('Partial Spearman(K_hat, K \\| label entropy, sensitivity)', 'partial', 'ci', 'g2_min_partial', 'g2_lb_above'),
    G3=('Spearman(K_hat, T*) on targets (alpha = 1, censored = +inf)', 'rho', 'ci', 'g3_min_rho', 'g3_lb_above'),
    G4a=('Diversity vs mixture entropy H, coverage mixes', 'rho_H', 'rho_H_ci', 'g4_min_rho_H', 'g4_rho_H_lb_above'),
    G4b=('rho_H - \\|rho_K\\|, coverage mixes', 'gap', 'gap_ci', 'g4_min_gap', 'g4_gap_lb_above'),
    G5=('L1(25% label noise) / L1(clean), median over tasks', 'ratio', 'ci', 'g5_min_ratio', 'g5_lb_above'))


def fmt(x, d=3):
    if x is None:
        return 'n/a'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return '[' + ', '.join(fmt(v, d) for v in x) + ']'
    if np.isnan(x):
        return 'n/a'
    if not np.isfinite(x):
        return '∞' if x > 0 else '−∞'
    return f'{x:.{d}g}'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(r, k):
    v = r.get(k, '')
    return float(v) if v not in ('', None) else np.nan


def rank_resid(v, Z):
    rv = rankdata(v)
    A = np.column_stack([np.ones(len(v))] + [rankdata(z) for z in Z.T])
    return rv - A @ np.linalg.lstsq(A, rv, rcond=None)[0]


# ---------------------------------------------------------------- figure

def figure(c, out, G, trows, drows):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for row, fam in enumerate(T.FAMILIES):
        tr = [r for r in trows if r['family'] == fam]
        K = np.array([num(r, 'K') for r in tr])
        kh = np.array([num(r, 'khat') for r in tr])
        L = np.array([[num(r, f'log_l1_{p}') for p in c['probe_seeds']] for r in tr])
        g = G['families'][fam]
        lv = np.unique(K)

        # (a) log L1 norm versus K, every probe (gray) and the mean over probes per level (blue)
        ax = axes[row, 0]
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(K))
        for i in range(L.shape[1]):
            ax.scatter(np.log2(K) + jit, L[:, i], s=8, color=PROBE, lw=0, zorder=2)
            ax.plot(np.log2(lv), [L[K == k, i].mean() for k in lv], color=PROBE, lw=1, zorder=2)
        m = [L[K == k].mean() for k in lv]
        ax.plot(np.log2(lv), m, color=BLUE, lw=2, marker='o', ms=8, zorder=3,
                markeredgecolor='white', markeredgewidth=1.5, label='mean over probes')
        ax.set_xticks(np.log2(lv), [str(int(k)) for k in lv])
        ax.set_xlabel('exact K (log scale)', fontsize=9, color=INK)
        ax.set_ylabel(f'{FAM_NAME[fam]}\nlog L1 norm of E0 embedding', fontsize=9, color=INK)
        ax.set_title(f"Gate 1: rho = {g['G1']['rho']:.2f} [{g['G1']['ci']['lo']:.2f}, {g['G1']['ci']['hi']:.2f}]",
                     fontsize=9, color=INK)
        if row == 0:
            ax.legend(frameon=False, fontsize=8, loc='best')
        style(ax)

        # (b) partial plot: ranks of K_hat and K after removing ranks of label entropy and sensitivity
        ax = axes[row, 1]
        Z = np.column_stack([[num(r, 'label_entropy') for r in tr], [num(r, 'sensitivity') for r in tr]])
        ax.scatter(rank_resid(K, Z), rank_resid(kh, Z), s=12, color=BLUE, lw=0, alpha=0.8)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel('rank of K, residual on covariates', fontsize=9, color=INK)
        ax.set_ylabel('rank of K_hat, residual on covariates', fontsize=9, color=INK)
        ax.set_title(f"Gate 2: partial = {g['G2']['partial']:.2f} "
                     f"[{g['G2']['ci']['lo']:.2f}, {g['G2']['ci']['hi']:.2f}]", fontsize=9, color=INK)
        style(ax)

        # (c) K_hat versus T*; censored targets drawn hollow at the right edge
        ax = axes[row, 2]
        tg = [r for r in tr if r['role'] == 'target']
        ts = np.array([num(r, 'tstar_median') for r in tg])
        kt = np.array([num(r, 'khat') for r in tg])
        fin = np.isfinite(ts)
        edge = (np.nanmax(ts[fin]) * 2.5) if fin.any() else 1e6
        ax.scatter(ts[fin], kt[fin], s=40, color=BLUE, lw=0, label='reached eps*')
        if (~fin).any():
            ax.scatter(np.full((~fin).sum(), edge), kt[~fin], s=40, facecolors='none', edgecolors=ORANGE, lw=1.5,
                       marker='s', label=f'censored ({(~fin).sum()})')
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.tick_params(axis='x', labelrotation=30)
        ax.set_xlabel(f"T* at alpha = 1 (eps* = {g['G3']['eps_star']:g} bits; censored at right)", fontsize=9,
                      color=INK)
        ax.set_ylabel('K_hat', fontsize=9, color=INK)
        ax.set_title(f"Gate 3: rho = {g['G3']['rho']:.2f} [{g['G3']['ci']['lo']:.2f}, {g['G3']['ci']['hi']:.2f}]",
                     fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8, loc='best')
        style(ax)

        # (d) diversity coefficient versus mixture entropy on coverage mixes, with per-mix intervals
        ax = axes[row, 3]
        if g['G4'].get('not_run'):
            ax.axis('off')
            ax.text(0.5, 0.5, 'Gate 4 (diversity vs mixture entropy)\nnot run: the wall-clock cap was\nreached before the diversity jobs',
                    ha='center', va='center', fontsize=9, color=MUTED, transform=ax.transAxes)
            continue
        cv = [r for r in drows if r['family'] == fam and r['mix_kind'] == 'coverage' and r['batches'] == 'interleaved']
        H = np.array([num(r, 'H') for r in cv])
        d = np.array([num(r, 'div') for r in cv])
        lo, hi = np.array([num(r, 'div_ci_lo') for r in cv]), np.array([num(r, 'div_ci_hi') for r in cv])
        x = H + np.random.default_rng(1).uniform(-0.08, 0.08, len(H))
        ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt='none', ecolor=PROBE, lw=1, zorder=1)
        ax.scatter(x, d, s=14, color=BLUE, lw=0, zorder=2)
        for w in ('lower', 'upper'):
            b = [r for r in drows if r['family'] == fam and r['mix_id'] == f'{fam}_bound_{w}']
            if b:
                ax.axhline(num(b[0], 'div'), color=MUTED, lw=0.8, ls='--')
                ax.annotate(f'{w} bound', (1.0, num(b[0], 'div')), xycoords=('axes fraction', 'data'), fontsize=7,
                            color=MUTED, ha='right', va='bottom' if w == 'upper' else 'top')
        ax.set_xlabel('mixture entropy H (bits), coverage mixes', fontsize=9, color=INK)
        ax.set_ylabel('diversity coefficient', fontsize=9, color=INK)
        ax.set_title('Gate 4: not run' if g['G4'].get('not_run') else
                     f"Gate 4: rho_H = {g['G4']['rho_H']:.2f}, rho_K = {g['G4']['rho_K']:.2f}", fontsize=9, color=INK)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / 'task2vec_vs_K.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- markdown

def gate_rows(c, G):
    L = ['| Gate | Statistic | Family | Estimate | 95% interval | Threshold | Family passes |',
         '|---|---|---|---|---|---|---|']
    th = G['thresholds']
    for gk, (text, est, cik, tmin, tlb) in GATE_TEXT.items():
        src = gk[:2]
        for fam in T.FAMILIES:
            g = G['families'][fam][src]
            if g.get('not_run'):
                L.append(f"| {src} | {text} | {fam.upper()} | not run | — | ≥ {th[tmin]:g}, lower bound > {th[tlb]:g} "
                         f"| — |")
                continue
            ci = g[cik]
            L.append(f"| {src} | {text} | {fam.upper()} | {fmt(g[est])} | [{fmt(ci['lo'])}, {fmt(ci['hi'])}] | "
                     f"≥ {th[tmin]:g}, lower bound > {th[tlb]:g} | {fmt(g['pass'])} |")
    L += ['', '| Gate | Verdict (both families must pass) |', '|---|---|']
    L += [f'| {k} | **{v}** |' for k, v in G['verdicts'].items()]
    return L


def variant_rows(A):
    names = [('E0', None), ('E1', 'E1'), ('E2', 'E2'), ('E3', 'E3'), ('E0, rescaled probe', 'rescaled_E0'),
             ('E0, L2 norm', 'E0_L2'), ('E0, log-sum', 'E0_logsum'), ('E0, block 1 only', 'E0_block1'),
             ('E0, block 2 only', 'E0_block2'), ('E0, random-init probe (point only)', 'random_init_E0')]
    L = ['| Embedding | Family | Gate 1 rho [CI] | Gate 2 partial [CI] | Gate 3 rho [CI] |', '|---|---|---|---|---|']
    for label, key in names:
        if key is None:
            continue
        for fam in T.FAMILIES:
            v = A[fam]['variants'][key]
            if v.get('not_run'):
                L.append(f'| {label} | {fam.upper()} | not run (wall cap) | | |')
                continue

            def cell(s, cik):
                if cik in v:
                    return f"{fmt(v[s])} [{fmt(v[cik]['lo'])}, {fmt(v[cik]['hi'])}]"
                return fmt(v[s])
            L.append(f"| {label} | {fam.upper()} | {cell('g1_rho', 'g1_ci')} | {cell('g2_partial', 'g2_ci')} | "
                     f"{cell('g3_rho', 'g3_ci')} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--notes', required=True)
    ap.add_argument('--tldr', required=True)
    a = ap.parse_args()
    c = T.load_config()
    out = HERE / a.results
    G = json.loads((out / 'gates.json').read_text())
    A = json.loads((out / 'analysis.json').read_text())
    trows, drows = read_csv(out / 'task_table.csv'), read_csv(out / 'diversity.csv')
    man = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    figure(c, out, G, trows, drows)

    L = ['# Experiment 14: does Task2Vec track exact complexity?', '', f'**TL;DR.** {a.tldr}', '',
         Path(a.notes).read_text().rstrip(), '',
         f'![Task2Vec versus exact K](expt_v1/{a.results}/task2vec_vs_K.png)', '',
         'Rows: DFA tasks (Experiment 11), DSL tasks (Experiment 12). Columns: log L1 norm of the E0 embedding '
         'against exact K (gray: each of the 4 probes; blue: mean over probes); the same after removing label '
         'entropy and sensitivity (rank residuals); K_hat against T* on the training targets (hollow squares: '
         'never reached eps*, treated as +inf); diversity coefficient against mixture entropy on the coverage mixes '
         '(gray bars: 95% bootstrap interval over batch pairs and probes; dashed: reference bounds).', '',
         '## Gates', ''] + gate_rows(c, G)
    h = G['handoff']
    L += ['', f"Handoff to real data (Gates 1-3 supported, Kendall's W ≥ {c['handoff']['min_kendall_w']}, rescaled "
          f"rank correlation ≥ {c['handoff']['min_rescaled_rho']}, non-negative incremental value in both families): "
          f"**{'met' if h['met'] else 'not met'}**.", '',
          '## Estimators and summaries (reported, not gated)', ''] + variant_rows(A)
    G5 = {f: G['families'][f]['G5'] for f in T.FAMILIES}
    failed = [f.upper() for f in T.FAMILIES if not G5[f].get('not_run') and not G5[f].get('pass')]
    if failed and any(G5[f].get('not_run') for f in T.FAMILIES):
        i = L.index('## Estimators and summaries (reported, not gated)')
        L[i:i] = [f"Gate 5 is NOT_RUN only because one family's inputs are missing; it cannot pass, since "
                  f"{', '.join(failed)} already fails its threshold and both families must pass.", '']
    L += ['', '## Probe and parameterization sensitivity', '',
          "| Family | Kendall's W | mean pairwise Spearman | probe pairs disagreeing on K-level order | "
          "Spearman(K_hat, rescaled K_hat) | Gate 1 rho change, rescaled | median L1 ratio rescaled/original |",
          '|---|---|---|---|---|---|---|']
    for fam in T.FAMILIES:
        p, q = A[fam]['probe_sensitivity'], A[fam]['parameterization']
        if q.get('not_run'):
            q = {k: None for k in ('spearman_khat_original_vs_rescaled', 'g1_rho_change',
                                   'median_l1_ratio_rescaled_over_original')}
        L.append(f"| {fam.upper()} | {fmt(p['kendall_w'])} | {fmt(p['mean_pairwise_spearman'])} | "
                 f"{fmt(p['frac_probe_pairs_disagree_on_level_order'])} | "
                 f"{fmt(q['spearman_khat_original_vs_rescaled'])} | {fmt(q['g1_rho_change'])} | "
                 f"{fmt(q['median_l1_ratio_rescaled_over_original'])} |")
    L += ['', '## Mechanism', '', '| Family | ' + ' | '.join(k.replace('_', ' ') for k in A['dfa']['mechanism']) +
          ' |', '|---' * (1 + len(A['dfa']['mechanism'])) + '|']
    for fam in T.FAMILIES:
        L.append(f'| {fam.upper()} | ' + ' | '.join(fmt(v) for v in A[fam]['mechanism'].values()) + ' |')
    L += ['', '## Incremental value over exact K for T*', '']
    for fam in T.FAMILIES:
        L.append(f'- {fam.upper()}: ' + '; '.join(f"{k.replace('_', ' ')} = {fmt(v)}"
                                                  for k, v in A[fam]['incremental'].items()))
    L += ['', '## E0 against the per-example Fisher (E2)', '']
    for fam in T.FAMILIES:
        e = A[fam]['e0_vs_e2']
        L.append(f"- {fam.upper()}: median (E0 L1 × batch size) / (E2 L1) = {fmt(e['median_ratio_E0xB_over_E2'])}; "
                 f"Spearman by probe {fmt(e['spearman_by_probe'])}")
    L += ['', '## Diversity coefficient (reported analyses)', '',
          '| Family | Mixes | Batches | rho with H | rho with K | rho with Rao Q | other |', '|---|---|---|---|---|---|---|']
    for fam in T.FAMILIES:
        dv = A[fam]['diversity']
        ci_, cs, ex = dv['coverage_interleaved'], dv['coverage_single'], dv['exp_interleaved']
        L.append(f"| {fam.upper()} | coverage ({ci_['n']}) | interleaved | {fmt(ci_['rho_H'])} | {fmt(ci_['rho_K'])} | "
                 f"{fmt(ci_['rho_Q'])} | range {fmt(ci_['div_range'])} |")
        L.append(f"| {fam.upper()} | coverage ({cs['n']}) | single-task | {fmt(cs['rho_H'])} | {fmt(cs['rho_K'])} | "
                 f"{fmt(cs['rho_Q'])} | range {fmt(cs['div_range'])} |")
        L.append(f"| {fam.upper()} | Experiment 11/12 training mixes ({ex['n']}) | interleaved | {fmt(ex['rho_H'])} | "
                 f"{fmt(ex['rho_K'])} | {fmt(ex['rho_Q'])} | rho with alpha {fmt(ex['rho_alpha'])}; alpha = 1 "
                 f"slice rho with K {fmt(dv['alpha1_slice']['rho_div_K'])} (n = {dv['alpha1_slice']['n']}) |")
        bd = {w: fmt(dv['bounds'][w]['div']) if w in dv['bounds'] else 'not run' for w in ('lower', 'upper')}
        L.append(f"| {fam.upper()} | reference bounds | — | — | — | — | lower {bd['lower']}, upper {bd['upper']} |")
    G5 = {f: G['families'][f]['G5'] for f in T.FAMILIES}
    L += ['', '## Label noise', '']
    for fam in T.FAMILIES:
        L.append(f"- {fam.upper()}: not run (wall cap)" if G5[fam].get('not_run') else
                 f"- {fam.upper()}: L1 ratio noisy/clean by rate {G5[fam]['ratio_at_other_rates']} "
                 f"({G5[fam]['n_tasks']} tasks)")
    L += ['', '## Censoring of T*', '']
    for fam in T.FAMILIES:
        g3 = G['families'][fam]['G3']
        L.append(f"- {fam.upper()}: {g3['n_censored_runs']} of {g3['n_runs']} alpha = 1 training runs never reached "
                 f"eps* = {g3['eps_star']:g} bits; {g3['n_censored_targets']} of {g3['n_targets']} targets have a "
                 f"censored median T* (ranked highest, tied).")
    if man:
        L += ['', '## Run', '', f"Wall time {fmt(man.get('wall_seconds_cumulative', float('nan')) / 3600, 3)} h on "
              f"a shared 4-core, 15 GB CPU machine (no GPU); versions {man.get('versions')}; "
              f"bootstrap {G['bootstrap']['draws']} draws, "
              f"seed {G['bootstrap']['seed']}. Full numbers: expt_v1/{a.results}/gates.json, analysis.json, "
              'task_table.csv, diversity.csv.']
        if man.get('complete') is False:
            miss = man.get('missing_jobs', [])
            kinds = {}
            for j in miss:
                k = j.split('_')[0] + ('_' + j.split('_')[3] if j.startswith('emb_') else '')
                k = dict(div='diversity', emb_main='random-init-probe embedding', emb_noise='label-noise embedding',
                         emb_rescaled='rescaled-probe embedding').get(k, k)
                kinds[k] = kinds.get(k, 0) + 1
            L += ['', f"**Incomplete run.** The run stopped at the {c['max_wall_seconds'] / 3600:g} h wall cap with "
                  f"{len(miss)} jobs not run ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}); "
                  'analyses that need them are marked not run. The run is resumable with `run.py --resume`.']
    (EXP / 'RESULTS_task2vec_complexity.md').write_text('\n'.join(L) + '\n')
    print('wrote', EXP / 'RESULTS_task2vec_complexity.md', out / 'task2vec_vs_K.png')


if __name__ == '__main__':
    main()
