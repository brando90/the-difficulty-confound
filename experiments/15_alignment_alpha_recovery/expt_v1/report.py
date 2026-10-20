"""Writes ../RESULTS_alpha_recovery.md and ../alpha_recovery.png from results/ (gates.json, laws.json, controls.csv,
decomposition.csv, alpha_hat.csv, graded/metrics.csv, manifest.json, verify_log.json) and the hand-written
results/results_notes.md (takeaway, Section 6 row, deviations).

Usage: report.py [--results results]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import common as C

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

MEASURE_NAMES = {
    'a': '(a) zstd dictionary compression ratio', 'aprime': "(a') dictionary minus no-dictionary size",
    'a2': '(a2) gzip NCD to references', 'b1': '(b1) probe-embedding max cosine', 'b2': '(b2) byte-histogram cosine',
    'c': '(c) = (a + b1) / 2, the paper\'s combined score', 'c2': '(c2) = (a + b2) / 2',
    'd': '(d) frozen-gradient cosine', 'e': '(e) Task2Vec set-level (E3)', 'e2': '(e2) Task2Vec set-level (E2)',
    'oracle': 'oracle (true label agreement)', 'random': 'random scores'}
CONTROL_NAMES = {'leak': 'task token left in (leak)', 'perm': 'permuted-label references', 'inp': 'input-only'}

# reference palette (light): chart chrome, two categorical slots, ordinal blue ramp
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
CAT = ['#2a78d6', '#eb6834']
RAMP = {4: ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'], 5: ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']}


def f3(x):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.3f}'


def ci(v):
    return f'[{f3(v[0])}, {f3(v[1])}]'


def pct(x):
    return f'{100 * x:+.1f}%'


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def notes_sections(res):
    """results_notes.md split on '## ' headings -> {heading: body}."""
    p = res / 'results_notes.md'
    if not p.exists():
        return {}
    out, cur = {}, None
    for line in p.read_text().splitlines():
        if line.startswith('## '):
            cur = line[3:].strip()
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return {k: '\n'.join(v).strip() for k, v in out.items()}


# ---------------------------------------------------------------- figure

def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def figure(c, res, gates, ah_rows, grows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), facecolor=SURF)
    # A, B: alpha-hat (c) vs alpha by K level
    for ax, fam, lab in ((axs[0, 0], 'dfa', 'A'), (axs[0, 1], 'dsl', 'B')):
        style(ax)
        rows = [r for r in ah_rows if r['arm'] == fam and r['measure'] == 'c' and r['probe'] == 'mean']
        Ks = sorted({int(r['K']) for r in rows})
        al = sorted({float(r['alpha']) for r in rows})
        for i, K in enumerate(Ks):
            y = [np.mean([float(r['alpha_hat']) for r in rows if int(r['K']) == K and float(r['alpha']) == a])
                 for a in al]
            ax.plot(al, y, '-o', color=RAMP[len(Ks)][i], lw=2, ms=6, label=f'K = {K}')
        ora = [r for r in ah_rows if r['arm'] == fam and r['measure'] == 'oracle' and r['probe'] == 'mean']
        yo = [np.mean([float(r['alpha_hat']) for r in ora if float(r['alpha']) == a]) for a in al]
        ax.plot(al, yo, ':', color=MUTED, lw=1.5, label='oracle (all K)')
        ax.set_xscale('log')
        ax.set_xticks(al)
        ax.set_xticklabels([f'{a:g}' for a in al])
        ax.set_xlabel('true alpha (share of target examples)')
        ax.set_ylabel('measured alpha-hat, measure (c)')
        ax.set_title(f'{lab}. {fam.upper()}: measured vs true alpha, by K', loc='left', fontsize=10)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    # C: Gate 1 min rho per measure; D: Gate 2 slope per measure
    ms = ['a', 'aprime', 'a2', 'b1', 'b2', 'c', 'c2', 'd', 'e', 'e2', 'oracle', 'random']
    pm = gates['per_measure']
    ax = axs[0, 2]
    style(ax)
    y = np.arange(len(ms))
    for j, fam in enumerate(C.FAMILIES):
        ax.scatter([pm[f'{fam}|{m}']['g1_min_rho'] for m in ms], y + (j - 0.5) * 0.25, s=36, color=CAT[j],
                   edgecolor=SURF, linewidth=1.5, label=fam.upper(), zorder=3)
    ax.axvline(c['gates']['g1_min_rho'], color=INK2, lw=1, ls='--')
    ax.text(c['gates']['g1_min_rho'], -0.7, 'Gate 1 bar ', color=INK2, fontsize=7, ha='right')
    ax.set_yticks(y)
    ax.set_yticklabels(ms, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Gate 1: lowest Spearman rho over K levels')
    ax.set_title('C. Recovery within K level, every measure', loc='left', fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc='upper left')
    ax = axs[1, 0]
    style(ax)
    band = c['gates']['g2_band']
    ax.axvspan(-band, band, color=GRID, alpha=0.8, lw=0)
    for j, fam in enumerate(C.FAMILIES):
        s = [pm[f'{fam}|{m}']['g2_slope'] for m in ms]
        lo = [pm[f'{fam}|{m}']['g2_ci'][0] for m in ms]
        hi = [pm[f'{fam}|{m}']['g2_ci'][1] for m in ms]
        yy = y + (j - 0.5) * 0.25
        ax.hlines(yy, lo, hi, color=CAT[j], lw=2)
        ax.scatter(s, yy, s=36, color=CAT[j], edgecolor=SURF, linewidth=1.5, label=fam.upper(), zorder=3)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(ms, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Gate 2: change in alpha-hat per doubling of K (95% CI)')
    ax.set_title('D. Complexity-neutrality, every measure (shaded: Gate 2 band)', loc='left', fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2, loc='lower left')
    # E: graded final loss vs alpha_soft; F: alpha-hat (c) vs alpha_soft
    Tm = max(r['T'] for r in grows)
    fin = {}
    for r in grows:
        if r['T'] == Tm:
            fin.setdefault((r['target_id'], r['mix']), []).append(r)
    shares = sorted({r['alpha_share'] for r in grows})
    ax = axs[1, 1]
    style(ax)
    for j, sh in enumerate(shares):
        pts = [(v[0]['alpha_soft'], np.mean([r['excess_bits'] for r in v])) for v in fin.values()
               if v[0]['alpha_share'] == sh]
        ax.scatter(*zip(*pts), s=40, color=CAT[j], edgecolor=SURF, linewidth=1.5, label=f'alpha_share = {sh:g}',
                   zorder=3)
    ax.set_xlabel('alpha_soft (chance-corrected relevance-weighted share)')
    ax.set_ylabel(f'excess loss at T = {Tm:,} (bits, mean over seeds)')
    ax.set_title('E. Graded arm: final loss vs graded relevance', loc='left', fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    ax = axs[1, 2]
    style(ax)
    gr = [r for r in ah_rows if r['arm'] == 'graded' and r['measure'] == 'c' and r['probe'] == 'mean']
    for j, sh in enumerate(shares):
        pts = [(float(r['alpha_soft']), float(r['alpha_hat'])) for r in gr if float(r['alpha']) == sh]
        ax.scatter(*zip(*pts), s=40, color=CAT[j], edgecolor=SURF, linewidth=1.5, label=f'alpha_share = {sh:g}',
                   zorder=3)
    ax.set_xlabel('alpha_soft')
    ax.set_ylabel('measured alpha-hat, measure (c)')
    ax.set_title('F. Graded arm: measured alpha vs graded relevance', loc='left', fontsize=10)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------- markdown

def report(c, res, outdir=EXP):
    gates = json.loads((res / 'gates.json').read_text())
    laws = json.loads((res / 'laws.json').read_text())
    ctrl = read_csv(res / 'controls.csv')
    dec = read_csv(res / 'decomposition.csv')
    ah_rows = read_csv(res / 'alpha_hat.csv')
    import laws as LW
    grows = LW.read_graded_metrics(res / 'graded' / 'metrics.csv')
    man = json.loads((res / 'manifest.json').read_text()) if (res / 'manifest.json').exists() else {}
    ver = json.loads((res / 'verify_log.json').read_text()) if (res / 'verify_log.json').exists() else None
    notes = notes_sections(res)
    figure(c, res, gates, ah_rows, grows, outdir / 'alpha_recovery.png')
    G1, G2, G3, G4a, G4b = (gates[k] for k in ('G1', 'G2', 'G3', 'G4a', 'G4b'))
    L = []
    w = L.append
    w('# Results: Can measured alignment recover the true alignment? (Experiment 15)')
    w('')
    w(f"Run on CPU (4 cores, 15 GB), {notes.get('Date', '').strip() or 'see CKPT'}. Protocol: [PROTOCOL.md](PROTOCOL.md) "
      '(frozen before any gated run); log: [CKPT_alpha_recovery.md](CKPT_alpha_recovery.md). Every number below is '
      'read from `expt_v1/results/` by `report.py`.')
    w('')
    if 'Takeaway' in notes:
        w('## Takeaway')
        w('')
        w(notes['Takeaway'])
        w('')
    w('## Gate verdicts')
    w('')
    w('All five gates are decided on measure (c), the paper\'s combined score (compression plus probe embedding), '
      'with 20,000 bootstrap draws and 95% percentile intervals.')
    w('')
    g1min = {fam: min(G1['levels'][fam], key=lambda x: x['rho']) for fam in C.FAMILIES}
    rows = [
        ['1 (primary): recovery within each K level', G1['threshold'],
         '; '.join(f"{fam.upper()} lowest rho {f3(g1min[fam]['rho'])} (K = {g1min[fam]['K']}, LB "
                   f"{f3(g1min[fam]['lb'])})" for fam in C.FAMILIES), f"**{G1['verdict']}**"],
        ['2 (primary): complexity-neutrality', G2['threshold'],
         '; '.join(f"{fam.upper()} slope {f3(v['slope'])} {ci(v['ci'])}, oracle {f3(v['oracle_slope'])}"
                   for fam, v in G2['families'].items()), f"**{G2['verdict']}**"],
        ['3 (primary): calibrated alpha-hat in the best law', G3['threshold'],
         '; '.join(f"{fam.upper()} Delta {pct(v['delta_calibrated'])} [{pct(v['delta_calibrated_ci'][0])}, "
                   f"{pct(v['delta_calibrated_ci'][1])}], {v['invalid']['calibrated']} invalid"
                   for fam, v in G3['families'].items()), f"**{G3['verdict']}**"],
        ['4a (secondary): graded relevance predicts loss', G4a['threshold'],
         f"RMSE reduction soft vs share {pct(G4a['reduction'])} [{pct(G4a['reduction_ci'][0])}, "
         f"{pct(G4a['reduction_ci'][1])}]; invalid share/soft {G4a['invalid']['share']}/{G4a['invalid']['soft']}",
         f"**{G4a['verdict']}**"],
        ['4b (secondary): measured alpha tracks graded relevance', G4b['threshold'],
         f"rho(soft) {f3(G4b['rho_soft'])}, rho(share) {f3(G4b['rho_share'])}, difference {f3(G4b['diff'])} "
         f"{ci(G4b['diff_ci'])}", f"**{G4b['verdict']}**"]]
    w(table(['Gate', 'Pass condition', 'Result', 'Verdict'], rows))
    w('')
    w('Read with the earlier experiments: Experiment 11\'s data-efficiency endpoint T\\* is censored by the 2^16 '
      'example budget, Experiment 13 came out the reverse of C2 (E = 0.80), and Experiment 14 is negative.')
    w('')
    if 'Section 6 row' in notes:
        w('**Which Section 6 outcome this is.** ' + notes['Section 6 row'])
        w('')
    if G2['verdict'] != 'SUPPORTED' and 'Gate 2 consequence' in notes:
        w('**Consequence for the alignment-scaling law (Gate 2 not established).** ' + notes['Gate 2 consequence'])
        w('')
    w('![Measured versus true alignment](alpha_recovery.png)')
    w('')
    w('*A, B: measured alpha-hat (measure c, mean over targets and probes) against the hand-set alpha, one line per '
      'complexity level K; the dotted line is the oracle. C: Gate 1\'s lowest within-level Spearman rho for every '
      'measure. D: Gate 2\'s slope per doubling of K with its 95% interval. E, F: the graded-relevance arm, '
      'final loss and measured alpha against alpha_soft.*')
    w('')

    # Gate 1
    w('## Gate 1: recovery within each K level (measure c)')
    w('')
    for fam in C.FAMILIES:
        w(f'**{fam.upper()}**')
        w('')
        w(table(['K', 'Spearman rho', '95% CI', 'independent rho'],
                [[x['K'], f3(x['rho']), ci(x['ci']), f3(x['rho_independent'])] for x in G1['levels'][fam]]))
        w('')
    # Gate 2
    w('## Gate 2: complexity-neutrality (slope of alpha-hat per doubling of K at fixed alpha)')
    w('')
    w(table(['Family', 'slope (c)', '95% CI', 'inside band', 'oracle slope', 'oracle 95% CI',
             'slope of (c) minus oracle', '95% CI'],
            [[fam.upper(), f3(v['slope']), ci(v['ci']), 'yes' if v['inside'] else 'no', f3(v['oracle_slope']),
              ci(v['oracle_ci']), f3(v['slope_minus_oracle']), ci(v['slope_minus_oracle_ci'])]
             for fam, v in G2['families'].items()]))
    w('')
    # Gate 3
    w('## Gate 3: measured alpha in the best Experiment 11/12 law (held-out cells)')
    w('')
    w('Raw alpha-hat is the realistic number: calibration uses the true alpha on fit cells, which real data never '
      'provides.')
    w('')
    rows = []
    for fam, v in G3['families'].items():
        rows.append([fam.upper(), f3(v['rmse']['true']), f3(v['rmse']['calibrated']), f3(v['rmse']['raw']),
                     f"{pct(v['delta_calibrated'])} [{pct(v['delta_calibrated_ci'][0])}, "
                     f"{pct(v['delta_calibrated_ci'][1])}]",
                     f"{pct(v['delta_raw'])} [{pct(v['delta_raw_ci'][0])}, {pct(v['delta_raw_ci'][1])}]",
                     f"{v['invalid']['true']} / {v['invalid']['calibrated']} / {v['invalid']['raw']}"])
    w(table(['Family', 'RMSE true alpha', 'RMSE calibrated (c)', 'RMSE raw (c)', 'Delta calibrated', 'Delta raw',
             'invalid (true / cal / raw)'], rows))
    w('')
    w('Calibrated alpha-hat of every other measure in the same law (reported, no verdict):')
    w('')
    ms = [m for m in MEASURE_NAMES if m != 'c']
    w(table(['Measure'] + [f'{fam.upper()} Delta (invalid)' for fam in C.FAMILIES],
            [[MEASURE_NAMES[m]] + [f"{pct(G3['families'][fam]['other_measures_calibrated'][m]['delta'])} "
                                   f"({G3['families'][fam]['other_measures_calibrated'][m]['invalid']})"
                                   for fam in C.FAMILIES] for m in ms]))
    w('')
    w(f"Law fits: laws.py versus independent_fit.py, max relative difference "
      + ', '.join(f"{fam.upper()} {laws[fam]['agreement']['max_rel_diff']:.1e}" for fam in C.FAMILIES)
      + f" (tolerance {c['law_rel_tol']:g}).")
    w('')
    # Gate 4a
    w('## Gate 4a: does graded relevance predict loss? (leave-one-mix-out, 160 runs)')
    w('')
    w(table(['alpha coordinate', 'pooled held-out RMSE', 'reduction vs share', 'invalid'],
            [['alpha_share (target share)', f3(G4a['rmse']['share']), '-', G4a['invalid']['share']],
             ['alpha_soft (chance-corrected relevance)', f3(G4a['rmse']['soft']),
              f"{pct(G4a['reduction'])} [{pct(G4a['reduction_ci'][0])}, {pct(G4a['reduction_ci'][1])}]",
              G4a['invalid']['soft']],
             ['alpha_agree (raw agreement)', f3(G4a['rmse']['agree']),
              f"{pct(G4a['reduction_agree'])} [{pct(G4a['reduction_agree_ci'][0])}, "
              f"{pct(G4a['reduction_agree_ci'][1])}]", G4a['invalid']['agree']]]))
    w('')
    G = json.loads((res / 'graded' / 'graded_tasks_full.json').read_text())
    rows = []
    for key in sorted(G['mixes']):
        m = G['mixes'][key]
        rows.append([m['target_id'], m['mix'], f"{m['alpha_share']:g}", f3(m['alpha_soft']), f3(m['alpha_agree']),
                     f3(G4a['final_loss_by_mix'][key])])
    w(f"Final excess loss (bits, T = {max(r['T'] for r in grows):,}, mean over 8 seeds) by mix:")
    w('')
    w(table(['target', 'mix', 'alpha_share', 'alpha_soft', 'alpha_agree', 'final excess loss'], rows))
    w('')
    # Gate 4b
    w('## Gate 4b: does measured alpha track graded relevance? (20 graded mixes)')
    w('')
    w(table(['Measure', 'rho(alpha-hat, alpha_soft)', 'rho(alpha-hat, alpha_share)', 'difference', '95% CI'],
            [[MEASURE_NAMES[m], f3(v['rho_soft']), f3(v['rho_share']), f3(v['diff']), ci(v['diff_ci'])]
             for m, v in G4b['all_measures'].items() if m in MEASURE_NAMES]))
    w('')
    # every measure
    w('## Every measure, Gates 1 and 2 statistics (no verdicts)')
    w('')
    pm = gates['per_measure']
    for fam in C.FAMILIES:
        st_al = sorted({float(r['alpha']) for r in ah_rows if r['arm'] == fam})
        w(f'**{fam.upper()}**')
        w('')
        w(table(['Measure', 'Gate 1 rho by K level', 'lowest LB', 'Gate 2 slope [95% CI]', 'slope minus oracle',
                 'mean alpha-hat at alpha = ' + ', '.join(f'{a:g}' for a in st_al)],
                [[MEASURE_NAMES[m], ', '.join(f3(x) for x in pm[f'{fam}|{m}']['g1_rho']),
                  f3(min(pm[f'{fam}|{m}']['g1_lb'])),
                  f"{f3(pm[f'{fam}|{m}']['g2_slope'])} {ci(pm[f'{fam}|{m}']['g2_ci'])}",
                  f3(pm[f'{fam}|{m}']['g2_slope_minus_oracle']),
                  ', '.join(f3(x) for x in pm[f'{fam}|{m}']['mean_alpha_hat_by_alpha'])]
                 for m in MEASURE_NAMES]))
        w('')
    # decomposition
    w('## Affine decomposition: alpha-hat = m_D + (m_T - m_D) alpha, versus K')
    w('')
    w('m_T is the token-weighted mean score of target examples, m_D that of distractor examples; their gap is how '
      'much of a unit of true alpha the measure can see. Means over the targets of each K level (and probes).')
    w('')
    for fam in C.FAMILIES:
        Ks = sorted({int(d['K']) for d in dec if d['arm'] == fam})
        rows = []
        for m in ('a', 'a2', 'b1', 'b2', 'c', 'd', 'oracle'):
            cells = []
            for K in Ks:
                ds = [d for d in dec if d['arm'] == fam and d['measure'] == m and int(d['K']) == K]
                cells.append(f"{np.mean([float(d['m_D']) for d in ds]):.3f} / "
                             f"{np.mean([float(d['separation']) for d in ds]):.3f}")
            res_ = np.mean([float(d['resolution_distinct_values']) for d in dec if d['arm'] == fam
                            and d['measure'] == m])
            rows.append([MEASURE_NAMES[m]] + cells + [f'{res_:,.0f}'])
        w(f'**{fam.upper()}** (cells: m_D / m_T - m_D)')
        w('')
        w(table(['Measure'] + [f'K = {K}' for K in Ks] + ['distinct score values in pool'], rows))
        w('')
    # controls
    w('## Controls')
    w('')
    w('Leak: the task-token byte left in (the identity cue the main scores strip). Permuted labels: references with '
      'shuffled labels. Input-only: labels removed from the serialization. (b1) reads inputs only, so its '
      'permuted-label and input-only versions equal (b1) by construction.')
    w('')
    w(table(['Family', 'Control', 'of measure', 'lowest Gate 1 rho (base)', 'mean Gate 1 rho (base)',
             'Gate 2 slope', 'mean m_T - m_D', 'alpha-hat at lowest / highest alpha'],
            [[r['family'].upper(), r['control'], r['base_measure'] or '-',
              f"{f3(float(r['g1_min_rho']))} ({f3(float(r['g1_min_rho_base']))})",
              f"{f3(float(r['g1_mean_rho']))} ({f3(float(r['g1_mean_rho_base']))})", f3(float(r['g2_slope'])),
              f3(float(r['mean_separation'])),
              f"{f3(float(r['alpha_hat_at_alpha_min']))} / {f3(float(r['alpha_hat_at_alpha_max']))}"]
             for r in ctrl]))
    w('')
    # inputs and checks
    w('## Inputs and checks')
    w('')
    inp = gates['inputs']
    for fam in C.FAMILIES:
        ov = [o for o in inp['overlap'] if o['family'] == fam]
        tv = [t['tv_inputs'] for t in inp['tv'] if t['family'] == fam]
        w(f"- {fam.upper()}: every reference is a fresh draw (no reference is a scored unit); exact-content overlap "
          f"between pool and references {min(o['exact_content_overlap'] for o in ov):.1%}-"
          f"{max(o['exact_content_overlap'] for o in ov):.1%}, reference coverage of pool inputs "
          f"{min(o['reference_coverage'] for o in ov):.1%}-{max(o['reference_coverage'] for o in ov):.1%}; input "
          f"total-variation distance target vs distractors at most {max(tv):.3f}; "
          f"{inp['mix_check'][fam]['n_mixes']} mixes rebuilt with counts equal to the upstream mixes.json.")
    gi = inp['graded']
    w(f"- Graded arm: construction attempt {len(gi['construction']) - 1} passed (mean rho by rewired transitions "
      + ', '.join(f"r = {k}: {v:.3f}" for k, v in gi['construction'][-1]['mean_rho_by_r'].items())
      + f"); {gi['n_rejections']} candidate distractors rejected; correlation of alpha_share and alpha_soft over the "
      f"20 mixes {gi['alpha_share_soft_correlation']:.3f}.")
    eq = gates['eq2_check']
    w(f"- Eq. 2 check: the stratified estimator against exact Eq. 2 over full training mixes (measure c, one seed): "
      f"max abs difference {eq['max_abs_diff']:.4f} (tolerance {eq['tol']:g}) -> {'pass' if eq['passed'] else 'FAIL'}.")
    for arm, cs in sorted(gates.get('constant_scores', {}).items()):
        if cs:
            ms = sorted({x[0] for x in cs})
            w(f"- Scores constant over a target's pool ({arm}): {len(cs)} (measure, target, probe) cases, measures "
              f"{', '.join(ms)}; normalized to 0 (Deviation 2), so their alpha-hat is 0 and their Spearman rho undefined.")
    if ver:
        w(f"- verify.py: **{ver['result']}** ({ver['seconds'] / 60:.0f} min).")
        for x in ver['checks']:
            w(f'  - {x}')
    w('')
    if man:
        w(f"Wall time: {man.get('wall_seconds_total', 0) / 3600:.2f} h of the {c['max_wall_seconds'] / 3600:.0f} h cap "
          '(scoring, graded training and the Eq. 2 check; analysis and verification extra, see CKPT).')
        w('')
    if 'Deviations' in notes:
        w('## Deviations from the protocol')
        w('')
        w(notes['Deviations'])
        w('')
    w('## What transfers to the real-text pipeline and what does not')
    w('')
    proto = (EXP / 'PROTOCOL.md').read_text()
    i = proto.index('### What transfers to the real-text pipeline')
    j = proto.index('**TLDR-end:**', i)
    w(proto[i:j].split('\n', 1)[1].strip())
    w('')
    if 'Scope' in notes:
        w(notes['Scope'])
        w('')
    (outdir / 'RESULTS_alpha_recovery.md').write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default=str(EXP), help='folder for the report and figure (default: the experiment)')
    a = ap.parse_args()
    report(C.load_config(), HERE / a.results, Path(a.out))
    print('wrote', Path(a.out) / 'RESULTS_alpha_recovery.md', Path(a.out) / 'alpha_recovery.png')


if __name__ == '__main__':
    sys.exit(main())
