"""Render RESULTS_vision.md (and the figure) from results/analysis.json only. Every number in the markdown is read
from analysis.json here, so verify.py can re-render and compare.

Usage (from expt_v1/): report.py [--analysis results/analysis.json] [--md ../RESULTS_vision.md] [--no-figures]
Text written by hand (takeaway, notes) lives in results/report_notes.json and is pasted in verbatim."""
import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, HERE)
import tasks as TK  # noqa: E402
NAMES = {'mnist': 'MNIST', 'fashion': 'Fashion-MNIST', 'kmnist': 'KMNIST', 'cifar10': 'CIFAR-10'}
COLORS = {'mnist': '#2a78d6', 'fashion': '#eb6834', 'kmnist': '#1baf7a', 'cifar10': '#eda100'}
LEVEL_RAMP = ['#a9c9f0', '#6aa3e6', '#2a78d6', '#164a8a']


def f3(x):
    if x is None:
        return 'n/a'
    if isinstance(x, float) and math.isinf(x):
        return 'inf'
    return f'{x:.3g}' if abs(x) < 1000 else f'{x:,.0f}'


def iv(c):
    return '[n/a]' if not c or c[0] is None else f'[{f3(c[0])}, {f3(c[1])}]'


def figure(A, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a in ax:
        a.set_facecolor('#fcfcfb')
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.grid(axis='y', color='#e4e3df', lw=0.8)
        a.set_axisbelow(True)
    # (a) E per dataset
    dss = [d for d in NAMES if d in A['main']]
    for i, d in enumerate(dss):
        g = A['main'][d]['G4']
        c = g['ci95_bits']
        ax[0].plot([i, i], c, color=COLORS[d], lw=2, solid_capstyle='round')
        ax[0].plot(i, g['E_bits'], 'o', ms=9, color=COLORS[d], mec='#fcfcfb', mew=2)
        ax[0].annotate(f"{g['E_bits']:.2f}", (i, g['E_bits']), xytext=(10, 0), textcoords='offset points',
                       va='center', color='#0b0b0b', fontsize=10)
    ax[0].axhline(1, color='#52514e', lw=1)
    ax[0].axhline(0.80, color='#52514e', lw=1, ls='--')
    ax[0].text(len(dss) - 0.5, 1, ' alignment dominates above', va='bottom', ha='right', color='#52514e', fontsize=9)
    ax[0].text(len(dss) - 0.5, 0.80, ' Exp 13-small: 0.80', va='top', ha='right', color='#52514e', fontsize=9)
    ax[0].set_xticks(range(len(dss)), [NAMES[d] for d in dss])
    ax[0].set_xlim(-0.5, len(dss) - 0.5)
    ax[0].set_ylabel('elasticity ratio E = e_alpha / e_K')
    ax[0].set_title('Does alignment dominate complexity? (95% seed CI)', fontsize=11, loc='left')
    # (b) hard-to-easy ratios
    hd = [d for d in NAMES if d in A['H'] and A['H'][d].get('joint', {}).get('S', {}).get('rho_HE', {}).get('ratio')]
    for i, d in enumerate(hd):
        S = A['H'][d]['joint']['S']
        for off, key, mk, lab in ((-0.12, 'rho_HE', 'o', 'hard-only / easy-only on easy targets'),
                                  (0.12, 'rho_EH', 's', 'easy-only / hard-only on hard targets')):
            r = S[key]
            ax[1].plot([i + off] * 2, r['ci95'], color=COLORS[d], lw=2)
            ax[1].plot(i + off, r['ratio'], mk, ms=9, color=COLORS[d], mec='#fcfcfb', mew=2,
                       label=lab if i == 0 else None)
            ax[1].annotate(f"{r['ratio']:.2f}", (i + off, r['ratio']), xytext=(8, 0), textcoords='offset points',
                           va='center', fontsize=9, color='#0b0b0b')
    ax[1].set_yscale('log', base=2)
    ax[1].axhline(1, color='#52514e', lw=1)
    ax[1].axhline(1.1, color='#52514e', lw=1, ls=':')
    ax[1].set_xticks(range(len(hd)), [NAMES[d] for d in hd])
    ax[1].set_xlim(-0.5, max(len(hd), 1) - 0.5)
    ax[1].set_ylabel('error ratio over the learning curve (log scale)')
    ax[1].set_title('Hard-to-easy: below 1.1 = carries (circle), joint alpha = 0.05', fontsize=11, loc='left')
    if hd:
        ax[1].legend(frameon=False, fontsize=9, loc='best')
    from matplotlib.ticker import FuncFormatter
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax[1].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    # (c) MNIST learning curves at alpha = 1 per level
    if 'mnist' in A['main']:
        cells = A['main']['mnist']['cells']
        lv = [ts for _, ts in TK.MAIN_LEVELS]
        labs = ['L1, 1 bit', 'L2, 2 bits', 'L3, 3.32 bits', 'L4, 6.64 bits']
        for j, ts in enumerate(lv):
            pts = {}
            for c in cells:
                if c['task_id'] in ts and c['alpha'] == 1.0:
                    pts.setdefault(c['T'], []).append(max(c['eps_mean'], 1e-3))
            T = sorted(pts)
            y = [math.exp(sum(math.log(v) for v in pts[t]) / len(pts[t])) for t in T]
            ax[2].plot(T, y, '-o', color=LEVEL_RAMP[j], lw=2, ms=4, label=labs[j])
        ax[2].axhline(A['main']['mnist']['eps_star'], color='#52514e', lw=1, ls='--')
        ax[2].set_xscale('log', base=2)
        ax[2].set_yscale('log')
        ax[2].set_xlabel('training pairs T')
        ax[2].set_ylabel('error / chance error')
        ax[2].set_title('MNIST, fully aligned: complexity costs data', fontsize=11, loc='left')
        ax[2].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='#fcfcfb')


def md(A, notes):
    L = []
    w = L.append
    w('# Results: vision replication of complexity versus alignment and hard-to-easy transfer (Experiment 18)')
    w('')
    w(notes.get('status', 'Status: results.'))
    w('')
    w('## Takeaway')
    w('')
    w(notes.get('takeaway', '(takeaway pending)'))
    w('')
    w('![Experiment 18 results](vision_results.png)')
    w('')
    w('## Elasticity grid (main arm, rotation mismatch)')
    w('')
    w('| Dataset | runs | eps* | G1 M_K (L4/L1, alpha=1) | G1 | G2 RMSE cut | G2 | E (K in bits) | E (K as classes) | E model-free | G4 | vs Exp 13-small |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for d in NAMES:
        if d not in A['main']:
            continue
        m = A['main'][d]
        g1, g2, g4 = m['G1'], m['G2'], m['G4']
        w(f"| {NAMES[d]} | {m['n_runs']}/{m['n_runs_expected']} | {f3(m['eps_star'])} | {f3(g1['M_K'])} "
          f"{iv(g1.get('ci95_seed'))} | {g1.get('verdict', 'n/a')} | {f3(100 * g2['reduction'])}% {iv([100 * x for x in g2['ci95']] if g2['ci95'][0] is not None else None)} "
          f"| {g2['verdict']} | {f3(g4['E_bits'])} {iv(g4['ci95_bits'])} | {f3(g4['E_classes'])} | {f3(g4['E_mf'])} "
          f"(at eps {f3(g4['E_mf_eps'])}) | {g4['verdict']} | {g4['replication']} |")
    w('')
    w('Elasticities per seed (e_alpha = eta, e_K = d ln T*/d ln K at K_ref), model-free T* censoring, rows dropped by the learning-regime filter (eps > 0.8), and G3 (reported, not gated):')
    w('')
    for d in NAMES:
        if d not in A['main']:
            continue
        m = A['main'][d]
        g4 = m['G4']
        w(f"- **{NAMES[d]}:** e_alpha {', '.join(f3(x) for x in g4['e_alpha'])}; e_K {', '.join(f3(x) for x in g4['e_K_bits'])}; "
          f"M_alpha (law, 20^eta) {f3(g4['M_alpha_law'])}; E at eps 0.05/0.1/0.2/0.4: "
          f"{', '.join(f3(v) for v in g4['E_curve'].values())}; T* cells ok/left/right {m['tstar_censored']['ok']}/"
          f"{m['tstar_censored']['left']}/{m['tstar_censored']['right']}; fit rows dropped {m['fit_rows_dropped_learning_regime']}; "
          f"G3 rho(exp/coef) {f3(m['G3_reported']['rho_exp_over_coef'])}; independent fitter agrees: {m['independent_fit']['agree']}.")
    w('')
    if A.get('mismatch_sweep'):
        w('## Mismatch-type sweep (MNIST, seeds 18101 and 18202, descriptive)')
        w('')
        w('| Mismatch | runs | eta per seed | E | M_alpha model-free by level (L1, L2, L3, L4) | eps at T_max, alpha 0.05 (L1..L4) |')
        w('|---|---|---|---|---|---|')
        for ty, v in A['mismatch_sweep'].items():
            ma = ', '.join(f3(v['M_alpha_mf_by_level'].get(l, {}).get('M_alpha_mf')) for l in ('L1', 'L2', 'L3', 'L4'))
            ee = ', '.join(f3(v['eps_at_Tmax_alpha005'].get(l)) for l in ('L1', 'L2', 'L3', 'L4'))
            w(f"| {ty} | {v['n_runs']} | {', '.join(f3(x) for x in v['eta_per_seed'])} | {f3(v['E'])} | {ma} | {ee} |")
        w('')
    w('## Hard-to-easy arm (matched compute)')
    w('')
    w('rho_HE = error of hard-only over error of easy-only on the two held-out easy targets; rho_EH = easy-only over hard-only on the two held-out hard targets. S = over the whole learning curve (primary); last = at the largest checkpoint.')
    w('')
    w('| Dataset | readout | runs | rho_HE (S) | rho_HE (last) | GH1 | rho_EH (S) | rho_EH (last) | GH2 | mixed/easy on easy (S) | mixed/hard on hard (S) |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for d in NAMES:
        h = A['H'].get(d)
        if not h:
            continue
        for mode in ('joint', 'probe'):
            M = h.get(mode)
            if not M or M['S']['rho_HE'].get('ratio') is None:
                continue
            S, La = M['S'], M['last']
            w(f"| {NAMES[d]} | {'joint alpha=0.05' if mode == 'joint' else 'linear probe'} | {h['n_runs']}/{h['n_runs_expected']} | "
              f"{f3(S['rho_HE']['ratio'])} {iv(S['rho_HE']['ci95'])} | {f3(La['rho_HE']['ratio'])} | {M.get('GH1')} | "
              f"{f3(S['rho_EH']['ratio'])} {iv(S['rho_EH']['ci95'])} | {f3(La['rho_EH']['ratio'])} | {M.get('GH2')} | "
              f"{f3(S['mixed_vs_easy_on_easy']['ratio'])} | {f3(S['mixed_vs_hard_on_hard']['ratio'])} |")
    w('')
    w('Mean error/chance over the learning curve, per pool and held-out target (joint alpha = 0.05):')
    w('')
    for d in NAMES:
        h = A['H'].get(d)
        if not h or not h.get('joint') or not h['joint']['S']['eps_mean']:
            continue
        e = h['joint']['S']['eps_mean']
        w(f"- **{NAMES[d]}:** " + '; '.join(f"{k.replace('|', ' on ')} {f3(v)}" for k, v in e.items()))
    w('')
    for k in ('compute', 'deviations', 'limits', 'files'):
        if notes.get(k):
            w(f"## {k.capitalize()}")
            w('')
            w(notes[k])
            w('')
    w('**TLDR-end:** ' + notes.get('tldr', ''))
    w('')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', default=os.path.join(HERE, 'results', 'analysis.json'))
    ap.add_argument('--md', default=os.path.join(HERE, '..', 'RESULTS_vision.md'))
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args()
    A = json.load(open(a.analysis))
    np_ = os.path.join(os.path.dirname(a.analysis), 'report_notes.json')
    notes = json.load(open(np_)) if os.path.exists(np_) else {}
    open(a.md, 'w').write(md(A, notes))
    if not a.no_figures:
        figure(A, os.path.join(HERE, '..', 'vision_results.png'))
    print('wrote', a.md)


if __name__ == '__main__':
    main()
