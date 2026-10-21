"""Experiment 23 report: writes RESULTS.md from the analysis outputs (no number is computed here that the analysis or
the aggregated metrics do not already hold, apart from cell means and their seed-bootstrap intervals).

Usage: report23.py --out OUT_ROOT [--results RESULTS.md]     OUT_ROOT holds e13/ and e17b/
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
SRC = {k: (HERE / v).resolve() for k, v in CFG['sources'].items()}


def jl(p):
    return json.loads(Path(p).read_text())


def f3(x):
    return f'{x:.3f}'


def ci_s(ci):
    return f'[{ci[0]:.3f}, {ci[1]:.3f}]'


def seed_ci(per_seed, draws=20000, seed=23):
    a = np.asarray(per_seed, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    m = a[idx].mean(1)
    return float(a.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def read_metrics(path, key_cols):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['slice'] == 'all':
                out.setdefault(tuple(r[k] for k in key_cols), []).append(r)
    return out


def decisions_table(root):
    lines = ['| stage | T | cells converged | decision |', '|---|---|---|---|']
    for p in sorted((root / 'decisions').glob('stage_*.json')):
        d = jl(p)
        if 'cells' in d:
            n = sum(v['converged'] for v in d['cells'].values())
            lines.append(f"| {d['stage']} | 2^{int(np.log2(d['T']))} | {n}/{len(d['cells'])} | {d['decision']} "
                         f"{d.get('reason', '')} |")
        else:
            lines.append(f"| {d['stage']} | 2^{int(np.log2(d['T']))} | - | {d['decision']} |")
    return lines


def worst_cells(root, k=4):
    p = sorted((root / 'decisions').glob('stage_*.json'))
    d = jl(p[-1]) if p else {}
    cells = sorted(d.get('cells', {}).items(), key=lambda kv: kv[1]['improvement'] - kv[1]['threshold'], reverse=True)
    return [f"{c}: {v['before']:.3f} -> {v['after']:.3f} bits (threshold {v['threshold']:.3f})" for c, v in cells[:k]]


def e13_section(root):
    L = []
    stop = jl(root / 'STOP')
    g, a = jl(root / 'gates.json'), jl(root / 'analysis.json')
    g110 = jl(root / 'eps_1.10' / 'gates.json')
    rule = jl(root / 'eps_star_rule.json')
    base = jl(SRC['exp13'] / 'results' / 'small' / 'gates.json')
    L += ['## Experiment 13 (e13): complexity versus alignment, 256 runs', '',
          f"Stopped at T_stop = 2^{int(np.log2(stop['T_stop']))} tokens ({stop['reason']}). "
          f"ε* = {rule['epsilon_star']} bits/token by the Section 5.1 rule (n = 32, α = 0.4, T = "
          f"2^{int(np.log2(rule['T']))}: median ε_learn {rule['median']:.3f}; share reaching each level "
          f"{', '.join(f'{k}: {v:.2f}' for k, v in rule['share_reached'].items())}).", '']
    L += ['Stopping decisions:', ''] + decisions_table(root) + ['']
    L += ['Least-converged cells at the last decision: ' + '; '.join(worst_cells(root)), '']
    d, d0 = g['detail'], base['detail']
    L += ['| gate | Exp 23 at ε* (gated) | Exp 23 at ε* = 1.10 | Exp 13-small (4 seeds, 2^20, ε* = 1.10) |',
          '|---|---|---|---|']

    def line(name, fmt):
        return (f"| {name} | {g['gates'][name]['verdict']}: {fmt(d[name])} | {g110['gates'][name]['verdict']}: "
                f"{fmt(g110['detail'][name])} | {base['gates'][name]['verdict']}: {fmt(d0[name])} |")
    L.append(line('G1', lambda x: f"M_K {x['M_K']:.2f} {ci_s(x['ci95_seed'])}, two-stage {ci_s(x['ci95_two_stage'])}"))
    L.append(line('G2', lambda x: f"{x['K_law']} cuts held-out RMSE {100 * x['reduction']:.1f}% "
                                  f"[{100 * x['ci95'][0]:.1f}, {100 * x['ci95'][1]:.1f}]"))
    L.append(line('G3a', lambda x: f"κ {x['kappa']['estimate']:.3f} {ci_s(x['kappa']['ci95'])}, γ₁ "
                                   f"{x['gamma1']['estimate']:.3f} {ci_s(x['gamma1']['ci95'])}"))
    L.append(line('G3b', lambda x: f"exp vs coef {100 * x['exp_vs_coef_reduction']:.1f}%"))
    L.append(line('G4', lambda x: f"E {x['E_bits']:.2f} {ci_s(x['ci95_bits'])}"))
    L.append(line('G5', lambda x: f"r_α early {x['r_alpha_early']['r_alpha']:.2f}, late "
                                  f"{x['r_alpha_late']['r_alpha']:.2f}"))
    L += ['', f"C3 (coefficient or exponent): {g.get('C3')}. Independent fitter: "
              f"{'AGREE' if a['independent_fit']['agree'] else 'DISAGREE'}.", '']
    # cell means at 2^20 and at T_stop, with the cooldown branch at T_stop
    m = read_metrics(root / 'metrics.csv', ('K', 'alpha', 'T'))
    cd = read_metrics(root / 'cooldown.csv', ('K', 'alpha', 'T'))
    T20, Ts = 2 ** 20, stop['T_stop']
    L += [f'Mean excess ε over 8 seeds × 2 targets (bits/token, all positions; 95% seed bootstrap), at 2^20 and at '
          f'T_stop, and after the cooldown branch at T_stop (reported only):', '',
          '| n | α | ε at 2^20 | ε at T_stop | ε_learn at T_stop | after cooldown |', '|---|---|---|---|---|---|']
    for K in sorted({int(k[0]) for k in m}):
        for al in CFG['e13']['alphas']:
            def cell(src, T, col='excess_bits'):
                rows = src.get((str(K), str(al), str(T)), [])
                by = {}
                for r in rows:
                    by.setdefault(r['seed'], []).append(float(r[col]))
                if not by:
                    return '-'
                mean, ci = seed_ci([np.mean(v) for v in by.values()])
                return f'{mean:.3f} {ci_s(ci)}'
            L.append(f'| {K} | {al} | {cell(m, T20)} | {cell(m, Ts)} | {cell(m, Ts, "learn_bits")} | {cell(cd, Ts)} |')
    L.append('')
    return L


def e17b_section(root):
    L = []
    stop = jl(root / 'STOP')
    r = jl(root / 'analysis_e17b.json')
    base = jl(SRC['exp17b'] / 'results' / 'analysis_v2.json')
    L += ['## Experiment 17b (e17b): coverage-matched easy pools, 160 runs', '',
          f"Stopped at T_stop = 2^{int(np.log2(stop['T_stop']))} tokens ({stop['reason']}). 16 (seed, target) pairs "
          f"per contrast, verdict needs ≥ {r['k_min']}/16.", '', 'Stopping decisions:', ''] + decisions_table(root)
    L += ['', 'Least-converged cells at the last decision: ' + '; '.join(worst_cells(root)), '']
    L += ['| pool | Q (hard-only still better zero-shot?) | H1′ | H2′ | gap closed (α = 0) | Exp 17b (3 seeds, 2^20) |',
          '|---|---|---|---|---|---|']
    for P in CFG['e17b']['new_pools']:
        gq, b = r['gates'][P], base['gates'][P]
        gc, gb = r['gap_closed'][P], base['gap_closed'][P]
        L.append(f"| {P}{' (primary)' if gq['primary'] else ''} | {gq['Q_hard_still_beats_easy_zero_shot']} | "
                 f"{gq['H1_prime']} | {gq['H2_prime']} | {gc['estimate']:.2f} {ci_s(gc['ci95'])} | Q "
                 f"{b['Q_hard_still_beats_easy_zero_shot']}, H1′ {b['H1_prime']}, H2′ {b['H2_prime']}, gap closed "
                 f"{gb['estimate']:.2f} |")
    L += ['', 'Contrasts at T_stop (positive = the coverage-matched pool wins; bits/token):', '',
          '| pool | targets | α | Δ mean | 95% CI | pairs > 0 | verdict | Exp 17b Δ (verdict) |',
          '|---|---|---|---|---|---|---|---|']
    for k, v in sorted(r['contrasts'].items()):
        P, which, al = k.split('|')
        b = base['contrasts'].get(k, {})
        L.append(f"| {P} | {which.split('_')[0]} | {al} | {v['mean']:.3f} | {ci_s(v['ci95'])} | "
                 f"{v['n_positive']}/{v['n']} | {v['verdict']} | {b.get('mean', float('nan')):.3f} "
                 f"({b.get('verdict', '-')}) |")
    pe = r['pool_excess_at_Tmax']
    L += ['', f"Training-pool excess at T_stop: mean {pe['mean']:.3f}, max {pe['max']:.3f} bits/token over "
              f"{pe['n']} (run, pool concept) pairs.", '']
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--results', default=str(HERE / 'RESULTS.md'))
    a = ap.parse_args()
    root = Path(a.out).expanduser()
    L = ['# Results: Experiment 23, Experiment 13-small and Experiment 17b to convergence, 8 seeds', '',
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by report23.py from {root}. "
         'Protocol: PROTOCOL.md (frozen before any job). Verdicts are the frozen gates; everything else is reported.',
         '']
    L += e13_section(root / 'e13')
    L += e17b_section(root / 'e17b')
    L += ['## Compute', '']
    for exp in ('e13', 'e17b'):
        t = jl(root / exp / 'manifest.json')['timing']
        L.append(f"- {exp}: {t['runs']} runs, {t['stages']} stages, {t['wall_hours']:.0f} core-hours of run wall time, "
                 f"median {t['median_tokens_per_second']:.0f} training tokens/s per run, hosts "
                 f"{', '.join(t['hosts'][:6])}{' ...' if len(t['hosts']) > 6 else ''}; non-finite runs: "
                 f"{len(t['nonfinite'])}.")
    L += ['', '## Verification', '']
    vp = root / 'verify_log.json'
    if vp.exists():
        v = jl(vp)
        L += [f"- {k}: {'PASS' if x['ok'] else 'FAIL'} ({x.get('note', '')})" for k, x in v['checks'].items()]
        L.append(f"- overall: {'PASS' if v['ok'] else 'FAIL'}")
    else:
        L.append('- verify23.py has not run.')
    L += ['', '## Deviations from PROTOCOL.md', '', '- None recorded by the pipeline; add any here.', '',
          '## Deviations from the source experiments (declared in PROTOCOL.md)', '',
          '- Experiment 17b: the hard and easy pools are re-run to T_stop instead of reused; the near pool and the α = 1 '
          'target-only runs are not run, so t_equiv is not reported.',
          '- Experiment 13: ε* comes from the Section 5.1 rule on the grid\'s own fit rows instead of the pilot rule.', '']
    Path(a.results).write_text('\n'.join(L))
    print(f'wrote {a.results}')


if __name__ == '__main__':
    main()
