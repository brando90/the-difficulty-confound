"""Markdown tables from results/analysis.json (pasted verbatim into RESULTS.md).

  python report.py --results results"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def f(x, d=3):
    return '—' if x is None else f'{x:.{d}f}'


def mc(m):
    return f"{f(m['mean'])} [{f(m['ci95'][0])}, {f(m['ci95'][1])}] (n={m['n']})" if m and m.get('n') else '—'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    r = json.load(open(os.path.join(a.results, 'analysis.json')))
    print(f"Runs: {r['n_runs']} ({', '.join(f'{k} {v}' for k, v in r['runs_by_phase'].items())}); "
          f"structure measures present: {r['structure_present']}\n")
    print('| phase | mean in-dist test acc | min | share best epoch = last | median train s |\n|---|---|---|---|---|')
    for ph, q in r['qc'].items():
        print(f"| {ph} | {f(q['indist_test_acc_mean'])} | {f(q['indist_test_acc_min'])} | "
              f"{f(q['best_epoch_at_last_eval_share'], 2)} | {q['train_sec_median']:.0f} |")
    print('\n## Pre-registered predictions (p = 0)\n')
    print('| arm | P1 shuffled-map transfer | real-map transfer | P2 Spearman rho (margin) | R² structure | R² complexity | R² diff [CI] | verdicts |')
    print('|---|---|---|---|---|---|---|---|')
    for arm, x in r['by_arm'].items():
        p2 = x['P2_P3_real_p0']
        rho = f"{f(p2.get('rho'))} [{f((p2.get('rho_ci95') or [None])[0])}, {f((p2.get('rho_ci95') or [None, None])[1])}] (n={p2['n']})"
        dci = p2.get('r2_diff_ci95') or [None, None]
        v = ', '.join(f'{k} {w}' for k, w in sorted(x['verdicts'].items())) or '—'
        print(f"| {arm} | {mc(x['P1_shuffled_p0_transfer'])} | {mc(x['real_p0_transfer'])} | {rho} | "
              f"{f(p2.get('r2_structure'))} | {f(p2.get('r2_complexity'))} | {f(p2.get('r2_diff'))} [{f(dci[0])}, {f(dci[1])}] | {v} |")
    print('\n## Secondary structure measures (real map, p = 0): Spearman rho with transfer\n')
    print('| arm | measure | rho [95% CI] | n |\n|---|---|---|---|')
    for arm, x in r['by_arm'].items():
        for m, c in x['secondary_real_p0'].items():
            ci = c.get('rho_ci95') or [None, None]
            print(f"| {arm} | {m} | {f(c.get('rho'))} [{f(ci[0])}, {f(ci[1])}] | {c['n']} |")
        c = x['pooled_real_shuffled_p0']
        ci = c.get('rho_ci95') or [None, None]
        print(f"| {arm} | margin, real + shuffled units pooled | {f(c.get('rho'))} [{f(ci[0])}, {f(ci[1])}] | {c['n']} |")
    print('\n## The p dial (share of held-out-class training items mixed in, fixed budget)\n')
    print('| arm | map / p | normalized transfer [95% CI] | held-out acc | in-dist acc |\n|---|---|---|---|---|')
    for arm, x in r['by_arm'].items():
        for k, d in x['p_dial'].items():
            print(f"| {arm} | {k} | {mc(d['transfer'])} | {f(d['heldout_acc'])} | {f(d['indist_acc'])} |")


if __name__ == '__main__':
    main()
