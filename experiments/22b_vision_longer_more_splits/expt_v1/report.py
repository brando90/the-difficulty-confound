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
    qb = r.get('qc_budget')
    if qb:
        print(f"\nBudget check: share of real-map vision runs whose best epoch is the last evaluation = "
              f"{f(qb['vis_real_best_epoch_last_share'], 3)} (limit {qb['max_allowed']}): "
              f"{'long enough' if qb['budget_long_enough'] else 'NOT long enough (disclosed limit)'}")
    print('\n## Pre-registered predictions (p = 0; 95% intervals resample whole splits)\n')
    print('| arm | P1 shuffled-map transfer | real-map transfer | P2 Spearman rho (margin) | R² structure | R² complexity | R² diff [CI] | verdicts |')
    print('|---|---|---|---|---|---|---|---|')
    for arm, x in r['by_arm'].items():
        p2 = x['P2_P3_real_p0']
        rho = (f"{f(p2.get('rho'))} split-level [{f((p2.get('rho_ci95') or [None])[0])}, {f((p2.get('rho_ci95') or [None, None])[1])}], "
               f"unit-level [{f((p2.get('rho_ci95_unit') or [None])[0])}, {f((p2.get('rho_ci95_unit') or [None, None])[1])}] (n={p2['n']}, {p2.get('n_splits')} splits)")
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
    print('\n## Descriptive (no threshold)\n')
    for arm, x in r['by_arm'].items():
        d = x.get('descriptive', {})
        for k in ('rho_splits_0_4', 'rho_splits_5_9'):
            if k in d:
                print(f"- {arm} {k}: rho {f(d[k]['rho'])} (n={d[k]['n']})")
        if 'class_level_rho' in d:
            c = d['class_level_rho']
            print(f"- {arm} class-level rho (each fine class's two splits averaged): {f(c['rho'])} [{f(c['ci95'][0])}, {f(c['ci95'][1])}] (n={c['n']})")
        for m in ('real', 'shuffled'):
            if f'seed_reliability_{m}' in d:
                c = d[f'seed_reliability_{m}']
                print(f"- {arm} {m} map seed-to-seed Spearman of unit transfers: {f(c['rho_between_seeds'])} (mean of 2 seeds: {f(c['spearman_brown_mean_of_2'])}, n={c['n']})")
            if f'per_split_mean_transfer_{m}' in d:
                print(f"- {arm} {m} per-split mean transfer: " + ', '.join(f"{k}: {f(v)}" for k, v in d[f'per_split_mean_transfer_{m}'].items()))
    print('\n## The p dial (share of held-out-class training items mixed in, fixed budget)\n')
    print('| arm | map / p | normalized transfer [95% CI] | held-out acc | in-dist acc |\n|---|---|---|---|---|')
    for arm, x in r['by_arm'].items():
        for k, d in x['p_dial'].items():
            print(f"| {arm} | {k} | {mc(d['transfer'])} | {f(d['heldout_acc'])} | {f(d['indist_acc'])} |")


if __name__ == '__main__':
    main()
