"""Reported comparisons for Experiment 20 (PROTOCOL Section 4, no gate), written to results/comparison.json.

  compare_exp11.py
1. Mean eps(2^16) by level and alpha for the LSTM (this experiment) and the transformer (Experiment 11).
2. Fraction of checkpoint rows below the resolution floor eps_res, by level and alpha.
3. Experiment 11 versus Experiment 20 gate verdicts and key estimates.
4. Post hoc (not pre-registered): effective-data check. For each (seed, target, alpha < 1), eps at T = 2^16 against
   the same seed's alpha = 1 curve read at T = alpha * 2^16 (log-log interpolation; the alpha = 1 curve starts at
   2^9, so alpha * 2^16 >= 3,277 is always inside it). Ratio log2(eps_alpha / eps_alpha1(alpha T)) > 0 means the
   mixed run is worse than its target examples alone predict (interference); < 0 means distractors helped.
"""
import collections
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
c = json.loads((HERE / 'config.json').read_text())
E11 = HERE.parent.parent / '11_dfa_exact_complexity_vs_alignment' / 'expt_v1' / 'results'


def load(p):
    with open(p) as f:
        return [dict(seed=int(r['seed']), target_id=r['target_id'], K=int(r['K']), alpha=float(r['alpha']),
                     T=int(r['T']), eps=float(r['excess_bits'])) for r in csv.DictReader(f)]


lstm, tf = load(HERE / 'results' / 'metrics.csv'), load(E11 / 'metrics.csv')
Tmax, res = c['T_checkpoints'][-1], c['eps_res']
out = dict(mean_eps_Tmax={}, frac_below_eps_res={}, effective_data_check={})
for name, rows in (('lstm', lstm), ('transformer', tf)):
    g = collections.defaultdict(list)
    for r in rows:
        if r['T'] == Tmax:
            g[(r['K'], r['alpha'])].append(r['eps'])
    out['mean_eps_Tmax'][name] = {str(n): {str(a): float(np.mean(g[(n, a)])) for a in c['alphas']} for n in c['n_levels']}
g = collections.defaultdict(list)
for r in lstm:
    g[(r['K'], r['alpha'])].append(r['eps'] < res)
out['frac_below_eps_res'] = {str(n): {str(a): float(np.mean(g[(n, a)])) for a in c['alphas']} for n in c['n_levels']}
out['frac_below_eps_res_all'] = float(np.mean([r['eps'] < res for r in lstm]))

curve = collections.defaultdict(dict)
for r in lstm:
    curve[(r['seed'], r['target_id'], r['alpha'])][r['T']] = (r['eps'], r['K'])
Ts = np.log2(c['T_checkpoints'])
ratios = collections.defaultdict(list)
for (s, t, a), cv in curve.items():
    if a == 1.0:
        continue
    e1 = np.log2([max(curve[(s, t, 1.0)][T][0], res) for T in c['T_checkpoints']])
    ref = np.interp(np.log2(a * Tmax), Ts, e1)
    ratios[(cv[Tmax][1], a)].append(float(np.log2(max(cv[Tmax][0], res)) - ref))
out['effective_data_check'] = {
    'note': 'post hoc; median over seeds and targets of log2(eps(alpha, 2^16) / eps(1, alpha 2^16)), both floored at eps_res',
    'median_log2_ratio': {str(n): {str(a): float(np.median(ratios[(n, a)])) for a in c['alphas'] if a < 1}
                          for n in c['n_levels']}}

A20 = json.loads((HERE / 'results' / 'analysis.json').read_text())
A11 = json.loads((E11 / 'analysis.json').read_text())
out['gates'] = {'exp11_transformer': A11['gates'], 'exp20_lstm': A20['gates']}
keys = [('G1', 'estimate'), ('G2', 'estimate'), ('G3', 'gamma1'), ('G3', 'kappa'), ('G3', 'exp_over_coef_gain'),
        ('G4', 'estimate_bits'), ('G4', 'estimate_states'), ('G4', 'E_mf'), ('G5', 'estimate'), ('G6', 'a1'),
        ('G6', 'estimate'), ('G7', 'estimate')]
out['key_estimates'] = {f'{k}.{f}': dict(exp11=A11[k][f], exp20=A20[k][f]) for k, f in keys}
out['eta'] = {k: dict(exp11=A11['alignment'][k], exp20=A20['alignment'][k])
              for k in ('eta_align_eff', 'eta_k_law', 'eta_refit_both')}
(HERE / 'results' / 'comparison.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('mean_eps_Tmax', 'frac_below_eps_res', 'frac_below_eps_res_all',
                                      'effective_data_check', 'gates', 'eta')}, indent=0))
