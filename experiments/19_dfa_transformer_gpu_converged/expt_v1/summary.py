"""Experiment 19 headline: convergence census, E next to Experiments 11 and 13-small, and the pre-registered verdict
on whether under-training explains E < 1 (PROTOCOL Sections 2 and 5). Reads metrics.csv, run_state.json,
analysis.json (and gpu_usage.json if present) in --results; writes summary.json there.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def converged(e_top, e_half, x):
    return e_top <= x['converged_learned_bits'] or (e_half - e_top) <= x['converged_last_doubling_bits']


def build(res):
    c = json.loads((HERE / 'config.json').read_text())
    x = c['exp19']
    st = json.loads((res / 'run_state.json').read_text())
    A = json.loads((res / 'analysis.json').read_text())
    T_top = int(st.get('T_top', st['T_done'][-1]))
    seeds = [s for s in st['seeds_planned'] if s not in st['seeds_dropped']]
    eps = {}
    K = {}
    with open(res / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            if int(r['seed']) in seeds:
                eps[(int(r['seed']), r['target_id'], float(r['alpha']), int(r['T']))] = float(r['excess_bits'])
                K[r['target_id']] = int(r['K'])
    cells = sorted({k[:3] for k in eps})
    census, runs = {}, []
    for s, tid, a in cells:
        et, eh = eps[(s, tid, a, T_top)], eps[(s, tid, a, T_top // 2)]
        cv = converged(et, eh, x)
        runs.append(dict(seed=s, target_id=tid, alpha=a, eps_top=et, eps_half=eh, converged=cv))
        key = f'n={K[tid]} alpha={a}'
        d = census.setdefault(key, dict(runs=0, converged=0, learned_le_0p1=0, stalled_gt_0p1=0, mean_eps_top=0.0))
        d['runs'] += 1
        d['converged'] += int(cv)
        d['learned_le_0p1'] += int(et <= 0.1)
        d['stalled_gt_0p1'] += int(cv and et > 0.1)
        d['mean_eps_top'] += et
    for d in census.values():
        d['mean_eps_top'] = round(d['mean_eps_top'] / d['runs'], 4)
    a1 = [r for r in runs if r['alpha'] == 1.0]
    f_a1 = float(np.mean([r['converged'] for r in a1]))
    f_all = float(np.mean([r['converged'] for r in runs]))
    grid_conv = f_a1 >= x['grid_converged_alpha1_frac'] and f_all >= x['grid_converged_all_frac']
    lv = A['E']['levels']
    eps_p = A['realized']['eps_star_primary']
    prim = lv[str(eps_p)] if str(eps_p) in lv else lv[[k for k in lv if float(k) == eps_p][0]]
    E, lo, hi = prim['E_bits'], prim['ci_bits'][0], prim['ci_bits'][1]
    if not grid_conv:
        verdict = 'NOT_CONVERGED: the under-training caveat is not removed'
    elif lo > 1:
        verdict = 'YES: once trained, alignment dominates; Experiment 11\'s E < 1 was an under-training artefact'
    elif hi < 1:
        verdict = 'NO: after convergence complexity still costs more data per log-unit than misalignment'
    else:
        verdict = 'UNDECIDED: converged, but the E interval contains 1'
    skipped = [f'seed {s}: all 64 cells after T = {2 ** x["subset_decision_log2T"]}' for s in st['seeds_dropped']]
    out = dict(T_top=T_top, stop_reason=st['stop_reason'], seeds_used=seeds, skipped_cells=skipped,
               gpu=st.get('gpu'), train_wall_seconds=st.get('wall_seconds'),
               converged_fraction_alpha1=f_a1, converged_fraction_all=f_all, grid_converged=grid_conv,
               census=census, eps_star_primary=eps_p, primary_rule_met=A['realized'].get('primary_rule_met'),
               E_bits=E, E_bits_ci=[lo, hi], E_states=prim.get('E_states'), E_by_level=lv,
               E_mf=A['E']['E_mf'], E_mf_eps_star=A['E']['E_mf_eps_star'],
               compare=x['compare'], verdict=verdict,
               gates=A['gates'])
    gu = res / 'gpu_usage.json'
    if gu.exists():
        out['gpu_usage'] = json.loads(gu.read_text())
    return out, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    res = Path(ap.parse_args().results)
    out, runs = build(res)
    (res / 'summary.json').write_text(json.dumps(out, indent=1))
    (res / 'convergence_runs.json').write_text(json.dumps(runs))
    print(json.dumps({k: out[k] for k in ['T_top', 'grid_converged', 'converged_fraction_alpha1', 'E_bits',
                                          'E_bits_ci', 'eps_star_primary', 'verdict']}, indent=1))


if __name__ == '__main__':
    main()
