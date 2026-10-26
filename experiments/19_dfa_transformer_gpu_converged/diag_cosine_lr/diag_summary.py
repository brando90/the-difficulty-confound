"""The pre-registered read-out of the AT-494 diagnostic (PREREG.md Section 3): summary.json from metrics.csv.

  python diag_summary.py --results results
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sign_test_p(k, n):
    """Two-sided exact binomial p for k of n successes at p = 1/2."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    a = ap.parse_args()
    res = Path(a.results)
    d = json.loads((HERE / 'diag_config.json').read_text())
    st = json.loads((res / 'run_state.json').read_text())
    rows = list(csv.DictReader(open(res / 'metrics.csv')))
    T_top = st['T_done'][-1]
    eps = d['eps_bits']
    bits = defaultdict(dict)   # (arm, n) -> {(seed, target): bits at T_top}
    curve = defaultdict(list)  # (arm, n, T) -> [bits]
    for r in rows:
        n = int(r['target_id'].split('_')[1][1:])
        curve[(r['arm'], n, int(r['T']))].append(float(r['excess_bits']))
        if int(r['T']) == T_top:
            bits[(r['arm'], n)][(r['seed'], r['target_id'])] = float(r['excess_bits'])
    levels = sorted({n for _, n in bits})
    arms = sorted({arm for arm, _ in bits})
    out = dict(T_top=T_top, reached_T_max=st['stop_reason'] == 'reached T_max', arms=arms,
               arms_dropped=st['arms_dropped'], eps_bits=eps, levels={})
    for n in levels:
        L = {}
        for arm in arms:
            v = np.array(sorted(bits[(arm, n)].values()))
            L[arm] = dict(n_cells=len(v), median=float(np.median(v)), mean=float(v.mean()),
                          below_eps=int((v < eps).sum()), min=float(v.min()), max=float(v.max()),
                          curve_median={str(T): float(np.median(curve[(arm, n, T)])) for T in st['T_done']
                                        if (arm, n, T) in curve})
        if 'constant' in arms and 'cosine' in arms:
            keys = sorted(bits[('cosine', n)])
            diff = np.array([bits[('cosine', n)][k] - bits[('constant', n)][k] for k in keys])
            k = int((diff < 0).sum())
            L['paired_cosine_minus_constant'] = dict(median=float(np.median(diff)), cosine_lower=k, n=len(diff),
                                                     sign_test_p=sign_test_p(k, len(diff)))
        out['levels'][str(n)] = L
    lv = [out['levels'][str(n)]['cosine'] for n in d['decision_levels']]
    passed = out['reached_T_max'] and all(x['median'] < eps for x in lv)
    out['decision'] = dict(
        rule=f"rerun the grid with cosine decay iff the cosine arm reached 2^{d['T_log2_max']} and its median excess "
             f"bits is below {eps} at every level in {d['decision_levels']}; otherwise retire the transformer arm",
        cosine_medians={str(n): x['median'] for n, x in zip(d['decision_levels'], lv)},
        verdict='RERUN_GRID_WITH_COSINE' if passed else 'RETIRE_TRANSFORMER_ARM')
    (res / 'summary.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out['decision'], indent=1))


if __name__ == '__main__':
    main()
