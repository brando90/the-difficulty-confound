"""Descriptive comparison (written after the results): Experiment 22 (100 epochs, splits 0-4), 22b FarmShare L40S, 22b mercury1 A4000.
Run from expt_v1/: python ../compare_runs.py"""
import json, numpy as np
def rk(v):
    o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
def rho(x, y): return np.corrcoef(rk(np.asarray(x)), rk(np.asarray(y)))[0, 1]
L = lambda p: json.load(open(p))
fs = L('results_farmshare/analysis.json'); me = L('results/analysis.json'); e22 = L('../../22_shared_structure_real_data/expt_v1/results/analysis.json')
def sel(a, kmax=10): return {(u['k'], u['h']): u['transfer'] for u in a['units'] if u['arm'] == 'vision' and u['map'] == 'real' and u['p'] == 0.0 and u['k'] < kmax}
def shift_ci(A, B, ks, seed=1):
    keys = [k for k in sorted(set(A) & set(B)) if k[0] in ks]; rng = np.random.default_rng(seed); bs = []
    for _ in range(5000):
        s = rng.choice(ks, len(ks)); bs.append(np.mean(np.concatenate([[A[k] - B[k] for k in keys if k[0] == kk] for kk in s])))
    return np.mean([A[k] - B[k] for k in keys]), np.percentile(bs, [2.5, 97.5])
F, M, E22 = sel(fs), sel(me), sel(e22, 5)
for name, A, B, ks in (('FarmShare vs mercury (200 units)', F, M, list(range(10))), ('FarmShare vs Exp22 (splits 0-4, 100 units)', F, E22, list(range(5))), ('mercury vs Exp22 (splits 0-4, 100 units)', M, E22, list(range(5)))):
    keys = [k for k in sorted(set(A) & set(B)) if k[0] in ks]
    print(name, 'n=%d spearman %.3f' % (len(keys), rho([A[k] for k in keys], [B[k] for k in keys])), 'mean shift %.4f CI [%.4f, %.4f]' % ((lambda m, c: (m, c[0], c[1]))(*shift_ci(A, B, ks))))
