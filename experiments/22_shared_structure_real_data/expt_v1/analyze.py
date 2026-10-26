"""Experiment 22 analysis (PROTOCOL Section 4): unit table, P1-P3 verdicts, the p dial. Deterministic.

  python analyze.py --results results              # writes results/analysis.json, prints a short summary"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'config.json')))
A = CFG['analysis']


def load_runs(results):
    runs = []
    for ph in CFG['grid']:
        for f in sorted(glob.glob(os.path.join(results, ph, '*.json'))):
            runs.append(json.load(open(f)))
    return runs


def unit_rows(runs):
    """One row per (run, held-out class): normalized transfer = (acc_h - 1/K) / (in-dist acc - 1/K)."""
    rows = []
    for r in runs:
        s = r['spec']
        den = r['indist_test_acc'] - r['chance']
        for h, v in r['heldout'].items():
            rows.append({'arm': s['arm'], 'map': s['map'], 'k': s['k'], 'p': s['p'], 'seed': s['seed'], 'h': int(h),
                         'acc': v['acc'], 'indist': r['indist_test_acc'], 'chance': r['chance'],
                         'transfer': (v['acc'] - r['chance']) / den if den > 0 else float('nan')})
    return rows


def seed_mean(rows):
    """Average over seeds -> one value per unit (arm, map, k, p, h)."""
    g = {}
    for r in rows:
        g.setdefault((r['arm'], r['map'], r['k'], r['p'], r['h']), []).append(r)
    out = []
    for key, rs in sorted(g.items()):
        out.append({'arm': key[0], 'map': key[1], 'k': key[2], 'p': key[3], 'h': key[4], 'n_seeds': len(rs),
                    'transfer': float(np.nanmean([r['transfer'] for r in rs])),
                    'acc': float(np.mean([r['acc'] for r in rs])), 'indist': float(np.mean([r['indist'] for r in rs]))})
    return out


def rowcorr(x, y):
    """Pearson correlation of matching rows of x and y (NaN where a row is constant)."""
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (x * y).sum(-1) / np.where(den > 0, den, 1), np.nan)


def spearman(a, b):
    """Spearman rho along the last axis (average ranks for ties)."""
    return rowcorr(rankdata(a, axis=-1), rankdata(b, axis=-1))


def boot(n, rng, B):
    return rng.integers(0, n, size=(B, n))


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [None, None]


def mean_ci(x, rng, B):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {'n': 0, 'mean': None, 'ci95': [None, None]}
    idx = boot(len(x), rng, B)
    return {'n': int(len(x)), 'mean': float(x.mean()), 'ci95': ci(x[idx].mean(1))}


def corr_block(units, struct, measure, rng, B, complexity='gzip_bits'):
    """Spearman rho(transfer, measure) and R^2(measure) - R^2(complexity), with a unit bootstrap."""
    ys, xs, cs = [], [], []
    for u in units:
        key = f"{u['arm']}/{u['map']}/{u['k']}/{u['h']}"
        if key in struct and np.isfinite(u['transfer']):
            ys.append(u['transfer'])
            xs.append(struct[key][measure])
            cs.append(struct[key][complexity])
    ys, xs, cs = (np.asarray(v, dtype=float) for v in (ys, xs, cs))
    n = len(ys)
    if n < 4 or np.std(xs) == 0 or np.std(ys) == 0:
        return {'n': n, 'rho': None}
    idx = boot(n, rng, B)
    Y, X, C = ys[idx], xs[idx], cs[idx]
    out = {'n': n, 'rho': float(spearman(ys, xs)), 'rho_ci95': ci(spearman(Y, X))}
    if np.std(cs) > 0:
        out['r2_structure'] = float(rowcorr(xs, ys) ** 2)
        out['r2_complexity'] = float(rowcorr(cs, ys) ** 2)
        out['r2_diff'] = out['r2_structure'] - out['r2_complexity']
        out['r2_diff_ci95'] = ci(rowcorr(X, Y) ** 2 - rowcorr(C, Y) ** 2)
    return out


def analyze(results):
    runs = load_runs(results)
    struct_path = os.path.join(results, 'structure.json')
    struct = json.load(open(struct_path))['units'] if os.path.exists(struct_path) else {}
    rows = unit_rows(runs)
    units = seed_mean(rows)
    rng = np.random.default_rng(A['seed'])
    B = A['bootstrap']
    p0 = A['primary_p']
    sel = lambda arm, m, p: [u for u in units if u['arm'] == arm and u['map'] == m and u['p'] == p]
    out = {'n_runs': len(runs), 'runs_by_phase': {}, 'qc': {}, 'by_arm': {}}
    for ph, g in CFG['grid'].items():
        rs = [r for r in runs if r['spec']['phase'] == ph]
        out['runs_by_phase'][ph] = len(rs)
        if rs:
            ep = CFG['vision' if g['arm'] == 'vision' else 'text']['epochs']
            out['qc'][ph] = {'indist_test_acc_mean': float(np.mean([r['indist_test_acc'] for r in rs])),
                             'indist_test_acc_min': float(np.min([r['indist_test_acc'] for r in rs])),
                             'best_epoch_at_last_eval_share': float(np.mean([r['best_epoch'] == ep for r in rs])),
                             'train_sec_median': float(np.median([r['train_sec'] for r in rs]))}
    for arm in ('vision', 'text'):
        a = {}
        # P1: shuffled map, p = 0
        sh = sel(arm, 'shuffled', p0)
        a['P1_shuffled_p0_transfer'] = mean_ci([u['transfer'] for u in sh], rng, B)
        re0 = sel(arm, 'real', p0)
        a['real_p0_transfer'] = mean_ci([u['transfer'] for u in re0], rng, B)
        if re0 and sh:
            a['real_minus_shuffled_p0'] = a['real_p0_transfer']['mean'] - a['P1_shuffled_p0_transfer']['mean']
        # P2 / P3: real map, p = 0, units; secondary measures and pooled real + shuffled
        measures = ['margin', 'nc_share', 'pad', 'gzip_bits'] + (['pixel_margin'] if arm == 'vision' else ['jaccard_top200'])
        a['P2_P3_real_p0'] = corr_block(re0, struct, 'margin', rng, B)
        a['secondary_real_p0'] = {m: corr_block(re0, struct, m, rng, B) for m in measures}
        a['pooled_real_shuffled_p0'] = corr_block(re0 + sh, struct, 'margin', rng, B)
        # the p dial
        dial = {}
        for m in ('real', 'shuffled'):
            for p in sorted({u['p'] for u in units if u['arm'] == arm and u['map'] == m}):
                us = sel(arm, m, p)
                dial[f'{m}/p={p:g}'] = {'transfer': mean_ci([u['transfer'] for u in us], rng, B),
                                        'heldout_acc': float(np.mean([u['acc'] for u in us])),
                                        'indist_acc': float(np.mean([u['indist'] for u in us]))}
        a['p_dial'] = dial
        a['verdicts'] = verdicts(a)
        out['by_arm'][arm] = a
    out['units'] = units
    out['structure_present'] = bool(struct)
    return out


def verdicts(a):
    v = {}
    p1 = a['P1_shuffled_p0_transfer']
    if p1['n']:
        v['P1'] = 'SUPPORTED' if abs(p1['mean']) <= A['P1_band'] else 'NOT_SUPPORTED'
    p2 = a['P2_P3_real_p0']
    if p2.get('rho') is not None:
        v['P2'] = 'SUPPORTED' if p2['rho'] >= A['P2_rho'] else 'NOT_SUPPORTED'
        lo, hi = p2.get('r2_diff_ci95', [None, None])
        if lo is not None:
            v['P3'] = 'SUPPORTED' if lo > 0 else ('CONTRADICTED' if hi < 0 else 'NOT_ESTABLISHED')
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out')
    a = ap.parse_args()
    r = analyze(a.results)
    op = a.out or os.path.join(a.results, 'analysis.json')
    with open(op + '.tmp', 'w') as f:
        json.dump(r, f, indent=1, sort_keys=True)
    os.replace(op + '.tmp', op)
    print('runs', r['runs_by_phase'])
    for arm, x in r['by_arm'].items():
        print(arm, 'verdicts', x['verdicts'], 'real p0', x['real_p0_transfer'], 'shuffled p0', x['P1_shuffled_p0_transfer'])
    print('wrote', op)


if __name__ == '__main__':
    main()
