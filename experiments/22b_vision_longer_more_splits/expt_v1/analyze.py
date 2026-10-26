"""Experiment 22b analysis (PROTOCOL Section 4): unit table, P1-P3 verdicts, the p dial. Deterministic.

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


class RaggedSplits(ValueError):
    pass


def split_boot(ks, rng, B):
    """Bootstrap index sets (B, n) that resample whole splits with replacement (the split-level / cluster bootstrap:
    units of a split share one training run, so they are not independent). ks = split of every unit; every split
    must hold the same number of units (20 here)."""
    ks = np.asarray(ks)
    sp = sorted(set(ks.tolist()))
    G = [np.where(ks == k)[0] for k in sp]
    if len({len(g) for g in G}) != 1:
        raise RaggedSplits(f'splits hold different numbers of units: {sorted({len(g) for g in G})}')
    G = np.stack(G)                                            # (splits, units per split)
    draw = rng.integers(0, len(sp), size=(B, len(sp)))
    return G[draw].reshape(B, -1)


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [None, None]


def mean_ci(x, ks, rng, B):
    """Mean with the split-level 95% interval (ci95) and, for comparison with Experiment 22, the unit-level one."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {'n': 0, 'mean': None, 'ci95': [None, None]}
    return {'n': int(len(x)), 'n_splits': len(set(ks)), 'mean': float(x.mean()),
            'ci95': ci(x[split_boot(ks, rng, B)].mean(1)), 'ci95_unit': ci(x[boot(len(x), rng, B)].mean(1))}


def corr_block(units, struct, measure, rng, B, complexity='gzip_bits'):
    """Spearman rho(transfer, measure) and R^2(measure) - R^2(complexity), with the split-level bootstrap (ci95) and
    the unit-level one of Experiment 22 (ci95_unit)."""
    ys, xs, cs, ks = [], [], [], []
    for u in units:
        key = f"{u['arm']}/{u['map']}/{u['k']}/{u['h']}"
        if key in struct and np.isfinite(u['transfer']):
            ys.append(u['transfer'])
            xs.append(struct[key][measure])
            cs.append(struct[key][complexity])
            ks.append(u['k'])
    ys, xs, cs = (np.asarray(v, dtype=float) for v in (ys, xs, cs))
    n = len(ys)
    if n < 4 or np.std(xs) == 0 or np.std(ys) == 0:
        return {'n': n, 'rho': None}
    iu = boot(n, rng, B)
    try:
        idx = split_boot(ks, rng, B)
    except RaggedSplits:   # only in a partial grid (the code-path pilot); a complete grid has 20 units per split
        return {'n': n, 'n_splits': len(set(ks)), 'rho': float(spearman(ys, xs)), 'rho_ci95': [None, None],
                'rho_ci95_unit': ci(spearman(ys[iu], xs[iu])), 'ragged': True}
    out = {'n': n, 'n_splits': len(set(ks)), 'rho': float(spearman(ys, xs)), 'rho_ci95': ci(spearman(ys[idx], xs[idx])),
           'rho_ci95_unit': ci(spearman(ys[iu], xs[iu]))}
    if np.std(cs) > 0:
        out['r2_structure'] = float(rowcorr(xs, ys) ** 2)
        out['r2_complexity'] = float(rowcorr(cs, ys) ** 2)
        out['r2_diff'] = out['r2_structure'] - out['r2_complexity']
        out['r2_diff_ci95'] = ci(rowcorr(xs[idx], ys[idx]) ** 2 - rowcorr(cs[idx], ys[idx]) ** 2)
        out['r2_diff_ci95_unit'] = ci(rowcorr(xs[iu], ys[iu]) ** 2 - rowcorr(cs[iu], ys[iu]) ** 2)
    return out


def descriptive(units, rows, struct, p0):
    """No threshold: rho by round (splits 0-4 / 5-9), class-level rho (each fine class's two splits averaged), the
    seed-to-seed reliability of the unit transfers and the per-split mean transfer (real map, p = 0)."""
    re0 = [u for u in units if u['map'] == 'real' and u['p'] == p0 and f"vision/real/{u['k']}/{u['h']}" in struct]
    out = {}
    for name, lo, hi in (('splits_0_4', 0, 4), ('splits_5_9', 5, 9)):
        us = [u for u in re0 if lo <= u['k'] <= hi]
        if len(us) >= 4:
            out['rho_' + name] = {'n': len(us), 'rho': float(spearman(np.array([u['transfer'] for u in us]),
                                  np.array([struct[f"vision/real/{u['k']}/{u['h']}"]['margin'] for u in us])))}
    byh = {}
    for u in re0:
        byh.setdefault(u['h'], []).append(u)
    hs = sorted(byh)
    if len(hs) >= 4:
        T = np.array([np.mean([u['transfer'] for u in byh[h]]) for h in hs])
        X = np.array([np.mean([struct[f"vision/real/{u['k']}/{u['h']}"]['margin'] for u in byh[h]]) for h in hs])
        ib = boot(len(hs), np.random.default_rng(A['seed']), A['bootstrap'])
        out['class_level_rho'] = {'n': len(hs), 'rho': float(spearman(T, X)), 'ci95': ci(spearman(T[ib], X[ib]))}
    for m in ('real', 'shuffled'):
        d = {}
        for r in rows:
            if r['map'] == m and r['p'] == p0:
                d.setdefault((r['k'], r['h']), {})[r['seed']] = r['transfer']
        seeds = sorted({s for v in d.values() for s in v})
        if len(seeds) >= 2 and d:
            keys = sorted(d)
            a, b = (np.array([d[k][s] for k in keys]) for s in seeds[:2])
            rel = float(spearman(a, b))
            out[f'seed_reliability_{m}'] = {'n': len(keys), 'rho_between_seeds': rel,
                                             'spearman_brown_mean_of_2': 2 * rel / (1 + rel)}
        per = {}
        for u in units:
            if u['map'] == m and u['p'] == p0:
                per.setdefault(u['k'], []).append(u['transfer'])
        out[f'per_split_mean_transfer_{m}'] = {str(k): float(np.mean(v)) for k, v in sorted(per.items())}
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
    tr = lambda us: ([u['transfer'] for u in us], [u['k'] for u in us])
    out = {'n_runs': len(runs), 'runs_by_phase': {}, 'qc': {}, 'by_arm': {}}
    for ph, g in CFG['grid'].items():
        rs = [r for r in runs if r['spec']['phase'] == ph]
        out['runs_by_phase'][ph] = len(rs)
        if rs:
            ep = CFG['vision']['epochs']
            out['qc'][ph] = {'indist_test_acc_mean': float(np.mean([r['indist_test_acc'] for r in rs])),
                             'indist_test_acc_min': float(np.min([r['indist_test_acc'] for r in rs])),
                             'best_epoch_at_last_eval_share': float(np.mean([r['best_epoch'] == ep for r in rs])),
                             'train_sec_median': float(np.median([r['train_sec'] for r in rs]))}
    if 'vis_real' in out['qc']:
        sh = out['qc']['vis_real']['best_epoch_at_last_eval_share']
        out['qc_budget'] = {'vis_real_best_epoch_last_share': sh, 'max_allowed': A['qc_best_epoch_last_max'],
                            'budget_long_enough': bool(sh <= A['qc_best_epoch_last_max'])}
    for arm in sorted({g['arm'] for g in CFG['grid'].values()}):
        a = {}
        # P1: shuffled map, p = 0
        sh = sel(arm, 'shuffled', p0)
        a['P1_shuffled_p0_transfer'] = mean_ci(*tr(sh), rng, B)
        re0 = sel(arm, 'real', p0)
        a['real_p0_transfer'] = mean_ci(*tr(re0), rng, B)
        if re0 and sh:
            a['real_minus_shuffled_p0'] = a['real_p0_transfer']['mean'] - a['P1_shuffled_p0_transfer']['mean']
        # P2 / P3: real map, p = 0, units; secondary measures and pooled real + shuffled
        measures = ['margin', 'nc_share', 'pad', 'gzip_bits', 'pixel_margin']
        a['P2_P3_real_p0'] = corr_block(re0, struct, 'margin', rng, B)
        a['secondary_real_p0'] = {m: corr_block(re0, struct, m, rng, B) for m in measures}
        a['pooled_real_shuffled_p0'] = corr_block(re0 + sh, struct, 'margin', rng, B)
        a['descriptive'] = descriptive(units, rows, struct, p0)
        # the p dial
        dial = {}
        for m in ('real', 'shuffled'):
            for p in sorted({u['p'] for u in units if u['arm'] == arm and u['map'] == m}):
                us = sel(arm, m, p)
                dial[f'{m}/p={p:g}'] = {'transfer': mean_ci(*tr(us), rng, B),
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
