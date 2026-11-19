"""Experiment 24 analysis (PROTOCOL Sections 4-5): data bills T*, censoring-aware (Tobit) elasticities per similarity
level, gates G1-G5 with a two-stage bootstrap (seeds, then targets within each level) and the every-seed rule, and the
reported analyses (secondary eps*, the fitted-law E, convergence, identification, distractor loss, Task2Vec).

Usage (from expt_v1/): analyze24.py [--results DIR] [--draws B] [--workers W]
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent
E13 = HERE.parent.parent / '13_ginc_hmm_complexity_vs_alignment' / 'expt_v1'
sys.path.insert(0, str(E13))
sys.path.insert(0, str(HERE))

SIMS = ['copy20', 'copy50', 'family', 'disjoint']
SHARED = {'copy20': 0.8, 'copy50': 0.5, 'family': 0.0, 'disjoint': 0.0, 'none': 0.0}


def load_config():
    return json.loads((HERE / 'config.json').read_text())


# ---------------------------------------------------------------- data bills

def read_runs(res):
    """run key (size, target_id, similarity, alpha, seed) -> dict(K, K_bits, Ts, learn, excess, ident, ...)."""
    runs = {}
    with open(Path(res) / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            if r['slice'] != 'all':
                continue
            k = (r['size'], r['target_id'], r['similarity'], float(r['alpha']), int(r['seed']))
            d = runs.setdefault(k, dict(K=int(r['K']), K_bits=int(r['K_bits']), pts=[]))
            d['pts'].append((int(r['T']), float(r['learn_bits']), float(r['excess_bits']), float(r['ident_bits'])))
    for d in runs.values():
        d['pts'].sort()
    return runs


def tstar(pts, eps):
    """First crossing of eps by the learning term, log-log interpolated. Returns (ln T*, status): status 0 = crossed,
    +1 = right-censored (never reached by the last checkpoint: ln T* >= ln T_last), -1 = left-censored (already
    below eps at the first checkpoint: ln T* <= ln T_first)."""
    Ts = [p[0] for p in pts]
    v = [max(p[1], 1e-6) for p in pts]
    if v[0] <= eps:
        return math.log(Ts[0]), -1
    for i in range(1, len(Ts)):
        if v[i] <= eps:
            a, b = math.log(v[i - 1]), math.log(v[i])
            fr = (math.log(eps) - a) / (b - a)
            return math.log(Ts[i - 1]) + fr * (math.log(Ts[i]) - math.log(Ts[i - 1])), 0
    return math.log(Ts[-1]), 1


def tstar_table(runs, eps):
    out = []
    for (size, tid, sim, alpha, seed), d in sorted(runs.items()):
        y, st = tstar(d['pts'], eps)
        out.append(dict(size=size, target_id=tid, similarity=sim, alpha=alpha, seed=seed, K=d['K'],
                        K_bits=d['K_bits'], lnT=y, status=st))
    return out


# ---------------------------------------------------------------- Tobit

def _nll(theta, X, y, st, w):
    beta, ls = theta[:-1], theta[-1]
    s = math.exp(ls)
    z = (y - X @ beta) / s
    ll = np.where(st == 0, -0.5 * z * z - ls - 0.5 * math.log(2 * math.pi),
                  np.where(st == 1, log_ndtr(-z), log_ndtr(z)))
    return -float(np.sum(w * ll))


def _start(X, y, w):
    W = np.sqrt(w)[:, None]
    b = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)[0]
    r = y - X @ b
    return np.r_[b, math.log(max(math.sqrt(np.sum(w * r * r) / max(np.sum(w), 1)), 1e-3))]


def tobit(X, y, st, w=None, start=None, method='BFGS'):
    w = np.ones(len(y)) if w is None else w
    th0 = _start(X, y, w) if start is None else start
    r = minimize(_nll, th0, args=(X, y, st, w), method=method, options=dict(gtol=1e-9, maxiter=5000))
    return r.x


def design(rows, kind, unit='n'):
    """kind 'pooled': [size dummies, ln alpha, ln K]; 'single': [1, ln alpha, ln K]; 'g3_0' / 'g3_1': the G3 laws."""
    la = np.array([math.log(r['alpha']) for r in rows])
    lk = np.array([math.log(r['K'] if unit == 'n' else r['K_bits']) for r in rows])
    if kind == 'single':
        cols = [np.ones(len(rows))]
    else:
        cols = [np.array([r['size'] == 'S' for r in rows], float), np.array([r['size'] == 'M' for r in rows], float)]
    cols += [la, lk]
    if kind == 'g3_1':
        cols += [np.array([SHARED[r['similarity']] for r in rows]) * la,
                 np.array([r['similarity'] == 'disjoint' for r in rows], float) * la]
    return np.column_stack(cols), np.array([r['lnT'] for r in rows]), np.array([r['status'] for r in rows])


def elasticities(beta, kind):
    j = 1 if kind == 'single' else 2
    e_a, e_k = -beta[j], beta[j + 1]
    return float(e_a), float(e_k), (float(e_a / e_k) if e_k > 0 else math.inf)


def cens_resid(pred, y, st):
    return np.where(st == 0, pred - y, np.where(st == 1, np.maximum(0.0, y - pred), np.maximum(0.0, pred - y)))


# ---------------------------------------------------------------- one evaluation of every gate statistic

def rows_for(tab, sim, size=None):
    return [r for r in tab if (r['similarity'] == sim or r['similarity'] == 'none')
            and (size is None or r['size'] == size)]


def stats(tab, weight=None, starts=None, unit='n'):
    """Every quantity the gates use, on one (possibly reweighted) table. weight(row) -> multiplicity."""
    wt = (lambda r: 1.0) if weight is None else weight
    out, fits = {}, {}
    for sim in SIMS:
        rs = [r for r in rows_for(tab, sim) if wt(r) > 0]
        X, y, st = design(rs, 'pooled', unit)
        b = tobit(X, y, st, np.array([wt(r) for r in rs]), None if starts is None else starts.get(('p', sim)))
        fits[('p', sim)] = b
        out[f'e_alpha|{sim}'], out[f'e_K|{sim}'], out[f'E|{sim}'] = elasticities(b, 'pooled')
        for size in ('S', 'M'):
            rs2 = [r for r in rows_for(tab, sim, size) if wt(r) > 0]
            X, y, st = design(rs2, 'single', unit)
            b = tobit(X, y, st, np.array([wt(r) for r in rs2]), None if starts is None else starts.get((size, sim)))
            fits[(size, sim)] = b
            out[f'e_alpha|{sim}|{size}'], out[f'e_K|{sim}|{size}'], out[f'E|{sim}|{size}'] = elasticities(b, 'single')
    out['G1_delta'] = out['e_alpha|disjoint'] - out['e_alpha|copy20']
    dl = {s: out[f'e_alpha|{s}|S'] - out[f'e_alpha|{s}|M'] for s in SIMS}
    out['G5_A'] = float(np.mean([dl['copy20'], dl['copy50'], dl['family']]))
    out['G5_B'] = out['G5_A'] - dl['disjoint']
    fit_rows = [r for r in tab if r['similarity'] != 'copy50' and wt(r) > 0]
    held = [r for r in tab if r['similarity'] == 'copy50' and wt(r) > 0]
    hw = np.array([wt(r) for r in held])
    rm = {}
    for kind in ('g3_0', 'g3_1'):
        X, y, st = design(fit_rows, kind, unit)
        b = tobit(X, y, st, np.array([wt(r) for r in fit_rows]), None if starts is None else starts.get(kind))
        fits[kind] = b
        Xh, yh, sth = design(held, kind, unit)
        e = cens_resid(Xh @ b[:-1], yh, sth)
        rm[kind] = math.sqrt(float(np.sum(hw * e * e) / np.sum(hw)))
    out['G3_rmse_alpha_only'], out['G3_rmse_similarity'] = rm['g3_0'], rm['g3_1']
    out['G3_reduction'] = 1 - rm['g3_1'] / rm['g3_0'] if rm['g3_0'] > 0 else float('nan')
    return out, fits


def _boot_job(args):
    tab, draws, seeds, levels, rng_seed, starts = args
    rng = np.random.default_rng(rng_seed)
    res = []
    for _ in range(draws):
        sc = dict(zip(*np.unique(rng.choice(seeds, len(seeds)), return_counts=True)))
        tc = {}
        for lv, ids in levels.items():
            for t, n in zip(*np.unique(rng.choice(ids, len(ids)), return_counts=True)):
                tc[t] = n
        wt = lambda r: float(sc.get(r['seed'], 0) * tc.get(r['target_id'], 0))  # noqa: E731
        try:
            res.append(stats(tab, wt, starts)[0])
        except Exception as e:  # a degenerate draw is recorded, never dropped silently
            res.append({'error': repr(e)})
    return res


def ci(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    v = np.where(np.isinf(v), np.sign(v) * 1e300, v)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------------------------------------------------------------- gates

def gates(point, boot, per_seed, cfg):
    B = [b for b in boot if 'error' not in b]
    get = lambda k: [b[k] for b in B]  # noqa: E731
    g = {}
    ea = [point[f'e_alpha|{s}'] for s in SIMS]
    g1 = dict(e_alpha={s: point[f'e_alpha|{s}'] for s in SIMS}, monotone=bool(all(ea[i] < ea[i + 1] for i in range(3))),
              delta=point['G1_delta'], ci95=ci(get('G1_delta')), per_seed=[p['G1_delta'] for p in per_seed])
    g1['every_seed'] = bool(all(x > 0 for x in g1['per_seed']))
    g1['verdict'] = 'SUPPORTED' if (g1['monotone'] and g1['delta'] > 0 and g1['ci95'][0] > 0 and g1['every_seed']) \
        else 'NOT_ESTABLISHED'
    g['G1'] = g1
    g2 = dict(E_disjoint=point['E|disjoint'], ci95_disjoint=ci(get('E|disjoint')), E_copy20=point['E|copy20'],
              ci95_copy20=ci(get('E|copy20')), E={s: point[f'E|{s}'] for s in SIMS},
              per_seed=[[p['E|disjoint'], p['E|copy20']] for p in per_seed])
    g2['every_seed'] = bool(all(a > 1 and b < 1 for a, b in g2['per_seed']))
    g2['verdict'] = 'SUPPORTED' if (g2['E_disjoint'] > 1 and g2['ci95_disjoint'][0] > 1 and g2['E_copy20'] < 1
                                    and g2['ci95_copy20'][1] < 1 and g2['every_seed']) else 'NOT_ESTABLISHED'
    lo = [ci(get(f'E|{s}')) for s in SIMS]
    g2['named'] = ('E > 1 at every similarity' if all(x[0] > 1 for x in lo) else
                   'E < 1 at every similarity' if all(x[1] < 1 for x in lo) else None)
    g['G2'] = g2
    g3 = dict(reduction=point['G3_reduction'], ci95=ci(get('G3_reduction')),
              rmse_alpha_only=point['G3_rmse_alpha_only'], rmse_similarity=point['G3_rmse_similarity'],
              per_seed=[p['G3_reduction'] for p in per_seed], threshold=cfg['analysis']['g3_reduction'])
    g3['every_seed'] = bool(all(x > 0 for x in g3['per_seed']))
    g3['verdict'] = 'SUPPORTED' if (g3['reduction'] >= g3['threshold'] and g3['ci95'][0] > 0 and g3['every_seed']) \
        else 'NOT_ESTABLISHED'
    g['G3'] = g3
    g4 = dict(e_K={s: point[f'e_K|{s}'] for s in SIMS}, ci95={s: ci(get(f'e_K|{s}')) for s in SIMS},
              per_seed=[{s: p[f'e_K|{s}'] for s in SIMS} for p in per_seed])
    g4['every_seed'] = bool(all(v > 0 for p in g4['per_seed'] for v in p.values()))
    g4['verdict'] = 'SUPPORTED' if (all(g4['e_K'][s] > 0 and g4['ci95'][s][0] > 0 for s in SIMS) and g4['every_seed']) \
        else 'NOT_ESTABLISHED'
    g['G4'] = g4
    g5 = dict(A=point['G5_A'], ci95_A=ci(get('G5_A')), B=point['G5_B'], ci95_B=ci(get('G5_B')),
              delta={s: point[f'e_alpha|{s}|S'] - point[f'e_alpha|{s}|M'] for s in SIMS},
              per_seed=[[p['G5_A'], p['G5_B']] for p in per_seed])
    g5['every_seed'] = bool(all(a > 0 and b > 0 for a, b in g5['per_seed']))
    g5['verdict'] = 'SUPPORTED' if (g5['A'] > 0 and g5['ci95_A'][0] > 0 and g5['B'] > 0 and g5['ci95_B'][0] > 0
                                    and g5['every_seed']) else 'NOT_ESTABLISHED'
    g['G5'] = g5
    g['bootstrap'] = dict(draws=len(boot), degenerate=len(boot) - len(B))
    return g


def full_analysis(cfg, tab, draws, workers, rng_seed):
    point, fits = stats(tab)
    seeds = sorted({r['seed'] for r in tab})
    per_seed = [stats([r for r in tab if r['seed'] == s], starts=fits)[0] for s in seeds]
    levels = defaultdict(list)
    for r in tab:
        if r['target_id'] not in levels[r['K']]:
            levels[r['K']].append(r['target_id'])
    levels = {k: sorted(v) for k, v in levels.items()}
    chunks = max(1, workers)
    sizes = [draws // chunks + (1 if i < draws % chunks else 0) for i in range(chunks)]
    jobs = [(tab, n, seeds, levels, [rng_seed, i], fits) for i, n in enumerate(sizes) if n]
    if workers > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(workers) as ex:
            boot = [b for part in ex.map(_boot_job, jobs) for b in part]
    else:
        boot = [b for j in jobs for b in _boot_job(j)]
    return point, fits, per_seed, boot, gates(point, boot, per_seed, cfg)


# ---------------------------------------------------------------- reported, not gated

def independent_check(tab, fits, tol):
    """Refit every point-estimate Tobit with a different optimizer from a different start."""
    worst, bad = 0.0, []
    for key, b in fits.items():
        if key[0] == 'p':
            rs, kind = rows_for(tab, key[1]), 'pooled'
        elif key[0] in ('S', 'M'):
            rs, kind = rows_for(tab, key[1], key[0]), 'single'
        else:
            rs, kind = [r for r in tab if r['similarity'] != 'copy50'], key
        X, y, st = design(rs, kind)
        b2 = tobit(X, y, st, start=np.r_[np.zeros(X.shape[1]), 0.0], method='L-BFGS-B')
        rel = float(np.max(np.abs(b2[:-1] - b[:-1]) / np.maximum(np.abs(b[:-1]), 1e-3)))
        worst = max(worst, rel)
        if rel > tol:
            bad.append(dict(fit=str(key), rel=rel))
    return dict(agree=not bad, worst_relative=worst, disagreements=bad)


def law_E(cfg, res, eps):
    """Experiment 13's M_K_both (identification-floor offset, complexity-dependent exponent) per (size, similarity),
    refit on all rows, E at eps* and K_ref (descriptive; the frozen-law reading of Experiment 13)."""
    import laws as L
    lc = json.loads((E13 / 'config.json').read_text())['shared']
    rows = defaultdict(list)
    with open(Path(res) / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            if r['slice'] == 'all':
                rows[r['size']].append(r)
    out = {}
    for size, rs in rows.items():
        kb = sorted({float(r['K_bits']) for r in rs})
        kref = float(np.exp(np.mean(np.log(kb))))
        T0 = min(int(r['T']) for r in rs)
        for sim in SIMS:
            sel = [dict(K_bits=float(r['K_bits']), alpha=float(r['alpha']), T=int(r['T']),
                        excess_bits=float(r['excess_bits']), ident_bits=float(r['ident_bits']))
                   for r in rs if r['similarity'] in (sim, 'none')]
            X = L.design(sel, 'K_bits', kb, kref, T0, cfg['analysis']['eps_res'])
            f = L.fit('M_K_both', X, lc)
            ea, ek = L.elasticities('M_K_both', f['params'], kref, eps, X)
            out[f'{size}|{sim}'] = dict(e_alpha=ea, e_K_bits=ek, E=(ea / ek if ek > 0 else math.inf), params=f['params'])
    return out


def convergence(cfg, runs):
    """Experiment 23's rule on each (size, n, similarity, alpha) cell between the last two even-power checkpoints."""
    rule = cfg['analysis']['convergence_rule']
    cells = defaultdict(list)
    for (size, tid, sim, alpha, seed), d in runs.items():
        pts = {T: l for T, l, _, _ in d['pts']}
        Ts = sorted(pts)
        cells[(size, d['K'], sim, alpha)].append((pts[Ts[-3]], pts[Ts[-1]]))
    out = {}
    for k, v in sorted(cells.items()):
        before, after = float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))
        out['|'.join(map(str, k))] = dict(before=before, after=after,
                                          converged=bool(before - after <= max(rule['min_drop_bits'], rule['rel_drop'] * after)))
    return dict(cells=out, converged=sum(c['converged'] for c in out.values()), total=len(out))


def reported(cfg, res, runs, tab, fits):
    rep = {}
    for eps in cfg['analysis']['eps_secondary']:
        t2 = tstar_table(runs, eps)
        p, _ = stats(t2)
        rep[f'eps_{eps}'] = dict(point=p, censored={f'{z}|{s}': sum(1 for r in t2 if r['size'] == z
                                                                    and r['similarity'] == s and r['status'] == 1)
                                                    for z in ('S', 'M') for s in SIMS + ['none']})
    p_bits, _ = stats(tab, unit='bits')
    rep['K_in_bits'] = {k: v for k, v in p_bits.items() if k.startswith(('e_K|', 'E|'))}
    rep['censoring'] = {f'{z}|{s}|{a}': [sum(1 for r in tab if r['size'] == z and r['similarity'] == s and r['alpha'] == a
                                             and r['status'] == st) for st in (0, 1, -1)]
                        for z in ('S', 'M') for s in SIMS + ['none'] for a in cfg['grid']['alphas']
                        if (s == 'none') == (a >= 1)}
    rep['convergence'] = convergence(cfg, runs)
    try:
        rep['law_E'] = law_E(cfg, res, cfg['analysis']['eps_star'])
    except Exception as e:
        rep['law_E'] = {'error': repr(e)}
    tasks = json.loads((Path(res) / 'tasks.json').read_text())
    rep['pools'] = {s: dict(shared_fraction=float(np.mean([t['shared_fraction'] for t in tasks
                                                           if t.get('similarity') == s])),
                            emission_overlap=float(np.mean([t['emission_overlap_with_target'] for t in tasks
                                                            if t.get('similarity') == s]))) for s in SIMS}
    o, mx = np.load(Path(res) / 'oracle.npz'), np.load(Path(res) / 'mixture.npz')
    eid = defaultdict(list)
    for k in mx.files:
        tid, s, a = k.split('|')
        eid[f'{s}|{a}'].append(float((mx[k] - o[tid]).mean()))
    rep['eps_id_mean'] = {k: float(np.mean(v)) for k, v in eid.items()}
    rep['posterior16'] = json.loads((Path(res) / 'posterior16.json').read_text())
    dl = defaultdict(list)
    with open(Path(res) / 'distractor_loss.csv') as f:
        for r in csv.DictReader(f):
            if int(r['T']) == max(cfg['grid']['T_checkpoints']):
                dl[f"{r['size']}|{r['similarity']}|{r['alpha']}"].append(float(r['model_bits']) - float(r['oracle_bits']))
    rep['distractor_excess_at_Tmax'] = {k: float(np.mean(v)) for k, v in dl.items()}
    t2v = Path(res) / 'task2vec.json'
    if t2v.exists():
        from scipy.stats import spearmanr
        tv = json.loads(t2v.read_text())
        pairs = [(tv['coef'][f"{r['target_id']}|{r['similarity']}|{r['alpha']:.2f}"], r['lnT']) for r in tab
                 if r['status'] == 0 and f"{r['target_id']}|{r['similarity']}|{r['alpha']:.2f}" in tv['coef']]
        sp = spearmanr([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) > 3 else None
        rep['task2vec'] = dict(n=len(pairs), spearman_coef_vs_lnTstar=None if sp is None else float(sp.statistic),
                               p=None if sp is None else float(sp.pvalue), by_similarity=tv.get('by_similarity'))
    else:
        rep['task2vec'] = 'not computed'
    rep['independent_fit'] = independent_check(tab, fits, cfg['analysis']['independent_fit_tol'])
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--draws', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cfg = load_config()
    res = Path(a.results)
    runs = read_runs(res)
    tab = tstar_table(runs, cfg['analysis']['eps_star'])
    with open(res / 'tstar.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(tab[0]))
        w.writeheader()
        w.writerows(tab)
    draws = a.draws or cfg['analysis']['bootstrap_draws']
    point, fits, per_seed, boot, g = full_analysis(cfg, tab, draws, a.workers, cfg['analysis']['analysis_seed'])
    rep = reported(cfg, res, runs, tab, fits)
    verdicts = {k: v['verdict'] for k, v in g.items() if isinstance(v, dict) and 'verdict' in v}
    (res / 'gates.json').write_text(json.dumps(dict(eps_star=cfg['analysis']['eps_star'], verdicts=verdicts,
                                                    detail=g), indent=1, default=float))
    (res / 'analysis.json').write_text(json.dumps(dict(point=point, per_seed=per_seed, gates=g, reported=rep,
                                                       n_runs=len(tab)), indent=1, default=float))
    for k, v in verdicts.items():
        print(k, v)
    print('independent fitter', 'AGREE' if rep['independent_fit']['agree'] else 'DISAGREE')


if __name__ == '__main__':
    main()
