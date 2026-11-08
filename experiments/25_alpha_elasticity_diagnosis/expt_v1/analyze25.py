"""Experiment 25 analysis (PROTOCOL Section 4): data bills T*, the per-cell learning-rate choice, censoring-aware
(Tobit) elasticities per arm, gates G1-G5 with a two-stage bootstrap (seeds, then targets within each level; the
learning-rate choice is redone in every draw) and the every-seed rule, and the reported analyses.

Usage (from expt_v1/): analyze25.py [--results DIR] [--draws B] [--workers W]
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E24 = HERE.parent.parent / '24_similarity_alignment_complexity' / 'expt_v1'
sys.path.insert(0, str(E24))
sys.path.insert(0, str(HERE))

import analyze24 as A24  # noqa: E402  (Tobit, the T* crossing rule, censored residuals, percentile intervals)

ARMS = ['hmm', 'iid', 'masked']
tobit, ci = A24.tobit, A24.ci


def load_config():
    return json.loads((HERE / 'config.json').read_text())


# ---------------------------------------------------------------- data bills

def read_runs(res):
    """run key (size, target_id, arm, alpha, lr_multiplier, seed) -> dict(K, pts, diverged)."""
    runs = {}
    with open(Path(res) / 'runs.csv') as f:
        for r in csv.DictReader(f):
            k = (r['size'], r['target_id'], r['arm'], float(r['alpha']), float(r['lr_multiplier']), int(r['seed']))
            runs[k] = dict(K=int(r['K']), pts=[], diverged=bool(r['diverged_at']))
    with open(Path(res) / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            if r['slice'] != 'all':
                continue
            k = (r['size'], r['target_id'], r['arm'], float(r['alpha']), float(r['lr_multiplier']), int(r['seed']))
            runs[k]['pts'].append((int(r['T']), float(r['learn_bits']), float(r['excess_bits']),
                                   float(r['ident_bits'])))
    for d in runs.values():
        d['pts'].sort()
    return runs


def tstar(d, eps, T_cap):
    """Experiment 24's crossing rule. A diverged run that had not crossed never crosses: right-censored at the cap."""
    if not d['pts']:
        return math.log(T_cap), 1
    y, st = A24.tstar(d['pts'], eps)
    if d['diverged'] and st == 1:
        return math.log(T_cap), 1
    return y, st


def tstar_table(runs, eps, T_cap):
    out = []
    for (size, tid, arm, alpha, m, seed), d in sorted(runs.items()):
        y, st = tstar(d, eps, T_cap)
        out.append(dict(size=size, target_id=tid, arm=arm, alpha=alpha, lr=m, seed=seed, K=d['K'], lnT=y, status=st))
    return out


# ---------------------------------------------------------------- the learning-rate choice

def choose_lr(tab, wt):
    """Per (size, n, arm, alpha) cell, the multiplier with the lowest weighted mean ln T* (a censored run counts at its
    bound); ties go to the smaller multiplier. Returns {(size, K, arm, alpha): multiplier}."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for r in tab:
        w = wt(r)
        if w > 0:
            a = acc[(r['size'], r['K'], r['arm'], r['alpha'], r['lr'])]
            a[0] += w * r['lnT']
            a[1] += w
    best = {}
    for (size, K, arm, alpha, m), (s, w) in sorted(acc.items()):
        key, v = (size, K, arm, alpha), s / w
        if key not in best or v < best[key][1] - 1e-12:
            best[key] = (m, v)
    return {k: m for k, (m, _) in best.items()}


def rows_for(tab, arm, policy, choice=None, size=None):
    """The rows of one arm (plus the shared alpha = 1 rows) under a learning-rate policy: 'tuned' (per-cell choice)
    or a fixed multiplier."""
    out = []
    for r in tab:
        if r['arm'] not in (arm, 'none') or (size is not None and r['size'] != size):
            continue
        m = choice[(r['size'], r['K'], r['arm'], r['alpha'])] if policy == 'tuned' else policy
        if r['lr'] == m:
            out.append(r)
    return out


def design(rows, kind, hinge_alpha=None):
    """'pooled': [S, M dummies, ln alpha, ln n]; 'single': [1, ln alpha, ln n]; 'hinge': pooled + min(ln a - ln k, 0)."""
    X, y, st = A24.design(rows, 'single' if kind == 'single' else 'pooled')
    if kind == 'hinge':
        h = np.minimum(np.log([r['alpha'] for r in rows]) - math.log(hinge_alpha), 0.0)
        X = np.column_stack([X, h])
    return X, y, st


def fit(rows, kind, wt, start, hinge_alpha=None):
    X, y, st = design(rows, kind, hinge_alpha)
    return tobit(X, y, st, np.array([wt(r) for r in rows]), start)


def elast(b, kind):
    j = 1 if kind == 'single' else 2
    e_a, e_k = float(-b[j]), float(b[j + 1])
    return e_a, e_k, (e_a / e_k if e_k > 0 else math.inf)


# ---------------------------------------------------------------- the gate statistics on one (reweighted) table

def stats(cfg, tab, weight=None, starts=None, full=True):
    wt = (lambda r: 1.0) if weight is None else weight
    st0 = starts or {}
    ref, k = cfg['grid']['reference_multiplier'], cfg['analysis']['hinge_alpha']
    choice = choose_lr(tab, wt)
    out, fits = {}, {}

    def one(key, rows, kind):
        rows = [r for r in rows if wt(r) > 0]
        b = fit(rows, kind, wt, st0.get(key), k)
        fits[key] = b
        return b

    for arm in ARMS:
        b = one(('tuned', arm), rows_for(tab, arm, 'tuned', choice), 'pooled')
        out[f'e_alpha|{arm}|tuned'], out[f'e_K|{arm}|tuned'], out[f'E|{arm}|tuned'] = elast(b, 'pooled')
    b = one(('fixed', 'hmm', ref), rows_for(tab, 'hmm', ref), 'pooled')
    out[f'e_alpha|hmm|x{ref:g}'], out[f'e_K|hmm|x{ref:g}'], out[f'E|hmm|x{ref:g}'] = elast(b, 'pooled')
    b = one(('hinge', 'hmm'), rows_for(tab, 'hmm', 'tuned', choice), 'hinge')
    out['e_alpha_high|hmm|tuned'] = float(-b[2])
    out['e_alpha_low|hmm|tuned'] = float(-b[2] - b[4])
    out['G1'] = out['e_alpha|hmm|tuned'] - out[f'e_alpha|hmm|x{ref:g}']
    out['G2'] = out['e_alpha|masked|tuned']
    out['G3'] = out['e_alpha|masked|tuned'] - out['e_alpha|hmm|tuned']
    out['G4'] = out['e_alpha|iid|tuned'] - out['e_alpha|hmm|tuned']
    out['G5'] = out['e_alpha_low|hmm|tuned'] - out['e_alpha_high|hmm|tuned']
    if full:  # reported, point only: every fixed multiplier, per size, and the hinge for the other arms
        for arm in ARMS:
            for m in cfg['grid']['lr_multipliers']:
                if arm == 'hmm' and m == ref:
                    continue
                b = one(('fixed', arm, m), rows_for(tab, arm, m), 'pooled')
                out[f'e_alpha|{arm}|x{m:g}'], out[f'e_K|{arm}|x{m:g}'], out[f'E|{arm}|x{m:g}'] = elast(b, 'pooled')
            for size in cfg['grid']['sizes']:
                b = one(('tuned', arm, size), rows_for(tab, arm, 'tuned', choice, size), 'single')
                (out[f'e_alpha|{arm}|tuned|{size}'], out[f'e_K|{arm}|tuned|{size}'],
                 out[f'E|{arm}|tuned|{size}']) = elast(b, 'single')
            if arm != 'hmm':
                b = one(('hinge', arm), rows_for(tab, arm, 'tuned', choice), 'hinge')
                out[f'e_alpha_high|{arm}|tuned'] = float(-b[2])
                out[f'e_alpha_low|{arm}|tuned'] = float(-b[2] - b[4])
    return out, fits, choice


def _boot_job(args):
    cfg, tab, draws, seeds, levels, rng_seed, starts = args
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
            res.append(stats(cfg, tab, wt, starts, full=False)[0])
        except Exception as e:  # a degenerate draw is recorded, never dropped silently
            res.append({'error': repr(e)})
    return res


# ---------------------------------------------------------------- gates

def gates(point, boot, per_seed):
    B = [b for b in boot if 'error' not in b]
    get = lambda k: [b[k] for b in B]  # noqa: E731
    g = {}

    def gate(name, claim, cond_point, cond_ci, cond_seed, extra=None):
        d = dict(claim=claim, value=point[name], ci95=ci(get(name)), per_seed=[p[name] for p in per_seed])
        d['every_seed'] = bool(all(cond_seed(x) for x in d['per_seed']))
        d['verdict'] = 'SUPPORTED' if (cond_point(d['value']) and cond_ci(d['ci95']) and d['every_seed']) \
            else 'NOT_ESTABLISHED'
        d.update(extra or {})
        g[name] = d
        return d

    pos = lambda x: x > 0  # noqa: E731
    gate('G1', 'one learning rate for every alpha biases e_alpha down: tuned minus reference > 0',
         pos, lambda c: c[0] > 0, pos)
    d = gate('G2', 'target-only learning is not purely token-limited: e_alpha(masked) < 1',
             lambda x: x < 1, lambda c: c[1] < 1, lambda x: x < 1)
    d['named'] = 'e_alpha(masked) > 1' if d['ci95'][0] > 1 and all(x > 1 for x in d['per_seed']) else None
    d = gate('G3', 'disjoint HMM distractors substitute for target data: e_alpha(masked) - e_alpha(hmm) > 0',
             pos, lambda c: c[0] > 0, pos)
    d['named'] = 'interference: distractors cost more than nothing' if (
        d['value'] < 0 and d['ci95'][1] < 0 and all(x < 0 for x in d['per_seed'])) else None
    gate('G4', 'the substitution is sequential structure: e_alpha(iid) - e_alpha(hmm) > 0', pos,
         lambda c: c[0] > 0, pos)
    gate('G5', 'the bill has a share-independent part: e_alpha below alpha 0.2 exceeds e_alpha above', pos,
         lambda c: c[0] > 0, pos)
    g['intervals'] = {k: ci(get(k)) for k in B[0] if k not in g} if B else {}
    g['bootstrap'] = dict(draws=len(boot), degenerate=len(boot) - len(B))
    return g


def levels_of(tab):
    lv = defaultdict(set)
    for r in tab:
        lv[r['K']].add(r['target_id'])
    return {k: sorted(v) for k, v in lv.items()}


def full_analysis(cfg, tab, draws, workers, rng_seed):
    point, fits, choice = stats(cfg, tab)
    seeds = sorted({r['seed'] for r in tab})
    per_seed = [stats(cfg, [r for r in tab if r['seed'] == s], starts=fits, full=False)[0] for s in seeds]
    levels = levels_of(tab)
    chunks = max(1, workers)
    sizes = [draws // chunks + (1 if i < draws % chunks else 0) for i in range(chunks)]
    jobs = [(cfg, tab, n, seeds, levels, [rng_seed, i], fits) for i, n in enumerate(sizes) if n]
    if workers > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(workers) as ex:
            boot = [b for part in ex.map(_boot_job, jobs) for b in part]
    else:
        boot = [b for j in jobs for b in _boot_job(j)]
    return point, fits, choice, per_seed, boot, gates(point, boot, per_seed)


# ---------------------------------------------------------------- reported, not gated

def independent_check(cfg, tab, fits, choice, tol):
    """Refit every point-estimate Tobit with L-BFGS-B from zero."""
    worst, bad = 0.0, []
    for key, b in fits.items():
        kind = key[0]
        if kind == 'fixed':
            rows, dk = rows_for(tab, key[1], key[2]), 'pooled'
        elif kind == 'hinge':
            rows, dk = rows_for(tab, key[1], 'tuned', choice), 'hinge'
        elif len(key) == 3:
            rows, dk = rows_for(tab, key[1], 'tuned', choice, key[2]), 'single'
        else:
            rows, dk = rows_for(tab, key[1], 'tuned', choice), 'pooled'
        X, y, st = design(rows, dk, cfg['analysis']['hinge_alpha'])
        b2 = tobit(X, y, st, start=np.r_[np.zeros(X.shape[1]), 0.0], method='L-BFGS-B')
        rel = float(np.max(np.abs(b2[:-1] - b[:-1]) / np.maximum(np.abs(b[:-1]), 1e-3)))
        worst = max(worst, rel)
        if rel > tol:
            bad.append(dict(fit=str(key), rel=rel))
    return dict(agree=not bad, worst_relative=worst, disagreements=bad)


def early(cfg, runs):
    """Mean learning term at the first checkpoints, per (size, n, arm, alpha, multiplier): the early-training
    penalty seen at alpha = 1 in Experiment 24."""
    acc = defaultdict(list)
    for (size, tid, arm, alpha, m, seed), d in runs.items():
        pts = {T: l for T, l, _, _ in d['pts']}
        acc[f'{size}|{d["K"]}|{arm}|{alpha}|x{m:g}'].append([pts.get(T, math.nan) for T in cfg['analysis']['early_T']])
    return {k: np.nanmean(np.array(v), 0).tolist() for k, v in sorted(acc.items())}


def convergence(cfg, runs):
    rule = cfg['analysis']['convergence_rule']
    cells = defaultdict(list)
    for (size, tid, arm, alpha, m, seed), d in runs.items():
        pts = {T: l for T, l, _, _ in d['pts']}
        Ts = sorted(pts)
        if len(Ts) >= 3:
            cells[(size, d['K'], arm, alpha, m)].append((pts[Ts[-3]], pts[Ts[-1]]))
    out = {}
    for k, v in sorted(cells.items()):
        before, after = float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))
        out['|'.join(map(str, k))] = dict(before=before, after=after, converged=bool(
            before - after <= max(rule['min_drop_bits'], rule['rel_drop'] * after)))
    return dict(cells=out, converged=sum(c['converged'] for c in out.values()), total=len(out))


def reported(cfg, res, runs, tab, fits, choice):
    T_cap = cfg['grid']['T_checkpoints'][-1]
    rep = dict(lr_choice={'|'.join(map(str, k)): v for k, v in sorted(choice.items())})
    edge = [k for k, v in choice.items() if v in (min(cfg['grid']['lr_multipliers']), max(cfg['grid']['lr_multipliers']))]
    rep['lr_choice_at_grid_edge'] = sorted('|'.join(map(str, k)) for k in edge)
    for eps in cfg['analysis']['eps_secondary']:
        p, _, _ = stats(cfg, tstar_table(runs, eps, T_cap), full=False)
        rep[f'eps_{eps}'] = p
    rep['censoring'] = {}
    for r in tab:
        key = f"{r['size']}|{r['K']}|{r['arm']}|{r['alpha']}|x{r['lr']:g}"
        c = rep['censoring'].setdefault(key, [0, 0, 0])
        c[{0: 0, 1: 1, -1: 2}[r['status']]] += 1
    rep['diverged'] = sorted('|'.join(map(str, k)) for k, d in runs.items() if d['diverged'])
    rep['early'] = early(cfg, runs)
    rep['convergence'] = convergence(cfg, runs)
    o, mx = np.load(Path(res) / 'oracle.npz'), np.load(Path(res) / 'mixture.npz')
    eid = defaultdict(list)
    for key in mx.files:
        tid, s, a = key.split('|')
        eid[f'{s}|{a}'].append(float((mx[key] - o[tid]).mean()))
    rep['eps_id_mean'] = {k: float(np.mean(v)) for k, v in eid.items()}
    dl = defaultdict(list)
    with open(Path(res) / 'distractor_loss.csv') as f:
        for r in csv.DictReader(f):
            if int(r['T']) == T_cap:
                dl[f"{r['size']}|{r['arm']}|{r['alpha']}"].append(float(r['model_bits']) - float(r['oracle_bits']))
    rep['distractor_excess_at_Tmax'] = {k: float(np.mean(v)) for k, v in dl.items()}
    rep['independent_fit'] = independent_check(cfg, tab, fits, choice, cfg['analysis']['independent_fit_tol'])
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
    tab = tstar_table(runs, cfg['analysis']['eps_star'], cfg['grid']['T_checkpoints'][-1])
    with open(res / 'tstar.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(tab[0]))
        w.writeheader()
        w.writerows(tab)
    draws = a.draws or cfg['analysis']['bootstrap_draws']
    point, fits, choice, per_seed, boot, g = full_analysis(cfg, tab, draws, a.workers, cfg['analysis']['analysis_seed'])
    rep = reported(cfg, res, runs, tab, fits, choice)
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
