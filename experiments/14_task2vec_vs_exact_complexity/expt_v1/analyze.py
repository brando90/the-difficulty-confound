"""Gates 1-5 and every reported analysis of PROTOCOL Section 4 from the saved embeddings.

  python analyze.py [--results results]

Writes results/gates.json, analysis.json, task_table.csv, diversity.csv and bootstrap.npz.
"""
import argparse
import csv
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

import stats as S
import tasks14 as T

HERE = Path(__file__).resolve().parent
FAM_INDEX = dict(dfa=0, dsl=1)


# ---------------------------------------------------------------- loading

class Store:
    """Saved task and batch embeddings of one results folder."""

    def __init__(self, c, out):
        self.c, self.out = c, Path(out)
        self.roster = T.load_roster(self.out / 'tasks14.json')[0]
        z = np.load(self.out / 'embeddings.npz', allow_pickle=False)      # arrays per family (dimensions differ)
        self.row, self.f, self.head_loss, self.head_error = {}, {}, {}, {}
        for fam in T.FAMILIES:
            self.row.update({k: (fam, i) for i, k in enumerate(z[f'{fam}_keys'].tolist())})
            self.f[fam], self.head_loss[fam], self.head_error[fam] = (z[f'{fam}_f'], z[f'{fam}_head_loss'],
                                                                      z[f'{fam}_head_error'])
        self.layout = {}
        for fam in T.FAMILIES:
            rec = json.loads((self.out / 'probes' / f"probe_{fam}_{c['probe_seeds'][0]}.json").read_text())
            self.layout[fam] = rec['layout']
        self._batch = {}

    def emb(self, fam, tid, probe, est='E0', variant='main'):
        f, i = self.row[f'{fam}|{tid}|{probe}|{est}|{variant}']
        return self.f[f][i].astype(np.float64)

    def head(self, fam, tid, probe, est='E0', variant='main'):
        f, i = self.row[f'{fam}|{tid}|{probe}|{est}|{variant}']
        return float(self.head_loss[f][i]), float(self.head_error[f][i])

    def has(self, fam, tid, probe, est='E0', variant='main'):
        return f'{fam}|{tid}|{probe}|{est}|{variant}' in self.row

    def batches(self, fam, probe):
        if (fam, probe) not in self._batch:
            if not (self.out / 'batch_embeddings' / f'{fam}_{probe}.npz').exists():
                return {}
            z = np.load(self.out / 'batch_embeddings' / f'{fam}_{probe}.npz', allow_pickle=False)
            d = {}
            for k, f in zip(z['keys'].tolist(), z['f']):
                _, _, mix_id, kind, i = k.split('|')
                d.setdefault((mix_id, kind), {})[int(i)] = f.astype(np.float64)
            self._batch[(fam, probe)] = {k: [v[i] for i in sorted(v)] for k, v in d.items()}
        return self._batch[(fam, probe)]


def summary(f, kind, layout=None):
    """Scalar summaries of an embedding: log L1 (primary), log L2, log-sum 0.5 * sum log(f + 1e-12), and log L1 of
    block 1 or block 2 units only."""
    if kind == 'l1':
        return float(np.log(f.sum()))
    if kind == 'l2':
        return float(np.log(np.sqrt((f * f).sum())))
    if kind == 'logsum':
        return float(0.5 * np.log(f + 1e-12).sum())
    blk = dict(block1='blocks.0.', block2='blocks.1.')[kind]
    idx = np.concatenate([np.arange(a, b) for name, a, b in layout if name.startswith(blk)])
    return float(np.log(f[idx].sum()))


def score_matrix(st, fam, tids, probes, est='E0', variant='main', kind='l1'):
    return np.array([[summary(st.emb(fam, t, p, est, variant), kind, st.layout[fam]) for t in tids] for p in probes])


# ---------------------------------------------------------------- outcomes from Experiments 11 and 12

def tstar_and_code(c, fam, targets):
    """(T* (n_targets, n_seeds) at alpha = 1 and eps* primary, +inf if censored; prequential code-length proxy
    (median over seeds of the trapezoid integral of eps over T); eps*)."""
    m = T.exp_modules(c)[fam]
    cfg = m['cfg']
    eps_star = float(cfg['eps_star_primary'])
    rows = []
    with open(m['dir'] / 'results' / 'metrics.csv') as f:
        for r in csv.DictReader(f):
            if fam == 'dsl' and r['arm'] != 'main':
                continue
            rows.append(r)
    seeds = cfg['seeds']
    ts = S.tstar_table(rows, targets, seeds, eps_star, 1.0)
    curves = {}
    for r in rows:
        if abs(float(r['alpha']) - 1.0) < 1e-9:
            curves.setdefault((r['target_id'], int(r['seed'])), []).append((int(r['T']), float(r['excess_bits'])))
    code = []
    for t in targets:
        v = []
        for s in seeds:
            cv = sorted(curves[(t, int(s))])
            Ts, e = np.array([a for a, _ in cv], float), np.array([b for _, b in cv])
            v.append(float(np.sum((e[1:] + e[:-1]) / 2 * np.diff(Ts))))
        code.append(float(np.median(v)))
    return ts, np.array(code), eps_star, seeds


# ---------------------------------------------------------------- helpers

def verdict(ok):
    return 'SUPPORTED' if ok else 'NOT_ESTABLISHED'


def not_run(**kw):
    """Placeholder for an analysis whose inputs were not computed (run stopped at the wall cap)."""
    return dict(not_run=True, reason='inputs not computed: the run stopped at the wall cap', **kw)


def ci_dict(draws):
    lo, hi, nn = S.ci(draws)
    return dict(lo=lo, hi=hi, n_undefined_draws=nn)


def pair_distances(F_):
    n = len(F_)
    out = []
    for i, j in combinations(range(n), 2):
        out.append(sym_cos(F_[i], F_[j]))
    return np.array(out)


def sym_cos(a, b, eps=1e-8):
    ha, hb = a / (a + b + eps), b / (a + b + eps)
    return float(1.0 - np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb)))


def mix_cells(cov):
    """cell index of (K, m) in sorted order and replicate per coverage mix (order as given)."""
    keys = sorted({(m['K'], m['m']) for m in cov})
    idx = {k: i for i, k in enumerate(keys)}
    return np.array([idx[(m['K'], m['m'])] for m in cov]), np.array([m['replicate'] for m in cov])


def loo_rmse(y, X):
    n = len(y)
    err = []
    for i in range(n):
        keep = np.arange(n) != i
        A = np.column_stack([np.ones(keep.sum()), X[keep]])
        coef = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        err.append(float(y[i] - np.r_[1.0, X[i]] @ coef))
    return float(np.sqrt(np.mean(np.square(err))))


# ---------------------------------------------------------------- per family

def analyze_family(c, st, fam, boot):
    fi = FAM_INDEX[fam]
    g = c['gates']
    R = st.roster
    probes = c['probe_seeds']
    tasks = T.embedded_tasks(R, fam)
    tids = [t['task_id'] for t in tasks]
    K = np.array([t['K'] for t in tasks], float)
    ent = np.array([t['label_entropy'] for t in tasks])
    sens = np.array([t['sensitivity'] for t in tasks])
    cov = np.column_stack([ent, sens])
    A = {}

    logn = score_matrix(st, fam, tids, probes)
    mu, sd = S.z_constants(logn)
    kh = S.khat(logn, mu, sd)

    # Gates 1 and 2
    g12 = S.gate12(c, fi, logn, K, cov, mu, sd)
    boot[f'{fam}_g1_rho'] = g12['rho_draws']
    boot[f'{fam}_g2_partial'] = g12['partial_draws']
    G1 = dict(rho=g12['rho'], ci=ci_dict(g12['rho_draws']), n_tasks=len(tids))
    G1['pass'] = bool(G1['rho'] >= g['g1_min_rho'] and G1['ci']['lo'] > g['g1_lb_above'])
    G2 = dict(partial=g12['partial'], ci=ci_dict(g12['partial_draws']), covariates=['label_entropy', 'sensitivity'])
    G2['pass'] = bool(G2['partial'] >= g['g2_min_partial'] and G2['ci']['lo'] > g['g2_lb_above'])

    # Gate 3
    tgt = [t for t in tasks if t['role'] == 'target']
    tgt_ids = [t['task_id'] for t in tgt]
    ti = np.array([tids.index(t) for t in tgt_ids])
    ts_seeds, code, eps_star, seeds = tstar_and_code(c, fam, tgt_ids)
    g3 = S.gate3(c, fi, logn[:, ti], K[ti], mu, sd, ts_seeds)
    boot[f'{fam}_g3_rho'] = g3['rho_draws']
    G3 = dict(rho=g3['rho'], ci=ci_dict(g3['rho_draws']), n_targets=len(ti), eps_star=eps_star,
              n_censored_targets=int(np.isinf(g3['tstar']).sum()),
              n_censored_runs=int(np.isinf(ts_seeds).sum()), n_runs=int(ts_seeds.size))
    G3['pass'] = bool(G3['rho'] >= g['g3_min_rho'] and G3['ci']['lo'] > g['g3_lb_above'])

    # Gate 4: interleaved diversity on coverage mixes
    cov_mixes = T.coverage_mixes(c, R, fam)
    exp_mixes = T.exp_mixes(c, R, fam)
    cell, rep = mix_cells(cov_mixes)
    nb = c['diversity_batches']

    def avail(mix_id, kind):
        return all(len(st.batches(fam, p).get((mix_id, kind), [])) == nb for p in probes)
    dist = {}
    for kind, mixes in (('interleaved', cov_mixes + exp_mixes), ('single', cov_mixes)):
        for m in mixes:
            if avail(m['mix_id'], kind):
                dist[(m['mix_id'], kind)] = np.stack([pair_distances(st.batches(fam, p)[(m['mix_id'], kind)])
                                                      for p in probes])
    bounds = {w: np.stack([pair_distances(st.batches(fam, p)[(f'{fam}_bound_{w}', 'bound')]) for p in probes])
              for w in ('lower', 'upper') if avail(f'{fam}_bound_{w}', 'bound')}
    n_cov = sum((m['mix_id'], 'interleaved') in dist for m in cov_mixes)
    if n_cov == len(cov_mixes):
        D4 = np.stack([dist[(m['mix_id'], 'interleaved')] for m in cov_mixes])      # (n_mix, 4, n_pairs)
        Hc = np.array([m['H'] for m in cov_mixes])
        Kc = np.array([m['K'] for m in cov_mixes], float)
        g4 = S.gate4(c, fi, D4, Hc, Kc, cell, rep)
        boot[f'{fam}_g4_rho_H'] = g4['rho_H_draws']
        boot[f'{fam}_g4_gap'] = g4['gap_draws']
        G4 = dict(rho_H=g4['rho_H'], rho_H_ci=ci_dict(g4['rho_H_draws']), rho_K=g4['rho_K'], gap=g4['gap'],
                  gap_ci=ci_dict(g4['gap_draws']), n_mixes=len(cov_mixes), n_pairs=int(D4.shape[2]))
        G4['pass'] = bool(G4['rho_H'] >= g['g4_min_rho_H'] and G4['rho_H_ci']['lo'] > g['g4_rho_H_lb_above']
                          and G4['gap'] >= g['g4_min_gap'] and G4['gap_ci']['lo'] > g['g4_gap_lb_above'])
    else:
        G4 = not_run(n_mixes=len(cov_mixes), n_mixes_available=n_cov, **{'pass': None})

    # Gate 5: label noise
    nt = T.noise_tasks(c, R, fam)
    nids = [t['task_id'] for t in nt]
    Kn = np.array([t['K'] for t in nt], float)
    hi_rate = max(c['noise_rates'])
    lr = {}
    for r in c['noise_rates']:
        if all(st.has(fam, t, p, 'E0', f'noise{r:.2f}') for t in nids for p in probes):
            lr[r] = np.array([[summary(st.emb(fam, t, p, 'E0', f'noise{r:.2f}'), 'l1') -
                               summary(st.emb(fam, t, p), 'l1') for t in nids] for p in probes])
    if hi_rate in lr:
        g5 = S.gate5(c, fi, lr[hi_rate], Kn)
        boot[f'{fam}_g5_ratio'] = g5['ratio_draws']
        G5 = dict(ratio=g5['ratio'], ci=ci_dict(g5['ratio_draws']), n_tasks=len(nids), rate=hi_rate,
                  ratio_at_other_rates={f'{r:.2f}': float(np.exp(np.median(v.mean(0)))) for r, v in lr.items()})
        G5['pass'] = bool(G5['ratio'] >= g['g5_min_ratio'] and G5['ci']['lo'] > g['g5_lb_above'])
    else:
        G5 = not_run(n_tasks=len(nids), rate=hi_rate, **{'pass': None})
    gates = dict(G1=G1, G2=G2, G3=G3, G4=G4, G5=G5)

    # ---- reported, not gated
    var = {}
    for name, est, variant, kind in (('E1', 'E1', 'main', 'l1'), ('E2', 'E2', 'main', 'l1'),
                                     ('E3', 'E3', 'main', 'l1'), ('rescaled_E0', 'E0', 'rescaled', 'l1'),
                                     ('E0_L2', 'E0', 'main', 'l2'), ('E0_logsum', 'E0', 'main', 'logsum'),
                                     ('E0_block1', 'E0', 'main', 'block1'), ('E0_block2', 'E0', 'main', 'block2')):
        if not all(st.has(fam, t, p, est, variant) for t in tids for p in probes):
            var[name] = not_run()
            continue
        ln = score_matrix(st, fam, tids, probes, est, variant, kind)
        m_, s_ = S.z_constants(ln)
        a = S.gate12(c, fi, ln, K, cov, m_, s_)
        b = S.gate3(c, fi, ln[:, ti], K[ti], m_, s_, ts_seeds)
        var[name] = dict(g1_rho=a['rho'], g1_ci=ci_dict(a['rho_draws']), g2_partial=a['partial'],
                         g2_ci=ci_dict(a['partial_draws']), g3_rho=b['rho'], g3_ci=ci_dict(b['rho_draws']))
    rp = c['random_init_seed']
    lnr = None
    if all(st.has(fam, t, rp) for t in tids):
        lnr = score_matrix(st, fam, tids, [rp])
        zr = (lnr[0] - lnr[0].mean()) / lnr[0].std()
        var['random_init_E0'] = dict(g1_rho=float(S.spearman(zr, K)),
                                     g2_partial=float(S.partial_spearman(zr, K, cov)),
                                     g3_rho=float(S.spearman(zr[ti], g3['tstar'])),
                                     note='one untrained probe: point estimates only')
    else:
        var['random_init_E0'] = not_run()
    A['variants'] = var

    # probe-seed sensitivity (L1 norms, E0)
    L1 = np.exp(logn)
    W = S.kendall_w(L1)
    pairs = list(combinations(range(len(probes)), 2))
    pw = [float(S.spearman(L1[i], L1[j])) for i, j in pairs]
    levels = np.unique(K)
    lvl_order = [tuple(np.argsort([L1[p][K == k].mean() for k in levels])) for p in range(len(probes))]
    disagree = float(np.mean([lvl_order[i] != lvl_order[j] for i, j in pairs]))
    A['probe_sensitivity'] = dict(kendall_w=W, pairwise_spearman=pw, mean_pairwise_spearman=float(np.mean(pw)),
                                  frac_probe_pairs_disagree_on_level_order=disagree,
                                  level_mean_l1_by_probe={str(p): [float(L1[i][K == k].mean()) for k in levels]
                                                          for i, p in enumerate(probes)})

    # parameterization sensitivity
    if var['rescaled_E0'].get('not_run'):
        A['parameterization'] = not_run()
    else:
        lres = score_matrix(st, fam, tids, probes, 'E0', 'rescaled')
        khr = S.khat(lres, *S.z_constants(lres))
        A['parameterization'] = dict(
            spearman_khat_original_vs_rescaled=float(S.spearman(kh, khr)),
            spearman_l1_original_vs_rescaled_by_probe=[float(S.spearman(logn[i], lres[i]))
                                                       for i in range(len(probes))],
            g1_rho_rescaled=var['rescaled_E0']['g1_rho'], g1_rho_change=var['rescaled_E0']['g1_rho'] - G1['rho'],
            median_l1_ratio_rescaled_over_original=float(np.exp(np.median(lres - logn))))

    # mechanism
    hl = np.array([[st.head(fam, t, p)[0] for t in tids] for p in probes]).mean(0)
    he = np.array([[st.head(fam, t, p)[1] for t in tids] for p in probes]).mean(0)
    A['mechanism'] = dict(
        rho_khat_head_loss=float(S.spearman(kh, hl)), rho_khat_head_error=float(S.spearman(kh, he)),
        rho_khat_label_entropy=float(S.spearman(kh, ent)), rho_khat_sensitivity=float(S.spearman(kh, sens)),
        rho_head_loss_K=float(S.spearman(hl, K)), rho_head_error_K=float(S.spearman(he, K)),
        rho_head_loss_sensitivity=float(S.spearman(hl, sens)),
        partial_head_loss_K_given_sensitivity=float(S.partial_spearman(hl, K, sens[:, None])),
        partial_khat_K_given_head_loss=float(S.partial_spearman(kh, K, hl[:, None])),
        rho_sensitivity_K=float(S.spearman(sens, K)), rho_label_entropy_K=float(S.spearman(ent, K)))

    # incremental value over exact K for T*
    ts = g3['tstar']
    kt = kh[ti]
    covt = np.column_stack([K[ti], ent[ti], sens[ti]])
    fin = np.isfinite(ts)
    inc = dict(partial_khat_tstar_given_K=float(S.partial_spearman(kt, ts, K[ti][:, None])),
               partial_khat_tstar_given_K_entropy_sensitivity=float(S.partial_spearman(kt, ts, covt)),
               rho_K_tstar=float(S.spearman(K[ti], ts)), n_finite_targets=int(fin.sum()),
               n_censored_targets=int((~fin).sum()))
    if fin.sum() >= 5:
        y = np.log(ts[fin])
        x1 = np.log2(K[ti][fin])[:, None]
        x2 = np.column_stack([np.log2(K[ti][fin]), logn[:, ti].mean(0)[fin]])
        inc.update(loo_rmse_logK=loo_rmse(y, x1), loo_rmse_logK_plus_lognorm=loo_rmse(y, x2))
    else:
        inc['loo_rmse_note'] = f'only {int(fin.sum())} uncensored targets; leave-one-out RMSE not computed'
    inc['rho_khat_code_length_proxy'] = float(S.spearman(kt, code))
    inc['rho_K_code_length_proxy'] = float(S.spearman(K[ti], code))
    A['incremental'] = inc

    # E0 versus E2 (E0 x batch size estimates the per-example FIM in expectation)
    l1e0 = np.exp(logn)
    l1e2 = np.exp(score_matrix(st, fam, tids, probes, 'E2'))
    A['e0_vs_e2'] = dict(median_ratio_E0xB_over_E2=float(np.median(l1e0 * c['estimators']['E0']['batch_size'] / l1e2)),
                         spearman_by_probe=[float(S.spearman(l1e0[i], l1e2[i])) for i in range(len(probes))])

    # diversity (reported analyses)
    E3 = {p: {t['task_id']: st.emb(fam, t['task_id'], p, 'E3') for t in tasks} for p in probes}

    def rao(m, p):
        Fm = [E3[p][t] for t in m['tasks']]
        D = np.array([[sym_cos(a, b) if i != j else 0.0 for j, b in enumerate(Fm)] for i, a in enumerate(Fm)])
        pi = np.asarray(m['shares'])
        return float(pi @ D @ pi)

    drows = []
    for kind, mixes in (('interleaved', cov_mixes + exp_mixes), ('single', cov_mixes)):
        for m in mixes:
            if (m['mix_id'], kind) not in dist:
                continue
            d = dist[(m['mix_id'], kind)]
            q = [rao(m, p) for p in probes]
            drows.append(dict(family=fam, mix_id=m['mix_id'], mix_kind=m['kind'], batches=kind, K=m['K'],
                              m=m.get('m', ''), replicate=m.get('replicate', ''), alpha=m.get('alpha', ''),
                              target_id=m.get('target_id', ''), H=m['H'], div=float(d.mean()),
                              div_ci_lo=0.0, div_ci_hi=0.0, rao_q=float(np.mean(q)),
                              **{f'div_{p}': float(d[i].mean()) for i, p in enumerate(probes)}))
    for w, d in bounds.items():
        drows.append(dict(family=fam, mix_id=f'{fam}_bound_{w}', mix_kind='bound', batches='bound', K='', m='',
                          replicate='', alpha='', target_id='', H='', div=float(d.mean()), div_ci_lo=0.0,
                          div_ci_hi=0.0, rao_q='', **{f'div_{p}': float(d[i].mean()) for i, p in enumerate(probes)}))
    # per-coefficient bootstrap interval over batch pairs (pairs and probes resampled)
    rng = np.random.default_rng([c['analysis_seed'], 40, fi])
    for r in drows:
        d = dist.get((r['mix_id'], r['batches'])) if r['batches'] != 'bound' else bounds[r['mix_id'].split('_')[-1]]
        pr = rng.integers(0, d.shape[0], (2000, d.shape[0]))
        pa = rng.integers(0, d.shape[1], (2000, d.shape[0], d.shape[1]))
        vals = np.take_along_axis(d[pr], pa, 2).mean((1, 2))
        r['div_ci_lo'], r['div_ci_hi'] = (float(v) for v in np.percentile(vals, [2.5, 97.5]))

    def col(rows, k):
        return np.array([float(r[k]) for r in rows])

    def sp(rows, a, b):
        return float(S.spearman(col(rows, a), col(rows, b))) if len(rows) >= 3 else None

    def rng_(rows):
        return [float(col(rows, 'div').min()), float(col(rows, 'div').max())] if rows else None
    ci_rows = [r for r in drows if r['mix_kind'] == 'coverage' and r['batches'] == 'interleaved']
    cs_rows = [r for r in drows if r['mix_kind'] == 'coverage' and r['batches'] == 'single']
    ex_rows = [r for r in drows if r['mix_kind'] == 'exp']
    a1 = [r for r in ex_rows if abs(float(r['alpha']) - 1.0) < 1e-9]
    A['diversity'] = dict(
        coverage_interleaved=dict(n=len(ci_rows), rho_H=sp(ci_rows, 'div', 'H'), rho_K=sp(ci_rows, 'div', 'K'),
                                  rho_Q=sp(ci_rows, 'div', 'rao_q'), div_range=rng_(ci_rows)),
        coverage_single=dict(n=len(cs_rows), rho_H=sp(cs_rows, 'div', 'H'), rho_K=sp(cs_rows, 'div', 'K'),
                             rho_Q=sp(cs_rows, 'div', 'rao_q'), div_range=rng_(cs_rows)),
        exp_interleaved=dict(n=len(ex_rows), rho_H=sp(ex_rows, 'div', 'H'), rho_K=sp(ex_rows, 'div', 'K'),
                             rho_alpha=sp(ex_rows, 'div', 'alpha'), rho_Q=sp(ex_rows, 'div', 'rao_q')),
        alpha1_slice=dict(n=len(a1), rho_div_K=sp(a1, 'div', 'K')),
        bounds={w: dict(div=float(d.mean()), by_probe=[float(x) for x in d.mean(1)]) for w, d in bounds.items()},
        n_mixes_expected=dict(coverage_interleaved=len(cov_mixes), coverage_single=len(cov_mixes),
                              exp_interleaved=len(exp_mixes), bounds=2))

    # task table rows
    trows = []
    for j, t in enumerate(tasks):
        r = dict(family=fam, task_id=t['task_id'], role=t['role'], source=t['source'], K=t['K'],
                 label_entropy=ent[j], sensitivity=sens[j], khat=float(kh[j]), head_loss=float(hl[j]),
                 head_error=float(he[j]))
        for i, p in enumerate(probes):
            r[f'log_l1_{p}'] = float(logn[i, j])
        r['log_l1_random_init'] = float(lnr[0, j]) if lnr is not None else ''
        if t['role'] == 'target':
            k = tgt_ids.index(t['task_id'])
            r['tstar_median'] = float(ts[k])
            r['code_length_proxy'] = float(code[k])
        trows.append(r)
    return gates, A, trows, drows, dict(K=K, khat=kh, logn=logn, ent=ent, sens=sens, ti=ti, tstar=ts,
                                        div_cov=col(ci_rows, 'div'), H_cov=col(ci_rows, 'H'), K_cov=col(ci_rows, 'K'),
                                        div_cov_single=col(cs_rows, 'div'), Q_cov_single=col(cs_rows, 'rao_q'),
                                        cov_resid=None)


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    a = ap.parse_args()
    t0 = time.time()
    c = T.load_config()
    out = HERE / a.results
    st = Store(c, out)
    boot, gates, analysis, trows, drows = {}, {}, {}, [], []
    for fam in T.FAMILIES:
        g, A_, tr, dr, _ = analyze_family(c, st, fam, boot)
        gates[fam], analysis[fam] = g, A_
        trows += tr
        drows += dr
    verdicts = {}
    for G in ('G1', 'G2', 'G3', 'G4', 'G5'):
        if any(gates[f][G].get('not_run') for f in T.FAMILIES):
            verdicts[G] = 'NOT_RUN'
        else:
            verdicts[G] = verdict(all(gates[f][G]['pass'] for f in T.FAMILIES))
    h = c['handoff']
    handoff = dict(
        g123_supported=all(verdicts[G] == 'SUPPORTED' for G in ('G1', 'G2', 'G3')),
        kendall_w_ok={f: analysis[f]['probe_sensitivity']['kendall_w'] >= h['min_kendall_w'] for f in T.FAMILIES},
        rescaled_rho_ok={f: (None if analysis[f]['parameterization'].get('not_run') else
                             analysis[f]['parameterization']['spearman_khat_original_vs_rescaled']
                             >= h['min_rescaled_rho']) for f in T.FAMILIES},
        incremental_nonnegative={f: analysis[f]['incremental']['partial_khat_tstar_given_K'] >= 0
                                 for f in T.FAMILIES})
    handoff['met'] = bool(handoff['g123_supported'] and all(handoff['kendall_w_ok'].values())
                          and all(v is True for v in handoff['rescaled_rho_ok'].values())
                          and all(handoff['incremental_nonnegative'].values()))
    if any(v is None for v in handoff['rescaled_rho_ok'].values()):
        handoff['note'] = 'rescaled-probe embeddings not computed (wall cap); the rescaling condition is untested'
    res = dict(verdicts=verdicts, families=gates, thresholds=c['gates'], handoff=handoff,
               bootstrap=dict(draws=c['bootstrap_draws'], seed=c['analysis_seed']),
               seconds=round(time.time() - t0, 1))
    (out / 'gates.json').write_text(json.dumps(res, indent=1))
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=1))
    write_csv(out / 'task_table.csv', trows)
    write_csv(out / 'diversity.csv', drows)
    with open(out / 'bootstrap.npz', 'wb') as f:
        np.savez(f, **boot)
    print(json.dumps(dict(verdicts=verdicts, handoff=handoff['met'], seconds=res['seconds'])))


if __name__ == '__main__':
    main()
