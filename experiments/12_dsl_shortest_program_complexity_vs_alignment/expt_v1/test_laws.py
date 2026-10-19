"""Unit tests for laws.py, independent_fit.py and the analysis logic, on synthetic data over the real design grid."""
import json
from pathlib import Path

import numpy as np
import pytest

import analyze as AN
import independent_fit as IF
import laws as L

C = json.loads((Path(__file__).parent / 'config.json').read_text())
KB = dict(zip(C['K_levels'], C['K_bits']))

TRUE = {
    'M_K_coef': dict(logA=-1.0, gamma=0.5, eta=1.0, kappa=1.0),
    'M_K_exp': dict(logA=1.0, gamma0=0.9, gamma1=-0.1, eta=1.0),
    'M_K_both': dict(logA=0.0, gamma0=0.8, gamma1=-0.08, eta=0.8, kappa=0.5),
}


def truth_eps(p, Kb, alpha, T):
    lK, la, lT = np.log2(Kb), np.log2(alpha), np.log2(T)
    g = p.get('gamma', p.get('gamma0', 0) + p.get('gamma1', 0) * lK)
    return 2 ** (p['logA'] + p.get('kappa', 0) * lK - g * (p['eta'] * la + lT))


def synthetic_rows(law, seeds=8, noise=0.05, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in range(seeds):
        for k in C['K_levels']:
            for j in range(C['targets_per_level']):
                for a in C['alphas']:
                    for T in C['T_checkpoints']:
                        e = truth_eps(TRUE[law], KB[k], a, T) * np.exp(rng.normal(0, noise))
                        rows.append(dict(arm='main', seed=C['seeds'][s], target_id=f'dsl_k{k}_t{j + 1}', K=k,
                                         K_bits=KB[k], alpha=a, T=T, excess_bits=float(e)))
    return rows


@pytest.mark.parametrize('law', ['M_K_coef', 'M_K_exp', 'M_K_both'])
def test_recovery(law):
    rows = synthetic_rows(law)
    est = []
    for s in C['seeds']:
        rs = [r for r in rows if r['seed'] == s and AN.is_fit(r, C)]
        assert len(rs) == 288
        est.append(L.fit(law, L.design(rs, 'K_bits', C), C)['params'])
    for k, v in TRUE[law].items():
        if k == 'logA':
            continue
        assert abs(np.mean([e[k] for e in est]) - v) < 0.05, (law, k, np.mean([e[k] for e in est]), v)
    assert abs(np.mean([e['logA'] for e in est]) - TRUE[law]['logA']) < 0.3


def test_heldout_grid_size():
    tasks = [dict(task_id=f'dsl_k{k}_t{j}', role='target', K=k, K_bits=KB[k], task_token=4 * i + j)
             for i, k in enumerate(C['K_levels']) for j in range(4)]
    assert len(AN.heldout_grid(C, tasks)) == 272


@pytest.mark.parametrize('law', L.LAWS)
def test_independent_fitter_agrees(law):
    rows = [r for r in synthetic_rows('M_K_both', seeds=1, rng_seed=3) if AN.is_fit(r, C)]
    X = L.design(rows, 'K_bits', C)
    a, b = L.fit(law, X, C), IF.fit(law, X, C)
    assert IF.agree(a['cost'], b['cost'], 1e-4), (a['cost'], b['cost'])
    grid = [dict(K=4, K_bits=20, alpha=0.15, T=65536, excess_bits=1.0)]
    Xh = L.design(grid, 'K_bits', C)
    assert IF.agree(L.predict_eps(law, a['theta'], Xh), L.predict_eps(law, b['theta'], Xh), 1e-4)


def test_centering_reports_uncentered_logA():
    rows = [r for r in synthetic_rows('M_K_coef', seeds=1, noise=1e-6) if AN.is_fit(r, C)]
    p = L.fit('M_K_coef', L.design(rows, 'K_bits', C), C)['params']
    assert abs(p['logA'] - TRUE['M_K_coef']['logA']) < 1e-3 and abs(p['kappa'] - 1.0) < 1e-3


def test_censored_residual_zero_below_floor():
    X = dict(y=np.log2(np.array([1e-4, 1e-5])), cens=np.array([True, True]), yres=np.log2(1e-2))
    assert np.all(L.censored_residual(np.log2(np.array([5e-3, 1e-6])), X) == 0)
    assert L.censored_residual(np.log2(np.array([2e-2])), dict(X, y=X['y'][:1], cens=X['cens'][:1]))[0] == 1.0


def test_t_star_exact_and_flags():
    Ts = np.array(C['T_checkpoints'], float)
    eps = 30.0 * Ts ** -0.5
    v, fl = AN.t_star(Ts, eps, 0.5)
    assert fl == '' and abs(v - (30 / 0.5) ** 2) / v < 1e-12
    assert AN.t_star(Ts, eps, 1e-3)[1] == 'right' and AN.t_star(Ts, eps, 10.0)[1] == 'left'
    X = L.design([dict(K=4, K_bits=20, alpha=1.0, T=512, excess_bits=1.0)], 'K_bits', C)
    p = dict(logA=np.log2(30.0), gamma=0.5)
    assert abs(L.t_star('M0_scale', p, 20, 1.0, 0.5, X) - 3600) < 1e-6


def test_analytic_e_K_matches_finite_difference():
    X = L.design([dict(K=4, K_bits=20, alpha=1.0, T=512, excess_bits=1.0)], 'K_bits', C)
    p = dict(logA=0.3, eta=0.9, kappa=0.4, gamma0=0.7, gamma1=-0.03)
    _, eK = L.elasticities('M_K_both', p, C['K_bits_center'], 1.0, X)
    h = 1e-6
    f = lambda k: np.log(L.t_star('M_K_both', p, k, 0.5, 1.0, X))  # noqa: E731
    kc = C['K_bits_center']
    fd = (f(kc * np.exp(h)) - f(kc * np.exp(-h))) / (2 * h)
    assert abs(eK - fd) < 1e-6


def test_align_add_negative_flagged_invalid():
    X = L.design([dict(K=4, K_bits=20, alpha=1.0, T=512, excess_bits=0.01)], 'K_bits', C)
    r, bad = L.scored_residual('M_align_add', np.array([0.0, 0.0, 1.0, 0.5]), X)
    assert bad.all() and np.isfinite(r).all()


def test_bootstrap_identical_seeds_collapses():
    b = AN.Boot(8, 500, 1)
    lo, hi = AN.ci(b.mean(np.full(8, 2.5)))
    assert abs(lo - 2.5) < 1e-12 and abs(hi - 2.5) < 1e-12


@pytest.mark.parametrize('law,expect', [('M_K_coef', ('SUPPORTED', 'NOT_ESTABLISHED')),
                                         ('M_K_exp', ('NOT_ESTABLISHED', 'SUPPORTED'))])
def test_g3_logic(law, expect):
    """M_K_both on fit rows per seed, held-out RMSE of M_K_coef and M_K_exp, then the G3 rule."""
    rows = synthetic_rows(law, rng_seed=7)
    held = [r for r in rows if not AN.is_fit(r, C)]
    boot = AN.Boot(8, 2000, C['analysis_seed'])
    kap, g1, sse = [], [], {'M_K_coef': np.zeros(8), 'M_K_exp': np.zeros(8)}
    nn = np.zeros(8)
    for i, s in enumerate(C['seeds']):
        X = L.design([r for r in rows if r['seed'] == s and AN.is_fit(r, C)], 'K_bits', C)
        Xh = L.design([r for r in held if r['seed'] == s], 'K_bits', C)
        p = L.fit('M_K_both', X, C)['params']
        kap.append(p['kappa'])
        g1.append(p['gamma1'])
        for m in sse:
            sse[m][i] = np.sum(L.scored_residual(m, L.fit(m, X, C)['theta'], Xh)[0] ** 2)
        nn[i] = len(Xh['y'])
    rc, re = (float(np.sqrt(sse[m].sum() / nn.sum())) for m in ('M_K_coef', 'M_K_exp'))
    gain, gain_ci = AN.rel_gain(sse['M_K_coef'], nn, sse['M_K_exp'], nn, boot)
    got = AN.gate_g3(AN.ci(boot.mean(kap)), AN.ci(boot.mean(g1)), rc, re, gain, gain_ci, C)
    assert got == expect, (got, np.mean(kap), np.mean(g1), rc, re, gain)


def _g3(kind, kappa=0.5, gamma1=-0.1):
    return dict(G3a='SUPPORTED' if kind == 'G3a' else 'NOT_ESTABLISHED',
                G3b='SUPPORTED' if kind == 'G3b' else 'NOT_ESTABLISHED', kappa=kappa, gamma1=gamma1)


def test_g9_mocks():
    assert AN.gate_g9({'G3': _g3('G3a', kappa=0.4)}, _g3('G3a', kappa=0.6), C)['verdict'] == 'SUPPORTED'  # 1.5
    assert AN.gate_g9({'G3': _g3('G3b', gamma1=-0.1)}, _g3('G3b', gamma1=-0.15), C)['verdict'] == 'SUPPORTED'
    assert AN.gate_g9({'G3': _g3('G3a', kappa=0.4)}, _g3('G3a', kappa=1.0), C)['verdict'] == 'NOT_ESTABLISHED'  # 2.5
    assert AN.gate_g9({'G3': _g3('G3a', kappa=0.4)}, _g3('G3a', kappa=-0.6), C)['verdict'] == 'NOT_ESTABLISHED'
    r = AN.gate_g9({'G3': _g3(None)}, _g3('G3a'), C)
    assert r['verdict'] == 'NOT_ESTABLISHED' and 'unresolved' in r['reason']
    assert AN.gate_g9({'G3': _g3('G3a')}, _g3('G3b'), C)['verdict'] == 'NOT_ESTABLISHED'
    r = AN.gate_g9(None, _g3('G3a'), C)
    assert r['verdict'] == 'NOT_ESTABLISHED' and r['reason'] == 'Experiment 11 unavailable'
