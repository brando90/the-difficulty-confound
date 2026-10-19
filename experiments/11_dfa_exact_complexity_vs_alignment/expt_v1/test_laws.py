"""Unit tests for laws.py, independent_fit.py and the analysis logic, on synthetic data over the real design grid."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import analyze as AN
import independent_fit as IF
import laws as L

C = json.loads((Path(__file__).parent / 'config.json').read_text())
KB = {4: 20, 8: 56, 144: 144, 16: 144, 32: 352}

TRUE = {
    'M_K_coef': dict(logA=-1.0, gamma=0.5, eta=1.0, kappa=0.5),
    'M_K_exp': dict(logA=1.0, gamma0=0.8, gamma1=-0.05, eta=1.0),
    'M_K_both': dict(logA=0.0, gamma0=0.7, gamma1=-0.04, eta=0.8, kappa=0.3),
}


def truth_eps(law, p, Kb, alpha, T):
    lK, la, lT = np.log2(Kb), np.log2(alpha), np.log2(T)
    g = p.get('gamma', p.get('gamma0', 0) + p.get('gamma1', 0) * lK)
    return 2 ** (p['logA'] + p.get('kappa', 0) * lK - g * (p['eta'] * la + lT))


def synthetic_rows(law, seeds=8, noise=0.05, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in range(seeds):
        for n in C['n_levels']:
            for j in range(4):
                for a in C['alphas']:
                    for T in C['T_checkpoints']:
                        e = truth_eps(law, TRUE[law], KB[n], a, T) * np.exp(rng.normal(0, noise))
                        rows.append(dict(seed=11101 + s, target_id=f'dfa_n{n:02d}_t{j + 1}', K=n, K_bits=KB[n],
                                         alpha=a, T=T, excess_bits=float(e), accuracy=1.0,
                                         excess_bits_by_length=[float(e)] * 17))
    return rows


@pytest.mark.parametrize('law', ['M_K_coef', 'M_K_exp', 'M_K_both'])
def test_recovery(law):
    rows = synthetic_rows(law)
    est = []
    for s in sorted({r['seed'] for r in rows}):
        rs = [r for r in rows if r['seed'] == s and AN.is_fit(r, C)]
        assert len(rs) == 252
        est.append(L.fit(law, L.design(rs, 'K_bits', C), C)['params'])
    for k, v in TRUE[law].items():
        if k == 'logA':
            continue
        assert abs(np.mean([e[k] for e in est]) - v) < 0.05, (law, k, np.mean([e[k] for e in est]), v)


@pytest.mark.parametrize('law', L.LAWS)
def test_independent_fitter_agrees(law):
    rows = [r for r in synthetic_rows('M_K_both', seeds=1, rng_seed=3) if AN.is_fit(r, C)]
    X = L.design(rows, 'K_bits', C)
    a, b = L.fit(law, X, C), IF.fit(law, X, C)
    assert IF.agree(a['cost'], b['cost'], 1e-4), (a['cost'], b['cost'])
    grid = [dict(K=16, K_bits=144, alpha=0.15, T=65536, excess_bits=1.0)]
    Xh = L.design(grid, 'K_bits', C)
    assert IF.agree(L.predict_eps(law, a['theta'], Xh), L.predict_eps(law, b['theta'], Xh), 1e-4)


def test_censored_residual_zero_below_floor():
    X = dict(y=np.log2(np.array([1e-4, 1e-5])), cens=np.array([True, True]), yres=np.log2(1e-3))
    assert np.all(L.censored_residual(np.log2(np.array([5e-4, 1e-6])), X) == 0)
    assert L.censored_residual(np.log2(np.array([2e-3])), dict(X, y=X['y'][:1], cens=X['cens'][:1]))[0] == 1.0


def test_t_star_exact_and_flags():
    Ts = np.array(C['T_checkpoints'], float)
    eps = 3.0 * Ts ** -0.5
    v, fl = AN.t_star(Ts, eps, 0.05)
    assert fl == '' and abs(v - (3 / 0.05) ** 2) / v < 1e-12
    assert AN.t_star(Ts, eps, 1e-6)[1] == 'right' and AN.t_star(Ts, eps, 10.0)[1] == 'left'
    X = L.design([dict(K=4, K_bits=20, alpha=1.0, T=512, excess_bits=1.0)], 'K_bits', C)
    p = dict(logA=np.log2(3.0), gamma=0.5)
    assert abs(L.t_star('M0_scale', p, 20, 1.0, 0.05, X) - 3600) < 1e-6


def test_analytic_e_K_matches_finite_difference():
    X = L.design([dict(K=4, K_bits=20, alpha=1.0, T=512, excess_bits=1.0)], 'K_bits', C)
    p = dict(logA=0.3, eta=0.9, kappa=0.4, gamma0=0.7, gamma1=-0.03)
    _, eK = L.elasticities('M_K_both', p, 86.8, 0.1, X)
    h = 1e-6
    f = lambda k: np.log(L.t_star('M_K_both', p, k, 0.5, 0.1, X))  # noqa: E731
    fd = (f(86.8 * np.exp(h)) - f(86.8 * np.exp(-h))) / (2 * h)
    assert abs(eK - fd) < 1e-6


def test_align_add_negative_flagged_invalid():
    X = L.design([dict(K=4, K_bits=20, alpha=1.0, T=512, excess_bits=0.01)], 'K_bits', C)
    r, bad = L.scored_residual('M_align_add', np.array([0.0, 0.0, 1.0, 0.5]), X)
    assert bad.all() and np.isfinite(r).all()


def test_bootstrap_identical_seeds_collapses():
    b = AN.Boot(8, 500, 1)
    lo, hi = AN.ci(b.mean(np.full(8, 2.5)))
    assert abs(lo - 2.5) < 1e-12 and abs(hi - 2.5) < 1e-12


def write_synthetic(tmp, law):
    rows = synthetic_rows(law, rng_seed=7)
    tasks = []
    for n in C['n_levels']:
        for j in range(4):
            tasks.append(dict(task_id=f'dfa_n{n:02d}_t{j + 1}', role='target', K=n, K_bits=KB[n],
                              task_token=len(tasks), visit_tail_steepness=0.5 + 0.1 * j + 0.01 * n,
                              sensitivity=0.1 * j + 0.01 * n))
    (tmp / 'tasks.json').write_text(json.dumps(tasks))
    cols = ['seed', 'target_id', 'K', 'K_bits', 'alpha', 'T', 'excess_bits', 'accuracy', 'excess_bits_by_length']
    with open(tmp / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, excess_bits=repr(r['excess_bits']), excess_bits_by_length=json.dumps(r['excess_bits_by_length'])))
    return tasks


@pytest.mark.parametrize('law,expect', [('M_K_coef', ('SUPPORTED', 'NOT_ESTABLISHED')),
                                         ('M_K_exp', ('NOT_ESTABLISHED', 'SUPPORTED'))])
def test_g3_logic_end_to_end(tmp_path, law, expect):
    c = copy.deepcopy(C)
    c['bootstrap_draws'] = 1000
    tasks = write_synthetic(tmp_path, law)
    h, out = AN.stage1(c, tmp_path, tasks)
    assert out['all_fitters_agree']
    A = AN.stage2(c, tmp_path, tasks, two_stage_draws=200)
    assert (A['G3']['G3a'], A['G3']['G3b']) == expect, A['G3']
    assert A['G2']['k_law'] == law
