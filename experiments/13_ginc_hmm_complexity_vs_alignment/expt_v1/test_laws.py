"""Unit tests 10-11 of PROMPT.md: law fitters recover known parameters, the independent fitter agrees, the
invalid-forecast rule triggers, and the position decomposition recovers known main effects."""
import json
import math
from pathlib import Path

import numpy as np

import analyze as AN
import independent_fit as IF
import laws as L

HERE = Path(__file__).resolve().parent
RAW = json.loads((HERE / 'config.json').read_text())
C = dict(RAW['shared'])
C.update(RAW['small'])
C['eps_res'] = 1e-4
KB = [240, 504, 1056, 2208]
KREF = math.exp(np.mean(np.log([240, 504, 2208])))
T0 = 65536


def synth(law, th, seed=0, sd=0.02):
    rng = np.random.default_rng(seed)
    rows = []
    for K in KB:
        for a in (0.05, 0.15, 0.4, 1.0):
            for T in (65536, 131072, 262144, 524288, 1048576):
                eid = 0.03 * math.log2(1 / a) / 4.32
                rows.append(dict(K_bits=K, alpha=a, T=T, ident_bits=eid, excess_bits=1.0))
    X = L.design(rows, 'K_bits', KB, KREF, T0, C['eps_res'])
    p = L.predict_eps(law, th, X)
    if law == 'M_align_add':
        eps = p + rng.normal(0, 0.002, len(p))
    else:
        eps = X['eid'] + (p - X['eid']) * 2.0 ** rng.normal(0, sd, len(p))
    for r, e in zip(rows, eps):
        r['excess_bits'] = float(e)
    return L.design(rows, 'K_bits', KB, KREF, T0, C['eps_res'])


TRUE = {
    'M0_scale': [math.log2(0.4), 0.5],
    'M_align_eff': [math.log2(0.4), 0.5, 0.8],
    'M_K_coef': [math.log2(0.4), 0.5, 0.8, 0.6],
    'M_K_exp': None,
    'M_K_both': None,
}


def exp_theta(g0, g1, X):
    return g0 + g1 * X['lKmin'], g0 + g1 * X['lKmax']


def test_recovery_log_laws():
    X0 = synth('M0_scale', TRUE['M0_scale'])
    cases = dict(TRUE)
    lo, hi = exp_theta(0.5, -0.1, X0)
    cases['M_K_exp'] = [math.log2(0.4), lo, hi, 0.8]
    cases['M_K_both'] = [math.log2(0.4), lo, hi, 0.8, 0.6]
    for law, th in cases.items():
        X = synth(law, th, seed=1)
        f = L.fit(law, X, C)
        p = L.public_params(law, th, X)
        for k in ('gamma', 'eta', 'kappa', 'gamma0', 'gamma1'):
            if k in p:
                est = f['params'][k]
                tol = 0.05 * abs(p[k]) if k != 'gamma1' else 0.05 * abs(p[k]) + 0.01
                assert abs(est - p[k]) <= tol, (law, k, est, p[k])


def test_recovery_align_add_and_invalid_rule():
    th = [0.05, 0.6, 0.03, 0.5]
    X = synth('M_align_add', th, seed=2)
    f = L.fit('M_align_add', X, C)
    assert abs(f['params']['gamma'] - 0.5) <= 0.05 * 0.5
    bad = [0.0, 0.01, 0.2, 0.5]  # forecasts <= 0 at high alpha / late T
    p = L.predict_eps('M_align_add', bad, X)
    inv = L.invalid_mask('M_align_add', p, X)
    assert inv.any() and np.all(inv[p <= 0])
    r, b = L.scored_residual('M_align_add', bad, X)
    assert np.all(np.isfinite(r)) and b.sum() == inv.sum()


def test_independent_fit_agrees():
    for law in L.LAWS:
        th = {'M_align_add': [0.05, 0.6, 0.03, 0.5]}.get(law)
        if th is None:
            X0 = synth('M0_scale', TRUE['M0_scale'])
            lo, hi = exp_theta(0.5, -0.05, X0)
            th = {'M0_scale': TRUE['M0_scale'], 'M_align_eff': TRUE['M_align_eff'], 'M_K_coef': TRUE['M_K_coef'],
                  'M_K_exp': [math.log2(0.4), lo, hi, 0.8], 'M_K_both': [math.log2(0.4), lo, hi, 0.8, 0.6]}[law]
        X = synth(law, th, seed=3)
        fit_m = X['alpha'] != 0.15
        Xf, Xh = L.subset(X, fit_m), L.subset(X, ~fit_m)
        a = L.fit(law, Xf, C)
        b = IF.fit(law, Xf, C)
        ra = IF.heldout_rmse(law, a['theta'], Xh)[0]
        rb = IF.heldout_rmse(law, b['theta'], Xh)[0]
        assert IF.agree(ra, rb, C['independent_fit_tol']), (law, ra, rb, a['cost'], b['cost'])


def test_tstar_and_elasticity():
    X = synth('M_K_both', [0.0, 0.5, 0.4, 1.0, 0.5], seed=4)
    p = L.public_params('M_K_both', [0.0, 0.5, 0.4, 1.0, 0.5], X)
    K = 1056.0
    ts = L.t_star('M_K_both', p, K, 0.4, 0.05, X)
    lK = math.log2(K / X['K_ref'])
    g = p['gamma0'] + p['gamma1'] * lK
    learn = 2 ** (p['logA'] + p['kappa'] * lK - g * (p['eta'] * math.log2(0.4) + math.log2(ts / X['T0'])))
    assert abs(learn - 0.05) < 1e-9
    ea, ek = L.elasticities('M_K_both', p, K, 0.05, X)
    h = 1e-6
    num = (math.log(L.t_star('M_K_both', p, K * math.exp(h), 0.4, 0.05, X)) - math.log(ts)) / h
    assert ea == p['eta'] and abs(ek - num) < 1e-4


def test_position_decomposition_recovers_effects():
    a_eff = {0.05: 1.0, 0.15: 0.4, 0.4: -0.3, 1.0: -1.1}
    k_eff = {4: -0.2, 8: -0.05, 16: 0.1, 32: 0.15}
    t_eff = {i: -0.5 * i for i in range(5)}
    ys, ai, ki, ti = [], [], [], []
    for a, ea in a_eff.items():
        for k, ek in k_eff.items():
            for t, et in t_eff.items():
                ys.append(3.0 + ea + ek + et)
                ai.append(a)
                ki.append(k)
                ti.append(t)
    share, ss = AN.main_effect_share(np.array(ys), np.array(ai), np.array(ki), np.array(ti))
    va = np.var(list(a_eff.values())) * 80
    vk = np.var(list(k_eff.values())) * 80
    assert abs(share - va / (va + vk)) < 1e-9


def test_model_free_tstar():
    Ts = [65536, 131072, 262144, 524288, 1048576]
    t, f = AN.mf_tstar(Ts, [0.4, 0.2, 0.1, 0.05, 0.025], 0.07)
    assert f == 'ok' and 262144 < t < 524288
    assert AN.mf_tstar(Ts, [0.4, 0.3, 0.2, 0.15, 0.12], 0.05) == (1048576.0, 'right')
    assert AN.mf_tstar(Ts, [0.01, 0.01, 0.01, 0.01, 0.01], 0.05) == (65536.0, 'left')
