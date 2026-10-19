"""Synthetic-recovery test: fake main-grid and H results from a known law; analyze.py must recover E and the H gates."""
import json, math, os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analyze as AN, run as R, tasks as TK


def fake(root, eta=0.5, kappa=1.0, gamma=0.5, noise=0.02):
    rng = np.random.default_rng(0)
    Ts = R.CFG['T_checkpoints']['mnist']
    Kref = AN.gm([1, 2, TK.k_bits('sum10')])
    for s in R.main_specs('mnist'):
        K = TK.k_bits(s['target'])
        rows = []
        for T in Ts:
            e = 0.6 * (K / Kref) ** kappa * (s['alpha'] ** eta * T / Ts[0]) ** (-gamma) * math.exp(noise * rng.normal())
            e = min(e, 1.0)
            rows.append({'T': T, f"eps__{s['target']}": e, f"err__{s['target']}": e * TK.chance_error(s['target'])})
        p = os.path.join(root, 'main', R.rid(s) + '.json'); os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(dict(spec=s, rows=rows), open(p, 'w'))
    for s in R.h_specs('mnist'):
        rows = []
        for T in Ts:
            r = {'T': T}
            for t in TK.H_TARGETS['easy'] + TK.H_TARGETS['hard'] + TK.H_EXTRA_ANCHORS:
                base = 0.5 * (T / Ts[0]) ** -0.5
                f = 1.0
                if s.get('pool') == 'easy' and t in TK.H_TARGETS['hard']:
                    f = 2.0
                r[f'probe_eps__{t}'] = base * f * math.exp(noise * rng.normal())
                r[f'eps__{t}'] = base * f * math.exp(noise * rng.normal())
            rows.append(r)
        if s['mode'] == 'pool_only':
            rows = [dict(rows[0], T=0)] + rows
        p = os.path.join(root, 'H', R.rid(s) + '.json'); os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(dict(spec=s, rows=rows), open(p, 'w'))


def test_recovery():
    d = tempfile.mkdtemp()
    fake(d)
    AN.NB = 2000
    A = AN.analyse_main(AN.load_runs(d, 'main', 'mnist'), 'mnist')
    E = A['G4']['E_bits']
    assert abs(E - 0.25) < 0.03, E
    assert A['G4']['replication'] == 'SAME_DIRECTION', A['G4']
    assert abs(A['G4']['E_mf'] - 0.25) < 0.05, A['G4']['E_mf']
    assert A['G1']['verdict'] == 'SUPPORTED'
    assert A['independent_fit']['agree'], A['independent_fit']
    H = AN.analyse_H(d, 'mnist')
    assert H['joint']['GH1'] == 'CARRIES' and H['joint']['GH2'] == 'SUPPORTED', H['joint']
    assert abs(H['joint']['S']['rho_EH']['ratio'] - 2.0) < 0.05
    assert H['mod_anchor']['sum10'] and H['n_runs'] == H['n_runs_expected'], H['n_runs']
    print('eps*', A['eps_star'], 'E', E, 'E_mf', A['G4']['E_mf'], 'M_K', A['G1']['M_K'], 'GH', H['joint']['GH1'], H['joint']['GH2'])


if __name__ == '__main__':
    test_recovery()
