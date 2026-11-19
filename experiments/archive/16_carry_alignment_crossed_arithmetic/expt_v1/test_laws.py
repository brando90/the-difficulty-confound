"""Unit test 8 of PROMPT.md (laws), plus an end-to-end analysis dry run on synthetic 16a metrics."""
import itertools
import json
from pathlib import Path

import numpy as np

import analyze as A
import independent_fit as IF
import laws as L

C = json.loads((Path(__file__).resolve().parent / 'config.json').read_text())
TRUE = {
    'M_align_eff': [1.0, -6.0, 0.5, 0.3, 0.9],
    'M_K_coef': [1.0, -6.0, 0.5, 0.3, 0.9, 0.8],
    'M_K_exp': [1.0, -6.0, 0.5, 0.35, 0.2, 0.9],
    'M_K_both': [1.0, -6.0, 0.5, 0.35, 0.2, 0.9, 0.8],
}


def grid_rows():
    return [dict(s=s, alpha=a, T=2 ** t, N=n, excess_bits=1.0)
            for s, a, t, n in itertools.product([1, 2, 3, 8], [0.05, 0.4, 1.0], [16, 18, 20, 22], [991616, 6318592, 25220096])]


def synth(law, noise, seed=0):
    rows = grid_rows()
    X = L.design(rows, [1, 2, 3, 8], A.gm([1, 2, 3, 8]), 2 ** 16, 991616, 1e-3)
    eps = L.predict_eps(law, TRUE[law], X) * 2 ** np.random.default_rng(seed).normal(0, noise, len(rows))
    for r, e in zip(rows, eps):
        r['excess_bits'] = float(e)
    return L.design(rows, [1, 2, 3, 8], A.gm([1, 2, 3, 8]), 2 ** 16, 991616, 1e-3)


def test_recovery_and_independent_agreement():
    for law, truth in TRUE.items():
        X = synth(law, 0.02)
        f = L.fit(law, X, C)
        names = L.NAMES[law]
        for name in ('gamma', 'g_lo', 'g_hi', 'eta', 'kappa'):
            if name in names:
                j = names.index(name)
                assert abs(f['theta'][j] - truth[j]) <= 0.05 * abs(truth[j]), (law, name, f['theta'][j], truth[j])
        g = IF.fit(law, X, C)
        a, b = IF.heldout_rmse(law, f['theta'], X)[0], IF.heldout_rmse(law, g['theta'], X)[0]
        assert abs(a - b) <= C['shared']['independent_fit_tol'] * max(a, b), (law, a, b)


def test_invalid_forecasts_detected():
    X = synth('M_align_eff', 0.02)
    p = L.predict_eps('M_align_add', [-5.0, 0.0, 0.0, 0.0, 0.5, 0.3], X)
    r, bad = L.scored_residual('M_align_add', [-5.0, 0.0, 0.0, 0.0, 0.5, 0.3], X)
    assert bad.all() and np.all(p <= 0) and np.all(np.isfinite(r))


def test_analysis_dry_run(tmp_path):
    """Synthetic 16a metrics from a known M_K_both law; the full analysis runs and the frozen-forecast hash holds."""
    cm = C['mlp']
    rng = np.random.default_rng(5)
    rows = []
    params = dict(zip(cm['widths'], cm['expected_params']))
    for seed in cm['seeds']:
        for target, cc in cm['chain'].items():
            for alpha in cm['alpha_grid']:
                for w in cm['widths']:
                    for step in cm['checkpoints_updates']:
                        T = step * cm['batch_size']
                        X = L.design([dict(s=cc + 1, alpha=alpha, T=T, N=params[w], excess_bits=1.0)], [1, 3], A.gm([1, 3]),
                                     25600, 4328, 1e-3)
                        e = float(L.predict_eps('M_K_both', [3.0, -4.0, 0.5, 0.6, 0.4, 0.9, 0.7], X)[0]) * 2 ** rng.normal(0, 0.05)
                        for stratum in cm['chain']:
                            rows.append(dict(seed=seed, target_id=target, K=cc + 1, K_bits='', alpha=alpha, T=T, step=step,
                                             excess_bits=e if stratum == target else e * 2, accuracy=0.5, n_test=720,
                                             arm='mlp', format='mlp', N=params[w], width=w, c=cc, s=cc + 1,
                                             eval_stratum=stratum, eval_c=cm['chain'][stratum],
                                             **{f'digit_bits_{j}': (e / 4 if j < 4 else '') for j in range(9)}))
    import run as R
    R.write_csv(tmp_path / 'metrics.csv', rows)
    sp = A.spec('mlp', C)
    sp['seeds'] = sp['seeds'][:3]
    data_rows = A.read_rows(tmp_path)
    fits, sel = A.stage1(data_rows, sp, C, tmp_path)
    fits = {int(k): v for k, v in json.loads((tmp_path / 'fits.json').read_text())['fits'].items()}
    res = A.stage2(data_rows, sp, C, tmp_path, fits, sel)
    assert res['independent_fit']['status'] == 'AGREE'
    assert res['G3']['picture'] in ('coefficient', 'exponent', 'unresolved')
    assert res['heldout_invalid']['M_K_both'] == 0
    json.dumps(A._jsonable(res))


def test_analysis_dry_run_transformer(tmp_path):
    """Synthetic 16b metrics (both formats at 6M): stage 1, stage 2 and G6/G7 run end to end."""
    t = C['transformer']
    rng = np.random.default_rng(6)
    sizes = dict(zip(t['size_order'], t['expected_nonembedding']))
    rows = []
    for seed in t['seeds']:
        for size in t['size_order']:
            for fmt in (['plain', 'reversed'] if size == t['control_size'] else ['plain']):
                kappa = 0.8 if fmt == 'plain' else 0.2
                for cc in t['train_levels']:
                    for alpha in t['alpha_grid']:
                        for k in t['T_checkpoints_log2']:
                            X = L.design([dict(s=cc + 1, alpha=alpha, T=2 ** k, N=sizes[size], excess_bits=1.0)],
                                         [1, 2, 3, 8], A.gm([1, 2, 3, 8]), 2 ** 16, 991616, 1e-3)
                            e = float(L.predict_eps('M_K_both', [3.0, -5.0, 0.5, 0.5, 0.3, 0.9, kappa], X)[0]) * 2 ** rng.normal(0, 0.05)
                            rows.append(dict(seed=seed, target_id=f'c{cc}', K=cc + 1, K_bits='', alpha=alpha, T=2 ** k,
                                             step=2 ** k // 256, excess_bits=e, accuracy=0.5, n_test=2000, arm='transformer',
                                             format=fmt, N=sizes[size], size=size, c=cc, s=cc + 1, eval_stratum=f'c{cc}',
                                             eval_c=cc, **{f'digit_bits_{j}': e / 9 for j in range(9)}))
    import run as R
    R.write_csv(tmp_path / 'metrics.csv', rows)
    sp = A.spec('transformer', C)
    data_rows = A.read_rows(tmp_path)
    for r in data_rows:
        r['size'] = r['size']
    fits, sel = A.stage1(data_rows, sp, C, tmp_path)
    fits = {int(k): v for k, v in json.loads((tmp_path / 'fits.json').read_text())['fits'].items()}
    res = A.stage2(data_rows, sp, C, tmp_path, fits, sel)
    assert res['independent_fit']['status'] == 'AGREE', res['independent_fit']
    assert res['G7']['log_ratio']['estimate'] > 1  # kappa 0.8 (plain) against 0.2 (reversed)
    assert 'G6' in res and res['distinct_resamples'] == 10
    json.dumps(A._jsonable(res))
