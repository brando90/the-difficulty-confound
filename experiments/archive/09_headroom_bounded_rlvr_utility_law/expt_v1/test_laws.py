"""Synthetic checks of the frozen laws; no experimental outcomes or tuning."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from laws import ADDITIVE_LAWS, BOUNDED_LAWS, bounded_forecast, bounded_inputs, fit_law, predict_law
from independent_fit import objective, refit

ROOT = Path(__file__).resolve().parent
c = json.loads((ROOT / 'config.json').read_text())
rng = np.random.default_rng(99)


def synthetic_rows(n_rows, group_sizes):
    rows = []
    for _ in range(n_rows):
        acc0 = float(rng.uniform(0.0, 1.0))
        G = int(rng.choice(group_sizes))
        r = dict(
            parameter_count=float(np.exp(rng.uniform(8, 12))),
            steps=float(rng.choice([100, 300, 1000])),
            carry_chain=int(rng.choice([1, 2])),
            starting_bits=float(np.exp(rng.uniform(-3, 2))),
            probe_pass1=float(rng.uniform(0.01, 0.99)),
            probe_rho=float(rng.uniform(0.01, 1.0)),
            pretrain_stratum_fraction=float(rng.uniform(0.1, 0.9)),
            group_size=G,
            start_accuracy=acc0,
            n=1440,
        )
        r['rl_utility'] = float(rng.uniform(-acc0, 1 - acc0))
        rows.append(r)
    return rows


rows = synthetic_rows(108, [4, 16])

# 1. Additive laws are byte-for-byte Experiment 08's: identical coefficients on identical rows.
spec = importlib.util.spec_from_file_location(
    'laws08', next(p for p in ROOT.parents if (p / 'experiments').is_dir()) / 'experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1/laws.py'
)
laws08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(laws08)
for law, law08 in ADDITIVE_LAWS.items():
    assert np.allclose(fit_law(rows, law, c)['coefficients'], laws08.fit_utility_law(rows, law08, c)['coefficients'], atol=0, rtol=0)

# 2. Bounds hold for any coefficients and features, including extreme ones.
for law, (form, names) in BOUNDED_LAWS.items():
    _, z, s, acc0, n, _, _ = bounded_inputs(rows, law, c)
    for scale in (0.0, 1.0, 50.0, 1e6):
        for _ in range(20):
            theta = rng.normal(0, scale, z.shape[1] + 1)
            pred = bounded_forecast(form, theta, z * scale, s, acc0, n)
            assert np.isfinite(pred).all()
            assert (pred <= 1 - acc0 + 1e-12).all() and (pred >= -acc0 - 1e-12).all()
            if form == 'gain':
                assert (pred >= 0).all()
    # Zero budget means zero forecast utility.
    assert np.allclose(bounded_forecast(form, rng.normal(0, 1, z.shape[1] + 1), z, 0 * s, acc0, n), 0, atol=1e-6)

# 3. Known parameters are recovered from noiseless simulated outcomes, and an independent optimizer agrees.
for law, (form, names) in BOUNDED_LAWS.items():
    g = [4, 16] if 'log_group_size' in names else [16]
    sim = synthetic_rows(108, g)
    _, z, s, acc0, n, _, _ = bounded_inputs(sim, law, c)
    truth = rng.normal(0, 0.5, z.shape[1] + 1)
    for r, u in zip(sim, bounded_forecast(form, truth, z, s, acc0, n)):
        r['rl_utility'] = float(u)
    # Recovery is checked without the ridge penalty, which deliberately shrinks coefficients.
    fit = fit_law(sim, law, {**c, 'ridge_lambda': 0.0})
    pred = predict_law(fit, sim, law, c)
    rmse = np.sqrt(np.mean((pred - np.array([r['rl_utility'] for r in sim])) ** 2))
    assert fit['converged'] and rmse < 1e-4, (law, rmse)
    # Parity is checked at the frozen penalty.
    fit = fit_law(sim, law, c)
    y = np.array([r['rl_utility'] for r in sim])
    _, ind = refit(form, z, s, acc0, n, y, c['ridge_lambda'], c['fit_starts'])
    ours = objective(np.array(fit['coefficients']), form, z, s, acc0, n, y, c['ridge_lambda'])
    assert ours <= ind + 1e-9, (law, ours, ind)
print('PASS: additive laws match Experiment 08, bounded forecasts stay within [-Acc0, 1-Acc0] (gain >= 0), parameter recovery and independent-optimizer parity.')
