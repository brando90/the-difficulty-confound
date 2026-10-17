"""Synthetic regression checks; no experimental outcomes or tuning."""
import json
from pathlib import Path
import numpy as np
from laws import LAWS, fit_law, design
from independent_fit import reconstruct

c = json.loads(Path(__file__).with_name('config.json').read_text())
rng = np.random.default_rng(95)
rows = []
for _ in range(54):
    p = float(np.exp(rng.uniform(7, 11)))
    t = float(np.exp(rng.uniform(4.5, 7)))
    d = float(np.exp(rng.uniform(-1, 2.5)))
    q = float(rng.uniform(0.1, 0.9))
    carry = int(rng.choice([0, 2]))
    loss = np.exp(7 - 0.3 * np.log(p) - 0.4 * np.log(t) - 1.8 * np.log(d) - 0.25 * np.log(q) + 0.1 * (carry == 2))
    rows.append(
        dict(
            parameter_count=p,
            steps=t,
            starting_bits=d,
            pretrain_stratum_fraction=q,
            carry_chain=carry,
            mean_bits=loss,
        )
    )
for law in LAWS:
    fit = fit_law(rows, law, c)
    coef = np.array(fit['coefficients'])
    ind, names = reconstruct(rows, law, c)
    assert np.max(abs(coef - ind)) < 1e-8, (law, coef, ind)
    if law.startswith('stable'):
        for n in ['log_parameters', 'log_updates', 'log_pretrain_exposure']:
            if n in names:
                assert coef[names.index(n)] <= 1e-10
        if law.endswith('difficulty'):
            assert -1e-10 <= coef[-1] <= 1 + 1e-10
            original = rows[0]
            stress = {
                **original,
                'parameter_count': original['parameter_count'] * 1e9,
                'steps': original['steps'] * 1e9,
                'starting_bits': original['starting_bits'] * 1e-9,
            }
            z, _ = design([original, stress], law)
            assert (z @ coef)[1] <= (z @ coef)[0] + 1e-9
assert fit_law(rows, 'legacy_difficulty', c)['coefficients'][-1] < -1
print('PASS: all six independent fits, coefficient bounds, inverse-difficulty failure case and monotonic extrapolation stress.')
