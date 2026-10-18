"""Frozen RL utility laws: Experiment 08's additive laws (unchanged) and headroom-bounded laws."""
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.special import expit

# --- Experiment 08 additive utility laws, copied unchanged --------------------------------------------------

# Protocol names -> Experiment 08 law names.
ADDITIVE_LAWS = {'additive_base': 'utility_base', 'additive_signal_difficulty': 'utility_signal_difficulty'}


def utility_design(rows, law, c):
    t0 = float(c.get('utility_budget_half_steps', 300.0))
    sc = np.array([r['steps'] / (r['steps'] + t0) for r in rows], float)
    names = ['intercept', 'log_parameters', 's_budget', 'two_carry']
    cols = [
        np.ones(len(rows)),
        np.log([r['parameter_count'] for r in rows]),
        sc,
        np.array([r['carry_chain'] == 2 for r in rows], float),
    ]
    if law == 'utility_oracle_history':
        names.append('s_neg_log_pretrain_exposure')
        cols.append(sc * (-np.log([r['pretrain_stratum_fraction'] for r in rows])))
    elif law == 'utility_signal_coverage':
        names.extend(['s_log_probe_rho', 's_probe_headroom'])
        cols.extend(
            [
                sc * np.log(np.clip([r['probe_rho'] for r in rows], 1e-6, 1.0)),
                sc * np.array([1.0 - r['probe_pass1'] for r in rows], float),
            ]
        )
    elif law == 'utility_signal_difficulty':
        names.extend(['s_log_probe_rho', 's_probe_headroom', 's_log_starting_bits'])
        cols.extend(
            [
                sc * np.log(np.clip([r['probe_rho'] for r in rows], 1e-6, 1.0)),
                sc * np.array([1.0 - r['probe_pass1'] for r in rows], float),
                sc * np.log([r['starting_bits'] for r in rows]),
            ]
        )
    return np.column_stack(cols), names


def fit_utility_law(rows, law, c):
    x, names = utility_design(rows, law, c)
    y = np.array([r['rl_utility'] for r in rows], float)
    center = x.mean(0)
    center[0] = 0.0
    scale = x.std(0)
    scale[0] = 1.0
    assert (scale > 0).all()
    z = (x - center) / scale
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    a = np.vstack([z / np.sqrt(len(x)), np.sqrt(c['ridge_lambda']) * penalty])
    b = np.r_[y / np.sqrt(len(x)), np.zeros(x.shape[1])]
    lower = np.full(x.shape[1], -np.inf)
    upper = np.full(x.shape[1], np.inf)
    for pos_name in ['s_budget', 's_neg_log_pretrain_exposure', 's_probe_headroom', 's_log_starting_bits']:
        if pos_name in names:
            lower[names.index(pos_name)] = 0.0
    result = lsq_linear(a, b, bounds=(lower, upper), method='bvls', tol=1e-12, max_iter=1000)
    assert result.success, (result.status, result.message)
    coef = result.x
    beta = coef / scale
    beta[0] -= np.sum(coef * center / scale)
    return dict(
        columns=names,
        coefficients=beta.tolist(),
        standardized_coefficients=coef.tolist(),
        feature_center=center.tolist(),
        feature_scale=scale.tolist(),
    )


def predict_utility_law(fit, rows, law, c):
    z, _ = utility_design(rows, law, c)
    return z @ np.array(fit['coefficients'])


# --- Headroom-bounded laws ----------------------------------------------------------------------------------
# gain:  U = (1 - Acc0) * (1 - exp(-s(C) * exp(theta . z)))           in [0, 1 - Acc0)
# logit: U = sigmoid(logit(Acc0) + s(C) * (theta . z)) - Acc0          in [-Acc0, 1 - Acc0]
# theta[0] is the intercept; z are rate features standardized on the fit rows.

BASE = ['log_parameters', 'two_carry']
SIGNAL = ['log_probe_rho', 'probe_headroom', 'log_starting_bits']
BOUNDED_LAWS = {
    'bounded_base': ('gain', BASE),
    'bounded_signal_difficulty': ('gain', BASE + SIGNAL),
    'logit_base': ('logit', BASE),
    'logit_signal_difficulty': ('logit', BASE + SIGNAL),
    # Group-size axis: both laws know G; the signal law adds log expected informative prompts per update,
    # log(B * rho(G)) = log(rollouts) - log G + log rho(G), so it tests rho(G) beyond G itself.
    'bounded_base_g': ('gain', BASE + ['log_group_size']),
    'bounded_signal_g': ('gain', BASE + ['log_group_size', 'log_probe_informative', 'probe_headroom', 'log_starting_bits']),
}


def bounded_features(rows, names, c):
    cols = {
        'log_parameters': lambda r: np.log(r['parameter_count']),
        'two_carry': lambda r: float(r['carry_chain'] == 2),
        'log_probe_rho': lambda r: np.log(np.clip(r['probe_rho'], 1e-6, 1.0)),
        'probe_headroom': lambda r: 1.0 - r['probe_pass1'],
        'log_starting_bits': lambda r: np.log(r['starting_bits']),
        'log_group_size': lambda r: np.log(r['group_size']),
        'log_probe_informative': lambda r: np.log(
            c['rollouts_per_update'] / r['group_size'] * np.clip(r['probe_rho'], 1e-6, 1.0)
        ),
    }
    return np.array([[cols[n](r) for n in names] for r in rows], float)


def budget(rows, c):
    return np.array([r['steps'] / (r['steps'] + c['utility_budget_half_steps']) for r in rows], float)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)  # finite at 0% and 100%; forecast at zero budget is within 1e-6 of zero
    return np.log(p / (1 - p))


def bounded_forecast(form, theta, z, s, acc0, n):
    linear = theta[0] + z @ theta[1:]
    if form == 'gain':
        rate = np.exp(np.clip(linear, -30, 30))  # clip only guards overflow
        return (1 - acc0) * (1 - np.exp(-s * rate))
    return expit(logit(acc0) + s * linear) - acc0


def bounded_objective(theta, form, z, s, acc0, n, y, lam):
    """Residual vector whose squared norm is mean squared error plus ridge on non-intercept coefficients."""
    fit = (bounded_forecast(form, theta, z, s, acc0, n) - y) / np.sqrt(len(y))
    return np.r_[fit, np.sqrt(lam) * theta[1:]]


def bounded_inputs(rows, law, c, center=None, scale=None):
    form, names = BOUNDED_LAWS[law]
    x = bounded_features(rows, names, c)
    if center is None:
        center, scale = x.mean(0), x.std(0)
        assert (scale > 0).all(), (law, names, scale)
    acc0 = np.array([r['start_accuracy'] for r in rows], float)
    n = np.array([r['n'] for r in rows], float)
    return form, (x - center) / scale, budget(rows, c), acc0, n, center, scale


def fit_bounded_law(rows, law, c):
    form, z, s, acc0, n, center, scale = bounded_inputs(rows, law, c)
    y = np.array([r['rl_utility'] for r in rows], float)
    best = None
    for start in c['fit_starts']:  # keep the lowest objective over a few intercept starts
        theta0 = np.zeros(z.shape[1] + 1)
        theta0[0] = start
        res = least_squares(bounded_objective, theta0, args=(form, z, s, acc0, n, y, c['ridge_lambda']), method='trf', max_nfev=20000)
        if best is None or res.cost < best.cost:
            best = res
    return dict(
        columns=['intercept'] + BOUNDED_LAWS[law][1],
        form=form,
        coefficients=best.x.tolist(),
        feature_center=center.tolist(),
        feature_scale=scale.tolist(),
        objective=float(2 * best.cost),
        converged=bool(best.success),
    )


def predict_bounded_law(fit, rows, law, c):
    form, z, s, acc0, n, _, _ = bounded_inputs(rows, law, c, np.array(fit['feature_center']), np.array(fit['feature_scale']))
    return bounded_forecast(form, np.array(fit['coefficients']), z, s, acc0, n)


def fit_law(rows, law, c):
    return fit_utility_law(rows, ADDITIVE_LAWS[law], c) if law in ADDITIVE_LAWS else fit_bounded_law(rows, law, c)


def predict_law(fit, rows, law, c):
    if law in ADDITIVE_LAWS:
        return predict_utility_law(fit, rows, ADDITIVE_LAWS[law], c)
    return predict_bounded_law(fit, rows, law, c)
