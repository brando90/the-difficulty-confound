"""EXPLORATORY, not prospective: candidate headroom-bounded RL utility laws refit on Experiment 08's observed outcomes.

Experiment 08's outcomes were already published, so nothing here is evidence for any law. This script only
(1) reproduces Experiment 08's additive utility-law RMSEs as a sanity check and (2) shows how candidate bounded
forms behave, to help choose which forms Experiment 09 freezes before training fresh models.

Every law is fitted per seed on the same 54 calibration fit rows Experiment 08 used (fit widths 32/64/128,
fit mixtures .1/.5/.9, checkpoints 100/300/1000, both strata) and scored on the same 192 joint held-out test
targets (widths 96/384 x mixtures .2/.8), with the same seed-block bootstrap.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[3]
E08 = REPO / 'experiments/archive/08_rlvr_verifiable_reward_utility_laws/expt_v1'
OUT = Path(__file__).resolve().parent / 'results'
sys.path.insert(0, str(E08))
from laws import fit_utility_law, utility_design  # noqa: E402  (Experiment 08's frozen additive laws)

KEYS = ['seed', 'width', 'mixture', 'steps', 'carry_chain']
BASE = ['log_parameters', 'two_carry']
SIGNAL = BASE + ['log_probe_rho', 'probe_headroom', 'log_starting_bits']
BOUNDED = {
    # name: (form, rate features)
    'gain_base': ('gain', BASE),
    'gain_signal_difficulty': ('gain', SIGNAL),
    'gain_rho': ('gain', BASE + ['log_probe_rho']),  # single-feature ablations of the gain form
    'gain_headroom': ('gain', BASE + ['probe_headroom']),
    'gain_difficulty': ('gain', BASE + ['log_starting_bits']),
    'twosided_base': ('twosided', BASE),
    'twosided_signal_difficulty': ('twosided', SIGNAL),
    'logit_base': ('logit', BASE),
    'logit_signal_difficulty': ('logit', SIGNAL),
}
ADDITIVE = ['utility_base', 'utility_signal_difficulty']
COMPARISONS = [
    ('utility_base', 'utility_signal_difficulty'),  # Experiment 08 Gate 2 (reproduction)
    ('utility_base', 'gain_base'),  # structural repair: cap only
    ('utility_base', 'twosided_base'),
    ('utility_base', 'logit_base'),
    ('gain_base', 'gain_signal_difficulty'),  # feature value inside the bounded form
    ('gain_base', 'gain_rho'),
    ('gain_base', 'gain_headroom'),
    ('gain_base', 'gain_difficulty'),
    ('twosided_base', 'twosided_signal_difficulty'),
    ('logit_base', 'logit_signal_difficulty'),
    ('utility_signal_difficulty', 'logit_signal_difficulty'),
]


def read(path):
    with Path(path).open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def key(r):
    return tuple(r[k] for k in KEYS)


def axis(r, c):
    w = r['width'] in c['heldout_widths']
    q = r['mixture'] in c['heldout_mixtures']
    return 'joint' if w and q else 'model_only' if w else 'mixture_only' if q else 'fit'


def add_accuracies(rows):
    for r in rows:
        r['acc_T'] = r['correct'] / r['n']
        r['acc_0'] = r['acc_T'] - r['rl_utility']  # rl_utility = acc_T - acc_0 on the same split


def raw_features(rows, names):
    cols = {
        'log_parameters': np.log([r['parameter_count'] for r in rows]),
        'two_carry': np.array([r['carry_chain'] == 2 for r in rows], float),
        'log_probe_rho': np.log(np.clip([r['probe_rho'] for r in rows], 1e-6, 1.0)),
        'probe_headroom': np.array([1.0 - r['probe_pass1'] for r in rows]),
        'log_starting_bits': np.log([r['starting_bits'] for r in rows]),
    }
    return np.column_stack([cols[n] for n in names])


def logit(p, n):
    p = np.clip(p, 0.5 / (n + 1), 1 - 0.5 / (n + 1))  # avoid infinite logits at exactly 0% or 100%
    return np.log(p / (1 - p))


def predict(form, theta, z, s, acc0, n):
    k = z.shape[1] + 1
    rate = np.exp(np.clip(theta[0] + z @ theta[1:k], -30, 30))
    if form == 'gain':
        return (1 - acc0) * (1 - np.exp(-s * rate))
    if form == 'twosided':
        forget = np.exp(np.clip(theta[k], -30, 30))
        return (1 - acc0) * (1 - np.exp(-s * rate)) - acc0 * (1 - np.exp(-s * forget))
    shift = theta[0] + z @ theta[1:k]  # logit form: signed shift, scaled by budget
    return 1 / (1 + np.exp(-(logit(acc0, n) + s * shift))) - acc0


def fit_bounded(rows, form, names, c):
    x = raw_features(rows, names)
    center, scale = x.mean(0), x.std(0)
    assert (scale > 0).all()
    z = (x - center) / scale
    t0 = c['utility_budget_half_steps']
    s = np.array([r['steps'] / (r['steps'] + t0) for r in rows])
    y = np.array([r['rl_utility'] for r in rows])
    acc0 = np.array([r['acc_0'] for r in rows])
    n = np.array([r['n'] for r in rows])
    k = len(names) + 1
    n_par = k + (form == 'twosided')
    lam = c['ridge_lambda']

    def residuals(theta):
        fit = (predict(form, theta, z, s, acc0, n) - y) / np.sqrt(len(y))
        penalty = np.sqrt(lam) * theta[1:k]  # same light ridge as Experiment 08; intercepts unpenalized
        return np.r_[fit, penalty]

    best = None
    for start in (-2.0, 0.0, 2.0):  # a few starts, keep the lowest objective
        theta0 = np.zeros(n_par)
        theta0[0] = start
        if form == 'twosided':
            theta0[k] = -3.0
        res = least_squares(residuals, theta0, method='trf', max_nfev=20000)
        if best is None or res.cost < best.cost:
            best = res
    return dict(theta=best.x, center=center, scale=scale, names=names, form=form, success=bool(best.success))


def forecast_bounded(fit, rows, c):
    z = (raw_features(rows, fit['names']) - fit['center']) / fit['scale']
    s = np.array([r['steps'] / (r['steps'] + c['utility_budget_half_steps']) for r in rows])
    acc0 = np.array([r['acc_0'] for r in rows])
    n = np.array([r['n'] for r in rows])
    return predict(fit['form'], fit['theta'], z, s, acc0, n)


def interval(point, samples):
    return dict(estimate=float(point), lo=float(np.quantile(samples, 0.025)), hi=float(np.quantile(samples, 0.975)))


def main():
    c = json.loads((E08 / 'config.json').read_text())
    cal = read(E08 / 'results/calibration_metrics.csv')
    test = read(E08 / 'results/test_metrics.csv')
    add_accuracies(cal)
    add_accuracies(test)
    test_map = {key(r): r for r in test}
    preds = []
    fit_failures = []
    for seed in c['seeds']:
        training = [r for r in cal if r['seed'] == seed and axis(r, c) == 'fit']
        targets = [r for r in test if r['seed'] == seed and axis(r, c) == 'joint']
        # Experiment 08 forecast from calibration-row features; probe features are identical across splits.
        feature_rows = [next(x for x in cal if key(x) == key(t)) for t in targets]
        assert len(training) == 54 and len(targets) == 24
        values = {}
        for law in ADDITIVE:
            fit = fit_utility_law(training, law, c)
            zz, _ = utility_design(feature_rows, law, c)
            values[law] = zz @ np.array(fit['coefficients'])
        for law, (form, names) in BOUNDED.items():
            fit = fit_bounded(training, form, names, c)
            if not fit['success']:
                fit_failures.append((seed, law))
            values[law] = forecast_bounded(fit, targets, c)  # cap uses each target's own pre-RL accuracy
        for i, t in enumerate(targets):
            obs = test_map[key(t)]['rl_utility']
            for law, v in values.items():
                preds.append(
                    dict(
                        seed=int(seed),
                        width=int(t['width']),
                        mixture=t['mixture'],
                        steps=int(t['steps']),
                        carry_chain=int(t['carry_chain']),
                        law=law,
                        acc_0=t['acc_0'],
                        predicted_utility=float(v[i]),
                        observed_utility=obs,
                        error=float(v[i] - obs),
                        above_headroom=bool(v[i] > 1 - t['acc_0'] + 1e-12),
                        below_floor=bool(v[i] < -t['acc_0'] - 1e-12),
                    )
                )

    laws = ADDITIVE + list(BOUNDED)
    draws = np.random.default_rng(c['analysis_seed']).integers(0, len(c['seeds']), (c['bootstrap_draws'], len(c['seeds'])))
    groups = {
        'joint': lambda r: True,
        'width_96': lambda r: r['width'] == 96,
        'width_384': lambda r: r['width'] == 384,
        'step_100': lambda r: r['steps'] == 100,
        'step_300': lambda r: r['steps'] == 300,
        'step_1000': lambda r: r['steps'] == 1000,
    }
    summary = dict(note='EXPLORATORY refit on already-observed Experiment 08 outcomes; not prospective evidence.',
                   fit_failures=fit_failures, rmse_pp={}, relative_rmse_reduction_pct={}, out_of_bounds={})
    for g, keep in groups.items():
        rows = [r for r in preds if keep(r)]
        mse = {law: np.array([np.mean([r['error'] ** 2 for r in rows if r['law'] == law and r['seed'] == s])
                              for s in c['seeds']]) for law in laws}
        summary['rmse_pp'][g] = {law: interval(100 * np.sqrt(mse[law].mean()), 100 * np.sqrt(mse[law][draws].mean(1)))
                                 for law in laws}
        summary['relative_rmse_reduction_pct'][g] = {
            f'{b}->{e}': interval(100 * (1 - np.sqrt(mse[e].mean() / mse[b].mean())),
                                  100 * (1 - np.sqrt(mse[e][draws].mean(1) / mse[b][draws].mean(1))))
            for b, e in COMPARISONS
        }
        summary['out_of_bounds'][g] = {
            law: dict(n=sum(r['law'] == law for r in rows),
                      above_headroom=sum(r['above_headroom'] for r in rows if r['law'] == law),
                      below_floor=sum(r['below_floor'] for r in rows if r['law'] == law))
            for law in laws
        }

    OUT.mkdir(exist_ok=True)
    with (OUT / 'exploratory_forecasts.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0]))
        w.writeheader()
        w.writerows(preds)
    (OUT / 'exploratory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    fmt = lambda d: f"{d['estimate']:7.2f} [{d['lo']:6.2f}, {d['hi']:6.2f}]"
    print(summary['note'])
    print('fit failures:', fit_failures or 'none')
    print('\nRMSE in percentage points (lower is better)')
    print(f"{'law':28s}" + ''.join(f'{g:>27s}' for g in ('joint', 'width_96', 'width_384')))
    for law in laws:
        print(f'{law:28s}' + ''.join(f"{fmt(summary['rmse_pp'][g][law]):>27s}" for g in ('joint', 'width_96', 'width_384')))
    print('\nRelative RMSE reduction, % (positive = extended law better)')
    for comp in summary['relative_rmse_reduction_pct']['joint']:
        print(f'{comp:52s}' + ''.join(f"{fmt(summary['relative_rmse_reduction_pct'][g][comp]):>27s}"
                                     for g in ('joint', 'width_96', 'width_384')))
    print('\nRMSE by RL checkpoint (pp)')
    for law in laws:
        print(f'{law:28s}' + ''.join(f"{summary['rmse_pp'][g][law]['estimate']:9.2f}" for g in ('step_100', 'step_300', 'step_1000')))
    # Sampling-noise floor: even a perfect law misses observed U by roughly the binomial noise of two test accuracies.
    print('\nApproximate sampling-noise floor of observed U (pp, independent-binomial upper bound)')
    for g in ('width_96', 'width_384'):
        rows = [r for r in preds if groups[g](r) and r['law'] == laws[0]]
        var = [(r['observed_utility'] + r['acc_0']) * (1 - r['observed_utility'] - r['acc_0']) / 1440
               + r['acc_0'] * (1 - r['acc_0']) / 1440 for r in rows]
        print(f'{g:12s} {100 * np.sqrt(np.mean(var)):.2f}')
    print('\nImpossible forecasts on 192 joint targets (above remaining headroom / below -Acc_0)')
    for law in laws:
        o = summary['out_of_bounds']['joint'][law]
        print(f"{law:28s} {o['above_headroom']:4d} {o['below_floor']:4d}")


if __name__ == '__main__':
    main()
