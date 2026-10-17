"""Prespecified paired contrasts, pure 0/1 RLVR remediation & retention, RL utility laws, and constrained held-out forecasts."""
import csv
import json
from pathlib import Path
import numpy as np
from addition import save_json, sha, write_csv

ROOT = Path(__file__).resolve().parent
from laws import LAWS, UTILITY_LAWS, design, fit_law, fit_utility_law, utility_design

KEYS = ['seed', 'width', 'mixture', 'steps', 'carry_chain']
COMPARISONS = [
    ('stable_base', 'stable_difficulty'),
    ('stable_base', 'stable_oracle_history'),
    ('legacy_difficulty', 'stable_difficulty'),
    ('legacy_difficulty', 'ridge_difficulty'),
    ('legacy_base', 'legacy_difficulty'),
    ('legacy_base', 'stable_base'),
    ('ridge_difficulty', 'stable_difficulty'),
    ('stable_oracle_history', 'stable_difficulty'),
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


def boot_draws(c):
    return np.random.default_rng(c['analysis_seed']).integers(0, len(c['seeds']), (c['bootstrap_draws'], len(c['seeds'])))


def interval(value, samples):
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(estimate=float(value), lo=float(lo), hi=float(hi))


def freeze(out, c):
    cal = read(out / 'calibration_metrics.csv')
    assert len(cal) == 1200 and len(set(map(key, cal))) == 1200
    outcome_cols = {'mean_bits', 'correct', 'n', 'online_active_group_fraction', 'rl_utility'}
    features = [{k: v for k, v in r.items() if k not in outcome_cols} for r in cal]
    assert all(not outcome_cols & r.keys() for r in features)
    fits = []
    predictions = []
    for seed in c['seeds']:
        training = [r for r in cal if r['seed'] == seed and axis(r, c) == 'fit']
        targets = [r for r in features if r['seed'] == seed and axis(r, c) != 'fit']
        assert len(training) == 54 and len(targets) == 96
        for law in LAWS:
            fit = fit_law(training, law, c)
            z, _ = design(targets, law)
            with np.errstate(over='ignore', under='ignore'):
                pred = np.exp(z @ np.array(fit['coefficients']))
            fits.append(dict(seed=seed, law=law, training_keys=[key(r) for r in training], **fit))
            predictions.extend(
                {**{k: r[k] for k in KEYS}, 'axis': axis(r, c), 'law': law, 'predicted_bits': float(v)}
                for r, v in zip(targets, pred)
            )
    frozen = dict(
        config_sha256=sha(ROOT / 'config.json'),
        analysis_sha256=sha(__file__),
        laws_sha256=sha(ROOT / 'laws.py'),
        calibration_sha256=sha(out / 'calibration_metrics.csv'),
        fits=fits,
    )
    target = out / 'frozen_forecasts.json'
    if target.exists():
        assert json.loads(target.read_text()) == frozen, 'Frozen forecasts changed'
    else:
        save_json(target, frozen)
    write_csv(out / 'forecasts.csv', predictions)
    return predictions


def intervention(out, c, pre_test, cell_test, draws):
    table = {(r['seed'], r['width'], r['mixture'], r['carry_chain']): r['correct'] / r['n'] for r in pre_test}
    ctable = {(r['seed'], r['width'], r['mixture'], int(r['cell_id'])): r['correct'] / r['n'] for r in cell_test if r['steps'] == 0}
    rows = []
    widths = c['train_widths'] + c['heldout_widths']
    for seed in c['seeds']:
        for w in widths:
            def acc(q, s):
                return table[seed, w, q, s]

            def cacc(q, cid):
                return ctable[seed, w, q, cid]

            low = acc(0.1, 1) - acc(0.1, 2)
            high = acc(0.9, 2) - acc(0.9, 1)
            pure_low = 0.5 * (cacc(0.1, 1) + cacc(0.1, 2)) - cacc(0.1, 3)
            pure_high = cacc(0.9, 3) - 0.5 * (cacc(0.9, 1) + cacc(0.9, 2))
            rows.append(
                dict(
                    seed=seed,
                    width=w,
                    e_low=low,
                    e_high=high,
                    alignment=(low + high) / 2,
                    structure=(low - high) / 2,
                    balanced_gap=acc(0.5, 1) - acc(0.5, 2),
                    pure_low=pure_low,
                    pure_high=pure_high,
                    pure_alignment=(pure_low + pure_high) / 2,
                    pure_structure=(pure_low - pure_high) / 2,
                )
            )
    write_csv(out / 'intervention_seed_width.csv', rows)
    stat_names = [
        'e_low',
        'e_high',
        'alignment',
        'structure',
        'balanced_gap',
        'pure_low',
        'pure_high',
        'pure_alignment',
        'pure_structure',
    ]
    seedrows = [
        {name: float(np.mean([r[name] for r in rows if r['seed'] == s])) for name in stat_names}
        for s in c['seeds']
    ]
    summaries = {}
    for name in stat_names:
        values = np.array([r[name] for r in seedrows])
        summaries[name] = interval(values.mean(), values[draws].mean(1))
    bywidth = {
        str(w): {n: float(np.mean([r[n] for r in rows if r['width'] == w])) for n in stat_names}
        for w in widths
    }
    supported = all(
        summaries[n]['estimate'] >= c['crossover_min_gap'] and summaries[n]['lo'] > 0 for n in ['e_low', 'e_high']
    ) and all(bywidth[str(w)][n] > 0 for w in widths for n in ['e_low', 'e_high'])
    pure_supported = all(
        summaries[n]['estimate'] >= c['pure_carry_crossover_min_gap'] and summaries[n]['lo'] > 0
        for n in ['pure_low', 'pure_high']
    ) and all(bywidth[str(w)][n] > 0 for w in widths for n in ['pure_low', 'pure_high'])
    obj = dict(
        status='SUPPORTED' if supported else 'NOT_ESTABLISHED',
        pure_carry_status='SUPPORTED' if pure_supported else 'NOT_ESTABLISHED',
        n_seed_blocks=8,
        n_widths=5,
        test_items_per_stratum=720,
        test_items_per_cell=360,
        interval='95% percentile bootstrap of complete seed blocks; fixed test set',
        p_value=None,
        threshold=c['crossover_min_gap'],
        pure_carry_threshold=c['pure_carry_crossover_min_gap'],
        statistics=summaries,
        by_width=bywidth,
    )
    save_json(out / 'intervention_summary.json', obj)
    return obj


def rl_utility_and_signal(out, c, post_test, draws):
    step = max(c['checkpoints'])
    u_table = {
        (r['seed'], r['width'], r['mixture'], r['carry_chain']): r['rl_utility']
        for r in post_test
        if r['steps'] == step
    }
    act_table = {
        (r['seed'], r['width'], r['mixture'], r['carry_chain']): r['online_active_group_fraction']
        for r in post_test
        if r['steps'] == step
    }
    rho_table = {
        (r['seed'], r['width'], r['mixture'], r['carry_chain']): r['probe_rho']
        for r in post_test
        if r['steps'] == step
    }
    widths = c['train_widths'] + c['heldout_widths']
    rows = []
    for seed in c['seeds']:
        for w in widths:
            # At q_pre=0.1, s=2 (paired zero/double carry) was scarce so RL utility U on s=2 > s=1
            u_low = u_table[seed, w, 0.1, 2] - u_table[seed, w, 0.1, 1]
            # At q_pre=0.9, s=1 (isolated single carry) was scarce so RL utility U on s=1 > s=2
            u_high = u_table[seed, w, 0.9, 1] - u_table[seed, w, 0.9, 2]
            rows.append(
                dict(
                    seed=seed,
                    width=w,
                    u_low=u_low,
                    u_high=u_high,
                    u_alignment=(u_low + u_high) / 2,
                    u_s1_q01=u_table[seed, w, 0.1, 1],
                    u_s2_q01=u_table[seed, w, 0.1, 2],
                    u_s1_q05=u_table[seed, w, 0.5, 1],
                    u_s2_q05=u_table[seed, w, 0.5, 2],
                    u_s1_q09=u_table[seed, w, 0.9, 1],
                    u_s2_q09=u_table[seed, w, 0.9, 2],
                    act_s1_q01=act_table[seed, w, 0.1, 1],
                    act_s2_q01=act_table[seed, w, 0.1, 2],
                    act_s1_q09=act_table[seed, w, 0.9, 1],
                    act_s2_q09=act_table[seed, w, 0.9, 2],
                    rho_s1_q01=rho_table[seed, w, 0.1, 1],
                    rho_s2_q01=rho_table[seed, w, 0.1, 2],
                    rho_s1_q09=rho_table[seed, w, 0.9, 1],
                    rho_s2_q09=rho_table[seed, w, 0.9, 2],
                )
            )
    write_csv(out / 'rl_utility_seed_width.csv', rows)
    stat_names = [k for k in rows[0] if k not in ('seed', 'width')]
    seedrows = [
        {name: float(np.mean([r[name] for r in rows if r['seed'] == s])) for name in stat_names}
        for s in c['seeds']
    ]
    summaries = {}
    for name in stat_names:
        values = np.array([r[name] for r in seedrows])
        summaries[name] = interval(values.mean(), values[draws].mean(1))
    bywidth = {
        str(w): {n: float(np.mean([r[n] for r in rows if r['width'] == w])) for n in stat_names}
        for w in widths
    }
    u_reversal_supported = (
        all(summaries[n]['estimate'] >= 0.10 and summaries[n]['lo'] > 0 for n in ['u_low', 'u_high'])
        and all(bywidth[str(w)][n] > 0 for w in widths for n in ['u_low', 'u_high'])
    )

    # Fit RL Environment Utility Scaling Laws on calibration fit rows and evaluate on held-out test rows
    cal = read(out / 'calibration_metrics.csv')
    test_map = {key(r): r for r in post_test}
    u_preds = []
    for seed in c['seeds']:
        training = [r for r in cal if r['seed'] == seed and axis(r, c) == 'fit']
        targets = [r for r in cal if r['seed'] == seed and axis(r, c) != 'fit']
        for law in UTILITY_LAWS:
            fit = fit_utility_law(training, law, c)
            z, _ = utility_design(targets, law, c)
            pred = z @ np.array(fit['coefficients'])
            for r, v in zip(targets, pred):
                obs = test_map[key(r)]['rl_utility']
                u_preds.append(
                    dict(
                        seed=seed,
                        width=r['width'],
                        mixture=r['mixture'],
                        steps=r['steps'],
                        carry_chain=r['carry_chain'],
                        axis=axis(r, c),
                        law=law,
                        predicted_utility=float(v),
                        observed_utility=float(obs),
                        error=float(v - obs),
                    )
                )
    write_csv(out / 'scored_utility_forecasts.csv', u_preds)
    u_metrics = {}
    u_comparisons = {}
    for group in ['joint', 'joint_width_96', 'joint_width_384']:
        rr = (
            [r for r in u_preds if r['axis'] == 'joint']
            if group == 'joint'
            else [r for r in u_preds if r['axis'] == 'joint' and r['width'] == int(group.rsplit('_', 1)[1])]
        )
        mses = {
            law: np.array([np.mean([r['error'] ** 2 for r in rr if r['law'] == law and r['seed'] == s]) for s in c['seeds']])
            for law in UTILITY_LAWS
        }
        u_metrics[group] = {
            law: interval(np.sqrt(mses[law].mean()), np.sqrt(mses[law][draws].mean(1))) for law in UTILITY_LAWS
        }
        u_comparisons[group] = {}
        for base, ext in [
            ('utility_base', 'utility_signal_coverage'),
            ('utility_base', 'utility_signal_difficulty'),
            ('utility_base', 'utility_oracle_history'),
            ('utility_oracle_history', 'utility_signal_difficulty'),
        ]:
            pt = 1 - np.sqrt(mses[ext].mean() / mses[base].mean())
            bt = 1 - np.sqrt(mses[ext][draws].mean(1) / mses[base][draws].mean(1))
            u_comparisons[group][f'{base}_to_{ext}'] = interval(pt, bt)

    main_gain = u_comparisons['joint']['utility_base_to_utility_signal_difficulty']
    w96_gain = u_comparisons['joint_width_96']['utility_base_to_utility_signal_difficulty']
    w384_gain = u_comparisons['joint_width_384']['utility_base_to_utility_signal_difficulty']
    u_law_supported = (
        main_gain['estimate'] >= c['predictive_min_relative_rmse_gain']
        and main_gain['lo'] > 0
        and w96_gain['estimate'] > 0
        and w384_gain['estimate'] > 0
    )
    obj = dict(
        utility_reversal_status='SUPPORTED' if u_reversal_supported else 'NOT_ESTABLISHED',
        utility_law_status='SUPPORTED' if u_law_supported else 'NOT_ESTABLISHED',
        statistics=summaries,
        by_width=bywidth,
        utility_law_metrics=u_metrics,
        utility_law_comparisons=u_comparisons,
    )
    save_json(out / 'rl_utility_summary.json', obj)
    return obj


def retention(out, c, pre_test, post_test, cell_test, draws):
    pre = {(r['seed'], r['width'], r['mixture'], r['carry_chain']): r['correct'] / r['n'] for r in pre_test}
    step = max(c['checkpoints'])
    post = {(r['seed'], r['width'], r['mixture'], r['carry_chain']): r['correct'] / r['n'] for r in post_test if r['steps'] == step}
    cpre = {(r['seed'], r['width'], r['mixture'], int(r['cell_id'])): r['correct'] / r['n'] for r in cell_test if r['steps'] == 0}
    cpost = {(r['seed'], r['width'], r['mixture'], int(r['cell_id'])): r['correct'] / r['n'] for r in cell_test if r['steps'] == step}
    widths = c['train_widths'] + c['heldout_widths']
    rows = []
    for seed in c['seeds']:
        for w in widths:
            c1_pre, t2_pre = pre[seed, w, 0.1, 1], pre[seed, w, 0.1, 2]
            c1_post, t2_post = post[seed, w, 0.1, 1], post[seed, w, 0.1, 2]
            gap0 = c1_pre - t2_pre
            gain2 = t2_post - t2_pre
            drop1 = c1_post - c1_pre
            pc_pre = 0.5 * (cpre[seed, w, 0.1, 1] + cpre[seed, w, 0.1, 2])
            pt_pre = cpre[seed, w, 0.1, 3]
            pc_post = 0.5 * (cpost[seed, w, 0.1, 1] + cpost[seed, w, 0.1, 2])
            pt_post = cpost[seed, w, 0.1, 3]
            pgap0 = pc_pre - pt_pre
            pgain = pt_post - pt_pre
            pdrop = pc_post - pc_pre
            rows.append(
                dict(
                    seed=seed,
                    width=w,
                    control_pre=c1_pre,
                    target_pre=t2_pre,
                    control_post=c1_post,
                    target_post=t2_post,
                    initial_gap=gap0,
                    target_gain=gain2,
                    recovery_fraction=gain2 / gap0 if gap0 > 0 else 0.0,
                    control_change=drop1,
                    pure_control_pre=pc_pre,
                    pure_target_pre=pt_pre,
                    pure_control_post=pc_post,
                    pure_target_post=pt_post,
                    pure_initial_gap=pgap0,
                    pure_target_gain=pgain,
                    pure_recovery_fraction=pgain / pgap0 if pgap0 > 0 else 0.0,
                    pure_control_change=pdrop,
                )
            )
    write_csv(out / 'retention_seed_width.csv', rows)
    seed_stats = []
    for s in c['seeds']:
        sr = [r for r in rows if r['seed'] == s]
        ig = float(np.mean([r['initial_gap'] for r in sr]))
        tg = float(np.mean([r['target_gain'] for r in sr]))
        pig = float(np.mean([r['pure_initial_gap'] for r in sr]))
        ptg = float(np.mean([r['pure_target_gain'] for r in sr]))
        seed_stats.append(
            dict(
                seed=s,
                control_pre=float(np.mean([r['control_pre'] for r in sr])),
                target_pre=float(np.mean([r['target_pre'] for r in sr])),
                control_post=float(np.mean([r['control_post'] for r in sr])),
                target_post=float(np.mean([r['target_post'] for r in sr])),
                initial_gap=ig,
                target_gain=tg,
                recovery_fraction=tg / ig,
                control_change=float(np.mean([r['control_change'] for r in sr])),
                pure_control_pre=float(np.mean([r['pure_control_pre'] for r in sr])),
                pure_target_pre=float(np.mean([r['pure_target_pre'] for r in sr])),
                pure_control_post=float(np.mean([r['pure_control_post'] for r in sr])),
                pure_target_post=float(np.mean([r['pure_target_post'] for r in sr])),
                pure_initial_gap=pig,
                pure_target_gain=ptg,
                pure_recovery_fraction=ptg / pig,
                pure_control_change=float(np.mean([r['pure_control_change'] for r in sr])),
            )
        )
    summaries = {}
    for name in [k for k in seed_stats[0] if k != 'seed']:
        vals = np.array([r[name] for r in seed_stats])
        summaries[name] = interval(vals.mean(), vals[draws].mean(1))
    ig_v = np.array([r['initial_gap'] for r in seed_stats])
    tg_v = np.array([r['target_gain'] for r in seed_stats])
    summaries['recovery_fraction'] = interval(tg_v.mean() / ig_v.mean(), tg_v[draws].mean(1) / ig_v[draws].mean(1))
    pig_v = np.array([r['pure_initial_gap'] for r in seed_stats])
    ptg_v = np.array([r['pure_target_gain'] for r in seed_stats])
    summaries['pure_recovery_fraction'] = interval(ptg_v.mean() / pig_v.mean(), ptg_v[draws].mean(1) / pig_v[draws].mean(1))
    by_width = {
        str(w): {
            n: float(np.mean([r[n] for r in rows if r['width'] == w]))
            for n in [
                'control_pre',
                'target_pre',
                'control_post',
                'target_post',
                'initial_gap',
                'target_gain',
                'recovery_fraction',
                'control_change',
                'pure_control_pre',
                'pure_target_pre',
                'pure_control_post',
                'pure_target_post',
                'pure_initial_gap',
                'pure_target_gain',
                'pure_recovery_fraction',
                'pure_control_change',
            ]
        }
        for w in widths
    }
    passed = (
        summaries['pure_initial_gap']['lo'] >= c['retention_min_initial_gap']
        and summaries['pure_recovery_fraction']['lo'] >= c['retention_min_recovery_fraction']
        and summaries['pure_control_change']['estimate'] >= -c['retention_max_control_drop']
        and summaries['pure_control_change']['lo'] >= -c['retention_max_control_drop']
    )
    obj = dict(
        status='SUPPORTED' if passed else 'NOT_ESTABLISHED',
        n_seed_blocks=8,
        n_widths=5,
        statistics=summaries,
        by_width=by_width,
        criteria=dict(
            min_initial_gap=c['retention_min_initial_gap'],
            min_recovery_fraction=c['retention_min_recovery_fraction'],
            max_control_drop=c['retention_max_control_drop'],
        ),
    )
    save_json(out / 'retention_summary.json', obj)
    return obj


def prediction(out, c, test, forecasts, draws):
    target = {key(r): r for r in test}
    scored = [
        {**r, 'observed_bits': target[key(r)]['mean_bits'], 'error': r['predicted_bits'] - target[key(r)]['mean_bits']}
        for r in forecasts
    ]
    write_csv(out / 'scored_forecasts.csv', scored)
    metrics = []
    comparisons = []
    for group in ['joint', 'model_only', 'mixture_only'] + [f'joint_width_{w}' for w in c['heldout_widths']]:
        rr = (
            [r for r in scored if r['axis'] == group]
            if group in ['joint', 'model_only', 'mixture_only']
            else [r for r in scored if r['axis'] == 'joint' and r['width'] == int(group.rsplit('_', 1)[1])]
        )
        mses = {}
        for law in LAWS:
            rows = [r for r in rr if r['law'] == law]
            err = np.array([r['error'] for r in rows])
            obs = np.array([r['observed_bits'] for r in rows])
            pred = np.array([r['predicted_bits'] for r in rows])
            mses[law] = np.array([np.mean([r['error'] ** 2 for r in rows if r['seed'] == s]) for s in c['seeds']])
            ci = interval(np.sqrt(mses[law].mean()), np.sqrt(mses[law][draws].mean(1)))
            metrics.append(
                dict(
                    axis=group,
                    law=law,
                    n_targets=len(rows),
                    n_seeds=8,
                    rmse=ci['estimate'],
                    rmse_lo=ci['lo'],
                    rmse_hi=ci['hi'],
                    mae=float(abs(err).mean()),
                    bias=float(err.mean()),
                    prediction_mean=float(pred.mean()),
                    outcome_mean=float(obs.mean()),
                    invalid_predictions=int((~np.isfinite(pred) | (pred <= 0)).sum()),
                )
            )
        for base, extra in COMPARISONS:
            point = 1 - np.sqrt(mses[extra].mean() / mses[base].mean())
            boot = 1 - np.sqrt(mses[extra][draws].mean(1) / mses[base][draws].mean(1))
            ci = interval(point, boot)
            comparisons.append(
                dict(
                    axis=group,
                    baseline=base,
                    extended=extra,
                    n_targets=len(rr) // len(LAWS),
                    relative_rmse_reduction=ci['estimate'],
                    lo=ci['lo'],
                    hi=ci['hi'],
                )
            )
    write_csv(out / 'prediction_metrics.csv', metrics)
    write_csv(out / 'prediction_comparisons.csv', comparisons)

    def gate_for(base, extended):
        chosen = [r for r in comparisons if r['baseline'] == base and r['extended'] == extended]
        main = next(r for r in chosen if r['axis'] == 'joint')
        finite = all(r['invalid_predictions'] == 0 for r in metrics if r['law'] in (base, extended))
        gate = (
            finite
            and main['relative_rmse_reduction'] >= c['predictive_min_relative_rmse_gain']
            and main['lo'] > 0
            and all(r['relative_rmse_reduction'] > 0 for r in chosen if r['axis'].startswith('joint_width_'))
        )
        return ('SUPPORTED' if gate else 'NOT_ESTABLISHED'), main

    status, main = gate_for('stable_base', 'stable_difficulty')
    repair, repair_main = gate_for('legacy_difficulty', 'stable_difficulty')
    oracle_status, oracle_main = gate_for('stable_base', 'stable_oracle_history')
    obj = dict(
        status=status,
        repair_status=repair,
        oracle_status=oracle_status,
        primary_secondary_comparison=main,
        repair_comparison=repair_main,
        oracle_comparison=oracle_main,
        criterion='>=10% joint gain, lower interval >0, both held-out widths improve; no invalid forecasts',
        n_seed_blocks=8,
        p_value=None,
        all_forecasts_positive=all(r['invalid_predictions'] == 0 for r in metrics),
        metrics=metrics,
        comparisons=comparisons,
    )
    save_json(out / 'prediction_summary.json', obj)
    return obj


def main():
    c = json.loads((ROOT / 'config.json').read_text())
    out = ROOT / 'results'
    forecasts = freeze(out, c)
    pre_test = read(out / 'pretrain_test_metrics.csv')
    test = read(out / 'test_metrics.csv')
    cell_test = read(out / 'cell_test_metrics.csv')
    assert len(pre_test) == 400 and len(test) == 1200 and len(set(map(key, test))) == 1200
    draws = boot_draws(c)
    a = intervention(out, c, pre_test, cell_test, draws)
    rl_u = rl_utility_and_signal(out, c, test, draws)
    ret = retention(out, c, pre_test, test, cell_test, draws)
    b = prediction(out, c, test, forecasts, draws)
    print(
        json.dumps(
            dict(
                intervention=a['status'],
                pure_carry_intervention=a['pure_carry_status'],
                rl_utility_reversal=rl_u['utility_reversal_status'],
                rl_utility_law=rl_u['utility_law_status'],
                retention=ret['status'],
                prediction=b['status'],
                repair=b['repair_status'],
            )
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
