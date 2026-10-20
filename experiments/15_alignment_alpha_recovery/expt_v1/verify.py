"""Independent verification of results/ (PROTOCOL Section 8). Prints PASS or the first failure.

Usage: verify.py [--results results] [--workers 4]
"""
import argparse
import concurrent.futures as cf
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import common as C
import graded as GR
import independent_fit as IFX
import jobs as J
import mixes15 as MX

HERE = Path(__file__).resolve().parent


class Fail(Exception):
    pass


def check(ok, msg):
    if not ok:
        raise Fail(msg)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12))) if a.size else 0.0


# ---------------------------------------------------------------- workers

def _probe_sample(args):
    arm, probe, res = args
    c = C.load_config()
    import torch
    torch.set_num_threads(1)
    st = J.Setup(c, arm, res)
    rng = np.random.default_rng([c['verify_sample_seed'], J.ARMS.index(arm), probe])
    k = max(1, int(round(c['verify_sample_fraction'] * st.nU)))
    sel = {u: np.sort(rng.choice(st.nU, k, replace=False)) for u in st.ids}
    r, _ = J.probe_scores(c, st, probe, unit_index=sel)
    worst = {}
    for m in J.probe_measures(arm):
        saved = np.load(Path(res) / 'scores' / f'{arm}_{m}_{probe}.npz')['raw']
        if m in ('b1_perm', 'b1_inp'):
            continue
        mx = mx_plain = mx_abs = 0.0
        for ti, T in enumerate(st.targets):
            lay = st.layout(T)
            pos = np.concatenate([b * st.nU + sel[u] for b, u in enumerate(lay)])
            # the saved scores are float32; compare the recompute at the same precision
            a = np.asarray(r[m][ti], np.float32).astype(np.float64)
            b = saved[ti][pos].astype(np.float64)
            # scale: the target's pool range, the scale min-max normalization divides by (Deviation 5 in RESULTS:
            # gradient cosines near 0 make the plain relative difference ill-conditioned)
            span = float(np.ptp(saved[ti])) or 1.0
            mx = max(mx, float(np.max(np.abs(a - b))) / span)
            mx_abs = max(mx_abs, float(np.max(np.abs(a - b))))
            mx_plain = max(mx_plain, rel(np.asarray(r[m][ti], np.float32), saved[ti][pos]))
        worst[m] = mx
        worst[m + ' (abs)'] = mx_abs
        worst[m + ' (plain relative)'] = mx_plain
    return arm, probe, worst


def _t2v_sample(args):
    arm, probe, target, res = args
    c = C.load_config()
    import torch
    torch.set_num_threads(1)
    st = J.Setup(c, arm, res)
    r = J.t2v_scores(c, st, probe, target)
    saved = json.loads((Path(res) / 'parts' / J.part_name('t2v', (arm, probe, target))).read_text())
    w = 0.0
    for est in ('e', 'e2'):
        a = [r[est]['cross_div'][k] for k in sorted(r[est]['cross_div'])]
        b = [saved[est]['cross_div'][k] for k in sorted(saved[est]['cross_div'])]
        w = max(w, rel(a, b), rel([r[est]['alpha_hat'][k] for k in sorted(r[est]['alpha_hat'])],
                                  [saved[est]['alpha_hat'][k] for k in sorted(saved[est]['alpha_hat'])]))
    return arm, probe, target, w


def _nonprobe(args):
    arm, target, res = args
    c = C.load_config()
    st = J.Setup(c, arm, res)
    r, _ = J.nonprobe_scores(c, st, target, None, only=['a', 'aprime', 'b2'])
    ti = st.targets.index(target)
    w = {}
    for m in ('a', 'aprime', 'b2'):
        saved = np.load(Path(res) / 'scores' / f'{arm}_{m}_all.npz')['raw'][ti]
        w[m] = rel(np.asarray(r[m], np.float32), saved)
    d_new = J.S.train_zstd_dictionary(c, J.ref_samples(target, C.serialize(c, st.fam, st.refs[target])),
                                      15000 + 100 * (st.code - 10) + st.token(target))
    w['dict_identical'] = d_new == J.dict_path(res, arm, target, 'a').read_bytes()
    return arm, target, w


def _replay(args):
    """Replay every checkpoint of the given graded runs on the target's test split (as Experiment 11's verify.py)."""
    res, names = args
    c = C.load_config()
    import importlib
    import torch
    torch.set_num_threads(1)
    R = GR.exp11_run(c)
    M = importlib.import_module('model')  # Experiment 11's model.py (its folder is on sys.path after the import)
    G = json.loads((Path(res) / 'graded' / 'graded_tasks_full.json').read_text())
    data = GR.load_data(Path(res) / 'graded', G['tasks'])
    c11 = R.load_config()
    tok = {t['task_id']: t['task_token'] for t in G['tasks']}
    wb = wp = 0.0
    n = 0
    for name in names:
        rec = json.loads((Path(res) / 'graded' / 'cells' / f'{name}.json').read_text())
        t = tok[rec['target_id']]
        pred = np.load(Path(res) / 'graded' / 'predictions' / f'{name}.npz')
        for i, row in enumerate(rec['rows']):
            m = M.load_npz(M.build_model(c11, c['graded']['n_task_tokens']),
                           Path(res) / 'graded' / 'checkpoints' / name / f"T_{row['T']}.npz")
            ev = M.evaluate(m, t, data['test_bits'], data['test_lens'], data['test_labels'][t])
            wb = max(wb, abs(ev['excess_bits'] - row['excess_bits']))
            wp = max(wp, float(np.max(np.abs(ev['per_string_bits'] - pred['bits'][i]))))
            n += 1
    return wb, wp, n


# ---------------------------------------------------------------- main

def verify(c, res, workers):
    res = Path(res)
    log = []

    def note(msg):
        print(msg, flush=True)
        log.append(msg)

    C.check_upstream(c)
    # 1. mixes
    for fam in C.FAMILIES:
        _, ok, _, n = MX.reconstruct(c, fam, workers=workers)
        check(ok, f'{fam}: rebuilt mix counts differ from mixes.json')
        note(f'mixes {fam}: {n} rebuilt, counts equal')
    G = json.loads((res / 'graded' / 'graded_tasks_full.json').read_text())
    mj = json.loads((res / 'graded' / 'mixes.json').read_text())
    D, c11 = GR.D_(), GR.c11()
    for name, m in mj.items():
        mix = D.assemble_mix(c11, GR.mix_tasks(G, f"{m['target_id']}_{m['mix']}"), m['target_id'], m['alpha'],
                             m['seed'])
        check(mix['counts'] == m['counts'], f'graded {name}: counts differ')
    note(f'mixes graded: {len(mj)} rebuilt, counts equal')
    # 2. disjoint, token-free
    for arm in J.ARMS:
        st = J.Setup(c, arm, res)
        lo = c['serialization']['task_token_first_byte']
        for T in st.targets:
            fi = st.ref_file_index[T]
            check(not np.intersect1d(st.unit_file_index[T], fi[fi >= 0]).size, f'{arm} {T}: references meet units')
            for raw in (st.refs[T],):
                check(all(max(b) < lo for b in C.serialize(c, st.fam, raw)), f'{arm} {T}: task token in references')
        for u in st.ids:
            check(all(max(b) < lo for b in C.serialize(c, st.fam, st.units[u])), f'{arm} {u}: task token in units')
    note('references and units disjoint; no task-token byte in any stripped serialization')
    # 3. scores
    with cf.ProcessPoolExecutor(workers) as ex:
        args = [(arm, T, str(res)) for arm in J.ARMS for T in J.arm_targets(c, res, arm)]
        worst = {}
        for arm, T, w in ex.map(_nonprobe, args):
            check(w.pop('dict_identical'), f'{arm} {T}: retrained zstd dictionary differs')
            for k, v in w.items():
                worst[k] = max(worst.get(k, 0.0), v)
        check(all(v <= c['verify_rel_tol'] for v in worst.values()), f'a / aprime / b2 recompute: {worst}')
        note(f'a, aprime, b2 recomputed for every target: max rel diff {worst}')
        pw = {}
        for arm, probe, w in ex.map(_probe_sample, [(arm, p, str(res)) for arm in J.ARMS for p in c['probes']]):
            for k, v in w.items():
                pw[k] = max(pw.get(k, 0.0), v)
        gated = {k: v for k, v in pw.items() if '(' not in k}
        check(all(v <= c['verify_rel_tol'] for v in gated.values()), f'b1 / d sample recompute: {pw}')
        note(f'b1, d (and controls) on a seeded 10% unit sample: max difference relative to the pool range {gated}; '
             f'also absolute and plain relative: { {k: v for k, v in pw.items() if "(" in k} }')
        rng = np.random.default_rng(c['verify_sample_seed'])
        tj = [(arm, p, T, str(res)) for arm in J.ARMS for p in c['probes'] for T in J.arm_targets(c, res, arm)]
        pick = sorted(rng.choice(len(tj), max(1, int(round(c['verify_sample_fraction'] * len(tj)))), replace=False))
        tw = max(w for *_, w in ex.map(_t2v_sample, [tj[i] for i in pick]))
        check(tw <= c['verify_rel_tol'], f'Task2Vec sample recompute: max rel diff {tw}')
        note(f'e, e2 on a seeded 10% sample of jobs ({len(pick)}): max rel diff {tw}')
    # 4. alpha-hat from saved scores
    import analyze as AN
    saved = {}
    with open(res / 'alpha_hat.csv') as f:
        for r in csv.DictReader(f):
            saved[(r['arm'], r['target'], r['mix'] or r['alpha'], r['measure'], r['probe'])] = float(r['alpha_hat'])
    worst = 0.0
    for arm in J.ARMS:
        st, A, _, _ = AN.compute_alpha_hat(c, res, arm)
        for m, arr in A.items():
            for ti, T in enumerate(st.targets):
                for ai, al in enumerate(st.alphas):
                    for pi in range(arr.shape[-1]):
                        p = str(c['probes'][pi]) if arr.shape[-1] > 1 else 'none'
                        k = (arm, T, al if arm == 'graded' else str(al), m, p)
                        worst = max(worst, abs(arr[ti, ai, pi] - saved[k]))
    check(worst <= 1e-12, f'alpha-hat recompute: max abs diff {worst}')
    note(f'every alpha-hat recomputed from saved scores: max abs diff {worst:.2e}')
    # 5. graded agreement and 7. K
    L0, L1 = c11['length_min'], c11['length_max']
    x = GR.mc_inputs(c, 2 ** 16, stream=7)
    A = GR.A_()
    mw, ew = 0.0, 0.0
    for t in G['tasks']:
        if t['role'] != 'distractor':
            continue
        T = A.DFA.from_json(next(u for u in G['tasks'] if u['task_id'] == t['of_target'])['definition'])
        d = A.DFA.from_json(t['definition'])
        check(A.hopcroft_minimize(d).n == t['K'] == A.moore_minimize(d).n, f"{t['task_id']}: K differs")
        ex_ = GR.agreement_exact(T, d, L0, L1)
        check(abs(ex_ - t['agree_exact_uniform']) < 1e-12, f"{t['task_id']}: exact agreement differs")
        mc, _ = GR.agreement_mc(x, T, d)
        mw = max(mw, abs(mc - ex_), abs(t['agree'] - ex_))
    check(mw <= c['graded']['verify_mc_tol'], f'graded agreement: exact vs Monte Carlo {mw}')
    note(f'graded agreement: exact DP vs 2^16 and 2^20 Monte Carlo max abs diff {mw:.4f}; every distractor K '
         'recomputed (Hopcroft and Moore)')
    # 6. graded checkpoints
    names = sorted(p.name for p in (res / 'graded' / 'checkpoints').iterdir())
    check(len(names) == len(mj), f'graded checkpoints: {len(names)} runs, mixes.json has {len(mj)}')
    with cf.ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_replay, [(str(res), names[i::workers]) for i in range(workers)]))
    wb, wp, n = max(o[0] for o in out), max(o[1] for o in out), sum(o[2] for o in out)
    check(wb < 1e-9 and wp < 1e-9, f'graded checkpoint replay: max abs diff {wb} bits, predictions {wp}')
    note(f'graded checkpoints replayed ({len(names)} runs, {n} checkpoints): max abs diff {wb:.2e} bits, '
         f'per-string predictions {wp:.2e}')
    # 8. laws and statistics
    laws = json.loads((res / 'laws.json').read_text())
    gates = json.loads((res / 'gates.json').read_text())
    tol = c['law_rel_tol']
    for fam in C.FAMILIES:
        amaps = {}
        y = gates['G3']['families'][fam]['calibrator']
        st = J.Setup(c, fam, res)
        pairs = [(T, al) for T in st.targets for al in st.alphas]
        fit_idx = [pairs.index(tuple(p)) for p in y['fit_pairs']]
        xa = np.array(y['x'])
        ya = np.array([al for _, al in pairs])
        cal = IFX.Isotonic(xa[fit_idx], ya[fit_idx])(xa)
        check(np.max(np.abs(cal - np.array(y['calibrated']))) <= c['stats_tol'], f'{fam}: calibrator differs')
        amaps = dict(true={(T, AN.key_alpha(al)): al for T, al in pairs},
                     calibrated={(T, AN.key_alpha(al)): cal[i] for i, (T, al) in enumerate(pairs)},
                     raw={(T, AN.key_alpha(al)): xa[i] for i, (T, al) in enumerate(pairs)})
        _, fi = AN.gate3_fits(c, fam, amaps, independent=True)
        for k in amaps:
            for a, b in zip(laws[fam]['fits'][k], fi[k]):
                check(abs(a['cost'] - b['cost']) <= tol * abs(a['cost']), f'{fam} {k}: law cost differs')
                check(rel(a['forecast'], b['forecast']) <= tol, f'{fam} {k}: forecasts differ')
    import laws as LW
    grows = LW.read_graded_metrics(res / 'graded' / 'metrics.csv')
    for key in ('share', 'soft', 'agree'):
        _, per = LW.gate4a_fits(c, grows, f'alpha_{key}', independent=True)
        for a, b in zip(laws['graded'][key], per):
            for fa, fb in zip(a['folds'], b['folds']):
                check(abs(fa['cost'] - fb['cost']) <= tol * abs(fa['cost']), f'graded {key}: law cost differs')
            check(abs(a['sse'] - b['sse']) <= tol * abs(a['sse']), f'graded {key}: held-out SSE differs')
    def same(x, y):
        # undefined in both (a score constant over the pool, Deviation 2) counts as agreement
        x, y = np.atleast_1d(np.asarray(x, np.float64)), np.atleast_1d(np.asarray(y, np.float64))
        both = np.isnan(x) & np.isnan(y)
        return bool(np.all(both | (np.abs(x - y) <= c['stats_tol'])))

    for k, v in gates['per_measure'].items():
        check(same(v['g1_rho'], v['g1_rho_independent']), f'{k}: Spearman differs')
        check(same(v['g2_slope'], v['g2_slope_independent']), f'{k}: slope differs')
    for m, v in gates['G4b']['all_measures'].items():
        check(same(v['rho_soft'], v['rho_soft_independent']), f'G4b {m}: Spearman differs')
    note(f'laws refit with independent_fit.py (Gate 3 true/calibrated/raw, Gate 4a share/soft/agree) within {tol}; '
         f'calibrator, Spearman and slopes within {c["stats_tol"]}')
    check(gates['eq2_check']['passed'], f"Eq. 2 check: {gates['eq2_check']['max_abs_diff']}")
    note(f"Eq. 2 stratified vs exact on full mixes: max abs diff {gates['eq2_check']['max_abs_diff']:.4f}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    c = C.load_config()
    t0 = time.time()
    try:
        log = verify(c, HERE / a.results, a.workers)
    except Fail as e:
        print(f'FAIL: {e}')
        return 1
    C.atomic_write_text(HERE / a.results / 'verify_log.json',
                        json.dumps(dict(result='PASS', checks=log, seconds=time.time() - t0), indent=1))
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
