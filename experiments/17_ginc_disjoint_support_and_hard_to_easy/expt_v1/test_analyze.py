"""Synthetic dry run of analyze.py: a known law and known pool effects must come back with the right verdicts."""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402
import run as R  # noqa: E402

COLS = R.ROW_COLS


def fake_results(tmp, eta_near, eta_far, easy_gap, Ts=None):
    c = R.load_config()
    Ts = Ts or c['T_checkpoints_primary']
    tasks = json.loads((HERE / 'prep' / 'tasks.json').read_text())
    out = Path(tmp)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tasks.json').write_text(json.dumps(tasks))
    rng = np.random.default_rng(0)
    tg = {t['task_id']: t for t in tasks if t['role'] == 'target'}
    K_ref = math.sqrt(224 * 2080)
    rows = []
    for s, pool, tid, a in R.grid_runs(c, tasks):
        for t in (tg.values() if tid == '-' else [tg[tid]]):
            for T in Ts:
                eta = {'near': eta_near, 'far': eta_far}.get(pool, eta_near)
                aa = max(a, 1e-3)
                lK = math.log2(t['K_bits'] / K_ref)
                eps = 2 ** (0.0 + 0.8 * lK - 0.6 * (eta * math.log2(aa) + math.log2(T / Ts[0])))
                if a == 0:
                    eps = {'far': 6.0, 'near': 2.0}.get(pool, 2.0)
                    if pool in ('easy', 'hard'):
                        matched = (pool == 'easy') == (t['K'] == 4)
                        eps = 2.0 - (easy_gap if matched else 0)
                elif pool in ('easy', 'hard'):
                    matched = (pool == 'easy') == (t['K'] == 4)
                    eps = eps * (1 - 0.1 * easy_gap) if matched else eps
                eps *= math.exp(rng.normal(0, 0.01))
                eid = 0.0 if a in (0, 1) else math.log2(1 / a) / 128
                rows.append(dict(seed=s, pool=pool, run_target=tid, target_id=t['task_id'], K=t['K'],
                                 K_bits=t['K_bits'], alpha=a, T=T, step=0, slice='all', excess_bits=eps + eid,
                                 ident_bits=eid, learn_bits=eps, model_bits=eps + 3, oracle_bits=3.0,
                                 mixture_bits=3 + eid, target_docs_seen=0, excess_se=0.01))
    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
    with open(out / 'pool_loss.csv', 'w', newline='') as f:
        f.write('run,pool_id,model_bits,oracle_bits\nx,y,3.1,3.0\n')
    return out


def run_analysis(out, eps_star=0.5):
    c = R.load_config()
    c['epsilon_star'] = eps_star
    tasks = json.loads((out / 'tasks.json').read_text())
    rows = A.read_rows(out)
    idx = A.index(rows)
    seeds = sorted({r['seed'] for r in rows})
    Ts = sorted({r['T'] for r in rows})
    rng = np.random.default_rng(1)
    return A.arm_H(c, idx, seeds, Ts, tasks, rng), A.arm_D(c, rows, idx, seeds, Ts, tasks, rng)


def test_far_raises_E_and_matched_pool_wins(tmp_path):
    out = fake_results(tmp_path / 'r1', eta_near=0.8, eta_far=1.6, easy_gap=0.5)
    H, D = run_analysis(out)
    assert H['H1_hard_does_not_subsume_easy'] == 'SUPPORTED'
    assert H['H2_crossover'] == 'SUPPORTED'
    assert D['D1_zero_shot']['verdict'] == 'SUPPORTED'
    # e_alpha = eta, e_K = kappa / gamma = 0.8 / 0.6 in bits units
    assert abs(D['near']['E_star'] - 0.8 / (0.8 / 0.6)) < 0.05, D['near']['E_star']
    assert abs(D['far']['E_star'] - 1.6 / (0.8 / 0.6)) < 0.08, D['far']['E_star']
    assert D['D2_E_far_above_1'] == 'SUPPORTED'
    assert D['D3_E_far_minus_near']['verdict'] == 'SUPPORTED'


def test_null_pools(tmp_path):
    out = fake_results(tmp_path / 'r2', eta_near=0.8, eta_far=0.8, easy_gap=0.0)
    H, D = run_analysis(out)
    assert H['H1_hard_does_not_subsume_easy'] == 'NOT_ESTABLISHED'
    assert D['D2_E_far_above_1'].startswith('NOT_ESTABLISHED')
    assert D['D3_E_far_minus_near']['verdict'] != 'SUPPORTED'


def test_reverse(tmp_path):
    out = fake_results(tmp_path / 'r3', eta_near=0.8, eta_far=0.8, easy_gap=-0.5)
    H, _ = run_analysis(out)
    assert H['H1_hard_does_not_subsume_easy'] == 'REFUTED'


def test_t_equiv():
    Ts = np.array([1, 2, 4, 8], float)
    e = np.array([8, 4, 2, 1], float)
    assert A.t_equiv(Ts, e, 9)[0] == '<'
    assert A.t_equiv(Ts, e, 0.5)[0] == '>'
    k, v = A.t_equiv(Ts, e, 2 ** 1.5)
    assert k == '=' and abs(v - 2 ** 1.5) < 1e-9


def test_report_runs(tmp_path):
    import report
    out = fake_results(tmp_path / 'r4', eta_near=0.8, eta_far=1.2, easy_gap=0.3)
    c = R.load_config()
    c['epsilon_star'] = 0.5
    tasks = json.loads((out / 'tasks.json').read_text())
    rows = A.read_rows(out)
    idx = A.index(rows)
    seeds = sorted({r['seed'] for r in rows})
    Ts = sorted({r['T'] for r in rows})
    rng = np.random.default_rng(1)
    a = dict(H=A.arm_H(c, idx, seeds, Ts, tasks, rng), D=A.arm_D(c, rows, idx, seeds, Ts, tasks, rng),
             T_checkpoints=Ts, seeds=seeds, cell_means={'near|4|0.05|%d' % Ts[-1]: 0.1},
             pool_excess_at_Tmax=dict(mean=0.1, max=0.2, n=1))
    a = json.loads(json.dumps(a))
    txt = report.tables(a, c, tasks)
    assert 'H1 hard does not subsume easy' in txt
    report.plot(a, c, tmp_path / 'x.png')
    assert (tmp_path / 'x.png').stat().st_size > 1000
