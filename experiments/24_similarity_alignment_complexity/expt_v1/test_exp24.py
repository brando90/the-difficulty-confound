"""Experiment 24 unit tests: distractor construction, shard and pilot maps, data bills and censoring, Tobit recovery,
the gate statistics on planted data, the pilot rule, and a tiny CPU shard -> VERIFIED -> merge round trip."""
import copy
import itertools
import json
import math

import numpy as np
import pytest

import analyze24 as A
import concepts24 as C
import hmm as H
import model as M
import run24 as R
import verify24 as VF

CFG = C.load_config()


def _target(n=8, seed=1):
    rng = np.random.default_rng(seed)
    while True:
        h = H.sample_candidate(rng, n, 150, 6)
        if H.strongly_connected_aperiodic(h) and H.process_rank(h)[0] == n:
            return h


def test_copies_share_exact_rows_and_disjoint_shifts_tokens():
    cc = C.concept_config(CFG)
    t = _target(32)
    rng = np.random.default_rng(0)
    for sim, f in (('copy20', 0.2), ('copy50', 0.5)):
        h = C.propose(cc, rng, t, sim)
        assert C.shared_fraction(t, h) >= 1 - round(f * 32) / 32 - 1e-12
        assert C.shared_fraction(t, h) < 1
    d = C.propose(cc, rng, t, 'disjoint')
    assert d['V'] == 300 and d['etok'].min() >= 150 and C.emission_overlap(t, d) == 0 and C.shared_fraction(t, d) == 0
    assert H.process_rank(d)[0] == H.process_rank({**d, 'etok': d['etok'] - 150, 'V': 150})[0]


def test_pool_is_deterministic_and_checked():
    cc = C.concept_config(CFG)
    task = dict(task_id='x', cid=3, definition=H.to_json(_target(8)))
    a = C.make_pool(cc, task, 'copy20', 2.86, 0.6, log=lambda m: None)
    b = C.make_pool(cc, task, 'copy20', 2.86, 0.6, log=lambda m: None)
    assert len(a) == 4 and all(H.same(x['h'], y['h']) for x, y in zip(a, b))
    assert all(H.strongly_connected_aperiodic(x['h']) and x['process_rank'] == 8 for x in a)


def test_shard_map_bijection():
    tasks = [dict(task_id=f'hmm_n{n:03d}_t{j}', role='target', K=n) for n in (8, 32, 64) for j in (1, 2, 3)]
    seen = set()
    for i in range(R.n_shards(CFG)):
        j = R.shard_job(CFG, tasks, i)
        seen.add((j['size'], j['target_id'], j['similarity'], j['alpha'], j['seed']))
        assert (j['similarity'] == 'none') == (j['alpha'] == 1.0)
    assert len(seen) == R.n_shards(CFG) == 918
    with pytest.raises(ValueError):
        R.shard_coords(CFG, 918)


def test_pilot_jobs_and_rule():
    tasks = [dict(task_id='hmm_n032_p1', role='pilot', K=32)]
    jobs = [R.pilot_job(CFG, tasks, j) for j in range(6)]
    assert [(j['size'], j['lr_multiplier']) for j in jobs] == list(itertools.product(['S', 'M'], [1, 2, 4]))
    res = [dict(size='S', lr_multiplier=1, excess_bits=0.30), dict(size='S', lr_multiplier=2, excess_bits=0.2015),
           dict(size='S', lr_multiplier=4, excess_bits=0.20), dict(size='M', lr_multiplier=1, excess_bits=0.1),
           dict(size='M', lr_multiplier=2, excess_bits=float('nan')), dict(size='M', lr_multiplier=4, excess_bits=0.2)]
    assert R.select_rule(CFG, res) == dict(S=2, M=1)


def test_tstar_interpolation_and_censoring():
    pts = [(2 ** 20, 1.0, 0, 0), (2 ** 21, 0.2, 0, 0), (2 ** 22, 0.05, 0, 0)]
    y, st = A.tstar(pts, 0.1)
    assert st == 0 and abs(y - (math.log(2 ** 21) + 0.5 * math.log(2))) < 1e-12
    assert A.tstar(pts, 0.01)[1] == 1 and A.tstar(pts, 2.0)[1] == -1


def _planted(e_alpha, e_k=0.5, sd=0.15, cap=math.log(2 ** 28), seed=0):
    rng = np.random.default_rng(seed)
    tab = []
    for size, n, j, seed_, in itertools.product(['S', 'M'], [8, 32, 64], [1, 2, 3], [1, 2, 3]):
        for sim, alpha in R.cells(CFG):
            ea = e_alpha[sim if sim != 'none' else 'family'][size]
            y = 15 + (0.3 if size == 'M' else 0) - ea * math.log(alpha) + e_k * math.log(n) + rng.normal(0, sd)
            st = 1 if y >= cap else 0
            tab.append(dict(size=size, target_id=f'n{n}_t{j}', similarity=sim, alpha=alpha, seed=seed_, K=n,
                            K_bits=n * 10, lnT=min(y, cap), status=st))
    return tab


def test_tobit_recovers_planted_slopes_with_censoring():
    ea = {s: {'S': v, 'M': v - 0.1} for s, v in zip(A.SIMS, [0.2, 0.4, 0.6, 0.9])}
    tab = _planted(ea, cap=17.5)
    assert 0.05 < np.mean([r['status'] for r in tab]) < 0.6
    p, _ = A.stats(tab)
    for s, v in zip(A.SIMS, [0.2, 0.4, 0.6, 0.9]):
        assert abs(p[f'e_alpha|{s}'] - (v - 0.05)) < 0.06, (s, p[f'e_alpha|{s}'])
        assert abs(p[f'e_K|{s}'] - 0.5) < 0.05
    assert p['G1_delta'] > 0.5 and abs(p['G5_A'] - 0.1) < 0.06
    assert p['G3_reduction'] > 0.1


def test_gates_on_planted_data():
    # planted: alignment sensitivity rises with dissimilarity; M drops it by 0.2 for similar pools, not for disjoint
    ea = {s: {'S': v + (0.2 if s != 'disjoint' else 0.0), 'M': v} for s, v in zip(A.SIMS, [0.15, 0.3, 0.45, 0.8])}
    tab = _planted(ea, sd=0.1)
    point, fits, per_seed, boot, g = A.full_analysis(CFG, tab, 200, 1, 5)
    assert g['G1']['verdict'] == 'SUPPORTED' and g['G4']['verdict'] == 'SUPPORTED'
    assert g['G2']['verdict'] == 'SUPPORTED', (g['G2']['E'], g['G2']['ci95_disjoint'], g['G2']['ci95_copy20'])
    assert g['G5']['verdict'] == 'SUPPORTED', (g['G5']['A'], g['G5']['B'], g['G5']['ci95_B'])
    assert A.independent_check(tab, fits, 1e-4)['agree']
    flat = {s: {'S': 0.4, 'M': 0.4} for s in A.SIMS}
    g2 = A.full_analysis(CFG, _planted(flat), 100, 1, 5)[4]
    assert g2['G1']['verdict'] == 'NOT_ESTABLISHED' and g2['G2']['verdict'] == 'NOT_ESTABLISHED'
    assert g2['G5']['verdict'] == 'NOT_ESTABLISHED'


def test_model_counts_with_301_tokens():
    for s in ('S', 'M'):
        M.build_model(R.model_config(CFG, s))


def _tiny(tmp_path, monkeypatch):
    monkeypatch.setenv('TDC_ALLOW_CPU', '1')
    cfg = copy.deepcopy(CFG)
    cfg['concepts'].update(levels=[4], targets_per_level=3, band_calibration_candidates=8,
                           entropy_band_bits=0.6, entropy_band_fallback=1.0, quick_entropy_seqs=2,
                           entropy_seqs=4, entropy_seq_len=400, pool_size=2)
    cfg['eval'].update(eval_docs_per_target=16, distractor_eval_docs=4)
    cfg['grid'].update(T_checkpoints=[1024, 2048, 4096], keep_checkpoints=[1024, 4096], seeds=[24101])
    cfg['model'].update(batch_docs=8, data_workers=2, warmup_steps=2, sizes=dict(S=[1, 16, 2], M=[1, 16, 2]))
    m = M.LM(301, 16, 1, 2, 64, 128)
    cfg['model']['expected_params'] = dict(S=M.param_count(m), M=M.param_count(m))
    cfg['model']['expected_nonembedding'] = dict(S=M.nonembedding_count(m), M=M.nonembedding_count(m))
    cfg['pilot'].update(level=4)
    cfg['lr_selected'] = dict(S=2, M=2)
    prep = tmp_path / 'prep'
    C.build_prep(cfg, prep, log=lambda s: None)
    return cfg, prep


def test_tiny_shard_verify_merge(tmp_path, monkeypatch):
    cfg, prep = _tiny(tmp_path, monkeypatch)
    tasks = json.loads((prep / 'tasks.json').read_text())
    assert len([t for t in tasks if t['role'] == 'distractor']) == 3 * 4 * 2 + 2
    jobs = {0: R.shard_job(cfg, tasks, 5), 1: R.shard_job(cfg, tasks, 0)}  # an alpha < 1 cell and an alpha = 1 cell
    assert jobs[0]['alpha'] < 1 and jobs[1]['alpha'] == 1
    monkeypatch.setattr(R, 'n_shards', lambda c: 2)
    monkeypatch.setattr(R, 'shard_job', lambda c, t, i: jobs[i])
    out = tmp_path / 'out'
    for i in (0, 1):
        R.run_shard(cfg, prep, out, i, allow_cpu=True)
        VF.verify_shard(cfg, out, i, prep, allow_cpu=True)
        v = json.loads((out / 'shards' / f'{i:05d}' / 'VERIFIED').read_text())
        assert v['max_abs_diff_bits'] < 1e-9 and len(v['removed_after_replay']) == 1
    rec = json.loads((out / 'shards' / '00000' / 'run.json').read_text())
    assert rec['job']['alpha'] < 1 and len(rec['pool']) == 2 and rec['finite']
    R.merge(cfg, out, prep, tmp_path / 'merged')
    rows = (tmp_path / 'merged' / 'metrics.csv').read_text().splitlines()
    assert len(rows) == 1 + 2 * 3 * 4
    (out / 'shards' / '00000' / 'VERIFIED').unlink()
    (out / 'shards' / '00000' / 'run.json').write_text('{}')
    with pytest.raises(VF.Fail):
        VF.verify_shard(cfg, out, 0, prep, allow_cpu=True)


def test_shard_refuses_without_lr(tmp_path, monkeypatch):
    cfg, prep = _tiny(tmp_path, monkeypatch)
    cfg['lr_selected'] = None
    with pytest.raises(SystemExit, match='lr_selected'):
        R.run_shard(cfg, prep, tmp_path / 'o', 0, allow_cpu=True)
