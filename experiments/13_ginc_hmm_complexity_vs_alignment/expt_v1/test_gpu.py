"""13-GPU code path: shard and pilot maps, the FLOP cap, the device-aware forward, the parallel training stream, the
pilot rule, the N laws, and a tiny end-to-end shard -> VERIFIED -> merge on CPU (TDC_ALLOW_CPU=1)."""
import itertools
import json
import math

import numpy as np
import pytest
import torch

import gpu_arm as G
import hmm as H
import independent_fit as IF
import laws as L
import model as M
import run as R
import verify_gpu as VG

C = G.load_config()


def test_shard_map_is_a_bijection_in_protocol_order():
    seen = set()
    for i in range(C['n_shards']):
        j = G.shard_coords(C, i)
        i_N, i_n = G.SIZES.index(j['size']), C['levels'].index(j['level'])
        i_a, i_s = C['alpha_grid'].index(j['alpha']), C['seeds'].index(j['seed'])
        assert i == 120 * i_N + 24 * i_n + 12 * j['target_index'] + 3 * i_a + i_s
        seen.add((j['size'], j['level'], j['target_index'], j['alpha'], j['seed']))
    assert len(seen) == 360
    assert [G.shard_coords(C, i)['size'] for i in (0, 119, 120, 239, 240, 359)] == ['S', 'S', 'M', 'M', 'L', 'L']
    with pytest.raises(ValueError):
        G.shard_coords(C, 360)


def test_flop_cap_checkpoints():
    full = [2 ** 22, 2 ** 24, 2 ** 26, 2 ** 28, 2 ** 30]
    assert G.checkpoints_for(C, 'S') == full and G.checkpoints_for(C, 'M') == full
    assert G.checkpoints_for(C, 'L') == [2 ** 22, 2 ** 24, 2 ** 26, 2 ** 28, 2 ** 29]
    for s in G.SIZES:
        assert 6 * C['expected_params'][G.SIZES.index(s)] * G.checkpoints_for(C, s)[-1] <= C['flop_cap']


def test_pilot_jobs():
    tasks = [dict(task_id='hmm_n032_p1', role='pilot', K=32), dict(task_id='hmm_n016_p1', role='pilot', K=16)]
    jobs = [G.pilot_job(C, tasks, j) for j in range(9)]
    assert [(j['size'], j['lr_multiplier']) for j in jobs] == list(itertools.product(G.SIZES, [0.5, 1, 2]))
    assert all(j['target_id'] == 'hmm_n032_p1' and j['alpha'] == 0.4 and j['seed'] == 13919 for j in jobs)


def test_gpu_sizes_and_forward_match_model_py():
    for s in G.SIZES:
        M.build_model(G.model_config(C, s))  # asserts both parameter counts
    torch.manual_seed(0)
    m = M.build_model(G.model_config(C, 'S')).double()
    x = torch.randint(0, 151, (3, 128))
    with torch.no_grad():
        assert torch.equal(m(x), G.lm_forward(m, x))
        assert torch.allclose(m(x), G.lm_forward(m, x, fast=True), atol=1e-10)
    docs = np.random.default_rng(0).integers(0, 150, size=(5, 128))
    m32 = M.build_model(G.model_config(C, 'S'))
    assert np.array_equal(G.code_lengths(m32, docs, 150, torch.device('cpu')), M.code_lengths(m32, docs, 150))


def _concepts(n_list, V=150, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for n in n_list:
        while True:
            h = H.sample_candidate(rng, n, V, 6)
            if H.strongly_connected_aperiodic(h) and H.process_rank(h)[0] == n:
                out.append(h)
                break
    return out


def test_parallel_stream_equals_stream_batches():
    hs = _concepts([4, 4, 4])
    defs = {str(i): H.to_json(h) for i, h in enumerate(hs)}
    cids, js = H.schedule(13101, 0, [1, 2], 0.4, 64)
    ref = [b.astype(np.int16) for b in H.stream_batches(dict(enumerate(hs)), 13101, cids, js, 128, 8)]
    st = G.Stream(2, (defs, 13101, 0, [1, 2], 0.4, 64, 128, 8), 8, depth=3)
    try:
        got = [st.get() for _ in range(8)]
    finally:
        st.close()
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))


def test_pilot_rule():
    c = dict(C)
    base = dict(S={0.5: 0.30, 1: 0.20, 2: 0.10}, M={0.5: 0.199, 1: 0.200, 2: 0.25}, L={0.5: 0.1, 1: float('nan'), 2: 0.3})
    res = [dict(size=s, lr_multiplier=m, excess_bits=v) for s, d in base.items() for m, v in d.items()]
    assert G.select_rule(c, res) == dict(S=2, M=1, L=0.5)


def test_n_law_recovery_and_independent_fit():
    Kb = [568, 1184, 5120, 10624]
    Kref = float(np.exp(np.mean(np.log(Kb))))
    rows = [dict(K_bits=K, alpha=a, T=T, N=N, ident_bits=0.01 * math.log2(1 / a), excess_bits=1.0)
            for K, a, T, N in itertools.product(Kb, [0.05, 0.4, 1.0], [2 ** 22, 2 ** 24, 2 ** 26, 2 ** 28],
                                                [3159552, 25220096, 85056000])]
    th = [math.log2(0.5), math.log2(0.05), 0.6, 0.35, 0.25, 0.8, 0.7]
    X = L.design(rows, 'K_bits', Kb, Kref, 2 ** 22, C['eps_res'])
    p = L.predict_eps('N_K_both', th, X) * 2 ** np.random.default_rng(0).normal(0, 0.02, len(rows))
    X = L.design([dict(r, excess_bits=float(v)) for r, v in zip(rows, p)], 'K_bits', Kb, Kref, 2 ** 22, C['eps_res'])
    f = L.fit('N_K_both', X, C)['params']
    for k, v in zip(L.NAMES['N_K_both'], th):
        if k in ('beta', 'g_lo', 'g_hi', 'eta', 'kappa'):
            assert abs(f[k] - v) / abs(v) < 0.05, (k, f[k], v)
    for law in ('N_K_both', 'N_AS_exp'):
        a = IF.heldout_rmse(law, L.fit(law, X, C)['theta'], X)[0]
        b = IF.heldout_rmse(law, IF.fit(law, X, C)['theta'], X)[0]
        assert IF.agree(a, b, C['independent_fit_tol'])
    bad = L.invalid_mask('N_AS_both', np.array([-0.1, 0.5]), dict(eid=np.array([0.0, 0.6])))
    assert bad.tolist() == [True, True]


def _tiny(tmp_path, monkeypatch):
    """A tiny config and prep (n=4 concepts, a 2-layer model, T to 2^12) for an end-to-end CPU shard."""
    monkeypatch.setenv('TDC_ALLOW_CPU', '1')
    c = dict(C, levels=[4, 8], heldout_level=8, targets_per_level=1, eval_docs_per_target=16, distractor_eval_docs=4,
             batch_docs=8, T_checkpoints=[1024, 4096], data_workers=2, warmup_steps=2, lr_selected=dict(S=1, M=1, L=1),
             sizes=dict(S=[1, 16, 2], M=[1, 16, 2], L=[1, 16, 2]), n_shards=1)
    m = M.LM(c['vocab'], 16, 1, 2, 64, 128)
    c['expected_params'] = [M.param_count(m)] * 3
    c['expected_nonembedding'] = [M.nonembedding_count(m)] * 3
    hs = _concepts([4, 4, 4, 4])
    tasks = []
    for i, (h, role) in enumerate(zip(hs, ['target', 'target', 'distractor', 'distractor'])):
        tasks.append(dict(task_id=f'hmm_n004_{role[0]}{i}', cid=i, role=role, K=4, K_bits=H.K_bits(4, 150, 6),
                          K_bits_canonical=1.0, definition=H.to_json(h), process_rank=4))
    prep = tmp_path / 'prep'
    prep.mkdir()
    (prep / 'tasks.json').write_text(json.dumps(tasks))
    (prep / 'acceptance_log.json').write_text('{}')
    R.build_references(c, tasks, prep, {'target': c['alpha_grid']})
    return c, tasks, prep


def test_tiny_shard_verify_merge(tmp_path, monkeypatch):
    c, tasks, prep = _tiny(tmp_path, monkeypatch)
    monkeypatch.setattr(G, 'shard_job', lambda c_, t_, i: dict(G.shard_coords(C, 0), target_id='hmm_n004_t0',
                                                               alpha=0.4, level=4))
    out = tmp_path / 'out'
    G.run_shard(c, prep, out, 0, allow_cpu=True)
    d = out / 'shards' / '00000'
    assert (d / 'DONE').exists()
    rec = json.loads((d / 'run.json').read_text())
    assert [r['T'] for r in rec['rows'] if r['slice'] == 'all'] == [1024, 4096] and rec['finite']
    G.run_shard(c, prep, out, 0, allow_cpu=True)  # DONE: skipped
    VG.verify_shard(c, out, 0, prep, allow_cpu=True)
    v = json.loads((d / 'VERIFIED').read_text())
    assert v['result'] == 'PASS' and v['max_abs_diff_bits'] < 1e-9
    (d / 'run.json').write_text((d / 'run.json').read_text().replace('"finite": true', '"finite": false'))
    with pytest.raises(VG.Fail):
        VG.verify_shard(c, out, 0, prep, allow_cpu=True)
    (d / 'run.json').write_text((d / 'run.json').read_text().replace('"finite": false', '"finite": true'))
    G.merge(c, out, prep, tmp_path / 'merged')
    rows = (tmp_path / 'merged' / 'metrics.csv').read_text().splitlines()
    assert rows[0].startswith('seed,target_id,K,K_bits,alpha,size,N,T') and len(rows) == 1 + 2 * 4
    man = json.loads((tmp_path / 'merged' / 'manifest.json').read_text())
    assert man['grid']['verified'] == 1


def test_shard_refuses_without_lr_selected(tmp_path, monkeypatch):
    c, _, prep = _tiny(tmp_path, monkeypatch)
    c['lr_selected'] = None
    with pytest.raises(SystemExit, match='lr_selected'):
        G.run_shard(c, prep, tmp_path / 'out', 0, allow_cpu=True)
