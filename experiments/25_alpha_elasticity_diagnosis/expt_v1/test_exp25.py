"""Experiment 25 unit tests: the iid concepts, the masked loss, the shard map, the learning-rate choice, data bills with
divergence, Tobit and gate recovery on planted data, and a tiny CPU shard -> VERIFIED -> merge round trip for every arm
(and a diverged run)."""
import copy
import itertools
import json
import math

import numpy as np
import pytest
import torch

import analyze25 as A
import concepts25 as C
import hmm as H
import model as M
import run25 as R
import verify25 as VF
import gpu_arm as G

CFG = C.load_config()


def _concept(n=8, seed=1, V=300, offset=150):
    rng = np.random.default_rng(seed)
    while True:
        h = H.sample_candidate(rng, n, 150, 6)
        if H.strongly_connected_aperiodic(h) and H.process_rank(h)[0] == n:
            h['etok'] = h['etok'] + offset
            h['V'] = V
            return h


def test_iid_concept_has_parent_unigram_and_no_structure():
    h = _concept(8)
    g, tv = C.iid_concept(h, 12, 128)
    assert g['n'] == 1 and int(g['emit'].sum()) == 4096 and tv < 0.01
    assert C.iid_matches_parent(g, h, 128, 0.02)
    docs = H.sample_docs(g, 5, 1000, range(200), 128)
    assert docs.min() >= 150 and set(np.unique(docs)) <= set(g['etok'][0].tolist())
    cl = H.forward_logprobs(g, docs[:4])  # i.i.d.: the code length of a token does not depend on its past
    p = dict(zip(g['etok'][0].tolist(), g['emit'][0] / 4096))
    assert np.allclose(cl, -np.log2(np.vectorize(p.get)(docs[:4])))
    assert np.allclose(H.forward_logspace(g, docs[:4]), cl)
    emp = np.bincount(docs.ravel(), minlength=300)[150:] / docs.size
    assert 0.5 * np.abs(emp - C.doc_unigram(h, 128)[150:]).sum() < 0.03


def test_masked_loss_ignores_distractor_documents():
    torch.manual_seed(0)
    V = 300
    m = M.LM(V + 1, 16, 1, 2, 64, 128)
    rng = np.random.default_rng(0)
    b1 = rng.integers(0, 150, size=(4, 16))
    b2 = b1.copy()
    b2[[1, 3]] = rng.integers(150, 300, size=(2, 16))  # change only the distractor documents
    rows = torch.tensor([True, False, True, False])
    grads = []
    for b in (b1, b2):
        x, y = M.inputs_targets(b, V)
        m.zero_grad()
        loss = R.batch_loss(G.lm_forward(m, x), y, V, rows)
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    assert torch.equal(grads[0], grads[1])
    x, y = M.inputs_targets(b1, V)
    lg = G.lm_forward(m, x)
    assert torch.allclose(R.batch_loss(lg, y, V, rows), R.batch_loss(lg[rows], y[rows], V))


def test_shard_map_bijection_and_smoke():
    tasks = [dict(task_id=f'hmm_n{n:03d}_t{j}', role='target', K=n) for n in (8, 32) for j in (1, 2, 3)]
    seen = set()
    for i in range(R.n_shards(CFG)):
        j = R.shard_job(CFG, tasks, i)
        seen.add((j['size'], j['target_id'], j['arm'], j['alpha'], j['lr_multiplier'], j['seed']))
        assert (j['arm'] == 'none') == (j['alpha'] == 1.0)
        assert R.shard_index(CFG, j['size'], j['target_index'], j['arm'], j['alpha'], j['lr_multiplier'],
                             j['seed']) == i
    assert len(seen) == R.n_shards(CFG) == 2880
    s0, s1 = [R.shard_job(CFG, tasks, i) for i in R.smoke_shards(CFG)]
    assert (s0['size'], s0['arm'], s0['lr_multiplier']) == ('S', 'none', 0.5)
    assert (s1['size'], s1['arm'], s1['alpha'], s1['lr_multiplier'], s1['level']) == ('M', 'masked', 0.02, 8, 32)
    with pytest.raises(ValueError):
        R.shard_coords(CFG, 2880)


def test_lr_choice_and_divergence_censoring():
    tab = [dict(size='S', K=8, arm='hmm', alpha=0.1, lr=m, lnT=v, seed=1, target_id='t')
           for m, v in ((0.5, 18.0), (1, 17.0), (2, 17.0), (4, 17.5))]
    assert A.choose_lr(tab, lambda r: 1.0) == {('S', 8, 'hmm', 0.1): 1}
    assert A.choose_lr(tab, lambda r: 0.0 if r['lr'] == 1 else 1.0) == {('S', 8, 'hmm', 0.1): 2}
    pts = [(2 ** 20, 1.0, 0, 0), (2 ** 21, 0.5, 0, 0)]
    assert A.tstar(dict(pts=pts, diverged=True), 0.1, 2 ** 28) == (math.log(2 ** 28), 1)
    assert A.tstar(dict(pts=[], diverged=True), 0.1, 2 ** 28) == (math.log(2 ** 28), 1)
    pts2 = pts + [(2 ** 22, 0.05, 0, 0)]
    assert A.tstar(dict(pts=pts2, diverged=True), 0.1, 2 ** 28)[1] == 0


def _planted(e_arm, curv, lr_opt_one, sd=0.1, cap=math.log(2 ** 28), seed=0):
    """ln T* = 15 + 0.3 [M] - e ln a + 0.5 ln n (+ curv * ln(a / 0.2) below 0.2) + 0.25 (log2 m - log2 m*)^2, with
    the best multiplier m* = lr_opt_one at alpha = 1 and 4 below."""
    rng = np.random.default_rng(seed)
    tab = []
    for size, n, j, s, m in itertools.product(['S', 'M'], [8, 32], [1, 2, 3], [1, 2, 3], CFG['grid']['lr_multipliers']):
        for arm, a in R.cells(CFG):
            e = e_arm['hmm' if arm == 'none' else arm]
            la = math.log(a)
            y = 15 + (0.3 if size == 'M' else 0) - e * la + 0.5 * math.log(n)
            y += -curv[arm if arm != 'none' else 'hmm'] * min(la - math.log(0.2), 0)
            opt = lr_opt_one if a == 1 else 4
            y += 0.25 * (math.log2(m) - math.log2(opt)) ** 2 + rng.normal(0, sd)
            tab.append(dict(size=size, target_id=f'n{n}_t{j}', arm=arm, alpha=a, lr=float(m), seed=s, K=n,
                            lnT=min(y, cap), status=int(y >= cap)))
    return tab


def test_gates_on_planted_data():
    tab = _planted(dict(hmm=0.6, iid=0.95, masked=0.85), dict(hmm=0.3, iid=0.0, masked=0.0), lr_opt_one=0.5)
    point, fits, choice, per_seed, boot, g = A.full_analysis(CFG, tab, 150, 1, 3)
    assert abs(point['e_alpha_high|hmm|tuned'] - 0.6) < 0.06 and abs(point['e_alpha_low|hmm|tuned'] - 0.9) < 0.08
    assert abs(point['e_alpha|iid|tuned'] - 0.95) < 0.06 and abs(point['e_alpha|masked|tuned'] - 0.85) < 0.06
    assert choice[('S', 8, 'none', 1.0)] == 0.5 and choice[('M', 32, 'hmm', 0.05)] == 4
    assert point['e_alpha|hmm|x4'] < point['e_alpha|hmm|tuned'] - 0.1
    for k in ('G1', 'G2', 'G3', 'G4', 'G5'):
        assert g[k]['verdict'] == 'SUPPORTED', (k, g[k]['value'], g[k]['ci95'])
    assert A.independent_check(CFG, tab, fits, choice, 1e-4)['agree']
    null = _planted(dict(hmm=0.5, iid=0.5, masked=0.5), dict(hmm=0.0, iid=0.0, masked=0.0), lr_opt_one=4)
    g0 = A.full_analysis(CFG, null, 100, 1, 3)[5]
    for k in ('G1', 'G3', 'G4', 'G5'):
        assert g0[k]['verdict'] == 'NOT_ESTABLISHED', (k, g0[k]['value'], g0[k]['ci95'])
    assert g0['G2']['verdict'] == 'SUPPORTED' and g0['G3']['named'] is None


def test_model_counts_with_301_tokens():
    for s in ('S', 'M'):
        M.build_model(R.model_config(CFG, s))


def _tiny(tmp_path, monkeypatch):
    monkeypatch.setenv('TDC_ALLOW_CPU', '1')
    c24 = C.exp24_config()
    c24['concepts'].update(levels=[4], targets_per_level=3, band_calibration_candidates=8, entropy_band_bits=0.6,
                           entropy_band_fallback=1.0, quick_entropy_seqs=2, entropy_seqs=4, entropy_seq_len=400,
                           pool_size=2)
    c24['pilot'].update(level=4)
    cfg = copy.deepcopy(CFG)
    cfg['concepts'].update(levels=[4], pool_size=2)
    cfg['eval'].update(eval_docs_per_target=16, distractor_eval_docs=4)
    cfg['grid'].update(alphas=[0.5, 1.0], lr_multipliers=[1, 2], T_checkpoints=[1024, 2048, 4096],
                       keep_checkpoints=[4096], seeds=[25101])
    cfg['model'].update(batch_docs=8, data_workers=2, warmup_steps=2, sizes=dict(S=[1, 16, 2], M=[1, 16, 2]))
    m = M.LM(301, 16, 1, 2, 64, 128)
    cfg['model']['expected_params'] = dict(S=M.param_count(m), M=M.param_count(m))
    cfg['model']['expected_nonembedding'] = dict(S=M.nonembedding_count(m), M=M.nonembedding_count(m))
    prep = tmp_path / 'prep'
    C.build_prep(cfg, prep, log=lambda s: None, cfg24=c24)
    monkeypatch.setattr(C, 'build_all', lambda cfg_, log=print, cfg24=None, _b=C.build_all: _b(cfg_, log, c24))
    return cfg, prep


def test_tiny_shards_verify_merge(tmp_path, monkeypatch):
    cfg, prep = _tiny(tmp_path, monkeypatch)
    tasks = json.loads((prep / 'tasks.json').read_text())
    assert len(C.targets(tasks)) == 3 and len([t for t in tasks if t['similarity'] == 'iid']) == 6
    assert R.n_shards(cfg) == 2 * 3 * 4 * 2 * 1
    out = tmp_path / 'out'
    picks = [R.shard_index(cfg, 'S', 0, arm, a, 1, 25101) for arm, a in R.cells(cfg)]
    for i in picks:
        R.run_shard(cfg, prep, out, i, allow_cpu=True)
        VF.verify_shard(cfg, out, i, prep, allow_cpu=True)
        v = json.loads((out / 'shards' / f'{i:05d}' / 'VERIFIED').read_text())
        assert v['max_abs_diff_bits'] < 1e-9 and len(v['removed_after_replay']) == 2
    rec = {R.shard_job(cfg, tasks, i)['arm']: json.loads((out / 'shards' / f'{i:05d}' / 'run.json').read_text())
           for i in picks}
    assert all(abs(r['ident_bits']) < 1e-12 for r in rec['masked']['rows'])
    assert all(r['ident_bits'] > 0 for r in rec['hmm']['rows'] + rec['iid']['rows'])
    assert rec['iid']['pool'] == [p + '_iid' for p in rec['hmm']['pool']] == [p + '_iid' for p in rec['masked']['pool']]
    assert rec['hmm']['stream_sha256'] == rec['masked']['stream_sha256'] != rec['iid']['stream_sha256']
    monkeypatch.setattr(R, 'n_shards', lambda c: len(picks))
    real = R.shard_job
    monkeypatch.setattr(R, 'shard_job', lambda c, t, i: real(c, t, picks[i]))
    out2 = tmp_path / 'out2'
    for k, i in enumerate(picks):
        (out2 / 'shards').mkdir(parents=True, exist_ok=True)
        (out / 'shards' / f'{i:05d}').rename(out2 / 'shards' / f'{k:05d}')
    R.merge(cfg, out2, prep, tmp_path / 'merged')
    rows = (tmp_path / 'merged' / 'metrics.csv').read_text().splitlines()
    assert len(rows) == 1 + len(picks) * 3 * 4
    assert len((tmp_path / 'merged' / 'runs.csv').read_text().splitlines()) == 1 + len(picks)
    runs = A.read_runs(tmp_path / 'merged')
    assert len(runs) == len(picks) and all(len(d['pts']) == 3 for d in runs.values())
    (out2 / 'shards' / '00000' / 'VERIFIED').unlink()
    (out2 / 'shards' / '00000' / 'run.json').write_text('{}')
    with pytest.raises(VF.Fail):
        VF.verify_shard(cfg, out2, 0, prep, allow_cpu=True)


def test_tiny_diverged_run_stops_and_verifies(tmp_path, monkeypatch):
    cfg, prep = _tiny(tmp_path, monkeypatch)
    real = R.G.code_lengths
    calls = []

    def flaky(model, docs, bos, device, batch=128):
        calls.append(1)
        cl = real(model, docs, bos, device, batch)
        return cl * np.inf if len(calls) == 2 else cl
    monkeypatch.setattr(R.G, 'code_lengths', flaky)
    i = R.shard_index(cfg, 'M', 2, 'hmm', 0.5, 2, 25101)
    R.run_shard(cfg, prep, tmp_path / 'o', i, allow_cpu=True)
    monkeypatch.setattr(R.G, 'code_lengths', real)
    rec = json.loads((tmp_path / 'o' / 'shards' / f'{i:05d}' / 'run.json').read_text())
    assert rec['diverged_at'] == 2048 and rec['T_done'] == [1024] and not rec['finite'] and rec['steps_run'] == 2
    VF.verify_shard(cfg, tmp_path / 'o', i, prep, allow_cpu=True)
    v = json.loads((tmp_path / 'o' / 'shards' / f'{i:05d}' / 'VERIFIED').read_text())
    assert v['replayed'] == [1024] and v['removed_after_replay'] == []
