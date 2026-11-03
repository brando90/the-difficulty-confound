"""CPU tests of the Experiment 23 runner (PROTOCOL Section 8). Run from the experiment folder: python -m pytest -q tests

The slow tests train for 2^14-2^16 tokens (a few seconds to a minute each on one core).
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run23 as R  # noqa: E402

PREP = HERE / 'prep'
needs_prep = pytest.mark.skipif(not (PREP / 'prep_manifest.json').exists() or not (PREP / 'e13' / 'tasks.json').exists(),
                                reason='prep/ not built (run23.py build-prep)')


@pytest.fixture(scope='module')
def outs(tmp_path_factory):
    root = tmp_path_factory.mktemp('out')
    for exp in R.EXPS:
        R.init_output(exp, PREP, root / exp)
    return root


def _ctx(outs, exp):
    out = outs / exp
    tasks = json.loads((out / 'tasks.json').read_text())
    return R.base_config(exp), tasks, R.load_refs(out, tasks)


def _metrics(path, **key):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if all(r[k] == str(v) for k, v in key.items()):
                rows.append(r)
    return rows


# ---------------------------------------------------------------- pure logic

def test_hmm_is_experiment13_superset():
    """Experiment 17's hmm.py (imported here) equals Experiment 13's on every function Experiment 13 defines."""
    import ast
    src13 = (R.SRC['exp13'] / 'hmm.py').read_text()
    src17 = (R.SRC['exp17'] / 'hmm.py').read_text()
    f13 = {n.name: ast.dump(n) for n in ast.parse(src13).body if isinstance(n, ast.FunctionDef)}
    f17 = {n.name: ast.dump(n) for n in ast.parse(src17).body if isinstance(n, ast.FunctionDef)}
    assert set(f13) <= set(f17)
    assert all(f13[k] == f17[k] for k in f13), [k for k in f13 if f13[k] != f17[k]]
    assert (R.SRC['exp13'] / 'model.py').read_text() == (R.SRC['exp17'] / 'model.py').read_text()


def test_stop_rule():
    rule = dict(abs_bits=0.02, rel=0.05)
    ok, t = R.stop_rule({'a': (0.50, 0.49), 'b': (1.00, 0.96)}, rule)  # 0.01 <= 0.0245; 0.04 <= 0.048
    assert ok and all(v['converged'] for v in t.values())
    ok, t = R.stop_rule({'a': (0.50, 0.49), 'b': (1.00, 0.90)}, rule)  # 0.10 > 0.045
    assert not ok and t['a']['converged'] and not t['b']['converged']
    ok, t = R.stop_rule({'a': (0.10, 0.075)}, rule)  # small values: absolute floor 0.02 < 0.025 improvement
    assert not ok
    ok, _ = R.stop_rule({'a': (0.30, 0.35)}, rule)  # got worse: converged (no further gain)
    assert ok


def test_stages_and_checkpoints_consistent():
    st, ck = R.CFG['stages_T'], R.CFG['T_checkpoints']
    assert all(T in ck for T in st) and ck[-1] == st[-1]
    w = 2 ** R.CFG['stop_rule']['window_doublings']
    assert all(T // w in ck for T in st)  # the rule's T_k / 4 is always a checkpoint
    assert all(T % 1024 == 0 and int(T * R.CFG['cooldown_fraction']) % 1024 == 0 for T in st)


@needs_prep
def test_grid_and_packing():
    for exp, n_runs, n_tasks in (('e13', 256, 64), ('e17b', 160, 40)):
        tasks = json.loads((PREP / exp / 'tasks.json').read_text())
        g = R.grid(exp, tasks)
        assert len(g) == n_runs and len(set(g)) == n_runs
        assert R.n_tasks(exp, tasks) == n_tasks
        packed = [r for i in range(n_tasks) for r in R.task_runs(exp, tasks, i)]
        assert packed == g
        assert len({R.run_name(exp, r) for r in g}) == n_runs
    fs = R.CFG['farmshare']
    total = (64 + 40) * len(R.CFG['stages_T']) + 2 * len(R.CFG['stages_T']) + 1
    assert total <= fs['qos_submit_limit'] and sum(fs['throttle'].values()) * R.CFG['runs_per_task'] <= 4 * fs[
        'qos_running_limit']
    tasks = json.loads((PREP / 'e17b' / 'tasks.json').read_text())
    cells = {R.monitor_cell('e17b', tasks, r) for r in R.grid('e17b', tasks)}
    assert len(cells) == 4 * 2 + 4  # (pool, n) at alpha_H, plus each pool at alpha = 0


@needs_prep
def test_validation_disjoint_from_eval_and_training():
    """Validation documents differ from every eval document (different stream seed) and from the training streams."""
    for exp in R.EXPS:
        tasks = json.loads((PREP / exp / 'tasks.json').read_text())
        v = np.load(PREP / exp / 'val.npz')
        for t in R.targets_of(tasks):
            vd = v[f"docs|{t['task_id']}"]
            ev = np.load(PREP / exp / 'data' / f"{t['task_id']}_test.npz")['docs']
            hv = {d.tobytes() for d in vd.astype(np.int16)}
            he = {d.tobytes() for d in ev.astype(np.int16)}
            assert len(hv) > 0.95 * len(vd) and not (hv & he)
            h = R.H.from_json(t['definition'])
            tr = R.H.sample_docs(h, R.CFG[exp]['seeds'][0], t['cid'], range(256), 128).astype(np.int16)
            assert not (hv & {d.tobytes() for d in tr})
            assert np.isfinite(v[f"oracle|{t['task_id']}"]).all()


def _shrink(out, n_eval=32, n_val=16):
    """Test-only: keep the first n_eval eval documents and n_val validation documents per concept in an output copy,
    so the miniature grids below evaluate in a fraction of a second."""
    for p in (out / 'data').glob('*_test.npz'):
        d = np.load(p)['docs'][:n_eval]
        with open(p, 'wb') as f:
            np.savez(f, docs=d)
    for name in ('oracle.npz', 'mixture.npz', 'val.npz'):
        z = np.load(out / name)
        arr = {k: z[k][:(n_val if name == 'val.npz' else n_eval)] for k in z.files}
        with open(out / name, 'wb') as f:
            np.savez(f, **arr)


# ---------------------------------------------------------------- training: exact replay and exact resume

@needs_prep
def test_exact_replay_experiment13_small(outs):
    """The first checkpoint (2^16 tokens) of an Exp 13-small run is reproduced from its committed metrics."""
    c, tasks, refs = _ctx(outs, 'e13')
    run = (13101, 'D', 'hmm_n004_t1', 0.05)
    rec = R.train_stage('e13', c, tasks, refs, run, 0, outs / 'e13', stages=[65536], ckpts=[65536],
                        save_weights=False)
    ref = _metrics(R.SRC['exp13'] / 'results' / 'small' / 'metrics.csv', seed=13101, target_id='hmm_n004_t1',
                   alpha=0.05, T=65536)
    assert len(ref) == 4
    for r in ref:
        mine = next(x for x in rec['rows'] if x['slice'] == r['slice'])
        for k in ('excess_bits', 'ident_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'excess_se'):
            assert abs(mine[k] - float(r[k])) < 1e-6, (r['slice'], k, mine[k], r[k])
        assert mine['N'] == int(r['N']) and mine['step'] == int(r['step']) and mine['n_tokens'] == int(r['n_tokens'])


@needs_prep
@pytest.mark.parametrize('run,source', [((17101, 'cov32', 'hmm_n032_t1', 0.05), 'exp17b'),
                                        ((17101, 'hard', '-', 0.0), 'exp17')])
def test_exact_replay_experiment17(outs, run, source):
    """The first checkpoint of an Exp 17b coverage run and of an Exp 17 hard-pool alpha = 0 run (scored on all four
    targets) is reproduced from their committed metrics."""
    c, tasks, refs = _ctx(outs, 'e17b')
    rec = R.train_stage('e17b', c, tasks, refs, run, 0, outs / 'e17b', stages=[65536], ckpts=[65536],
                        save_weights=False)
    ref = _metrics(R.SRC[source] / 'results' / 'metrics.csv', seed=run[0], pool=run[1], run_target=run[2], T=65536)
    assert len(ref) == 4 * len(R.scored_targets(tasks, run))
    for r in ref:
        mine = next(x for x in rec['rows'] if x['slice'] == r['slice'] and x['target_id'] == r['target_id'])
        for k in ('excess_bits', 'ident_bits', 'model_bits', 'oracle_bits', 'mixture_bits', 'excess_se'):
            assert abs(mine[k] - float(r[k])) < 1e-6, (r['target_id'], r['slice'], k, mine[k], r[k])
        assert mine['target_docs_seen'] == int(r['target_docs_seen'])


@needs_prep
def test_resume_is_bitwise(outs, tmp_path):
    """Two stages (2^13 then 2^14 tokens, resumed from the saved state) equal one continuous 2^14-token run."""
    c, tasks, refs = _ctx(outs, 'e13')
    run = (13202, 'D', 'hmm_n008_t2', 0.15)
    a, b = tmp_path / 'a', tmp_path / 'b'
    for d in (a, b):
        shutil.copytree(outs / 'e13', d, ignore=shutil.ignore_patterns('state', 'runs', 'checkpoints'))
    R.train_stage('e13', c, tasks, refs, run, 0, a, stages=[8192, 16384], ckpts=[8192, 16384])
    r2 = R.train_stage('e13', c, tasks, refs, run, 1, a, stages=[8192, 16384], ckpts=[8192, 16384])
    r1 = R.train_stage('e13', c, tasks, refs, run, 0, b, stages=[16384], ckpts=[8192, 16384])
    name = R.run_name('e13', run)
    sa = torch.load(a / 'state' / name / 'stage_1.pt', weights_only=True)
    sb = torch.load(b / 'state' / name / 'stage_0.pt', weights_only=True)
    for k in sa['model']:
        assert torch.equal(sa['model'][k], sb['model'][k]), k
    for i in sa['opt']['state']:
        for k in sa['opt']['state'][i]:
            assert torch.equal(sa['opt']['state'][i][k], sb['opt']['state'][i][k]), (i, k)
    ra = [x for x in r2['rows'] if x['T'] == 16384]
    rb = [x for x in r1['rows'] if x['T'] == 16384]
    assert [x['excess_bits'] for x in ra] == [x['excess_bits'] for x in rb]
    # the cooldown branch never touches the saved state: stage 1 started from exactly what stage 0 saved
    assert r2['T_start'] == 8192 and r2['cooldown_tokens'] == 2048
    assert [x['excess_bits'] for x in r2['cooldown_rows']] == [x['excess_bits'] for x in r1['cooldown_rows']]


@needs_prep
def test_interrupted_stage_resumes_bitwise(outs, tmp_path):
    """A stage interrupted after its 2^13 checkpoint resumes from the partial state and ends identical to an
    uninterrupted stage (weights, optimizer, rows, stream hash)."""
    c, tasks, refs = _ctx(outs, 'e13')
    run = (13303, 'D', 'hmm_n016_t1', 0.4)
    a, b = tmp_path / 'a', tmp_path / 'b'
    for d in (a, b):
        shutil.copytree(outs / 'e13', d, ignore=shutil.ignore_patterns('state', 'runs', 'checkpoints'))
    kw = dict(stages=[16384], ckpts=[4096, 8192, 16384])
    with pytest.raises(R._Interrupt):
        R.train_stage('e13', c, tasks, refs, run, 0, a, _interrupt_at=8192, **kw)
    name = R.run_name('e13', run)
    assert (a / 'state' / name / 'partial_0.json').exists()
    ra = R.train_stage('e13', c, tasks, refs, run, 0, a, **kw)
    rb = R.train_stage('e13', c, tasks, refs, run, 0, b, **kw)
    assert ra['resumed_at'] == [8192] and rb['resumed_at'] == []
    assert not (a / 'state' / name / 'partial_0.json').exists()
    sa = torch.load(a / 'state' / name / 'stage_0.pt', weights_only=True)
    sb = torch.load(b / 'state' / name / 'stage_0.pt', weights_only=True)
    assert all(torch.equal(sa['model'][k], sb['model'][k]) for k in sa['model'])
    assert ra['stream_sha256'] == rb['stream_sha256']
    assert [x['excess_bits'] for x in ra['rows']] == [x['excess_bits'] for x in rb['rows']]
    assert [x['excess_bits'] for x in ra['cooldown_rows']] == [x['excess_bits'] for x in rb['cooldown_rows']]


@needs_prep
def test_stage_decide_aggregate_end_to_end(tmp_path, monkeypatch):
    """A miniature grid (2 seeds, stages 2^13/2^14) through cmd_stage, cmd_decide (CONTINUE, then STOP at the cap) and
    cmd_aggregate; the aggregated metrics are in Experiment 13's format and readable by its analysis loader."""
    cfg = json.loads(json.dumps(R.CFG))
    cfg['e13']['seeds'] = cfg['e13']['seeds'][:1]
    cfg['e13']['alphas'] = [0.4, 1.0]
    cfg['stages_T'] = [8192, 16384]
    cfg['T_checkpoints'] = [2048, 4096, 8192, 16384]
    cfg['stop_rule'] = dict(window_doublings=2, abs_bits=0.0, rel=0.0)  # never converged -> stop at the cap
    monkeypatch.setattr(R, 'CFG', cfg)
    out = tmp_path / 'e13'
    R.init_output('e13', PREP, out)
    _shrink(out)
    tasks = json.loads((out / 'tasks.json').read_text())
    n = R.n_tasks('e13', tasks)
    assert n == 4
    for k in range(2):
        for i in range(n):
            assert R.cmd_stage('e13', k, i, out, 4) == 0
        assert R.cmd_decide('e13', k, out) == 0
    d0 = json.loads((out / 'decisions' / 'stage_0.json').read_text())
    assert d0['decision'] == 'CONTINUE' and len(d0['cells']) == 8
    stop = json.loads((out / 'STOP').read_text())
    assert stop == {**stop, 'T_stop': 16384, 'stage': 1, 'reason': 'cap'}
    assert R.cmd_stage('e13', 1, 0, out, 4) == 0  # resume-safe: nothing left to do
    R.cmd_aggregate('e13', out)
    rows = list(csv.DictReader(open(out / 'metrics.csv')))
    assert list(rows[0]) == R.E13_COLS and len(rows) == 16 * 4 * 4
    assert {int(r['T']) for r in rows} == {2048, 4096, 8192, 16384}
    cool = list(csv.DictReader(open(out / 'cooldown.csv')))
    assert len(cool) == 16 * 2 * 4 and {r['phase'] for r in cool} == {'cooldown'}
    dl = list(csv.DictReader(open(out / 'distractor_loss.csv')))
    assert len(dl) == 16 * 4 * 8


@needs_prep
def test_decide_halts_on_missing_runs(tmp_path, monkeypatch):
    cfg = json.loads(json.dumps(R.CFG))
    cfg['e13']['seeds'] = cfg['e13']['seeds'][:1]
    cfg['e13']['alphas'] = [1.0]
    cfg['stages_T'] = [8192, 16384]
    cfg['T_checkpoints'] = [2048, 8192, 16384]
    monkeypatch.setattr(R, 'CFG', cfg)
    out = tmp_path / 'e13'
    R.init_output('e13', PREP, out)
    _shrink(out)
    assert R.cmd_stage('e13', 0, 0, out, 4) == 0  # 4 of 8 runs
    assert R.cmd_decide('e13', 0, out) == 1
    assert not (out / 'STOP').exists()
    assert R.cmd_stage('e13', 1, 1, out, 4) == 2  # stage 1 refuses without stage 0's state
