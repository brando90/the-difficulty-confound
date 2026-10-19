"""CPU checks that the stacked ensemble is Experiment 11's model, step for step.

Run: python -m pytest -q test_ensemble.py   (needs ../../11_dfa_exact_complexity_vs_alignment/expt_v1 for model.py)
"""
import copy

import numpy as np
import torch

import common as C
import ensemble as E

torch.set_num_threads(1)
c11 = C.load_c11()


def _models(seeds):
    out = []
    for s in seeds:
        torch.manual_seed(s)
        out.append(C.M11.build_model(c11))
    return out


def _batch(R, B, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (R, B, L + 1), generator=g)
    x[:, :, 0] = 2 + torch.randint(0, 32, (R, B), generator=g)
    y = torch.randint(0, 2, (R, B), generator=g)
    return x, y


def test_forward_matches_model_py():
    ms = _models([11101, 11202, 11303])
    P = E.init_stack(ms, 'cpu')
    for L in (8, 17, 24):
        x, _ = _batch(3, 16, L, L)
        got = E.forward(P, x, c11['n_layers'], c11['n_heads'])
        for r, m in enumerate(ms):
            want = m(x[r])
            assert torch.allclose(got[r], want, atol=1e-5), (L, r, (got[r] - want).abs().max())


def test_training_steps_match_model_py():
    """Five AdamW steps with warmup and per-model clipping equal five separate Experiment 11 steps."""
    ms = _models([11101, 11505])
    ref = [copy.deepcopy(m) for m in ms]
    P = E.init_stack(ms, 'cpu')
    opt = torch.optim.AdamW(list(P.values()), lr=c11['lr'], betas=tuple(c11['betas']),
                            weight_decay=c11['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c11['warmup_steps']))
    ropt = [torch.optim.AdamW(m.parameters(), lr=c11['lr'], betas=tuple(c11['betas']),
                              weight_decay=c11['weight_decay']) for m in ref]
    rsch = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / c11['warmup_steps'])) for o in ropt]
    for step in range(5):
        x, y = _batch(2, 64, 12 + step, 100 + step)
        logits = E.forward(P, x, c11['n_layers'], c11['n_heads'])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), reduction='none')
        loss.reshape(2, -1).mean(1).sum().backward()
        E.clip_per_model_(P, 0.05)  # small max_norm so clipping is active
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        for r, m in enumerate(ref):
            lo = torch.nn.functional.cross_entropy(m(x[r]), y[r])
            lo.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.05)
            ropt[r].step(); rsch[r].step(); ropt[r].zero_grad(set_to_none=True)
    for r, m in enumerate(ref):
        for k, v in m.state_dict().items():
            assert torch.allclose(P[k][r], v, atol=2e-5), (r, k, (P[k][r] - v).abs().max())


def test_export_roundtrip():
    ms = _models([11101])
    P = E.init_stack(ms, 'cpu')
    m = E.load_one(C.M11.build_model(c11), E.export_one(P, 0))
    for k, v in ms[0].state_dict().items():
        assert torch.equal(m.state_dict()[k], v)


def test_gpu_labels_match_run_dfa():
    tasks = C.load_tasks()
    tb = C.TaskTables(tasks, 'cpu')
    rng = np.random.default_rng(3)
    for L in (8, 15, 24):
        bits = torch.from_numpy(rng.integers(0, 2 ** L, (4, 50)).astype(np.int64))
        tok = torch.from_numpy(rng.integers(0, 32, (4, 50)).astype(np.int64))
        got = tb.labels(tok, bits, L)
        for t in tasks:
            sel = tok == t['task_token']
            if sel.any():
                want = C.D11.labels_for(t, bits[sel].numpy().astype(np.uint64), np.full(int(sel.sum()), L))
                assert np.array_equal(got[sel].numpy(), want.astype(np.int64))


def test_test_mask_matches_hash():
    rng = np.random.default_rng(5)
    for L in (8, 13, 20):
        mask = C.test_mask_for_length(L, c11)
        cand = rng.integers(0, 2 ** L, 2000, dtype=np.uint64)
        want = C.D11.in_test_partition(cand, L, c11)
        assert np.array_equal(mask[cand.astype(np.int64)], want)


def test_schedule_exact_shares():
    for alpha in (0.05, 0.15, 0.4, 1.0):
        k = C.target_counts(alpha, 64, 2 ** 12 // 64)
        cum = np.cumsum(k)
        for s in range(len(k)):
            T = 64 * (s + 1)
            assert cum[s] == int(np.floor(alpha * T + 0.5))
