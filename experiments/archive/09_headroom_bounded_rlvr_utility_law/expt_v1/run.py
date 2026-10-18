"""Carry-marginal matched heterogeneous pretraining, then Verifiable-Reward RL (RLVR / GRPO) at several rollout group sizes.

Each starting checkpoint is RL-trained once per group size G from identical saved weights, with prompts per
update B = rollouts_per_update / G, so every update draws the same number of rollouts.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import math
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from addition import carry_chain, column_addition, digits_of, model, save_json, sha, write_csv

ROOT = Path(__file__).resolve().parent


def group_sizes(c):
    return sorted(c['train_group_sizes'] + c['heldout_group_sizes'])


def carry_cells(a, b):
    da, db = digits_of(a, 3), digits_of(b, 3)
    c0 = (da[:, 0] + db[:, 0]) >= 10
    c1 = (da[:, 1] + db[:, 1] + c0) >= 10
    c2 = (da[:, 2] + db[:, 2] + c1) >= 10
    cell = np.full(len(a), -1, dtype=int)
    valid = ((a + b) < 1000) & (~c2)
    cell[valid & (~c0) & (~c1)] = 0
    cell[valid & c0 & (~c1)] = 1
    cell[valid & (~c0) & c1] = 2
    cell[valid & c0 & c1] = 3
    stratum = np.where(np.isin(cell, [1, 2]), 1, np.where(np.isin(cell, [0, 3]), 2, -1))
    return cell, stratum, c0, c1, c2


def dataset(c):
    a, b = np.triu_indices(1000)
    cell, stratum, _, _, _ = carry_cells(a, b)
    y = a + b
    counts = np.stack([np.bincount(y[cell == cid], minlength=1000) for cid in range(4)], axis=1)
    answers = np.flatnonzero(counts.min(axis=1) >= c['minimum_pairs_per_answer_cell'])
    rng = np.random.default_rng(c['split_seed'])
    splits = {name: [] for name in ('test', 'calibration', 'probe')}
    pools = {}
    for answer in answers:
        for cid in range(4):
            population = np.flatnonzero((y == answer) & (cell == cid))
            ids = rng.permutation(population)
            off = 0
            for name in splits:
                n = c[name + '_per_answer_cell']
                splits[name].extend(ids[off : off + n])
                off += n
            chosen = ids[:off]
            pools[int(answer), cid] = rng.permutation(np.setdiff1d(population, chosen))
    return a, b, cell, stratum, answers, pools, {n: np.asarray(v) for n, v in splits.items()}


def training_ids(c, pools, seed, q):
    rng = np.random.default_rng(seed)
    chosen = []
    half = c['train_per_answer'] // 2
    for (answer, cid), ids in sorted(pools.items()):
        perm = rng.permutation(ids)
        n = round(half * (q if cid in (0, 3) else 1 - q))
        chosen.extend(perm[:n])
    chosen = np.asarray(chosen)
    return np.random.default_rng(seed + 1000).permutation(chosen)


def check(c):
    a, b, cell, stratum, answers, pools, splits = dataset(c)
    _, _, c0, c1, c2 = carry_cells(a, b)
    assert len(answers) == 180 and np.all(a <= b)
    sets = [set(v) for v in splits.values()] + [set(np.concatenate(list(pools.values())))]
    assert sum(map(len, sets)) == len(set.union(*sets))
    assert all(column_addition(int(x), int(y), 3) == x + y for x, y in zip(a, b))
    assert carry_chain(np.array([123, 109, 199, 999]), np.array([234, 1, 1, 1]), 3).tolist() == [0, 1, 2, 3]
    for name, ids in splits.items():
        assert len(ids) == 4 * len(answers) * c[name + '_per_answer_cell']
        for cid in range(4):
            vals, cnt = np.unique((a + b)[ids[cell[ids] == cid]], return_counts=True)
            assert np.array_equal(vals, answers) and np.all(cnt == c[name + '_per_answer_cell'])
    for seed in c['seeds']:
        for q in c['train_mixtures'] + c['heldout_mixtures']:
            ids = training_ids(c, pools, seed, q)
            assert len(set(ids)) == len(ids) == 7200
            assert not set(ids) & set.union(*sets[:3])
            vals, cnt = np.unique((a + b)[ids], return_counts=True)
            assert np.array_equal(vals, answers) and np.all(cnt == 40)
            assert int(c0[ids].sum()) == 3600
            assert int(c1[ids].sum()) == 3600
            assert int(c2[ids].sum()) == 0
    assert set(c['train_widths']).isdisjoint(c['heldout_widths'])
    assert set(c['train_mixtures']).isdisjoint(c['heldout_mixtures'])
    # Fixed rollout budget: every group size divides it, and the primary slice keeps Experiment 08's B=256, G=16.
    assert set(c['train_group_sizes']).isdisjoint(c['heldout_group_sizes'])
    assert all(c['rollouts_per_update'] % g == 0 for g in group_sizes(c))
    assert c['primary_group_size'] in c['train_group_sizes']
    assert c['rollouts_per_update'] // c['primary_group_size'] == c['batch_size']
    return dict(
        oracle_pairs=500500,
        eligible_answers=len(answers),
        train_size=7200,
        test=len(splits['test']),
        calibration=len(splits['calibration']),
        probe=len(splits['probe']),
        split_overlap=0,
        exact_answer_matching=True,
        ones_carry_rate=0.5,
        tens_carry_rate=0.5,
        hundreds_carry_rate=0.0,
        post_mixture=c['post_mixture'],
        group_sizes=group_sizes(c),
        prompts_per_update={g: c['rollouts_per_update'] // g for g in group_sizes(c)},
        verifiable_exact_sum_reward=True,
    )


@torch.no_grad()
def evaluate(net, x, y, ids):
    preds, bits = [], []
    for off in range(0, len(ids), 1024):
        ii = ids[off : off + 1024]
        z = net(x[ii]).reshape(-1, 4, 10)
        preds.append((z.argmax(-1) * 10 ** torch.arange(4)).sum(-1).numpy())
        bits.append(
            F.cross_entropy(z.reshape(-1, 10), y[ii].reshape(-1), reduction='none')
            .reshape(-1, 4)
            .sum(1)
            .numpy()
            / math.log(2)
        )
    return np.concatenate(preds), np.concatenate(bits)


def advance_supervised(net, x, y, ids, c, generator, steps):
    opt = torch.optim.AdamW(net.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    tx, ty = x[ids], y[ids]
    for _ in range(1, steps + 1):
        batch = torch.randint(len(tx), (c['batch_size'],), generator=generator)
        loss = F.cross_entropy(net(tx[batch]).reshape(-1, 10), ty[batch].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite supervised training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def advance_rlvr(net, x, y, ids, stratum_ids, cell_ids, c, generator, steps, G, on_step=None):
    opt = torch.optim.AdamW(net.parameters(), lr=c['rl_learning_rate'], weight_decay=c['weight_decay'])
    tx, ty = x[ids], y[ids]
    s_arr = stratum_ids[ids]
    c_arr = cell_ids[ids]
    B = c['rollouts_per_update'] // G  # fixed rollout budget per update
    cum_s_active = {1: 0, 2: 0}
    cum_s_groups = {1: 0, 2: 0}
    cum_c_active = {cid: 0 for cid in range(4)}
    cum_c_groups = {cid: 0 for cid in range(4)}
    for step in range(1, steps + 1):
        batch = torch.randint(len(tx), (B,), generator=generator)
        logits = net(tx[batch]).reshape(B, 4, 10)
        logp = F.log_softmax(logits, dim=-1)
        cdf = logp.detach().exp().cumsum(dim=-1)
        u = torch.rand((B, G, 4, 1), generator=generator)
        actions = (u > cdf.unsqueeze(1)).sum(dim=-1).clamp(max=9)
        traj_logp = logp.unsqueeze(1).expand(-1, G, -1, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
        # Verifiable 0/1 exact-sum reward: 1 iff all 4 digits match exact sum, 0 otherwise
        rew = (actions == ty[batch].unsqueeze(1)).all(dim=-1).float()
        rmean = rew.mean(dim=1, keepdim=True)
        adv = (G / (G - 1.0)) * (rew - rmean)
        loss = -(adv.detach() * traj_logp).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite RLVR training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if on_step is not None:
            active_mask = ((rmean.squeeze(1) > 0.0) & (rmean.squeeze(1) < 1.0)).numpy()
            b_np = batch.numpy()
            bs = s_arr[b_np]
            bc = c_arr[b_np]
            for s in (1, 2):
                m = bs == s
                cum_s_active[s] += int(active_mask[m].sum())
                cum_s_groups[s] += int(m.sum())
            for cid in range(4):
                mc = bc == cid
                cum_c_active[cid] += int(active_mask[mc].sum())
                cum_c_groups[cid] += int(mc.sum())
            on_step(step, cum_s_active, cum_s_groups, cum_c_active, cum_c_groups)


def run_seed(seed, c, out):
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    torch.set_num_threads(c['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a, b, cell_arr, stratum, answers, pools, splits = dataset(c)
    xd = np.concatenate([digits_of(a, 3), digits_of(b, 3)], 1)
    x = F.one_hot(torch.from_numpy(xd).long(), 10).float().reshape(-1, 60)
    y = torch.from_numpy(digits_of(a + b, 4)).long()
    widths = c['train_widths'] + c['heldout_widths']
    mixtures = c['train_mixtures'] + c['heldout_mixtures']
    trains = {q: training_ids(c, pools, seed, q) for q in mixtures}
    post_ids = trains[c['post_mixture']]
    np.savez_compressed(out / f'train_{seed}.npz', **{f'q{q:.2f}': ids for q, ids in trains.items()})
    marginals = []
    for q, ids in trains.items():
        for column in range(6):
            hist = np.bincount(xd[ids, column], minlength=10)
            marginals.extend(
                dict(seed=seed, mixture=q, column=column, digit=d, count=int(hist[d]), n=len(ids)) for d in range(10)
            )
    probes = []
    pre_metrics = {name: [] for name in ('calibration', 'test')}
    metrics = {name: [] for name in ('calibration', 'test')}
    cell_metrics = []
    seed_cells = []
    started = time.monotonic()
    q_post = float(c['post_mixture'])
    for width in widths:
        for q_pre in mixtures:
            cell_start = time.monotonic()
            torch.manual_seed(seed)
            net = model(dict(hidden_width=width))
            count = sum(p.numel() for p in net.parameters())
            advance_supervised(
                net,
                x,
                y,
                trains[q_pre],
                c,
                torch.Generator().manual_seed(seed + int(round(q_pre * 1000)) + 20000),
                c['pretrain_steps'],
            )
            warm_state = {n: v.clone() for n, v in net.state_dict().items()}
            torch.save(warm_state, out / f'warm_{seed}_{width}_{q_pre:.2f}.pt')

            # Independent probe before any RL: starting bits, sampled pass@1, and per-G signal density.
            pred_probe, bits_probe = evaluate(net, x, y, splits['probe'])
            np.savez_compressed(
                out / f'probe_{seed}_{width}_{q_pre:.2f}.npz',
                item_id=splits['probe'],
                prediction=pred_probe,
                bits=bits_probe,
            )
            p_item = np.power(2.0, -bits_probe)
            probe = {}
            for s in (1, 2):
                pmask = stratum[splits['probe']] == s
                probe[s] = dict(starting_bits=float(bits_probe[pmask].mean()), probe_pass1=float(p_item[pmask].mean()))
                for G in group_sizes(c):
                    pg = p_item[pmask]
                    probe[s][f'probe_rho_g{G}'] = float(np.mean(1.0 - pg**G - (1.0 - pg) ** G))
                    probe[s][f'probe_coverage_g{G}'] = float(np.mean(1.0 - (1.0 - pg) ** G))
                probes.append(dict(seed=seed, width=width, mixture=q_pre, carry_chain=s, **probe[s], n=int(pmask.sum())))

            def eval_split(name, step, G):
                ids = splits[name]
                prediction, loss = evaluate(net, x, y, ids)
                tag = '0' if step == 0 else f'g{G}_{step}'
                np.savez_compressed(out / f'{name}_{seed}_{width}_{q_pre:.2f}_{tag}.npz', item_id=ids, prediction=prediction, bits=loss)
                return ids, prediction, loss

            # Pre-RL evaluation, shared by every group size.
            base_acc = {name: {} for name in ('calibration', 'test')}
            for name in pre_metrics:
                ids, prediction, loss = eval_split(name, 0, None)
                for s in (1, 2):
                    mask = stratum[ids] == s
                    corr = int(np.sum(prediction[mask] == (a + b)[ids[mask]]))
                    base_acc[name][s] = corr / int(mask.sum())
                    pre_metrics[name].append(
                        dict(
                            seed=seed,
                            width=width,
                            mixture=q_pre,
                            steps=0,
                            carry_chain=s,
                            parameter_count=count,
                            pretrain_stratum_fraction=q_pre if s == 2 else 1 - q_pre,
                            **probe[s],
                            mean_bits=float(loss[mask].mean()),
                            correct=corr,
                            n=int(mask.sum()),
                        )
                    )
                if name == 'test':
                    for cid in range(4):
                        cmask = cell_arr[ids] == cid
                        cell_metrics.append(
                            dict(
                                seed=seed,
                                width=width,
                                mixture=q_pre,
                                group_size=0,
                                steps=0,
                                cell_id=cid,
                                online_active_group_fraction=0.0,
                                mean_bits=float(loss[cmask].mean()),
                                correct=int(np.sum(prediction[cmask] == (a + b)[ids[cmask]])),
                                n=int(cmask.sum()),
                            )
                        )

            for G in group_sizes(c):
                net.load_state_dict(warm_state)  # every group size starts from the identical checkpoint

                def on_step(step, cum_s_act, cum_s_grp, cum_c_act, cum_c_grp):
                    if step not in c['checkpoints']:
                        return
                    for name in metrics:
                        ids, prediction, loss = eval_split(name, step, G)
                        for s in (1, 2):
                            mask = stratum[ids] == s
                            corr = int(np.sum(prediction[mask] == (a + b)[ids[mask]]))
                            n_s = int(mask.sum())
                            metrics[name].append(
                                dict(
                                    seed=seed,
                                    width=width,
                                    mixture=q_pre,
                                    group_size=G,
                                    prompts_per_update=c['rollouts_per_update'] // G,
                                    steps=step,
                                    carry_chain=s,
                                    parameter_count=count,
                                    pretrain_stratum_fraction=q_pre if s == 2 else 1 - q_pre,
                                    starting_bits=probe[s]['starting_bits'],
                                    probe_pass1=probe[s]['probe_pass1'],
                                    probe_rho=probe[s][f'probe_rho_g{G}'],
                                    probe_coverage=probe[s][f'probe_coverage_g{G}'],
                                    start_accuracy=base_acc[name][s],  # pre-RL accuracy on the same split
                                    online_active_group_fraction=float(cum_s_act[s] / max(cum_s_grp[s], 1)),
                                    rl_utility=float(corr / n_s - base_acc[name][s]),
                                    mean_bits=float(loss[mask].mean()),
                                    correct=corr,
                                    n=n_s,
                                )
                            )
                        if name == 'test':
                            for cid in range(4):
                                cmask = cell_arr[ids] == cid
                                cell_metrics.append(
                                    dict(
                                        seed=seed,
                                        width=width,
                                        mixture=q_pre,
                                        group_size=G,
                                        steps=step,
                                        cell_id=cid,
                                        online_active_group_fraction=float(cum_c_act[cid] / max(cum_c_grp[cid], 1)),
                                        mean_bits=float(loss[cmask].mean()),
                                        correct=int(np.sum(prediction[cmask] == (a + b)[ids[cmask]])),
                                        n=int(cmask.sum()),
                                    )
                                )

                advance_rlvr(
                    net,
                    x,
                    y,
                    post_ids,
                    stratum,
                    cell_arr,
                    c,
                    torch.Generator().manual_seed(seed + 10000 + G),
                    max(c['checkpoints']),
                    G,
                    on_step,
                )
                torch.save(net.state_dict(), out / f'model_{seed}_{width}_{q_pre:.2f}_g{G}.pt')
            c_rec = dict(
                seed=seed,
                width=width,
                mixture=q_pre,
                status='complete',
                parameter_count=count,
                elapsed_seconds=time.monotonic() - cell_start,
            )
            seed_cells.append(c_rec)
            print(json.dumps(c_rec), flush=True)
    return dict(
        seed=seed,
        marginals=marginals,
        probes=probes,
        pre_calibration=pre_metrics['calibration'],
        pre_test=pre_metrics['test'],
        calibration=metrics['calibration'],
        test=metrics['test'],
        cell_test=cell_metrics,
        cells=seed_cells,
        elapsed=time.monotonic() - started,
    )


def smoke_config(c):
    """Tiny end-to-end run of the full grid for pipeline checks and timing; never used for results."""
    return {**c, 'seeds': c['seeds'][:2], 'pretrain_steps': 30, 'checkpoints': [5, 10, 20], 'smoke': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='2 seeds, 30 pretraining and 20 RL updates')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    c = json.loads((ROOT / 'config.json').read_text())
    checks = check(c)
    print(json.dumps(checks), flush=True)
    if args.check_only:
        return
    if args.smoke:
        c = smoke_config(c)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'runtime.json').exists():
        raise RuntimeError('Refusing to overwrite run')
    save_json(out / 'config.json', c)  # effective config read by analyze.py and verify.py
    started = time.monotonic()
    runtime = dict(
        host=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        device='cpu',
        threads=c['threads'],
        workers=c['workers'],
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={p.name: sha(p) for p in ROOT.glob('*.py')},
        config_sha256=sha(ROOT / 'config.json'),
        smoke=bool(args.smoke),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        checks=checks,
        expected_cells=len(c['seeds']) * 25,
        expected_trajectories=len(c['seeds']) * 25 * len(group_sizes(c)),
    )
    save_json(out / 'runtime.json', runtime)
    a, b, cell_arr, stratum, answers, pools, splits = dataset(c)
    np.savez_compressed(out / 'heldout_items.npz', a=a, b=b, cell=cell_arr, carry=stratum, answers=answers, **splits)
    write_csv(
        out / 'answer_counts.csv',
        [
            dict(
                answer=int(ans),
                cell_id=cid,
                population=int(np.sum((a + b == ans) & (cell_arr == cid))),
                test=4,
                calibration=2,
                probe=2,
                pool=len(pools[int(ans), cid]),
            )
            for ans in answers
            for cid in range(4)
        ],
    )
    with ProcessPoolExecutor(max_workers=c['workers']) as ex:
        results = list(ex.map(partial(run_seed, c=c, out=out), c['seeds']))
    marginals, probes, pre_cal, pre_test, cal, test, cell_test, cells = [], [], [], [], [], [], [], []
    for r in results:
        marginals.extend(r['marginals'])
        probes.extend(r['probes'])
        pre_cal.extend(r['pre_calibration'])
        pre_test.extend(r['pre_test'])
        cal.extend(r['calibration'])
        test.extend(r['test'])
        cell_test.extend(r['cell_test'])
        cells.extend(r['cells'])
    write_csv(out / 'input_digit_marginals.csv', marginals)
    write_csv(out / 'probe_features.csv', probes)
    write_csv(out / 'pretrain_calibration_metrics.csv', pre_cal)
    write_csv(out / 'pretrain_test_metrics.csv', pre_test)
    write_csv(out / 'calibration_metrics.csv', cal)
    write_csv(out / 'test_metrics.csv', test)
    write_csv(out / 'cell_test_metrics.csv', cell_test)
    save_json(out / 'cells.json', cells)
    runtime.update(
        completed_cells=len(cells),
        elapsed_seconds=time.monotonic() - started,
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )
    save_json(out / 'runtime.json', runtime)
    print('TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
