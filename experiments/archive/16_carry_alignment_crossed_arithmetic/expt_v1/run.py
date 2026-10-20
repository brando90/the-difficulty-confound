"""Experiment 16 command-line interface (PROMPT.md "run.py"; launch/ scripts call the transformer modes).

    run.py --arm mlp|transformer --check-only
    run.py --arm mlp --pilot                         # 16a pilot into results_pilot/mlp
    run.py --arm mlp --output DIR [--workers 4] [--resume]
    run.py --arm transformer --shard I --output DIR
    run.py --arm transformer --pilot --pilot-job J --output DIR
    run.py --arm transformer --pilot-select DIR
    run.py --arm transformer --merge DIR
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import carry  # noqa: E402


def load_config():
    return json.loads((ROOT / 'config.json').read_text())


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        return 'unknown'


def runtime_info(c):
    import torch
    return dict(host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, git_revision=git_revision(),
                config_sha256=sha_bytes((ROOT / 'config.json').read_bytes()),
                source_sha256={p.name: sha_bytes(p.read_bytes()) for p in sorted(ROOT.glob('*.py'))},
                imports_sha256=carry.import_hashes(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def write_csv(path, rows):
    rows = list(rows)
    keys = list(rows[0])
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    tmp.replace(path)


# --- shared checks ------------------------------------------------------------------------------------------

def check_dp():
    assert sum(carry.stratum_counts(8)) == 10 ** 16
    assert carry.stratum_counts(3) == carry.brute_counts(3)
    assert carry.carry_chain(np.array([123, 109, 199, 999]), np.array([234, 1, 1, 1]), 3).tolist() == [0, 1, 2, 3]
    return dict(dp_total_8_digits=10 ** 16, dp_matches_brute_3_digits=True,
                natural_shares=[round(x / 1e16, 6) for x in carry.stratum_counts(8)])


# --- 16a ----------------------------------------------------------------------------------------------------

def mlp_check(c):
    import torch
    import mlp_arm as M
    cm = c['mlp']
    d = M.data(cm, c['shared']['split_seed'])
    test = d['splits']['test']
    assert len(test) == 2880 and np.all(np.bincount(d['cell'][test], minlength=4) == 720)
    held = set(np.concatenate(list(d['splits'].values())).tolist())
    mixes = 0
    for seed in cm['seeds'] + [cm['pilot_seed']]:
        for target in M.TARGETS:
            prev = None
            for alpha in sorted(cm['alpha_grid']):
                ids = M.training_ids(cm, d, seed, target, alpha)
                tot = M.cell_totals(cm, target, alpha)
                assert len(ids) == len(set(ids.tolist())) == cm['train_size'] and not held & set(ids.tolist())
                assert all(np.bincount(d['cell'][ids], minlength=4)[k] == v for k, v in tot.items())
                _, cnt = np.unique((d['a'] + d['b'])[ids], return_counts=True)
                assert len(cnt) == 180 and np.all(cnt == cm['train_per_answer'])
                mixes += 1
    assert all(column_ok for column_ok in [carry.column_addition(int(x), int(y), 3) == x + y for x, y in zip(d['a'], d['b'])])
    params = [sum(p.numel() for p in carry.addition06.model(dict(hidden_width=w)).parameters()) for w in cm['widths']]
    assert params == cm['expected_params'], params
    n_runs = len(M.grid(cm, cm['seeds']))
    return dict(test_pairs=2880, per_cell=720, mixes_checked=mixes, train_pairs_per_mix=cm['train_size'],
                per_answer=cm['train_per_answer'], split_overlap=0, params=params, runs=n_runs,
                estimate_minutes=round(n_runs * 1.5 / cm['workers'] / 60, 1))


def _mlp_worker(job):
    import mlp_arm as M
    c, out, (seed, target, alpha, width) = job
    cm = c['mlp']
    d = M.data(cm, c['shared']['split_seed'])
    rid = f'{seed}_{target}_{alpha}_w{width}'
    rows, ids, info = M.run_one(cm, d, seed, target, alpha, width, ckpt_dir=out / 'checkpoints' / rid)
    mix = dict(run=rid, counts={str(k): int(v) for k, v in enumerate(np.bincount(d['cell'][ids], minlength=4))},
               carry_rates=M.carry_rates(d, ids), ids_sha256=sha_bytes(np.sort(ids).tobytes()), **info)
    (out / 'runs').mkdir(exist_ok=True)
    save_json(out / 'runs' / f'{rid}.json', dict(rows=rows, mix=mix))  # done marker, written last
    return rid


def mlp_run(c, out, workers, resume, pilot=False):
    import mlp_arm as M
    cm = c['mlp']
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f'{out} is not empty; pass --resume to continue it')
    out.mkdir(parents=True, exist_ok=True)
    if pilot:
        jobs = [(cm['pilot_seed'], t, a, cm['pilot_width']) for t in M.TARGETS for a in cm['pilot_alphas']]
    else:
        jobs = M.grid(cm, cm['seeds'])
    done = {p.stem for p in (out / 'runs').glob('*.json')} if (out / 'runs').exists() else set()
    todo = [j for j in jobs if f'{j[0]}_{j[1]}_{j[2]}_w{j[3]}' not in done]
    state_path = out / 'run_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(chunks=[], runtime=runtime_info(c))
    started = time.monotonic()
    finished = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_mlp_worker, (c, out, j)) for j in todo]
        for f in futures:
            remaining = cm['max_wall_seconds'] - (time.monotonic() - started)
            try:
                finished.append(f.result(timeout=max(remaining, 1)))
            except TimeoutError:
                for g in futures:
                    g.cancel()
                print(f'stopped at max_wall_seconds; rerun with --resume', flush=True)
                break
    state['chunks'].append(dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), runs=len(finished),
                                seconds=time.monotonic() - started))
    save_json(state_path, state)
    collect_mlp(c, out, jobs)
    return len(finished)


def collect_mlp(c, out, jobs):
    """Assemble metrics.csv, mixes.json, tasks.json, data/ and manifest.json from finished runs."""
    import mlp_arm as M
    cm = c['mlp']
    runs = sorted((out / 'runs').glob('*.json'))
    if len(runs) < len(jobs):
        print(f'{len(runs)}/{len(jobs)} runs finished', flush=True)
        return
    rows, mixes = [], []
    for p in runs:
        r = json.loads(p.read_text())
        rows.extend(r['rows'])
        mixes.append(r['mix'])
    write_csv(out / 'metrics.csv', rows)
    save_json(out / 'mixes.json', mixes)
    d = M.data(cm, c['shared']['split_seed'])
    (out / 'data').mkdir(exist_ok=True)
    tasks = []
    strat = M.stratum_of_cells(cm)
    for name, cells in cm['strata'].items():
        ids = np.array([i for i in d['splits']['test'] if strat[int(d['cell'][i])] == name])
        np.savez(out / 'data' / f'{name}_test.npz', a=d['a'][ids], b=d['b'][ids], item_id=ids)
        pool = np.concatenate([d['pools'][int(a), k] for a in d['answers'] for k in cells])
        np.savez(out / 'data' / f'{name}_train.npz', a=d['a'][pool], b=d['b'][pool], item_id=pool)
        tasks.append(dict(task_id=name, family='addition_carry', role='target or distractor', K=cm['chain'][name] + 1,
                          K_bits=None, definition=dict(digits=3, cells=cells, chain_length=cm['chain'][name]),
                          test_pairs=len(ids), train_pool=len(pool)))
    save_json(out / 'tasks.json', tasks)
    manifest = dict(runtime=json.loads((out / 'run_state.json').read_text())['runtime'], runs=len(runs),
                    metrics_rows=len(rows), files={p.relative_to(out).as_posix(): sha_bytes(p.read_bytes())
                                                   for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    save_json(out / 'manifest.json', manifest)
    print(f'collected {len(runs)} runs, {len(rows)} metric rows', flush=True)


def mlp_pilot_rule(c, out):
    """PROTOCOL 4.4: if the cascade target at alpha=0.1, width 64, is not below 2*eps* by 1,000 updates, raise eps*
    to its value there rounded up to 0.05."""
    cm = c['mlp']
    rows = [r for p in sorted((Path(out) / 'runs').glob('*.json')) for r in json.loads(p.read_text())['rows']]
    v = next(r['excess_bits'] for r in rows if r['target_id'] == 'cascade' and r['eval_stratum'] == 'cascade'
             and r['alpha'] == 0.1 and r['step'] == 1000)
    eps = cm['epsilon_star']
    fired = v > 2 * eps
    new = math.ceil(v / 0.05) * 0.05 if fired else eps
    return dict(cascade_alpha_0_1_at_1000=v, threshold=2 * eps, fired=fired, epsilon_star=round(new, 2))


# --- 16b ----------------------------------------------------------------------------------------------------

def transformer_check(c):
    import transformer_arm as TA
    t = c['transformer']
    pairs, reject = TA.test_set(t, c['shared']['split_seed'])
    assert all(len(a) == t['test_per_stratum'] for a, _ in pairs.values())
    for cc, (a, b) in pairs.items():
        assert np.all(carry.carry_chain(a, b, 8) == cc)
        assert all(carry.column_addition(int(x), int(y), 8) == x + y for x, y in zip(a[:200], b[:200]))
    stream = TA.StratumStream(t['seeds'][0], 7, t['stream_block'], reject)
    a, b = stream.take(0, 4096)
    assert not np.isin(a * 10 ** 8 + b, reject).any() and not np.isin(b * 10 ** 8 + a, reject).any()
    counts = [TA.param_counts(TA.build(t, s)) for s in t['size_order']]
    assert [x[0] for x in counts] == t['expected_params'] and [x[1] for x in counts] == t['expected_nonembedding']
    shards = [TA.shard(t, i) for i in range(t['n_shards'])]
    assert len({(s['size'], s['format'], s['c'], s['alpha'], s['seed']) for s in shards}) == t['n_shards']
    return dict(test_pairs=sum(len(a) for a, _ in pairs.values()), test_rejection_keys=len(reject), params=counts,
                shards=t['n_shards'], shard_examples=[shards[i] for i in (0, 59, 60, 179, 180, 239)],
                lr_selected=t['lr_selected'], primary_format_final=t['primary_format_final'],
                estimate_a100_hours=90)


def device_or_die(allow_cpu):
    import torch
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        cap = torch.cuda.get_device_capability(0)
        assert cap >= (8, 0), f'GPU compute capability {cap} < 8.0: bfloat16 needs >= 8.0'
        return dev, dict(gpu=torch.cuda.get_device_name(0), capability=list(cap), cuda=torch.version.cuda,
                         driver=_driver())
    if allow_cpu:
        return torch.device('cpu'), dict(gpu=None)
    raise SystemExit('CUDA is required for 16b (set TDC_ALLOW_CPU=1 only for tests)')


def _driver():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    except Exception:
        return None


def finish_dir(d, extra):
    """Write DONE last, holding the SHA-256 of every file the job wrote."""
    files = {p.relative_to(d).as_posix(): sha_bytes(p.read_bytes()) for p in sorted(d.rglob('*')) if p.is_file()}
    save_json(d / 'DONE', dict(files=files, **extra))


def transformer_job(c, out, kind, index, allow_cpu, T_override=None):
    import torch
    import transformer_arm as TA
    t = c['transformer']
    if kind == 'shard':
        job = TA.shard(t, index)
        if t['lr_selected'] is None:
            raise SystemExit('transformer.lr_selected is unset: make the GPU pilot disclosure commit first')
        lr = t['lr_selected'][job['size']] if job['format'] == (t['primary_format_final'] or 'plain') else t['lr_selected'][t['control_size']]
        T_ck = [2 ** k for k in t['T_checkpoints_log2']]
        d = Path(out) / 'shards' / f'{index:05d}'
    else:
        job = TA.pilot_job(t, index)
        lr = job['lr']
        T_ck = [2 ** t['pilot_T_log2']]
        d = Path(out) / f'{index:05d}'
    T_ck = T_override or T_ck
    if (d / 'DONE').exists():
        print(f'{kind} {index} already DONE', flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(job['seed'])
    torch.use_deterministic_algorithms(True, warn_only=True)
    device, hw = device_or_die(allow_cpu)
    pairs, reject = TA.test_set(t, c['shared']['split_seed'])
    log = lambda m: print(f'[{kind} {index}] {m}', flush=True)
    rows, info = TA.train_run(t, job['size'], job['format'], job['c'], job['alpha'], job['seed'], lr, T_ck, pairs,
                              reject, device, ckpt_dir=d / 'checkpoints', log=log, run_meta=dict(job=job, lr=lr))
    write_csv(d / 'metrics.csv', rows)
    save_json(d / 'run.json', dict(info, hardware=hw, runtime=runtime_info(c)))
    finish_dir(d, dict(job=job))
    log(f'DONE in {info["seconds"]:.0f}s')


def pilot_select(c, pilot_dir):
    import transformer_arm as TA
    t = c['transformer']
    results = []
    for j in range(9):
        d = Path(pilot_dir) / f'{j:05d}'
        assert (d / 'DONE').exists(), f'pilot job {j} is not DONE'
        rows = list(csv.DictReader((d / 'metrics.csv').open()))
        job = TA.pilot_job(t, j)
        v = next(float(r['excess_bits']) for r in rows if r['eval_stratum'] == f"c{t['pilot_c']}" and float(r['T']) == 2 ** t['pilot_T_log2'])
        results.append(dict(job=j, size=job['size'], lr_multiplier=job['lr_multiplier'], excess_bits=v))
    sel = TA.pilot_select(t, results)
    sel['pilot_results'] = results
    save_json(Path(pilot_dir) / 'pilot_selection.json', sel)
    print(json.dumps(sel, indent=1))
    return sel


def merge(c, out):
    """Collect DONE shards into results/transformer (metrics, mixes, tasks, data); checkpoints stay on the cluster."""
    import transformer_arm as TA
    t = c['transformer']
    out = Path(out)
    dest = ROOT / 'results' / 'transformer'
    dest.mkdir(parents=True, exist_ok=True)
    rows, mixes, ckpt_sha = [], [], {}
    for i in range(t['n_shards']):
        d = out / 'shards' / f'{i:05d}'
        assert (d / 'DONE').exists(), f'shard {i} is not DONE'
        done = json.loads((d / 'DONE').read_text())
        rows.extend(csv.DictReader((d / 'metrics.csv').open()))
        run = json.loads((d / 'run.json').read_text())
        mixes.append(dict(shard=i, job=done['job'], used=run['used'], rejections=run['rejections'], seconds=run['seconds'],
                          hardware=run['hardware'], verified=(d / 'VERIFIED').exists()))
        ckpt_sha[i] = {k: v for k, v in done['files'].items() if k.startswith('checkpoints/')}
    write_csv(dest / 'metrics.csv', rows)
    save_json(dest / 'mixes.json', mixes)
    save_json(dest / 'checkpoint_sha256.json', ckpt_sha)
    pairs, _ = TA.test_set(t, c['shared']['split_seed'])
    (dest / 'data').mkdir(exist_ok=True)
    tasks = []
    for cc, (a, b) in pairs.items():
        np.savez(dest / 'data' / f'c{cc}_test.npz', a=a, b=b)
        tasks.append(dict(task_id=f'c{cc}', family='addition_carry', role='target' if cc in t['train_levels'] else 'probe',
                          K=cc + 1, K_bits=None, definition=dict(digits=8, chain_length=cc),
                          natural_share=carry.stratum_counts(8)[cc] / 1e16, covariates=carry.stratum_covariates(a, b, 8)))
    save_json(dest / 'tasks.json', tasks)
    print(f'merged {t["n_shards"]} shards, {len(rows)} rows into {dest}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['mlp', 'transformer'], required=True)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--pilot-job', type=int)
    ap.add_argument('--pilot-select', type=Path)
    ap.add_argument('--merge', type=Path)
    ap.add_argument('--shard', type=int)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    c = load_config()
    allow_cpu = os.environ.get('TDC_ALLOW_CPU') == '1'
    if args.check_only:
        res = dict(dp=check_dp(), **(mlp_check(c) if args.arm == 'mlp' else transformer_check(c)))
        print(json.dumps(res, indent=1, default=str))
        return
    if args.arm == 'mlp':
        if args.pilot:
            out = args.output or ROOT / 'results_pilot' / 'mlp'
            mlp_run(c, out, args.workers, args.resume, pilot=True)
            rule = mlp_pilot_rule(c, out)
            save_json(Path(out) / 'pilot_rule.json', rule)
            print(json.dumps(rule), flush=True)
        else:
            mlp_run(c, args.output, args.workers, args.resume)
        return
    if args.pilot_select:
        pilot_select(c, args.pilot_select)
    elif args.merge:
        merge(c, args.merge)
    elif args.pilot:
        transformer_job(c, args.output, 'pilot', args.pilot_job, allow_cpu)
    elif args.shard is not None:
        transformer_job(c, args.output, 'shard', args.shard, allow_cpu)
    else:
        ap.error('nothing to do')


if __name__ == '__main__':
    sys.exit(main())
