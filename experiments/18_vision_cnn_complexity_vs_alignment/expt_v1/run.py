"""Run list, one training run, and the local worker pool / Slurm shard driver (PROTOCOL Sections 2 and 7).

  python run.py --list main|mis|H|pilot [--dataset D]     print run ids
  python run.py --pool 4 --phase mnist_main ...            run phases locally, resumable, under scratch/at343 CPU lock
  python run.py --runlist runs.txt --shard i --nshards n   one Slurm array task (Sherlock / SNAP)
  python run.py --check-only                               data, task and model checks, no training
"""
import argparse
import glob
import json
import math
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import tasks as TK  # noqa: E402

CFG = json.load(open(os.path.join(HERE, 'config.json')))
OUT = os.environ.get('EXP18_OUT', os.path.join(HERE, 'results'))
LOCKDIR = os.environ.get('CPU_LOCK_DIR', '/var/lib/agent/scratch/at343')
MY_LOCK = os.path.join(LOCKDIR, 'CPU_LOCK_AT-439')


# ---------------------------------------------------------------- run specs
def rid(s):
    if s['arm'] in ('main', 'mis', 'pilot'):
        return f"{s['arm']}__{s['dataset']}__{s['mismatch']}__{s['target']}__a{s['alpha']}__s{s['seed']}"
    return f"H__{s['dataset']}__{s['mode']}__{s.get('pool', 'none')}__{s.get('target', 'none')}__s{s['seed']}"


def main_specs(ds, seeds=None):
    out = []
    for s in seeds or CFG['seeds']:
        for _, ts in TK.MAIN_LEVELS:
            for t in ts:
                for a in CFG['alphas']:
                    out.append(dict(arm='main', dataset=ds, mismatch=CFG['primary_mismatch'], target=t, alpha=a, seed=s))
    return out


def mis_specs():
    m = CFG['mismatch_sweep']
    return [dict(arm='mis', dataset=m['dataset'], mismatch=ty, target=t, alpha=a, seed=s)
            for s in m['seeds'] for ty in m['types'] for _, ts in TK.MAIN_LEVELS for t in ts for a in m['alphas']]


def h_specs(ds, seeds=None):
    out = []
    tg = TK.H_TARGETS['easy'] + TK.H_TARGETS['hard']
    for s in seeds or CFG['seeds']:
        for p in TK.POOLS:
            out.append(dict(arm='H', dataset=ds, mode='pool_only', pool=p, seed=s))
        for p in TK.POOLS:
            for t in tg:
                out.append(dict(arm='H', dataset=ds, mode='joint', pool=p, target=t, alpha=CFG['H']['alpha_joint'], seed=s))
        for t in tg:
            out.append(dict(arm='H', dataset=ds, mode='anchor', target=t, alpha=1.0, seed=s))
    return out


def pilot_specs():
    s = CFG['pilot_seed']
    out = [dict(arm='pilot', dataset='mnist', mismatch='rot90', target=t, alpha=a, seed=s)
           for t in ('par_a', 'sum10') for a in (0.05, 0.4)]
    out += [dict(arm='pilot', dataset=d, mismatch='rot90', target='sum10', alpha=0.4, seed=s)
            for d in ('fashion', 'kmnist', 'cifar10')]
    out.append(dict(arm='H', dataset='mnist', mode='pool_only', pool='hard', seed=s))
    return out


PHASES = {
    'pilot': pilot_specs,
    'mnist_main': lambda: main_specs('mnist'), 'mnist_H': lambda: h_specs('mnist'),
    'fashion_main': lambda: main_specs('fashion'), 'fashion_H': lambda: h_specs('fashion'),
    'kmnist_main': lambda: main_specs('kmnist'), 'kmnist_H': lambda: h_specs('kmnist'),
    'mis': mis_specs,
    'cifar10_main': lambda: main_specs('cifar10'), 'cifar10_H': lambda: h_specs('cifar10'),
}
ORDER = ['mnist_main', 'mnist_H', 'fashion_main', 'kmnist_main', 'fashion_H', 'kmnist_H', 'mis', 'cifar10_main',
         'cifar10_H']


def seed_major(specs):
    seeds = sorted({s['seed'] for s in specs}, key=lambda x: CFG['seeds'].index(x) if x in CFG['seeds'] else -1)
    return [s for sd in seeds for s in specs if s['seed'] == sd]


def out_path(s):
    return os.path.join(OUT, s['arm'] if s['arm'] != 'pilot' else 'pilot', rid(s) + '.json')


# ---------------------------------------------------------------- one run
def probe_err(ztr, ytr, zte, yte, C, l2, iters):
    import torch
    ztr = torch.as_tensor(ztr, dtype=torch.float64)
    zte = torch.as_tensor(zte, dtype=torch.float64)
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-6
    ztr, zte = (ztr - mu) / sd, (zte - mu) / sd
    W = torch.zeros(ztr.shape[1], C, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(C, dtype=torch.float64, requires_grad=True)
    yt = torch.as_tensor(ytr)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn='strong_wolfe', tolerance_grad=1e-9,
                            tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(ztr @ W + b, yt) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return float(((zte @ W + b).argmax(1).numpy() != yte).mean())


def to_tensor(x, d):
    import torch
    t = torch.from_numpy(x).float().div_(255.0)
    m = torch.from_numpy(d['mean']).view(1, 1, -1, 1, 1)
    s = torch.from_numpy(d['std']).view(1, 1, -1, 1, 1)
    return (t - m) / s


def features(model, x, d, bs=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model.features(to_tensor(x[i:i + bs], d)).double().numpy())
    return np.concatenate(out)


def logits_all(model, x, d, bs=512):
    import torch
    out = {}
    with torch.no_grad():
        for i in range(0, len(x), bs):
            for t, v in model(to_tensor(x[i:i + bs], d)).items():
                out.setdefault(t, []).append(torch.log_softmax(v.double(), 1).numpy())
    return {t: np.concatenate(v) for t, v in out.items()}


def score(lp, y):
    err = float((lp.argmax(1) != y).mean())
    ce_bits = float(-lp[np.arange(len(y)), y].mean() / math.log(2))
    return err, ce_bits


def train(spec, verbose=False):
    import torch
    from model import PairNet, WIDTH, n_params
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ds, seed = spec['dataset'], spec['seed']
    d = D.load(ds)
    Ts = list(CFG['T_checkpoints'][ds])
    if spec['arm'] == 'pilot' and ds == 'cifar10':
        Ts.append(CFG['pilot_cifar_extra_T'])
    n = max(Ts)
    B = CFG['batch']
    assert all(t % B == 0 for t in Ts)
    draws = D.base_draws(ds, seed, n)
    mismatch = spec.get('mismatch', 'rot90')
    # task of each example (index into run_tasks) and whether its images are aligned
    if spec['arm'] in ('main', 'mis', 'pilot'):
        run_tasks = [spec['target']]
        task_of = np.zeros(n, np.int64)
        aligned = D.aligned_mask(n, spec['alpha'])
    else:
        aligned = np.ones(n, bool)
        if spec['mode'] == 'anchor':
            run_tasks = [spec['target']]
            task_of = np.zeros(n, np.int64)
        else:
            pool = TK.POOLS[spec['pool']]
            run_tasks = list(pool) + ([spec['target']] if spec['mode'] == 'joint' else [])
            tmask = D.aligned_mask(n, spec['alpha']) if spec['mode'] == 'joint' else np.zeros(n, bool)
            task_of = np.empty(n, np.int64)
            task_of[tmask] = len(pool)
            nt = int((~tmask).sum())
            task_of[~tmask] = np.arange(nt) % len(pool)
    heads = {t: TK.n_classes(t) for t in run_tasks}
    cin = d['x_train'].shape[1]
    model = PairNet(cin, WIDTH[ds], heads, seed)
    decay = [p for nme, p in model.named_parameters() if p.ndim > 1]
    nodecay = [p for nme, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': CFG['weight_decay']},
                             {'params': nodecay, 'weight_decay': 0.0}], lr=CFG['lr'], betas=tuple(CFG['betas']))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / CFG['warmup_steps']))
    xe, ye, _ = D.eval_pairs(ds)
    probe = spec['arm'] == 'H' and spec['mode'] in ('pool_only', 'joint')
    if probe:
        xp, yp, _ = D.probe_pairs(ds, seed)
        ptargets = (TK.H_TARGETS['easy'] + TK.H_TARGETS['hard']) if spec['mode'] == 'pool_only' else [spec['target']]
    tdist = spec['arm'] in ('main', 'mis', 'pilot') and spec['alpha'] < 1
    if tdist:
        xm, ym = D.traindist_pairs(ds, mismatch, seed)
    rows = []
    t0 = time.time()
    train_time = 0.0

    def evaluate(T):
        model.eval()
        r = dict(T=T)
        lp = logits_all(model, xe, d)
        for t in run_tasks:
            err, ce = score(lp[t], TK.label(t, ye[:, 0], ye[:, 1]))
            r[f'err__{t}'], r[f'ce_bits__{t}'] = err, ce
            r[f'eps__{t}'] = err / TK.chance_error(t)
        if tdist:
            lm = logits_all(model, xm, d)
            t = run_tasks[0]
            r[f'traindist_err__{t}'] = score(lm[t], TK.label(t, ym[:, 0], ym[:, 1]))[0]
        if probe:
            ztr, zte = features(model, xp, d), features(model, xe, d)
            for t in ptargets:
                e = probe_err(ztr, TK.label(t, yp[:, 0], yp[:, 1]), zte, TK.label(t, ye[:, 0], ye[:, 1]),
                              TK.n_classes(t), CFG['H']['probe_l2'], CFG['H']['probe_iters'])
                r[f'probe_err__{t}'] = e
                r[f'probe_eps__{t}'] = e / TK.chance_error(t)
        model.train()
        return r
    if probe and spec['mode'] == 'pool_only':
        rows.append(evaluate(0))
    task_ids = np.array(run_tasks)
    for step in range(n // B):
        ts = time.time()
        sl = slice(step * B, (step + 1) * B)
        x, cls = D.gather(ds, mismatch, draws, sl, aligned[sl])
        if ds == 'cifar10':
            x = D.augment(x, draws, sl)
        xt = to_tensor(x, d)
        out = model(xt)
        to = task_of[sl]
        loss = 0.0
        for j in np.unique(to):
            m = to == j
            t = task_ids[j]
            y = torch.from_numpy(TK.label(t, cls[m, 0], cls[m, 1]))
            loss = loss + torch.nn.functional.cross_entropy(out[t][torch.from_numpy(m)], y, reduction='sum')
        loss = loss / B
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
        opt.step()
        sched.step()
        train_time += time.time() - ts
        if not math.isfinite(loss.item()):
            raise RuntimeError(f'nonfinite loss at step {step} in {rid(spec)}')
        T = (step + 1) * B
        if T in Ts:
            rows.append(evaluate(T))
            if verbose:
                print(rid(spec), {k: round(v, 4) for k, v in rows[-1].items() if 'err' in k}, flush=True)
    res = dict(run_id=rid(spec), spec=spec, run_tasks=run_tasks, task_info={t: TK.info(t) for t in run_tasks},
               n_params=n_params(model), n_params_trunk=n_params(model, True), rows=rows,
               aligned_count_at_T={str(T): int(aligned[:T].sum()) for T in Ts}, T_checkpoints=Ts,
               target_count_at_T={str(T): int((task_of[:T] == len(run_tasks) - 1).sum()) for T in Ts},
               wall_s=time.time() - t0, train_s=train_time, pairs_per_s=n / train_time, host=socket.gethostname(),
               finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    res['_state'] = {k: v.clone() for k, v in model.state_dict().items()}
    return res


def run_one(spec):
    p = out_path(spec)
    if os.path.exists(p):
        return rid(spec), 'skip'
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        res = train(spec)
    except Exception as e:  # recorded, never silently dropped
        with open(p + '.error', 'w') as f:
            f.write(repr(e))
        return rid(spec), 'error ' + repr(e)
    import torch
    ck = os.path.join(os.path.dirname(p), 'ckpt', rid(spec) + '.pt')
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save(res.pop('_state'), ck + '.tmp')
    os.replace(ck + '.tmp', ck)
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f)
    os.replace(tmp, p)
    return rid(spec), f"ok {res['wall_s']:.0f}s {res['pairs_per_s']:.0f}/s"


# ---------------------------------------------------------------- drivers
def other_locks():
    return [p for p in glob.glob(os.path.join(LOCKDIR, 'CPU_LOCK_*')) if os.path.abspath(p) != MY_LOCK]


def pool_phases(phases, workers):
    import multiprocessing as mp
    for ph in phases:
        specs = [s for s in seed_major(PHASES[ph]()) if not os.path.exists(out_path(s))]
        if not specs:
            print(ph, 'done already', flush=True)
            continue
        while other_locks():
            print(time.strftime('%H:%M:%S'), 'waiting for', [os.path.basename(p) for p in other_locks()], flush=True)
            time.sleep(60)
        with open(MY_LOCK, 'w') as f:
            f.write(f"AT-439 Exp 18 vision phase {ph} ({len(specs)} runs, {workers} workers), started "
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        try:
            print(time.strftime('%H:%M:%S'), ph, len(specs), 'runs', flush=True)
            with mp.get_context('fork').Pool(workers, maxtasksperchild=8) as P:
                for r, st in P.imap_unordered(run_one, specs):
                    print(time.strftime('%H:%M:%S'), r, st, flush=True)
        finally:
            if os.path.exists(MY_LOCK):
                os.remove(MY_LOCK)


def check_only():
    import torch
    from model import PairNet, WIDTH, n_params
    rep = {}
    for k, (l, r) in {'par_a': (2, 1), 'sum10': (10, 10), 'a_lt_b': (10, 10), 'ge5_xor': (2, 2), 'id_b': (1, 10)}.items():
        assert TK.slot_classes(k) == (l, r), k
    for ds in CFG['datasets']:
        d = D.load(ds)
        xe, ye, idx = D.eval_pairs(ds)
        assert idx.max() < len(d['y_test'])
        xp, _, _ = D.probe_pairs(ds, CFG['seeds'][0])
        # training images come only from the train split and evaluation images only from the test split
        trh = {hash(x.tobytes()) for x in d['x_train']}
        dup = sum(hash(x.tobytes()) in trh for x in d['x_test'])
        m = PairNet(d['x_train'].shape[1], WIDTH[ds], {'sum10': 10}, 1)
        rep[ds] = dict(test_images_identical_to_a_train_image=dup, n_params=n_params(m),
                       n_params_trunk=n_params(m, True))
        for mm in ['rot90'] + (CFG['mismatch_sweep']['types'] if ds == CFG['mismatch_sweep']['dataset'] else []):
            mx, bc = D.misaligned_train(ds, mm)
            assert mx.shape[1:] == d['x_train'].shape[1:], (ds, mm)
            if mm in D.TRANSFORMS:
                assert not np.array_equal(mx[:100], d['x_train'][:100]), (ds, mm)
    for a in CFG['alphas']:
        for n in (1000, 131072):
            mk = D.aligned_mask(n, a)
            for m_ in (17, 640, n):
                assert mk[:m_].sum() == math.floor(m_ * float(__import__('fractions').Fraction(str(a)))), (a, m_)
    # common random numbers: identical draws across calls
    a1, a2 = D.base_draws('mnist', 18101, 4096), D.base_draws('mnist', 18101, 4096)
    assert np.array_equal(a1['idx'], a2['idx'])
    rep['n_runs'] = {ph: len(PHASES[ph]()) for ph in ORDER}
    rep['n_runs_total'] = sum(rep['n_runs'].values())
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(HERE, 'check_only.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list')
    ap.add_argument('--pool', type=int)
    ap.add_argument('--phase', nargs='*')
    ap.add_argument('--runlist')
    ap.add_argument('--shard', type=int)
    ap.add_argument('--nshards', type=int)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--one')
    a = ap.parse_args()
    if a.check_only:
        return check_only()
    if a.list:
        for s in seed_major(PHASES[a.list]()):
            print(json.dumps(s))
        return
    if a.runlist:
        specs = [json.loads(l) for l in open(a.runlist) if l.strip()]
        for s in specs[a.shard::a.nshards]:
            print(time.strftime('%H:%M:%S'), *run_one(s), flush=True)
        return
    if a.one:
        r = train(json.loads(a.one), verbose=True)
        r.pop('_state')
        print(json.dumps(r['rows'][-1]))
        return
    pool_phases(a.phase or ORDER, a.pool or CFG['workers'])


if __name__ == '__main__':
    main()
