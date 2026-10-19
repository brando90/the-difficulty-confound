"""Graded-relevance arm (PROTOCOL Section 2.5): 4 Experiment 11 targets, 32 rewired distractors, label agreement and
chance-corrected relevance, the 5 mixes per target, and training with Experiment 11's train_cell (imported unchanged;
only the task-token table is sized for the 36 graded tasks).

Shared contract under results/graded/: tasks.json, data/<task>_{train,test}.npz, mixes.json, metrics.csv,
cells/<run>.json, predictions/<run>.npz, checkpoints/<seed>_<target>_<mix>/T_<T>.npz.
"""
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import common as C
import mixes15 as MX

GRADED_STREAM = 15


def A_():
    return C.mods()['dfa']['gen']


def D_():
    return C.mods()['dfa']['data']


def c11():
    return C.mods()['dfa']['cfg']


# ---------------------------------------------------------------- targets and distractors

def pick_targets(c):
    """First Experiment 11 targets (tasks.json order) with acceptance in the band, per K level."""
    g = c['graded']
    lo, hi = g['target_accept_band']
    tasks = json.loads((C.results_dir(c, 'dfa') / 'tasks.json').read_text())
    out = []
    for K in g['K_levels']:
        q = [t for t in tasks if t['role'] == 'target' and t['K'] == K and lo <= t['acceptance_rate'] <= hi]
        if len(q) < g['targets_per_level']:
            raise RuntimeError(f'only {len(q)} Experiment 11 targets qualify at K={K}; fresh draws not implemented')
        out += q[: g['targets_per_level']]
    return out


def rewire(dfa, r, rng):
    """r (state, symbol) entries sent to a uniformly random different state; 'all' redraws every entry uniformly over
    the states (state count and accept set kept)."""
    A = A_()
    trans = dfa.trans.copy()
    n = dfa.n
    if r == 'all':
        trans = rng.integers(0, n, size=trans.shape).astype(np.int64)
    else:
        cells = rng.choice(2 * n, int(r), replace=False)
        for k in cells:
            s, a = divmod(int(k), 2)
            choices = [q for q in range(n) if q != trans[s, a]]
            trans[s, a] = choices[int(rng.integers(len(choices)))]
    return A.DFA(trans, dfa.accept.copy())


def draw_distractors(c, target, attempt, taken_keys):
    """2 distractors per rewiring level for one target. Returns (list of dict, log)."""
    A = A_()
    g = c['graded']
    lo, hi = g['distractor_accept_band']
    L0, L1 = c11()['length_min'], c11()['length_max']
    T = A.DFA.from_json(target['definition'])
    tkey = A.canonical_form(T).key()
    rng = np.random.default_rng([c['task_sampling_seed'], GRADED_STREAM, target['task_token'], attempt])
    out, log = [], []
    for r in g['rewire_levels']:
        for j in range(g['distractors_per_level']):
            for tries in range(1, g['max_tries'] + 1):
                raw = rewire(T, r, rng)
                m = A.hopcroft_minimize(raw)
                key = m.key()
                acc = A.acceptance_rate(m, L0, L1)
                why = None
                if key == tkey:
                    why = 'same language as target'
                elif key in taken_keys:
                    why = 'same language as another distractor'
                elif not lo <= acc <= hi:
                    why = 'acceptance outside band'
                if why is None:
                    break
                log.append(dict(target=target['task_id'], r=str(r), index=j + 1, attempt=attempt, try_=tries,
                                rejected=why))
            else:
                raise RuntimeError(f'no distractor for {target["task_id"]} r={r} after {g["max_tries"]} tries')
            taken_keys.add(key)
            changed = int((raw.trans != T.trans).sum())
            out.append(dict(r=str(r), index=j + 1, raw=raw, dfa=m, acceptance=acc, tries=tries, changed=changed))
    return out, log


# ---------------------------------------------------------------- agreement

def agreement_exact(d1, d2, lmin, lmax):
    """Pr[f1(x) = f2(x)] with length uniform on lmin..lmax and the string uniform given length (product-automaton
    dynamic programming)."""
    n1, n2 = d1.n, d2.n
    p = np.zeros((n1, n2))
    p[0, 0] = 1.0
    same = (d1.accept[:, None] == d2.accept[None, :])
    vals = []
    for L in range(lmax + 1):
        if L >= lmin:
            vals.append(float((p * same).sum()))
        q = np.zeros_like(p)
        for a in range(2):
            np.add.at(q, (d1.trans[:, a][:, None].repeat(n2, 1), d2.trans[:, a][None, :].repeat(n1, 0)), 0.5 * p)
        p = q
    return float(np.mean(vals))


def mc_inputs(c, n, stream=0):
    """n inputs from Experiment 11's training-partition sampler (shared by every agreement and acceptance estimate)."""
    return MX.fresh_dfa_inputs(c, GRADED_STREAM * 100 + stream, 0, n)


def agreement_mc(x, d1, d2):
    """Monte Carlo agreement over draws x of Experiment 11's training-partition input sampler, with its SE."""
    A = A_()
    y1 = A.run_dfa(d1, x['bits'], x['lens'])
    y2 = A.run_dfa(d2, x['bits'], x['lens'])
    a = float((y1 == y2).mean())
    return a, float(np.sqrt(a * (1 - a) / len(y1)))


def kappa(agree, pT, pj):
    a0 = pT * pj + (1 - pT) * (1 - pj)
    return max(0.0, (agree - a0) / (1 - a0)), a0


def accept_rate_mc(x, d):
    return float(A_().run_dfa(d, x['bits'], x['lens']).mean())


# ---------------------------------------------------------------- construction

def build(c, log=print):
    """Targets, distractors (with the construction check and redraws), agreement, relevance and mixes."""
    A = A_()
    g = c['graded']
    L0, L1 = c11()['length_min'], c11()['length_max']
    targets = pick_targets(c)
    attempt = 0
    rej_all, construction = [], []
    while True:
        taken = set()
        drawn = {}
        for t in targets:
            drawn[t['task_id']], rej = draw_distractors(c, t, attempt, taken)
            rej_all += rej
        rows = []
        x = mc_inputs(c, g['agreement_mc_draws'])
        for ti, t in enumerate(targets):
            T = A.DFA.from_json(t['definition'])
            pT_mc = accept_rate_mc(x, T)
            for di, d in enumerate(drawn[t['task_id']]):
                a_mc, se = agreement_mc(x, T, d['dfa'])
                pj_mc = accept_rate_mc(x, d['dfa'])
                a_ex = agreement_exact(T, d['dfa'], L0, L1)
                rho, a0 = kappa(a_mc, pT_mc, pj_mc)
                rho_ex, _ = kappa(a_ex, A.acceptance_rate(T, L0, L1), d['acceptance'])
                rows.append(dict(target=t['task_id'], r=d['r'], index=d['index'], K=int(d['dfa'].n),
                                 changed_entries=d['changed'], tries=d['tries'],
                                 acceptance_exact=d['acceptance'], acceptance_mc=pj_mc, target_acceptance_mc=pT_mc,
                                 agree=a_mc, agree_se=se, agree_exact_uniform=a_ex, a0=a0, rho=rho,
                                 rho_exact_uniform=rho_ex, definition=d['dfa'].to_json()))
        means = {r: float(np.mean([x['rho'] for x in rows if x['r'] == str(r)])) for r in g['rewire_levels']}
        seq = [means[r] for r in g['rewire_levels']]
        ok = (all(a > b for a, b in zip(seq[:-1], seq[1:])) and seq[0] >= g['construction_min_rho_r1']
              and seq[-1] <= g['construction_max_rho_all'])
        construction.append(dict(attempt=attempt, mean_rho_by_r={str(k): v for k, v in means.items()}, passed=ok))
        log(f'graded construction attempt {attempt}: mean rho by r {means} -> {"PASS" if ok else "redraw"}')
        if ok:
            break
        attempt += 1
        if attempt >= 20:
            raise RuntimeError('graded construction check failed 20 times')
    return assemble(c, targets, rows, construction, rej_all)


def assemble(c, targets, rows, construction, rejections):
    """Graded task table (tokens: targets 0-3, then each target's 8 distractors) and the mixes."""
    g = c['graded']
    tasks = []
    for i, t in enumerate(targets):
        tasks.append(dict(task_id=t['task_id'], family='dfa', role='target', K=t['K'], K_bits=t['K_bits'],
                          definition=t['definition'], task_token=i, exp11_task_token=t['task_token'],
                          acceptance_rate=t['acceptance_rate']))
    k_bits = A_().k_bits
    for r in rows:
        tok = len(tasks)
        tid = f"{r['target']}_r{r['r']}_{r['index']}"
        tasks.append(dict(task_id=tid, family='dfa', role='distractor', of_target=r['target'], r=r['r'],
                          K=r['K'], K_bits=int(k_bits(r['K'])), definition=r['definition'], task_token=tok,
                          pool_token=g['distractor_pool_token_offset'] + tok, acceptance_rate=r['acceptance_exact'],
                          agree=r['agree'], agree_se=r['agree_se'], agree_method='monte_carlo_train_sampler',
                          agree_exact_uniform=r['agree_exact_uniform'], a0=r['a0'], rho=r['rho'],
                          changed_entries=r['changed_entries'], tries=r['tries']))
    assert len(tasks) == g['n_task_tokens']
    mixes = {}
    for t in targets:
        ds = [x for x in tasks if x.get('of_target') == t['task_id']]
        for m, spec in g['mixes'].items():
            sel = []
            for r, k in spec['distractors'].items():
                q = [x for x in ds if x['r'] == r]
                assert len(q) == k
                sel += q
            a = spec['alpha_share']
            pi = (1 - a) / len(sel)
            mixes[f"{t['task_id']}_{m}"] = dict(
                target_id=t['task_id'], mix=m, alpha_share=a, distractors=[x['task_id'] for x in sel],
                pi=pi, alpha_soft=a + pi * sum(x['rho'] for x in sel),
                alpha_agree=a + pi * sum(x['agree'] for x in sel))
    return dict(tasks=tasks, mixes=mixes, construction=construction, rejections=rejections)


def check_alpha_soft_spread(G):
    """Correlation between alpha_share and alpha_soft over the 20 mixes (design aim: far from collinear)."""
    a = np.array([m['alpha_share'] for m in G['mixes'].values()])
    s = np.array([m['alpha_soft'] for m in G['mixes'].values()])
    return float(np.corrcoef(a, s)[0, 1])


# ---------------------------------------------------------------- data in the shared contract

def write_data(c, G, out):
    out = Path(out)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    D, c1 = D_(), c11()
    te = np.load(C.results_dir(c, 'dfa') / 'data' / f"{G['tasks'][0]['task_id']}_test.npz")
    tb, tl = te['bits'], te['length']
    for t in G['tasks']:
        p_tr = out / 'data' / f"{t['task_id']}_train.npz"
        p_te = out / 'data' / f"{t['task_id']}_test.npz"
        if p_tr.exists() and p_te.exists():
            continue
        if t['role'] == 'target':
            z = np.load(C.results_dir(c, 'dfa') / 'data' / f"{t['task_id']}_train.npz")
            b, l, y = z['bits'], z['length'], z['label']
            yt = np.load(C.results_dir(c, 'dfa') / 'data' / f"{t['task_id']}_test.npz")['label']
        else:
            b, l = D.build_pool(c1, dict(task_token=t['pool_token']))
            y = D.labels_for(t, b, l)
            yt = D.labels_for(t, tb, tl)
        for p, arrs in ((p_tr, dict(bits=b, length=l, label=y)), (p_te, dict(bits=tb, length=tl, label=yt))):
            tmp = Path(str(p) + '.tmp')
            with open(tmp, 'wb') as f:
                np.savez(f, **arrs)
            os.replace(tmp, p)


def load_data(out, tasks):
    out = Path(out)
    order = sorted(tasks, key=lambda t: t['task_token'])
    assert [t['task_token'] for t in order] == list(range(len(order)))
    tr = [np.load(out / 'data' / f"{t['task_id']}_train.npz") for t in order]
    te = [np.load(out / 'data' / f"{t['task_id']}_test.npz") for t in order]
    return dict(tasks=tasks, pool_bits=np.stack([z['bits'] for z in tr]),
                pool_lens=np.stack([z['length'] for z in tr]), pool_labels=np.stack([z['label'] for z in tr]),
                test_bits=te[0]['bits'], test_lens=te[0]['length'], test_labels=np.stack([z['label'] for z in te]))


def mix_tasks(G, key):
    """Task list for one mix: the target (role target) and the mix's distractors (role distractor)."""
    m = G['mixes'][key]
    keep = {m['target_id']} | set(m['distractors'])
    return [dict(t) for t in G['tasks'] if t['task_id'] in keep]


def run_name(seed, target_id, mix):
    return f'{seed}_{target_id}_{mix}'


# ---------------------------------------------------------------- training

_RUN = {}


def exp11_run(c):
    """Experiment 11's run.py, imported under its own name (hash-checked); it puts its folder first on sys.path."""
    if 'm' not in _RUN:
        C.check_upstream(c)
        d = C.exp_dirs(c)['dfa']
        spec = importlib.util.spec_from_file_location('e11_run', d / 'run.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['e11_run'] = mod
        spec.loader.exec_module(mod)
        _RUN['m'] = mod
    return _RUN['m']


def train_one(c, G, data, seed, key, out):
    """One graded run in a private folder, then moved into the shared-contract layout."""
    R = exp11_run(c)
    import torch
    torch.set_num_threads(c['threads'])
    out = Path(out)
    m = G['mixes'][key]
    name = run_name(seed, m['target_id'], m['mix'])
    final = out / 'cells' / f'{name}.json'
    if final.exists():
        return json.loads(final.read_text())
    tmp = out / 'runs_tmp' / name
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    rec = R.train_cell(R.load_config(), mix_tasks(G, key), m['target_id'], m['alpha_share'], seed, tmp,
                       n_task_tokens=c['graded']['n_task_tokens'], data=data)
    cell = R.cell_name(seed, m['target_id'], m['alpha_share'])
    for sub in ('checkpoints', 'predictions'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dst = out / 'checkpoints' / name
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp / 'checkpoints' / cell, dst)
    os.replace(tmp / 'predictions' / f'{cell}.npz', out / 'predictions' / f'{name}.npz')
    rec = dict(rec, run=name, mix=m['mix'], alpha_soft=m['alpha_soft'], alpha_agree=m['alpha_agree'])
    (out / 'cells').mkdir(parents=True, exist_ok=True)
    C.atomic_write_text(final, json.dumps(rec))
    shutil.rmtree(tmp, ignore_errors=True)
    return rec


_W = {}


def _init(out):
    c = C.load_config()
    G = json.loads((Path(out) / 'graded_tasks_full.json').read_text())
    _W.update(c=c, G=G, data=load_data(out, G['tasks']), out=out)
    exp11_run(c)


def _work(args):
    seed, key = args
    t0 = time.time()
    rec = train_one(_W['c'], _W['G'], _W['data'], seed, key, _W['out'])
    return rec['run'], time.time() - t0


def run_grid(c, out, runs, max_seconds, log=print, workers=None):
    """Train the (seed, mix key) runs not done yet, stopping cleanly when the time budget is spent."""
    import concurrent.futures as cf
    out = Path(out)
    todo = [r for r in runs if not (out / 'cells' / f"{run_name(r[0], *r[1].rsplit('_', 1))}.json").exists()]
    log(f'graded: {len(runs) - len(todo)} done, {len(todo)} to run')
    t0 = time.time()
    done = []
    with cf.ProcessPoolExecutor(workers or c['workers'], initializer=_init, initargs=(str(out),)) as ex:
        pending = set()
        it = iter(todo)
        for _ in range(workers or c['workers']):
            r = next(it, None)
            if r is not None:
                pending.add(ex.submit(_work, r))
        while pending:
            fin, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                name, sec = f.result()
                done.append((name, sec))
                log(f'graded run {name} {sec:.1f} s')
            if time.time() - t0 < max_seconds:
                for _ in fin:
                    r = next(it, None)
                    if r is not None:
                        pending.add(ex.submit(_work, r))
    return done, len(todo) - len(done)


def aggregate(c, out, G, seeds):
    """metrics.csv from the cell records (rows of Experiment 11's form plus mix and the three alphas)."""
    out = Path(out)
    rows = []
    for seed in seeds:
        for key, m in sorted(G['mixes'].items()):
            rec = json.loads((out / 'cells' / f"{run_name(seed, m['target_id'], m['mix'])}.json").read_text())
            for r in rec['rows']:
                rows.append(dict(seed=seed, target_id=m['target_id'], mix=m['mix'], K=r['K'], K_bits=r['K_bits'],
                                 alpha=m['alpha_share'], alpha_share=m['alpha_share'], alpha_soft=m['alpha_soft'],
                                 alpha_agree=m['alpha_agree'], T=r['T'], excess_bits=r['excess_bits'],
                                 accuracy=r['accuracy'], target_examples_seen=r['target_examples_seen'],
                                 distractor_examples_seen=r['distractor_examples_seen'],
                                 wall_seconds=r['wall_seconds']))
    tmp = out / 'metrics.csv.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out / 'metrics.csv')
    return rows


def write_contract(c, G, out, seeds):
    """tasks.json and mixes.json of the shared contract (mixes.json: per run the counts at every checkpoint)."""
    out = Path(out)
    C.atomic_write_text(out / 'tasks.json', json.dumps(G['tasks'], indent=1))
    D = D_()
    mj = {}
    for seed in seeds:
        for key, m in sorted(G['mixes'].items()):
            mix = D.assemble_mix(c11(), mix_tasks(G, key), m['target_id'], m['alpha_share'], seed)
            mj[run_name(seed, m['target_id'], m['mix'])] = dict(
                target_id=m['target_id'], mix=m['mix'], seed=seed, alpha=m['alpha_share'],
                alpha_share=m['alpha_share'], alpha_soft=m['alpha_soft'], alpha_agree=m['alpha_agree'],
                distractors=m['distractors'], counts=mix['counts'], data_order_seed=mix['data_order_seed'])
    C.atomic_write_text(out / 'mixes.json', json.dumps(mj, indent=1))
    return mj
