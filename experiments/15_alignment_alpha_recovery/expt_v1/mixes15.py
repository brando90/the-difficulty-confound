"""Mix reconstruction, references and scored units (PROTOCOL Sections 2.1-2.2).

Every Experiment 11/12 training multiset is rebuilt with the experiment's own assemble_mix (imported, hash-checked);
its per-task counts at every checkpoint must equal mixes.json. References R_T are the target's unused training
examples first (file order), then fresh draws from the target's own input sampler (seed 20261015), logged. Scored
units U_u are 4,096 training examples per task from a fixed permutation (the whole file if smaller), disjoint from R_T
as index sets.
"""
import json

import numpy as np

import common as C


def mixes_json(c, fam):
    mx = json.loads((C.results_dir(c, fam) / 'mixes.json').read_text())
    return mx['main'] if fam == 'dsl' else mx


def upstream_cfg(c, fam):
    return C.mods(c)[fam]['cfg']


def rebuild_mix(c, fam, tasks, target_id, alpha, seed):
    M = C.mods(c)[fam]
    return M['data'].assemble_mix(upstream_cfg(c, fam), tasks, target_id, alpha, seed)


def mix_cells(c, fam):
    """[(cell name, target_id, alpha, seed)] of the gated grid, from mixes.json."""
    return sorted((k, v['target_id'], float(v['alpha']), int(v['seed'])) for k, v in mixes_json(c, fam).items())


def check_mix(c, fam, tasks, cell, recorded):
    """Rebuild one cell; return (ok, used target pool indices)."""
    _, tid, alpha, seed = cell
    mix = rebuild_mix(c, fam, tasks, tid, alpha, seed)
    ok = mix['counts'] == recorded['counts']
    tok = next(t['task_token'] for t in tasks if t['task_id'] == tid)
    sel = mix['task_token'] == tok
    used = mix['pool_index'][sel] if fam == 'dfa' else mix['input_index'][sel]
    return ok, np.unique(used)


def tmax_counts(c, fam):
    """(target_id, alpha) -> {task_id: mean count over the 8 seeds at T_max}, and the max seed-to-seed spread."""
    cfg = upstream_cfg(c, fam)
    Tm = str(cfg['T_checkpoints'][-1])
    acc, spread = {}, 0
    for k, v in mixes_json(c, fam).items():
        key = (v['target_id'], float(v['alpha']))
        acc.setdefault(key, []).append(v['counts'][Tm])
    out = {}
    for key, lst in acc.items():
        ids = sorted(set().union(*lst))
        arr = np.array([[d.get(i, 0) for i in ids] for d in lst], float)
        spread = max(spread, int((arr.max(0) - arr.min(0)).max()))
        out[key] = dict(zip(ids, arr.mean(0).tolist()))
    return out, spread


def fresh_dfa_inputs(c, fam_code, token, n):
    """n inputs from Experiment 11's training-partition sampler: length uniform on 8..24, then a uniform string."""
    M = C.mods(c)['dfa']
    cfg = M['cfg']
    rng = np.random.default_rng([c['task_sampling_seed'], fam_code, int(token)])
    lens = rng.integers(cfg['length_min'], cfg['length_max'] + 1, n)
    bits = np.zeros(n, np.uint64)
    for L in np.unique(lens):
        idx = np.flatnonzero(lens == L)
        bits[idx] = M['data'].sample_strings(rng, len(idx), int(L), 'train', cfg)
    return dict(bits=bits.astype(np.uint32), lens=lens.astype(np.uint8))


def fresh_references(c, fam, task, n, stream=1):
    """Fresh target examples from the target's own generator and input sampler (seed 20261015)."""
    if fam == 'dfa':
        raw = fresh_dfa_inputs(c, stream, task['task_token'], n)
        raw['y'] = C.labels_for(c, fam, task, raw)
        return raw
    M = C.mods(c)['dsl']
    s = M['data'].example_stream(M['cfg'], task, n, c['task_sampling_seed'], 'train')
    return dict(x=s['x'].astype(np.uint16), yw=s['y'].astype(np.uint16))


def unit_perm(c, fam, task, n_file):
    return np.random.default_rng([c['task_sampling_seed'], C.FAM_CODE[fam], 2, int(task['task_token'])]).permutation(
        n_file)


def build_refs_units(c, fam, used_by_target, log):
    """References per target and scored units per task. used_by_target: target_id -> used pool indices (union
    over all mixes and seeds)."""
    tasks = C.load_tasks(c, fam)
    R, U = {}, {}
    nR, nU = c['references_per_target'], c['scored_units_per_task']
    for t in tasks:
        f = C.load_train_file(c, fam, t['task_id'])
        n_file = C.n_examples(f)
        ref_idx = np.empty(0, np.int64)
        if t['role'] == 'target':
            unused = np.setdiff1d(np.arange(n_file), used_by_target[t['task_id']])
            ref_idx = unused[:nR]
            parts = [C.take(f, ref_idx)]
            n_fresh = nR - len(ref_idx)
            if n_fresh:
                parts.append(fresh_references(c, fam, t, n_fresh))
            raw = C.concat(parts) if len(parts) > 1 else parts[0]
            assert C.n_examples(raw) == nR
            src = np.r_[ref_idx, np.full(n_fresh, -1)]
            R[t['task_id']] = dict(raw=raw, file_index=src)
            log.append(dict(family=fam, target=t['task_id'], unused_file_examples=int(len(unused)),
                            refs_from_file=int(len(ref_idx)), refs_fresh=int(n_fresh)))
        perm = unit_perm(c, fam, t, n_file)
        perm = perm[~np.isin(perm, ref_idx)]
        uidx = perm[:nU]
        U[t['task_id']] = dict(raw=C.take(f, uidx), file_index=uidx)
        assert not np.intersect1d(uidx, ref_idx).size
    return tasks, R, U


def reconstruct(c, fam, workers=1, log=print):
    """Rebuild every gated mix; assert counts; return (tasks, all_ok, used_by_target, n_cells)."""
    import concurrent.futures as cf
    tasks = C.load_tasks(c, fam)
    mj = mixes_json(c, fam)
    cells = mix_cells(c, fam)
    used = {t['task_id']: np.empty(0, np.int64) for t in tasks if t['role'] == 'target'}
    ok_all = True
    if workers > 1:
        with cf.ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_check_one, [(fam, cell) for cell in cells], chunksize=8))
    else:
        res = [_check_one((fam, cell)) for cell in cells]
    for cell, (ok, u) in zip(cells, res):
        ok_all &= ok
        used[cell[1]] = np.union1d(used[cell[1]], u)
    return tasks, ok_all, used, len(cells)


def _check_one(args):
    fam, cell = args
    c = C.load_config()
    tasks = C.load_tasks(c, fam)
    rec = mixes_json(c, fam)[cell[0]]
    return check_mix(c, fam, tasks, cell, rec)


def pool_layout(tasks, target_id):
    """Order of P_T blocks: the target, then the distractors by token."""
    d = [t['task_id'] for t in tasks if t['role'] == 'distractor']
    return [target_id] + d
