"""Group maps, held-out members and the p dial (PROTOCOL Sections 2.2-2.4).

A *map* puts the base classes (CIFAR-100 fine classes, 20NG subgroups) into groups (the labels the model learns).
real     = the dataset's own hierarchy (CIFAR-100 superclasses; comp/rec/sci/talk).
shuffled = the same group sizes with members assigned by a fixed random permutation (structure removed).
Split k holds out member k % 5 of every group (members in the map's order; 22b: vision splits 0-4 and 5-9 are two
partitions of the 100 classes, the second in a seeded random member order). The held-out members' training items are
outside training at p = 0; at p > 0 a share p of them is mixed in at a fixed total budget."""
import numpy as np
import data as D

MAP_SEED = {'vision': 23000, 'text': 22500}   # 22b: vision shuffled maps are new permutations
REAL_ORDER_SEED = 23500                        # 22b: member order of the second real-map round (splits 5-9)


def groups(arm, map_kind, k):
    """List of groups; each group is a list of base-class ids in member order."""
    if arm == 'vision':
        coarse = D.coarse_of_fine()
        real = [sorted(f for f in range(100) if coarse[f] == c) for c in range(20)]
        sizes = [5] * 20
        base = list(range(100))
    else:
        real = [[D.SUBGROUPS.index(s) for s in D.TOP[g]] for g in ('comp', 'rec', 'sci', 'talk')]
        sizes = [len(g) for g in real]
        base = list(range(len(D.SUBGROUPS)))
    if map_kind == 'real':
        if arm == 'vision' and k >= 5:  # 22b: round k // 5 visits the members of every superclass in its own random order
            rng = np.random.default_rng(REAL_ORDER_SEED + k // 5)
            return [[g[i] for i in rng.permutation(len(g))] for g in real]
        return real
    assert map_kind == 'shuffled'
    perm = np.random.default_rng(MAP_SEED[arm] + k).permutation(base).tolist()
    out, i = [], 0
    for s in sizes:
        out.append(perm[i:i + s])
        i += s
    return out


def heldout(gs, k):
    """Held-out base class of each group: member k (every group used here has more than k members)."""
    return [g[k % 5] for g in gs]


def label_of(gs):
    """base class -> group index."""
    return {b: gi for gi, g in enumerate(gs) for b in g}


def stratified_take(idx_by_class, n_total, rng):
    """Take n_total items spread over the classes as evenly as possible (largest-remainder by class size), each
    class's items in a random order from rng. Returns sorted indices."""
    classes = sorted(idx_by_class)
    sizes = np.array([len(idx_by_class[c]) for c in classes], dtype=float)
    if n_total <= 0:
        return np.zeros(0, dtype=np.int64)
    share = sizes / sizes.sum() * n_total
    take = np.floor(share).astype(int)
    rem = int(n_total - take.sum())
    order = np.argsort(-(share - take), kind='stable')
    take[order[:rem]] += 1
    take = np.minimum(take, sizes.astype(int))
    out = []
    for c, t in zip(classes, take):
        ids = np.asarray(idx_by_class[c])
        out.append(ids[rng.permutation(len(ids))[:t]])
    return np.sort(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def make_split(base_train, arm, map_kind, k, p, seed, val_rule):
    """base_train: base-class id of every training item. Returns dict with train/val indices and their group labels.

    val: a fixed share of the *seen* base classes' training items (val_rule(n) items of a class with n items), drawn
    with a seed that depends on (arm, map, k) only, so every p and seed of a split validates on the same items.
    Budget N = seen training items minus val. At p, n_h = round(p * (held-out training items)) held-out items are
    mixed in (stratified over held-out classes) and the seen pool is subsampled to N - n_h (stratified), so every
    run of a split trains on exactly N items."""
    gs = groups(arm, map_kind, k)
    ho = set(heldout(gs, k))
    lab = label_of(gs)
    base_train = np.asarray(base_train)
    by = {}
    for i, b in enumerate(base_train.tolist()):
        by.setdefault(b, []).append(i)
    vrng = np.random.default_rng([22, 0 if arm == 'vision' else 1, 0 if map_kind == 'real' else 1, k])
    val, seen_pool = [], {}
    for b in sorted(by):
        if b in ho:
            continue
        ids = np.asarray(by[b])
        perm = vrng.permutation(len(ids))
        nv = val_rule(len(ids))
        val.append(ids[perm[:nv]])
        seen_pool[b] = ids[perm[nv:]].tolist()
    val = np.sort(np.concatenate(val))
    N = sum(len(v) for v in seen_pool.values())
    ho_pool = {b: by[b] for b in sorted(ho)}
    n_ho_total = sum(len(v) for v in ho_pool.values())
    n_h = int(round(p * n_ho_total))
    rng = np.random.default_rng([seed, 7, k])
    tr_h = stratified_take(ho_pool, n_h, rng)
    tr_s = stratified_take(seen_pool, N - len(tr_h), rng)
    train = np.sort(np.concatenate([tr_s, tr_h]))
    return {'groups': gs, 'heldout': sorted(ho), 'label': lab, 'train': train, 'val': val, 'N': N, 'n_h': int(len(tr_h)),
            'y_train': np.array([lab[b] for b in base_train[train].tolist()], dtype=np.int64),
            'y_val': np.array([lab[b] for b in base_train[val].tolist()], dtype=np.int64)}
