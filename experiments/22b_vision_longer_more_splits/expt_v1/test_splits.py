"""Checks of the maps, held-out members and the p dial (run in the cluster prep job and before the freeze)."""
import json, os
import numpy as np
import data as D
import splits as S

CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')))


def check_arm(arm, base_train, val_rule, n_base, sizes):
    for m in ('real', 'shuffled'):
        ks = sorted({k for g in CFG['grid'].values() if g['arm'] == arm and g['map'] == m for k in g['splits']})
        for k in ks:
            gs = S.groups(arm, m, k)
            assert sorted(b for g in gs for b in g) == list(range(n_base)), 'map is not a partition'
            assert sorted(len(g) for g in gs) == sorted(sizes)
            ho = S.heldout(gs, k)
            assert len(set(ho)) == len(gs)
            prev_val, N = None, None
            for p in (0.0, 0.05, 1.0):
                for seed in (1, 2):
                    sp = S.make_split(base_train, arm, m, k, p, seed, val_rule)
                    tb = base_train[sp['train']]
                    assert len(sp['train']) == sp['N'], 'fixed budget broken'
                    assert len(set(sp['train'].tolist()) & set(sp['val'].tolist())) == 0
                    assert not np.isin(base_train[sp['val']], ho).any(), 'held-out class in val'
                    n_ho = int(np.isin(tb, ho).sum())
                    n_ho_all = int(np.isin(base_train, ho).sum())
                    assert n_ho == int(round(p * n_ho_all)), (p, n_ho, n_ho_all)
                    if prev_val is None:
                        prev_val, N = sp['val'], sp['N']
                    assert np.array_equal(prev_val, sp['val']) and N == sp['N'], 'val/budget must not depend on p, seed'
                    lab = sp['label']
                    assert all(lab[b] == y for b, y in zip(tb.tolist(), sp['y_train'].tolist()))
    if arm == 'vision':
        coarse = D.coarse_of_fine()
        assert all(coarse[f] == gi for gi, g in enumerate(S.groups('vision', 'real', 0)) for f in g)
    assert S.groups(arm, 'shuffled', 0) != S.groups(arm, 'shuffled', 1)
    assert S.groups(arm, 'shuffled', 0) != S.groups(arm, 'real', 0)


def main():
    d = D.load_cifar100()
    check_arm('vision', d['fine_train'], lambda n: CFG['vision']['val_per_class'], 100, [5] * 20)
    ho_real = {r: sorted(h for k in range(5 * r, 5 * r + 5) for h in S.heldout(S.groups('vision', 'real', k), k)) for r in (0, 1)}
    assert ho_real[0] == ho_real[1] == list(range(100)), 'each real round must hold out every fine class exactly once'
    assert S.groups('vision', 'real', 5) != S.groups('vision', 'real', 0), 'round 2 must use another member order'
    assert all(S.heldout(S.groups('vision', 'real', k), k) != S.heldout(S.groups('vision', 'real', k - 5), k - 5) for k in range(5, 10))
    sh = [S.groups('vision', 'shuffled', k) for k in range(10)]
    assert all(sh[i] != sh[j] for i in range(10) for j in range(i)), 'shuffled maps must differ'
    print('test_splits OK')


if __name__ == '__main__':
    main()
