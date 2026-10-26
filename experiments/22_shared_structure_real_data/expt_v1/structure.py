"""Measured shared structure and complexity of every unit (PROTOCOL Section 2.5), from inputs only: no Exp 22 model
and no evaluation accuracy is used. A unit is (arm, map, split k, held-out base class h); its group is the group h
belongs to under that map.

  python structure.py --results results            # writes results/structure.json

Vision features: the frozen CIFAR-10 ViT's [CLS] features of the CIFAR-100 training images (run.py feat).
Text features: TF-IDF (unigrams+bigrams) fit on the training documents of all 17 subgroups."""
import argparse, gzip, json, os
import numpy as np
import data as D
import splits as S

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'config.json')))
TOPN = 200


def unit_keys(arm):
    out = []
    for g in CFG['grid'].values():
        if g['arm'] == arm:
            out += [(g['map'], k) for k in g['splits']]
    return sorted(set(out))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def centroid_stats(F, base, gs, k, nc_rows=None):
    """margin = cos(c_h, C_own) - mean cos(c_h, C_other); nc_share = share of h's items whose nearest seen-group
    centroid (cosine) is their own group. C_g = mean feature of the seen (not held-out) members of group g."""
    ho = S.heldout(gs, k)
    C = np.stack([F[np.isin(base, [b for b in g if b not in ho])].mean(0) for g in gs])
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    out = {}
    for gi, h in enumerate(ho):
        Fh = F[base == h]
        ch = Fh.mean(0)
        c = [cos(ch, C[j]) for j in range(len(gs))]
        margin = c[gi] - float(np.mean([c[j] for j in range(len(gs)) if j != gi]))
        Fhn = Fh / (np.linalg.norm(Fh, axis=1, keepdims=True) + 1e-12)
        nc = float(np.mean((Fhn @ Cn.T).argmax(1) == gi))
        out[h] = {'group': gi, 'margin': margin, 'nc_share': nc}
    return out


def pad(Xa, Xb, seed):
    """Proxy A-distance 2(1 - 2 err) of a 2-fold logistic regression telling Xa from Xb (balanced by subsampling)."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = min(Xa.shape[0], Xb.shape[0])
    ia, ib = rng.permutation(Xa.shape[0])[:n], rng.permutation(Xb.shape[0])[:n]
    X = _vstack(Xa[ia], Xb[ib])
    y = np.r_[np.zeros(n), np.ones(n)]
    fold = np.r_[rng.permutation(n) % 2, rng.permutation(n) % 2]
    err = 0
    for f in (0, 1):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X[fold != f], y[fold != f])
        err += int((clf.predict(X[fold == f]) != y[fold == f]).sum())
    return float(2 * (1 - 2 * err / (2 * n)))


def _vstack(a, b):
    import scipy.sparse as sp
    return sp.vstack([a, b]).tocsr() if sp.issparse(a) else np.vstack([a, b])


def gzip_bits(items):
    return float(np.mean([8 * len(gzip.compress(x, 9, mtime=0)) for x in items]))


def vision(results):
    d = D.load_cifar100()
    base = d['fine_train']
    feats = np.load(os.path.join(results, 'feat', 'cifar100_features_from_cifar10_vit.npz'))['train'].astype(np.float64)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    x = d['x_train'].astype(np.float64) / 255.0
    pix = x.reshape(-1, 8, 4, 8, 4, 3).mean((2, 4)).reshape(len(x), -1)  # 8x8x3 average-pooled pixels
    pix = (pix - pix.mean(0)) / (pix.std(0) + 1e-8)
    gz = {f: gzip_bits([im.tobytes() for im in d['x_test'][d['fine_test'] == f]]) for f in range(100)}
    out = {}
    for m, k in unit_keys('vision'):
        gs = S.groups('vision', m, k)
        ho = S.heldout(gs, k)
        a, b = centroid_stats(feats, base, gs, k), centroid_stats(pix, base, gs, k)
        for gi, h in enumerate(ho):
            seen_own = [x for x in gs[gi] if x not in ho]
            r = {'group': gi, 'margin': a[h]['margin'], 'nc_share': a[h]['nc_share'],
                 'pixel_margin': b[h]['margin'], 'gzip_bits': gz[h],
                 'pad': pad(feats[base == h], feats[np.isin(base, seen_own)], 1000 * k + h)}
            out[f'vision/{m}/{k}/{h}'] = r
    return out


def text():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import chi2
    ng = D.load_20ng()
    base = np.array([D.SUBGROUPS.index(x['g']) for x in ng['train']])
    base_te = np.array([D.SUBGROUPS.index(x['g']) for x in ng['test']])
    vec = TfidfVectorizer(tokenizer=D.tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=3,
                          max_features=100000, sublinear_tf=True, dtype=np.float64)
    X = vec.fit_transform([x['t'] for x in ng['train']])
    gz = {b: gzip_bits([ng['test'][i]['t'].encode('latin-1') for i in np.where(base_te == b)[0]])
          for b in range(len(D.SUBGROUPS))}
    out = {}
    for m, k in unit_keys('text'):
        gs = S.groups('text', m, k)
        ho = S.heldout(gs, k)
        C = np.stack([np.asarray(X[np.isin(base, [b for b in g if b not in ho])].mean(0)).ravel() for g in gs])
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
        for gi, h in enumerate(ho):
            Xh = X[base == h]
            ch = np.asarray(Xh.mean(0)).ravel()
            c = [cos(ch, C[j]) for j in range(len(gs))]
            margin = c[gi] - float(np.mean([c[j] for j in range(len(gs)) if j != gi]))
            sims = np.asarray(Xh @ Cn.T)
            nc = float(np.mean(sims.argmax(1) == gi))
            seen_own = [x for x in gs[gi] if x not in ho]
            other_seen = [x for j, g in enumerate(gs) if j != gi for x in g if x not in ho]
            top = []
            for pos in ([h], seen_own):
                rows = np.isin(base, pos + other_seen)
                sc, _ = chi2(X[rows], np.isin(base[rows], pos))
                sc = np.nan_to_num(sc, nan=-1.0)
                top.append(set(np.argsort(-sc, kind='stable')[:TOPN].tolist()))
            jac = len(top[0] & top[1]) / len(top[0] | top[1])
            out[f'text/{m}/{k}/{h}'] = {'group': gi, 'margin': margin, 'nc_share': nc, 'jaccard_top200': jac,
                                        'gzip_bits': gz[h], 'pad': pad(Xh, X[np.isin(base, seen_own)], 1000 * k + h)}
    return out


def compute(results):
    import sklearn
    return {'units': {**vision(results), **text()}, 'sklearn': sklearn.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out')
    a = ap.parse_args()
    r = compute(a.results)
    op = a.out or os.path.join(a.results, 'structure.json')
    with open(op + '.tmp', 'w') as f:
        json.dump(r, f, indent=1, sort_keys=True)
    os.replace(op + '.tmp', op)
    print('wrote', op, len(r['units']), 'units')


if __name__ == '__main__':
    main()
