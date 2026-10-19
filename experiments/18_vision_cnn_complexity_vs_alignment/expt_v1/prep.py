"""Convert the raw MNIST, Fashion-MNIST, KMNIST (idx.gz) and CIFAR-10 (HuggingFace parquet) files into one uint8 npz
per dataset: x_train (N,H,W[,3]), y_train, x_test, y_test. Records the SHA-256 of every raw file and every npz."""
import argparse, gzip, hashlib, io, json, os
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def idx(p):
    with gzip.open(p, 'rb') as f:
        d = f.read()
    nd = d[3]
    dims = [int.from_bytes(d[4 + 4 * i:8 + 4 * i], 'big') for i in range(nd)]
    return np.frombuffer(d, np.uint8, offset=4 + 4 * nd).reshape(dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='/var/lib/agent/scratch/vision_data')
    ap.add_argument('--out', default='/var/lib/agent/scratch/vision_data/npz')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {'raw': {}, 'npz': {}}
    for name in ('mnist', 'fashion', 'kmnist'):
        p = {k: os.path.join(a.raw, name, f'{k}.gz') for k in
             ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')}
        for k, v in p.items():
            rec['raw'][f'{name}/{k}.gz'] = sha(v)
        d = dict(x_train=idx(p['train-images-idx3-ubyte']), y_train=idx(p['train-labels-idx1-ubyte']),
                 x_test=idx(p['t10k-images-idx3-ubyte']), y_test=idx(p['t10k-labels-idx1-ubyte']))
        assert d['x_train'].shape == (60000, 28, 28) and d['x_test'].shape == (10000, 28, 28)
        assert set(np.unique(d['y_train'])) == set(range(10))
        np.savez(os.path.join(a.out, f'{name}.npz'), **d)
    import pyarrow.parquet as pq
    from PIL import Image
    d = {}
    for split, f in (('train', 'train-00000-of-00001.parquet'), ('test', 'test-00000-of-00001.parquet')):
        fp = os.path.join(a.raw, 'cifar_hf', f)
        rec['raw'][f'cifar_hf/{f}'] = sha(fp)
        t = pq.read_table(fp).to_pydict()
        ik = [k for k in t if 'im' in k][0]
        lk = [k for k in t if 'label' in k][0]
        d[f'x_{split}'] = np.stack([np.array(Image.open(io.BytesIO(r['bytes'])).convert('RGB')) for r in t[ik]])
        d[f'y_{split}'] = np.array(t[lk], np.uint8)
    assert d['x_train'].shape == (50000, 32, 32, 3) and d['x_test'].shape == (10000, 32, 32, 3)
    assert np.bincount(d['y_test']).tolist() == [1000] * 10
    np.savez(os.path.join(a.out, 'cifar10.npz'), **d)
    for f in sorted(os.listdir(a.out)):
        if f.endswith('.npz'):
            rec['npz'][f] = sha(os.path.join(a.out, f))
    json.dump(rec, open(os.path.join(a.out, 'SHA256.json'), 'w'), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
