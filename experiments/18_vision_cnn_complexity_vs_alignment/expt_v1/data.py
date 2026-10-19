"""Datasets, mismatch transforms, the training-pair stream (alignment construction and common random numbers),
the held-out evaluation pairs and the probe pairs (PROTOCOL Sections 2.3-2.5)."""
import hashlib
import os
from fractions import Fraction

import numpy as np

DATA = os.environ.get('VISION_DATA', '/var/lib/agent/scratch/vision_data/npz')
FILES = {'mnist': 'mnist.npz', 'fashion': 'fashion.npz', 'kmnist': 'kmnist.npz', 'cifar10': 'cifar10.npz'}
EVAL_SEED = 20261018 + 7
N_EVAL = 4096
N_PROBE = 1000
N_TRAINDIST = 1024
_cache = {}


def stable(*parts):
    """A 63-bit integer from strings and ints, stable across processes and machines."""
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], 'big') >> 1


def rng(*parts):
    return np.random.Generator(np.random.PCG64(stable(*parts)))


def load(name):
    if name not in _cache:
        z = np.load(os.path.join(DATA, FILES[name]))
        d = {k: z[k] for k in ('x_train', 'y_train', 'x_test', 'y_test')}
        if d['x_train'].ndim == 3:
            d['x_train'] = d['x_train'][:, None]
            d['x_test'] = d['x_test'][:, None]
        else:
            d['x_train'] = np.ascontiguousarray(d['x_train'].transpose(0, 3, 1, 2))
            d['x_test'] = np.ascontiguousarray(d['x_test'].transpose(0, 3, 1, 2))
        d['y_train'] = d['y_train'].astype(np.int64)
        d['y_test'] = d['y_test'].astype(np.int64)
        x = d['x_train'].astype(np.float64) / 255.0
        d['mean'] = x.mean(axis=(0, 2, 3)).astype(np.float32)
        d['std'] = x.std(axis=(0, 2, 3)).astype(np.float32)
        d['by_class'] = [np.flatnonzero(d['y_train'] == c) for c in range(10)]
        _cache[name] = d
    return _cache[name]


# ---------------------------------------------------------------- mismatch transforms (uint8 in, uint8 out, N x C x H x W)
def _rot90(x):
    return np.ascontiguousarray(np.rot90(x, 1, axes=(2, 3)))


def _scale05(x):
    n, c, h, w = x.shape
    small = x.reshape(n, c, h // 2, 2, w // 2, 2).astype(np.float32).mean(axis=(3, 5))
    out = np.zeros_like(x)
    out[:, :, h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2] = np.round(small).astype(np.uint8)
    return out


def _thick(x):
    from scipy.ndimage import grey_dilation
    return grey_dilation(x, size=(1, 1, 3, 3))


def _thin(x):
    from scipy.ndimage import grey_erosion
    return grey_erosion(x, size=(1, 1, 2, 2))


TRANSFORMS = {'rot90': _rot90, 'scale05': _scale05, 'thick': _thick, 'thin': _thin}
SWAPS = {'swapF': 'fashion', 'swapK': 'kmnist', 'swapM': 'mnist'}
_mis_cache = {}


def misaligned_train(name, mismatch):
    """(images, by_class) of the misaligned source. For a transform, image i is train image i transformed (so it keeps
    class y_train[i]); for a swap, it is the other dataset's train split, used through its per-class index lists."""
    key = (name, mismatch)
    if key not in _mis_cache:
        if mismatch in TRANSFORMS:
            d = load(name)
            _mis_cache[key] = (TRANSFORMS[mismatch](d['x_train']), d['by_class'])
        else:
            o = load(SWAPS[mismatch])
            _mis_cache[key] = (o['x_train'], o['by_class'])
    return _mis_cache[key]


# ---------------------------------------------------------------- stream
def aligned_mask(n, alpha):
    """Example k is a target/aligned slot iff floor((k+1) alpha) - floor(k alpha) = 1 (exact rational arithmetic), so
    after m examples exactly floor(m alpha) are aligned, at every checkpoint."""
    f = Fraction(str(alpha))
    k = np.arange(n + 1, dtype=np.int64)
    fl = (k * f.numerator) // f.denominator
    return (fl[1:] - fl[:-1]) == 1


def base_draws(name, seed, n):
    """Per-seed common random numbers: the k-th example's two train-image indices, the uniform draws used to pick
    swap images of the same class, and the augmentation draws (CIFAR). Identical for every run of (dataset, seed)."""
    d = load(name)
    r = rng('pairs', name, seed)
    idx = r.integers(0, len(d['y_train']), size=(n, 2))
    u = rng('swap', name, seed).random((n, 2))
    aug = rng('aug', name, seed).integers(0, 9, size=(n, 2, 2))
    flip = rng('flip', name, seed).integers(0, 2, size=(n, 2))
    return dict(idx=idx, cls=d['y_train'][idx], u=u, aug=aug, flip=flip)


def gather(name, mismatch, draws, sl, aligned):
    """uint8 pair images (B, 2, C, H, W) and class pairs (B, 2) for stream positions sl."""
    d = load(name)
    idx, cls = draws['idx'][sl], draws['cls'][sl]
    x = d['x_train'][idx]
    m = ~aligned
    if m.any():
        mx, mcls = misaligned_train(name, mismatch)
        if mismatch in TRANSFORMS:
            x[m] = mx[idx[m]]
        else:
            u = draws['u'][sl][m]
            c = cls[m]
            sel = np.empty(c.shape, np.int64)
            for j in range(10):
                q = c == j
                if q.any():
                    lst = mcls[j]
                    sel[q] = lst[np.minimum((u[q] * len(lst)).astype(np.int64), len(lst) - 1)]
            x[m] = mx[sel]
    return x, cls


def augment(x, draws, sl):
    """CIFAR only: random crop with 4-pixel zero padding and horizontal flip, from the per-seed draws."""
    b, s, c, h, w = x.shape
    p = np.zeros((b, s, c, h + 8, w + 8), x.dtype)
    p[:, :, :, 4:4 + h, 4:4 + w] = x
    aug, flip = draws['aug'][sl], draws['flip'][sl]
    out = np.empty_like(x)
    for i in range(b):
        for j in range(s):
            dy, dx = aug[i, j]
            im = p[i, j, :, dy:dy + h, dx:dx + w]
            out[i, j] = im[:, :, ::-1] if flip[i, j] else im
    return out


def eval_pairs(name):
    """Fixed held-out pairs from the test split (test images never appear in training)."""
    d = load(name)
    idx = rng('eval', EVAL_SEED, len(d['y_test'])).integers(0, len(d['y_test']), size=(N_EVAL, 2))
    return d['x_test'][idx], d['y_test'][idx], idx


def probe_pairs(name, seed):
    d = load(name)
    idx = rng('probe', name, seed).integers(0, len(d['y_train']), size=(N_PROBE, 2))
    return d['x_train'][idx], d['y_train'][idx], idx


def traindist_pairs(name, mismatch, seed):
    """Misaligned-source pairs (the training mix's other part), for the not-gated training-distribution error."""
    d = load(name)
    r = rng('traindist', name, mismatch, seed)
    dr = dict(idx=r.integers(0, len(d['y_train']), size=(N_TRAINDIST, 2)), u=r.random((N_TRAINDIST, 2)))
    dr['cls'] = d['y_train'][dr['idx']]
    return gather(name, mismatch, dr, slice(0, N_TRAINDIST), np.zeros(N_TRAINDIST, bool))
