"""Data for Experiment 22: CIFAR-100 (20 superclasses), CIFAR-10 (feature extractor only) and 20 Newsgroups (4 top-level
groups). Raw files -> prepared files (prep.py) -> loaders here. Every prepared array is hashed in data_sha256.json."""
import hashlib, io, json, os, re, tarfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('EXP22_DATA', os.path.join(HERE, '..', 'data'))

RAW = {  # file name under the raw folder -> URL (sha256 in data_sha256.json)
    'cifar100_train.parquet': 'https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet',
    'cifar100_test.parquet': 'https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet',
    'cifar10_train.parquet': 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet',
    'cifar10_test.parquet': 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet',
    '20news-bydate.tar.gz': 'https://ndownloader.figshare.com/files/5975967',
}

# 20 Newsgroups: the 17 subgroups under the four top-level hierarchies used here (alt.atheism, misc.forsale and
# soc.religion.christian have no sibling under their top level and are left out).
TOP = {
    'comp': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
    'rec': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
    'sci': ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
    'talk': ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'],
}
SUBGROUPS = [s for g in TOP.values() for s in g]  # 17, in this fixed order
MIN_TOKENS = 5  # documents with fewer word tokens after stripping are dropped in prep (disclosed in PROTOCOL 2.1)

# --- sklearn 1.x's 20 Newsgroups cleaning (BSD-3), copied so the result does not depend on the installed version ---
_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)')


def strip_header(text):
    return text.partition('\n\n')[2]


def strip_quoting(text):
    return '\n'.join(l for l in text.split('\n') if not _QUOTE_RE.search(l))


def strip_footer(text):
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        if lines[line_num].strip().strip('-') == '':
            break
    return '\n'.join(lines[:line_num]) if line_num > 0 else text


_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text):
    return _TOK.findall(text.lower())


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_array(a):
    return sha_bytes(np.ascontiguousarray(a).tobytes())


# ---------------------------------------------------------------- prep (raw -> prepared)
def _parquet_images(path, label_cols):
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path).to_pydict()
    x = np.stack([np.asarray(Image.open(io.BytesIO(d['bytes'])).convert('RGB'), dtype=np.uint8) for d in t['img']])
    return x, [np.asarray(t[c], dtype=np.int64) for c in label_cols]


def prep_cifar100(raw, out):
    d = {}
    for split in ('train', 'test'):
        x, (yf, yc) = _parquet_images(os.path.join(raw, f'cifar100_{split}.parquet'), ['fine_label', 'coarse_label'])
        d[f'x_{split}'], d[f'fine_{split}'], d[f'coarse_{split}'] = x, yf, yc
    np.savez(os.path.join(out, 'cifar100.npz'), **d)


def prep_cifar10(raw, out):
    d = {}
    for split in ('train', 'test'):
        x, (y,) = _parquet_images(os.path.join(raw, f'cifar10_{split}.parquet'), ['label'])
        d[f'x_{split}'], d[f'y_{split}'] = x, y
    np.savez(os.path.join(out, 'cifar10.npz'), **d)


def prep_20ng(raw, out):
    docs = {'train': [], 'test': []}
    dropped = {'train': 0, 'test': 0}
    with tarfile.open(os.path.join(raw, '20news-bydate.tar.gz')) as tf:
        files = {}
        for m in tf:  # archive order: a sorted walk would re-read the gzip stream for every member
            parts = m.name.split('/')
            if m.isfile() and len(parts) == 3 and parts[1] in SUBGROUPS:
                files[m.name] = tf.extractfile(m).read()
        for name in sorted(files):
            parts = name.split('/')
            split = 'train' if parts[0].endswith('train') else 'test'
            text = files[name].decode('latin-1')
            text = strip_quoting(strip_footer(strip_header(text)))
            if len(tokenize(text)) < MIN_TOKENS:
                dropped[split] += 1
                continue
            docs[split].append({'g': parts[1], 'id': parts[2], 't': text})
    with open(os.path.join(out, '20ng.json'), 'w') as f:
        json.dump({'docs': docs, 'dropped': dropped}, f, sort_keys=True)


def prepared_hashes():
    """sha256 of every prepared array / document list, the values data_sha256.json['prepared'] pins."""
    h = {}
    for name in ('cifar100', 'cifar10'):
        d = np.load(os.path.join(DATA, f'{name}.npz'))
        for k in sorted(d.files):
            h[f'{name}/{k}'] = sha_array(d[k])
    ng = load_20ng()
    for split in ('train', 'test'):
        h[f'20ng/{split}'] = sha_bytes(json.dumps(ng[split], sort_keys=True).encode())
    return h


# ---------------------------------------------------------------- loaders
_cache = {}


def load_cifar100():
    if 'c100' not in _cache:
        d = np.load(os.path.join(DATA, 'cifar100.npz'))
        _cache['c100'] = {k: d[k] for k in d.files}
    return _cache['c100']


def load_cifar10():
    if 'c10' not in _cache:
        d = np.load(os.path.join(DATA, 'cifar10.npz'))
        _cache['c10'] = {k: d[k] for k in d.files}
    return _cache['c10']


def load_20ng():
    if '20ng' not in _cache:
        with open(os.path.join(DATA, '20ng.json')) as f:
            j = json.load(f)
        _cache['20ng'] = {'train': j['docs']['train'], 'test': j['docs']['test'], 'dropped': j['dropped']}
    return _cache['20ng']


def coarse_of_fine():
    """CIFAR-100 fine class -> superclass, read from the data (it is a fixed map)."""
    d = load_cifar100()
    m = {}
    for f, c in zip(d['fine_train'].tolist(), d['coarse_train'].tolist()):
        assert m.setdefault(f, c) == c
    return [m[f] for f in range(100)]
