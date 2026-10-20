"""Shared plumbing for Experiment 15: config, hash-checked imports of Experiments 11, 12 and 14, the task tables, the
example sets, serialization and small I/O helpers.

Upstream code is imported read-only through Experiment 14's tasks14.exp_modules (which hash-checks Experiment 11/12
config.json, dfa.py / dsl.py, data.py and model.py against their manifests); everything else imported here is checked
against the SHA-256 values pinned in upstream_sha256.json (written once, before the freeze, by `common.py --pin`).
Experiment 12's laws.py differs from its run manifest by the documented post-run polish of M_align_add (Experiment 12
RESULTS item 7); the pinned hash is the committed file's.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
FAMILIES = ('dfa', 'dsl')
FAM_CODE = dict(dfa=11, dsl=12)


def load_config():
    return json.loads((HERE / 'config.json').read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def atomic_write_text(path, text):
    path = Path(path)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def exp_dirs(c):
    return dict(dfa=(HERE / c['exp11_dir']).resolve(), dsl=(HERE / c['exp12_dir']).resolve(),
                e14=(HERE / c['exp14_dir']).resolve())


def upstream_files(c):
    d = exp_dirs(c)
    out = {}
    for key, fam in (('exp11_sources', 'dfa'), ('exp12_sources', 'dsl'), ('exp14_sources', 'e14')):
        for s in c[key]:
            out[f'{fam}/{s}'] = d[fam] / s
    for fam in ('dfa', 'dsl'):
        for s in ('tasks.json', 'mixes.json', 'metrics.csv', 'manifest.json'):
            out[f'{fam}/results/{s}'] = d[fam] / 'results' / s
    for fam in ('dfa', 'dsl'):
        for p in c['probes']:
            out[f'e14/results/probes/{fam}_{p}.npz'] = d['e14'] / 'results' / 'probes' / f'{fam}_{p}.npz'
    return out


def pin_upstream(c):
    pins = {k: sha256_file(p) for k, p in upstream_files(c).items()}
    atomic_write_text(HERE / 'upstream_sha256.json', json.dumps(pins, indent=1, sort_keys=True))
    return pins


_CHECKED = {}


def check_upstream(c=None, keys=None):
    """Assert every pinned upstream file still has its pinned SHA-256 (and probes their Experiment 14 artifact hash)."""
    c = c or load_config()
    pins = json.loads((HERE / 'upstream_sha256.json').read_text())
    files = upstream_files(c)
    bad = []
    for k in (keys or pins):
        if k in _CHECKED:
            continue
        h = sha256_file(files[k])
        if h != pins[k]:
            bad.append((k, h, pins[k]))
        _CHECKED[k] = h
    if bad:
        raise RuntimeError(f'upstream files changed: {bad}')
    art = json.loads((exp_dirs(c)['e14'].parent / 'ARTIFACT_SHA256.json').read_text())['files']
    for fam in FAMILIES:
        for p in c['probes']:
            k = f'probes/{fam}_{p}.npz'
            if art[k]['sha256'] != pins[f'e14/results/{k}']:
                raise RuntimeError(f'{k} differs from Experiment 14 ARTIFACT_SHA256.json')
    return pins


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_MODS = {}


def mods(c=None):
    """dict: e14 (tasks14, probe, task2vec, cfg), dfa / dsl (gen, data, model, cfg, dir, laws, IF)."""
    if _MODS:
        return _MODS
    c = c or load_config()
    check_upstream(c)
    d = exp_dirs(c)
    if str(d['e14']) not in sys.path:
        sys.path.append(str(d['e14']))
    import tasks14  # noqa: E402  (Experiment 14, hash-checked above)
    import probe  # noqa: E402
    import task2vec  # noqa: E402
    em = tasks14.exp_modules(tasks14.load_config())
    _MODS['e14'] = dict(T=tasks14, probe=probe, t2v=task2vec, cfg=tasks14.load_config(), dir=d['e14'])
    for fam in FAMILIES:
        m = dict(em[fam])
        saved = sys.modules.pop('laws', None)
        laws = _load('laws', d[fam] / 'laws.py')
        IF = _load(f'e_{fam}_independent_fit', d[fam] / 'independent_fit.py')
        sys.modules.pop('laws')
        sys.modules[f'e_{fam}_laws'] = laws
        if saved is not None:
            sys.modules['laws'] = saved
        m['laws'], m['IF'] = laws, IF
        _MODS[fam] = m
    return _MODS


def results_dir(c, fam):
    return exp_dirs(c)[fam] / 'results'


def load_tasks(c, fam):
    """Targets and distractors of Experiment 11 (dfa) or 12 (dsl), ordered by task token."""
    t = json.loads((results_dir(c, fam) / 'tasks.json').read_text())
    t = [x for x in t if x['role'] in ('target', 'distractor')]
    return sorted(t, key=lambda x: x['task_token'])


def load_train_file(c, fam, task_id):
    """The task's Experiment 11/12 training file as raw example arrays (file order)."""
    z = np.load(results_dir(c, fam) / 'data' / f'{task_id}_train.npz')
    if fam == 'dfa':
        return dict(bits=z['bits'].astype(np.uint32), lens=z['length'].astype(np.uint8),
                    y=z['label'].astype(np.int64))
    return dict(x=z['x'].astype(np.uint16), yw=z['y'].astype(np.uint16))


def take(raw, idx):
    return {k: np.asarray(v)[idx] for k, v in raw.items()}


def concat(raws):
    return {k: np.concatenate([r[k] for r in raws]) for k in raws[0]}


def n_examples(raw):
    return len(next(iter(raw.values())))


# ---------------------------------------------------------------- serialization (task token stripped)

def byte_table(c, fam):
    s = c['serialization'][fam]
    return {k: v.encode('latin-1') for k, v in s.items()}


def task_token_byte(c, token):
    b = c['serialization']['task_token_first_byte'] + int(token)
    assert 161 <= b <= 255, b
    return bytes([b])


def serialize(c, fam, raw, token=None, input_only=False):
    """List of byte strings: [task token byte if token is given] input symbols [sep label] (one byte per token).

    DFA: characters a / b, separator, accept / reject. DSL: the 10 input bits, separator, the 10 output bits."""
    tb = byte_table(c, fam)
    n = n_examples(raw)
    toks = None if token is None else np.broadcast_to(np.asarray(token), (n,))
    out = []
    if fam == 'dfa':
        bits, lens, y = raw['bits'].astype(np.uint64), raw['lens'], raw['y']
        A, B = tb['a'][0], tb['b'][0]
        sh = np.arange(24, dtype=np.uint64)
        ch = np.where((bits[:, None] >> sh) & np.uint64(1), B, A).astype(np.uint8)
        for i in range(n):
            s = ch[i, : int(lens[i])].tobytes()
            if not input_only:
                s += tb['sep'] + (tb['accept'] if y[i] else tb['reject'])
            out.append(s if toks is None else task_token_byte(c, toks[i]) + s)
    else:
        x, yw = raw['x'].astype(np.int64), raw['yw'].astype(np.int64)
        sh = np.arange(10)
        xi = np.where((x[:, None] >> sh) & 1, tb['in1'][0], tb['in0'][0]).astype(np.uint8)
        yo = np.where((yw[:, None] >> sh) & 1, tb['out1'][0], tb['out0'][0]).astype(np.uint8)
        for i in range(n):
            s = xi[i].tobytes()
            if not input_only:
                s += tb['sep'] + yo[i].tobytes()
            out.append(s if toks is None else task_token_byte(c, toks[i]) + s)
    return out


def n_tokens(fam, raw):
    """t(x): tokens in the stripped serialization."""
    if fam == 'dfa':
        return raw['lens'].astype(np.float64) + 2.0
    return np.full(n_examples(raw), 21.0)


def content_keys(fam, raw, input_only=False):
    """Integer keys identifying an example's content (input and label, or input only)."""
    if fam == 'dfa':
        k = raw['lens'].astype(np.int64) * (1 << 25) + raw['bits'].astype(np.int64)
        return k if input_only else k * 2 + raw['y'].astype(np.int64)
    k = raw['x'].astype(np.int64)
    return k if input_only else k * 1024 + raw['yw'].astype(np.int64)


def labels_for(c, fam, task, raw):
    """Labels of `task` on the inputs of `raw` (DFA accept bit; DSL output word)."""
    M = mods(c)[fam]
    if fam == 'dfa':
        return M['gen'].run_dfa(M['gen'].DFA.from_json(task['definition']), raw['bits'], raw['lens']).astype(np.int64)
    return np.asarray(task['truth_table'], np.uint16)[raw['x'].astype(np.int64)]


def relabel(c, fam, task, raw):
    out = dict(raw)
    if fam == 'dfa':
        out['y'] = labels_for(c, fam, task, raw)
    else:
        out['yw'] = labels_for(c, fam, task, raw)
    return out


def probe_input(c, fam, raw, token=None):
    """Experiment 14 probe encoding: dict(X, last, y) with the null token unless token (Experiment 11/12 task
    token index, the leak control) is given."""
    T = mods(c)['e14']['T']
    c14 = mods(c)['e14']['cfg']
    if fam == 'dfa':
        r = dict(bits=raw['bits'], lens=raw['lens'], y=raw['y'])
    else:
        M = mods(c)['dsl']
        r = dict(bits_x=M['data'].to_bits(raw['x']), y=M['data'].to_bits(raw['yw']).astype(np.int64))
    tok = None if token is None else np.broadcast_to(np.asarray(token, np.int64), (n_examples(raw),))
    return T.encode(c14, fam, r, token_index=tok)


def load_probe(c, fam, seed):
    m = mods(c)['e14']
    return m['probe'].load_probe(m['cfg'], fam, m['dir'] / 'results' / 'probes' / f'{fam}_{seed}.npz')


if __name__ == '__main__':
    if '--pin' in sys.argv:
        print(json.dumps(pin_upstream(load_config()), indent=1))
