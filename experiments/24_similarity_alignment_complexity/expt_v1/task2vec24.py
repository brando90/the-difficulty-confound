"""Task2Vec alignment coefficient of every Experiment 24 training mix with its target (reported, never gated;
PROTOCOL Section 4.5), after the alignment coefficient of Chawla, Miranda, Koyejo et al. (arXiv 2501.08496):
coef(mix, target) = 1 - E[cosine distance between Task2Vec embeddings of a batch of the mix and a batch of the target].

Probe network: a size-S model (vocabulary 301) trained once, seed 24999, on a uniform mixture of all 160 concepts for
2^24 tokens at the S learning rate selected by the pilot. Embedding of a batch of 64 documents: the diagonal empirical
Fisher information (mean over documents of the squared gradient of the document's log-likelihood) of the probe's last
transformer block and final LayerNorm, L2-normalized. Mix batches: documents 0..255 of the mix's training stream
(seed 24101, the run's exact schedule); target batches: held-out documents 0..255. 4 x 4 batch pairs per mix.

Usage (from expt_v1/): task2vec24.py --results DIR [--device cpu|cuda]
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run24 as R  # noqa: E402
import concepts24 as C  # noqa: E402
import gpu_arm as G  # noqa: E402
import hmm as H  # noqa: E402
import model as M  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PROBE_SEED, MIX_SEED, BATCH, NB = 24999, 24101, 64, 4
PROBE_T = int(__import__('os').environ.get('TDC_T2V_TOKENS', 2 ** 24))  # override for smoke tests only


def train_probe(cfg, tasks, lr, device, log):
    torch.manual_seed(PROBE_SEED)
    m = M.build_model(R.model_config(cfg, 'S')).to(device)
    opt = G.make_optimizer(R.train_config(cfg), m, lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg['model']['warmup_steps']))
    L, Bd, V = cfg['concepts']['doc_len'], cfg['model']['batch_docs'], cfg['concepts']['V_model_tokens']
    cm = {t['cid']: H.from_json(t['definition']) for t in tasks}
    order = np.random.default_rng(PROBE_SEED).permutation(sorted(cm))
    used = defaultdict(int)
    m.train()
    for step in range(PROBE_T // (L * Bd)):
        batch = np.empty((Bd, L), np.int64)
        for k in range(Bd):
            c = int(order[(step * Bd + k) % len(order)])
            batch[k] = H.sample_docs(cm[c], PROBE_SEED, c, [used[c]], L)[0]
            used[c] += 1
        x, y = M.inputs_targets(batch, V)
        loss = F.cross_entropy(G.lm_forward(m, x.to(device), fast=True).reshape(-1, V + 1), y.to(device).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), cfg['model']['grad_clip'])
        opt.step()
        sched.step()
        if step % 256 == 0:
            log(f'probe step {step} loss {loss.item() / math.log(2):.3f} bits')
    return m


def embed(m, docs, V, device):
    params = list(m.blocks[-1].parameters()) + list(m.ln_f.parameters())
    acc = None
    for doc in docs:
        x, y = M.inputs_targets(doc[None], V)
        nll = F.cross_entropy(G.lm_forward(m, x.to(device)).reshape(-1, V + 1), y.to(device).reshape(-1), reduction='sum')
        g = torch.autograd.grad(nll, params)
        v = torch.cat([t.reshape(-1) for t in g]) ** 2
        acc = v if acc is None else acc + v
    e = (acc / len(docs)).double()
    return (e / e.norm()).cpu().numpy()


def coefficient(ea, eb):
    d = [1 - float(a @ b) for a in ea for b in eb]  # embeddings are unit vectors and non-negative
    return 1 - float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    cfg = C.load_config()
    res = Path(a.results)
    tasks = json.loads((res / 'tasks.json').read_text())
    device = torch.device(a.device)
    t0 = time.time()
    log = lambda s: print(f'[{time.time() - t0:6.0f}s] {s}', flush=True)  # noqa: E731
    lr = cfg['model']['lr_table']['S'] * (cfg['lr_selected'] or {'S': 2})['S']
    probe = train_probe(cfg, tasks, lr, device, log)
    probe.eval()
    V, L = cfg['concepts']['V_model_tokens'], cfg['concepts']['doc_len']
    cm = {t['cid']: H.from_json(t['definition']) for t in tasks}
    out, by = {}, defaultdict(list)
    for t in R.targets(tasks):
        td = np.load(res / 'data' / f"{t['task_id']}_test.npz")['docs'].astype(np.int64)[:BATCH * NB]
        et = [embed(probe, td[i * BATCH:(i + 1) * BATCH], V, device) for i in range(NB)]
        for sim, alpha in R.cells(cfg):
            pool = C.pool_of(tasks, t['task_id'], sim) if alpha < 1 else []
            cids, js = H.schedule(MIX_SEED, t['cid'], [p['cid'] for p in pool], alpha, BATCH * NB)
            docs = np.empty((BATCH * NB, L), np.int64)
            for c in np.unique(cids):
                msk = cids == c
                docs[msk] = H.sample_docs(cm[int(c)], MIX_SEED, int(c), js[msk], L)
            em = [embed(probe, docs[i * BATCH:(i + 1) * BATCH], V, device) for i in range(NB)]
            k = f"{t['task_id']}|{sim}|{alpha:.2f}"
            out[k] = coefficient(em, et)
            by[f'{sim}|{alpha:.2f}'].append(out[k])
        log(f"{t['task_id']} done")
    rec = dict(coef=out, by_similarity={k: float(np.mean(v)) for k, v in by.items()}, probe_seed=PROBE_SEED,
               probe_tokens=PROBE_T, mix_seed=MIX_SEED, batch=BATCH, batches=NB, seconds=round(time.time() - t0, 1))
    (res / 'task2vec.json').write_text(json.dumps(rec, indent=1))
    log('wrote task2vec.json')


if __name__ == '__main__':
    main()
