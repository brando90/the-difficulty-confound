"""Experiment 22 runs (PROTOCOL Sections 2-3). One run = one (phase, split k, p, seed) cell, or the feature extractor.

  python run.py --list vis_real > runs_vis_real.txt          # the frozen run list of a phase (JSON lines)
  python run.py --runlist runs_vis_real.txt --shard 3 --nshards 50   # what one Slurm array task does
  python run.py --pilot                                       # tiny CPU pilot of every code path -> results_pilot/

Resume-safe: a run whose result JSON exists is skipped; results are written atomically; a failed run leaves
<run_id>.error with the traceback and exits non-zero."""
import argparse, copy, json, math, os, sys, time, traceback
import numpy as np
import torch
import torch.nn.functional as F

import data as D
import splits as S
from models import ViT, TextTransformer, n_params

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'config.json')))
ORDER = ['feat', 'vis_real', 'vis_shuf']
RESULTS = os.path.join(HERE, 'results')


def run_id(s):
    if s['phase'] == 'feat':
        return 'feat_cifar10_vit'
    return f"{s['phase']}_k{s['k']}_p{s['p']:g}_s{s['seed']}"


def specs(phase):
    if phase == 'feat':
        return [{'phase': 'feat', 'arm': 'feat', 'seed': CFG['feat']['seed']}]
    g = CFG['grid'][phase]
    out = []
    for seed in g['seeds']:          # seed-major so a partial grid still covers every split and p
        for k in g['splits']:
            for p in g['p']:
                out.append({'phase': phase, 'arm': g['arm'], 'map': g['map'], 'k': k, 'p': p, 'seed': seed})
    return out


def out_path(s, root=None):
    return os.path.join(root or RESULTS, s['phase'], run_id(s) + '.json')


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def amp_dtype(dev):
    """bf16 autocast where the GPU supports it (FarmShare L40S), else fp16 with loss scaling; CPU runs in fp32."""
    if dev.type != 'cuda':
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(dev):
    return torch.autocast('cuda', dtype=amp_dtype(dev) or torch.float16, enabled=dev.type == 'cuda')


def setup(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def lr_at(step, total, warm, base):
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def make_opt(model, lr, wd):
    decay, no = [], []
    for n, p in model.named_parameters():
        (decay if p.ndim >= 2 and not n.endswith(('pos', 'cls')) and 'emb' not in n else no).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': no, 'weight_decay': 0.0}],
                             lr=lr, betas=(0.9, 0.999))


# ---------------------------------------------------------------- vision helpers
def to_dev_images(x_uint8, dev):
    return torch.from_numpy(x_uint8).to(dev).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW


def norm_images(xb):
    m = torch.tensor(CFG['cifar_mean'], device=xb.device).view(1, 3, 1, 1)
    s = torch.tensor(CFG['cifar_std'], device=xb.device).view(1, 3, 1, 1)
    return (xb.float() / 255.0 - m) / s


def augment(xb, gen, pad):
    """Random crop with zero padding `pad` and horizontal flip, from a CPU generator (reproducible on any device)."""
    b = xb.shape[0]
    oy = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(xb.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), generator=gen).to(xb.device)
    flip = (torch.rand(b, generator=gen) < 0.5).to(xb.device)
    xp = F.pad(xb, (pad, pad, pad, pad))
    ar = torch.arange(32, device=xb.device)
    rows = (oy[:, None] + ar[None, :])
    cols = (ox[:, None] + ar[None, :])
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    bi = torch.arange(b, device=xb.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[bi, rows[:, :, None], cols[:, None, :]]  # B,32,32,C
    return out.permute(0, 3, 1, 2)


@torch.no_grad()
def predict(model, X, dev, kind, bs=1024):
    model.eval()
    out = []
    for i in range(0, X.shape[0], bs):
        xb = X[i:i + bs]
        with autocast(dev):
            lo = model(norm_images(xb) if kind == 'vision' else xb)
        out.append(lo.float().argmax(1).cpu())
    model.train()
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.int64)


def train_loop(model, Xtr, ytr, Xval, yval, hp, seed, dev, kind, extra_eval=None):
    """Cosine schedule with warmup, AdamW, bf16 autocast on GPU (fp16 + loss scaling where bf16 is missing). Keeps
    the weights of the epoch with the best in-distribution validation accuracy (ties -> earlier epoch); held-out
    accuracy is only logged (extra_eval)."""
    model.to(dev).train()
    opt = make_opt(model, hp['lr'], hp['wd'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype(dev) == torch.float16)
    n = Xtr.shape[0]
    spe = max(1, math.ceil(n / hp['batch']))
    total, warm = hp['epochs'] * spe, hp['warmup_epochs'] * spe
    gen = torch.Generator().manual_seed(seed)
    ytr_d = torch.from_numpy(ytr).to(dev)
    best = (-1.0, -1, None)
    curve, step = [], 0
    t0 = time.time()
    for ep in range(1, hp['epochs'] + 1):
        perm = torch.randperm(n, generator=gen).to(dev)
        tot_loss = 0.0
        for i in range(spe):
            idx = perm[i * hp['batch']:(i + 1) * hp['batch']]
            xb = Xtr[idx]
            if kind == 'vision':
                xb = norm_images(augment(xb, gen, hp['crop_pad']))
            for g in opt.param_groups:
                g['lr'] = lr_at(step, total, warm, hp['lr'])
            with autocast(dev):
                loss = F.cross_entropy(model(xb).float(), ytr_d[idx], label_smoothing=hp['label_smoothing'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot_loss += loss.item()
            step += 1
        if ep % hp['eval_every'] == 0 or ep == hp['epochs']:
            row = {'epoch': ep, 'train_loss': tot_loss / spe, 'sec': round(time.time() - t0, 1)}
            if Xval is not None:
                row['val_acc'] = float((predict(model, Xval, dev, kind) == yval).mean())
                if row['val_acc'] > best[0]:
                    best = (row['val_acc'], ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if extra_eval is not None:
                row.update(extra_eval(model))
            curve.append(row)
            print(json.dumps(row), flush=True)
    if best[2] is not None:
        model.load_state_dict(best[2])
    return {'best_val_acc': best[0], 'best_epoch': best[1], 'curve': curve, 'train_sec': round(time.time() - t0, 1)}


def hparams(arm, pilot):
    hp = copy.deepcopy(CFG[arm])
    if pilot:
        hp.update({k: v for k, v in CFG['pilot'][arm].items() if not k.startswith('n_')})
    return hp


def pilot_subset(idx, n, seed):
    if n is None or len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


# ---------------------------------------------------------------- the three kinds of run
def run_vision(s, pilot, dev):
    hp = hparams('vision', pilot)
    d = D.load_cifar100()
    sp = S.make_split(d['fine_train'], 'vision', s['map'], s['k'], s['p'], s['seed'], lambda n: hp['val_per_class'])
    tr, va, ytr, yva = sp['train'], sp['val'], sp['y_train'], sp['y_val']
    if pilot:
        pc = CFG['pilot']['vision']
        keep = pilot_subset(np.arange(len(tr)), pc['n_train'], 1)
        tr, ytr = tr[keep], ytr[keep]
        keep = pilot_subset(np.arange(len(va)), pc['n_val'], 2)
        va, yva = va[keep], yva[keep]
    lab, ho = sp['label'], set(sp['heldout'])
    te = np.arange(len(d['fine_test']))
    if pilot:
        te = pilot_subset(te, CFG['pilot']['vision']['n_test'], 4)
    fine_te = d['fine_test'][te]
    seen_te = np.array([f not in ho for f in fine_te.tolist()])
    y_te = np.array([lab[f] for f in fine_te.tolist()])
    Xall_tr = to_dev_images(d['x_train'], dev)
    Xte = to_dev_images(d['x_test'][te], dev)
    ho_te = np.where(~seen_te)[0]
    Xho = Xte[torch.from_numpy(ho_te).to(dev)]

    def extra(model):
        return {'heldout_acc': float((predict(model, Xho, dev, 'vision') == y_te[ho_te]).mean())}

    setup(s['seed'])
    model = ViT(20, hp['patch'], hp['dim'], hp['depth'], hp['heads'], hp['mlp'], hp['drop'])
    tl = train_loop(model, Xall_tr[torch.from_numpy(tr).to(dev)], ytr, Xall_tr[torch.from_numpy(va).to(dev)], yva,
                    hp, s['seed'], dev, 'vision', extra)
    pred = predict(model, Xte, dev, 'vision')
    return finish(s, sp, tl, model, pred, y_te, fine_te, seen_te, 20, len(tr))


def build_vocab(token_lists, size, min_df):
    df = {}
    for toks in token_lists:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    words = sorted((w for w, c in df.items() if c >= min_df), key=lambda w: (-df[w], w))[:size]
    return {w: i + 3 for i, w in enumerate(words)}


def encode(token_lists, vocab, max_len):
    X = np.zeros((len(token_lists), max_len), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = [TextTransformer.CLS] + [vocab.get(t, TextTransformer.UNK) for t in toks[:max_len - 1]]
        X[i, :len(ids)] = ids
    return X


def run_text(s, pilot, dev):
    hp = hparams('text', pilot)
    ng = D.load_20ng()
    base_tr = np.array([D.SUBGROUPS.index(x['g']) for x in ng['train']])
    te_docs = ng['test']
    if pilot:
        te_docs = [te_docs[i] for i in pilot_subset(np.arange(len(te_docs)), CFG['pilot']['text']['n_test'], 4)]
    base_te = np.array([D.SUBGROUPS.index(x['g']) for x in te_docs])
    sp = S.make_split(base_tr, 'text', s['map'], s['k'], s['p'], s['seed'],
                      lambda n: int(round(hp['val_frac'] * n)))
    tr, va, ytr, yva = sp['train'], sp['val'], sp['y_train'], sp['y_val']
    if pilot:
        pc = CFG['pilot']['text']
        keep = pilot_subset(np.arange(len(tr)), pc['n_train'], 1)
        tr, ytr = tr[keep], ytr[keep]
        keep = pilot_subset(np.arange(len(va)), pc['n_val'], 2)
        va, yva = va[keep], yva[keep]
    toks_tr = [D.tokenize(x['t']) for x in ng['train']]
    toks_te = [D.tokenize(x['t']) for x in te_docs]
    vocab = build_vocab([toks_tr[i] for i in tr], hp['vocab'], hp['min_df'])  # from this run's training docs only
    Xtr_all = encode(toks_tr, vocab, hp['max_len'])
    Xte_np = encode(toks_te, vocab, hp['max_len'])
    lab, ho = sp['label'], set(sp['heldout'])
    seen_te = np.array([b not in ho for b in base_te.tolist()])
    y_te = np.array([lab[b] for b in base_te.tolist()])
    oov = {}
    for b in sorted(ho):
        toks = [t for i in np.where(base_te == b)[0] for t in toks_te[i]]
        oov[str(b)] = float(np.mean([t not in vocab for t in toks])) if toks else None
    Xall = torch.from_numpy(Xtr_all).to(dev)
    Xte = torch.from_numpy(Xte_np).to(dev)
    ho_te = np.where(~seen_te)[0]
    Xho = Xte[torch.from_numpy(ho_te).to(dev)]

    def extra(model):
        return {'heldout_acc': float((predict(model, Xho, dev, 'text') == y_te[ho_te]).mean())}

    setup(s['seed'])
    model = TextTransformer(len(vocab) + 3, 4, hp['max_len'], hp['dim'], hp['depth'], hp['heads'], hp['mlp'], hp['drop'])
    tl = train_loop(model, Xall[torch.from_numpy(tr).to(dev)], ytr, Xall[torch.from_numpy(va).to(dev)], yva,
                    hp, s['seed'], dev, 'text', extra)
    pred = predict(model, Xte, dev, 'text')
    r = finish(s, sp, tl, model, pred, y_te, base_te, seen_te, 4, len(tr))
    r['vocab_size'] = len(vocab)
    r['heldout_oov_rate'] = oov
    return r


def finish(s, sp, tl, model, pred, y_te, base_te, seen_te, K, n_train):
    correct = pred == y_te
    heldout = {}
    for b in sp['heldout']:
        m = base_te == b
        heldout[str(b)] = {'acc': float(correct[m].mean()), 'n': int(m.sum()), 'group': int(sp['label'][b])}
    return {'spec': s, 'run_id': run_id(s), 'K': K, 'chance': 1.0 / K,
            'indist_test_acc': float(correct[seen_te].mean()), 'n_indist_test': int(seen_te.sum()),
            'heldout_acc': float(correct[~seen_te].mean()), 'heldout': heldout,
            'groups': sp['groups'], 'N_budget': int(sp['N']), 'n_train': int(n_train), 'n_heldout_in_train': sp['n_h'],
            'n_val': int(len(sp['val'])), 'best_val_acc': tl['best_val_acc'], 'best_epoch': tl['best_epoch'],
            'curve': tl['curve'], 'train_sec': tl['train_sec'], 'n_params': n_params(model),
            'pred_sha256': D.sha_bytes(pred.astype(np.int64).tobytes())}


def run_feat(s, pilot, dev, root):
    """A ViT trained only on CIFAR-10 (no CIFAR-100 image or label), frozen; its [CLS] features of every CIFAR-100
    image are what Section 2.5's vision structure measures use."""
    hp = hparams('vision', False)
    hp.update({'epochs': CFG['feat']['epochs'], 'warmup_epochs': CFG['feat']['warmup_epochs'], 'eval_every': 10})
    if pilot:
        hp.update({k: v for k, v in CFG['pilot']['feat'].items() if not k.startswith('n_')})
        hp['eval_every'] = 1
    c10, c100 = D.load_cifar10(), D.load_cifar100()
    tr = np.arange(len(c10['y_train']))
    if pilot:
        tr = pilot_subset(tr, CFG['pilot']['feat']['n_train'], 3)
    setup(s['seed'])
    model = ViT(10, hp['patch'], hp['dim'], hp['depth'], hp['heads'], hp['mlp'], hp['drop'])
    X = to_dev_images(c10['x_train'], dev)[torch.from_numpy(tr).to(dev)]
    Xte = to_dev_images(c10['x_test'], dev)
    tl = train_loop(model, X, c10['y_train'][tr], None, None, hp, s['seed'], dev, 'vision')
    acc = float((predict(model, Xte, dev, 'vision') == c10['y_test']).mean())
    model.eval()
    feats = {}
    with torch.no_grad():
        for split in ('train', 'test'):
            Xs = to_dev_images(c100[f'x_{split}'], dev)
            fs = []
            for i in range(0, Xs.shape[0], 1024):
                with autocast(dev):
                    fs.append(model.features(norm_images(Xs[i:i + 1024])).float().cpu())
            feats[split] = torch.cat(fs).numpy().astype(np.float32)
    fp = os.path.join(root, 'feat', 'cifar100_features_from_cifar10_vit.npz')
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    np.savez(fp + '.tmp.npz', train=feats['train'], test=feats['test'])
    os.replace(fp + '.tmp.npz', fp)
    return {'spec': s, 'run_id': run_id(s), 'cifar10_test_acc': acc, 'curve': tl['curve'], 'train_sec': tl['train_sec'],
            'n_params': n_params(model), 'features_file': os.path.basename(fp),
            'features_sha256': {k: D.sha_array(v) for k, v in feats.items()}}


def run_one(s, pilot=False, root=None):
    root = root or (os.path.join(HERE, 'results_pilot') if pilot else RESULTS)
    op = out_path(s, root)
    if os.path.exists(op):
        print('skip (done):', run_id(s), flush=True)
        return
    os.makedirs(os.path.dirname(op), exist_ok=True)
    dev = device()
    t0 = time.time()
    try:
        if s['phase'] == 'feat':
            r = run_feat(s, pilot, dev, root)
        elif s['arm'] == 'vision':
            r = run_vision(s, pilot, dev)
        else:
            r = run_text(s, pilot, dev)
    except Exception:
        with open(op[:-5] + '.error', 'w') as f:
            f.write(traceback.format_exc())
        raise
    r.update({'pilot': pilot, 'wall_sec': round(time.time() - t0, 1), 'device': str(dev),
              'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else None, 'amp': str(amp_dtype(dev)),
              'torch': torch.__version__, 'numpy': np.__version__, 'host': os.uname().nodename,
              'slurm_job': os.environ.get('SLURM_JOB_ID')})
    with open(op + '.tmp', 'w') as f:
        json.dump(r, f, indent=1, sort_keys=True)
    os.replace(op + '.tmp', op)
    if os.path.exists(op[:-5] + '.error'):
        os.remove(op[:-5] + '.error')
    print('done:', run_id(s), f"{r['wall_sec']} s", flush=True)


PILOT_CELLS = [  # every code path once: feature extractor, both arms, both maps, p = 0 and p > 0
    {'phase': 'feat', 'arm': 'feat'},
    {'phase': 'vis_real', 'arm': 'vision', 'map': 'real', 'k': 0, 'p': 0.0},
    {'phase': 'vis_real', 'arm': 'vision', 'map': 'real', 'k': 0, 'p': 0.2},
    {'phase': 'vis_shuf', 'arm': 'vision', 'map': 'shuffled', 'k': 0, 'p': 0.0},
    {'phase': 'vis_real', 'arm': 'vision', 'map': 'real', 'k': 5, 'p': 0.0},
    {'phase': 'vis_shuf', 'arm': 'vision', 'map': 'shuffled', 'k': 9, 'p': 0.0},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', choices=ORDER)
    ap.add_argument('--runlist')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--root', help='results folder (default results/, or results_pilot/ with --pilot)')
    a = ap.parse_args()
    if a.list:
        for s in specs(a.list):
            print(json.dumps(s, sort_keys=True))
        return
    if a.pilot:
        for c in PILOT_CELLS:
            run_one(dict(c, seed=CFG['pilot']['seed']), pilot=True, root=a.root)
        return
    todo = [json.loads(l) for l in open(a.runlist) if l.strip()][a.shard::a.nshards]
    for s in todo:
        run_one(s, root=a.root)


if __name__ == '__main__':
    main()
