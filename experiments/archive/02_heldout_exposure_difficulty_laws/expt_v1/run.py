"""Prospective fixed-skill training; final-test outcomes are separate from fit inputs."""
import argparse
import json
import math
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from addition import carry_chain, column_addition, digits_of, model, save_json, sha, write_csv

ROOT = Path(__file__).resolve().parent


def dataset(c):
    a, b = np.triu_indices(1000)
    k = carry_chain(a, b, 3)
    rng = np.random.default_rng(c['split_seed'])
    splits = {name: [] for name in ['test', 'calibration', 'probe']}
    pools = []
    for s in range(4):
        ids = rng.permutation(np.flatnonzero(k == s))
        offset = 0
        for name, count in [('test', c['test_per_stratum']), ('calibration', c['validation_per_stratum']), ('probe', c['probe_per_stratum'])]:
            splits[name].extend(ids[offset:offset + count])
            offset += count
        pools.append(ids[offset:])
    return a, b, k, pools, {name: np.array(ids) for name, ids in splits.items()}


def check(c):
    a, b, k, pools, splits = dataset(c)
    sets = [set(v) for v in splits.values()] + [set(np.concatenate(pools))]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 500500
    assert len(splits['test']) == 4000 and len(splits['calibration']) == len(splits['probe']) == 1000
    assert np.all(a <= b)
    assert all(column_addition(int(x), int(y), 3) == x + y for x, y in zip(a, b))
    assert carry_chain(np.array([123, 109, 199, 999]), np.array([234, 1, 1, 1]), 3).tolist() == [0, 1, 2, 3]
    for q in c['train_mixtures'] + c['heldout_mixtures']:
        counts = train_counts(c, q)
        assert sum(counts) == c['train_size'] and min(counts) > 0
    assert set(c['train_widths']).isdisjoint(c['heldout_widths'])
    assert set(c['train_mixtures']).isdisjoint(c['heldout_mixtures'])
    return {'oracle_pairs': len(a), 'split_overlap': 0, 'test': 4000, 'calibration': 1000, 'probe': 1000}


def train_counts(c, q):
    counts = np.rint(np.array([(1-q)/3]*3 + [q])*c['train_size']).astype(int)
    assert sum(counts) == c['train_size']
    return counts


def training_ids(c, pools, seed, q):
    rng = np.random.default_rng(seed)
    counts = train_counts(c, q)
    ids = np.concatenate([rng.permutation(pool)[:n] for pool, n in zip(pools, counts)])
    return ids[np.random.default_rng(seed+1000).permutation(len(ids))]


@torch.no_grad()
def evaluate(net, x, y, ids):
    predictions, losses = [], []
    for offset in range(0, len(ids), 1024):
        ii = ids[offset:offset+1024]
        z = net(x[ii]).reshape(-1, 4, 10)
        predictions.append((z.argmax(-1) * 10**torch.arange(4)).sum(-1).numpy())
        losses.append(F.cross_entropy(z.reshape(-1, 10), y[ii].reshape(-1), reduction='none').reshape(-1, 4).sum(1).numpy()/math.log(2))
    return np.concatenate(predictions), np.concatenate(losses)


def advance(net, x, y, ids, c, generator, steps, on_step=None):
    optimizer = torch.optim.AdamW(net.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    tx, ty = x[ids], y[ids]
    for step in range(1, steps+1):
        batch = torch.randint(len(tx), (c['batch_size'],), generator=generator)
        loss = F.cross_entropy(net(tx[batch]).reshape(-1, 10), ty[batch].reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if on_step:
            on_step(step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'results')
    args = p.parse_args()
    c = json.loads((ROOT/'config.json').read_text())
    checks = check(c)
    print(json.dumps({'checks': checks}), flush=True)
    if args.check_only:
        return
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    if (out/'runtime.json').exists():
        raise RuntimeError('Refusing to overwrite a run')
    torch.set_num_threads(c['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    runtime = dict(hostname=socket.gethostname(), pid=os.getpid(), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, threads=c['threads'], device='cpu', git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}, config_sha256=sha(ROOT/'config.json'), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), checks=checks, expected_cells=200)
    save_json(out/'runtime.json',runtime)
    a,b,k,pools,splits=dataset(c)
    xd=np.concatenate([digits_of(a,3),digits_of(b,3)],axis=1)
    x=F.one_hot(torch.from_numpy(xd).long(),10).float().reshape(-1,60)
    y=torch.from_numpy(digits_of(a+b,4)).long()
    np.savez_compressed(out/'heldout_items.npz',**splits,a=a,b=b,carry=k)
    widths=c['train_widths']+c['heldout_widths']
    mixtures=c['train_mixtures']+c['heldout_mixtures']
    cells=[dict(seed=s,width=w,mixture=q,status='pending') for s in c['seeds'] for w in widths for q in mixtures]
    save_json(out/'cells.json',cells)
    metrics={'calibration':[], 'test':[]}
    probes=[]
    for seed in c['seeds']:
        trainsets={q:training_ids(c,pools,seed,q) for q in set(mixtures+[.25])}
        np.savez_compressed(out/f'train_{seed}.npz',**{f'q{q:.2f}':ids for q,ids in trainsets.items()})
        for width in widths:
            torch.manual_seed(seed)
            net=model(dict(hidden_width=width))
            advance(net,x,y,trainsets[.25],c,torch.Generator().manual_seed(seed+20000),c['warmup_steps'])
            initial={name:value.clone() for name,value in net.state_dict().items()}
            torch.save(initial,out/f'warm_{seed}_{width}.pt')
            parameter_count=sum(p.numel() for p in net.parameters())
            pred,bits=evaluate(net,x,y,splits['probe'])
            np.savez_compressed(out/f'probe_{seed}_{width}.npz',item_id=splits['probe'],prediction=pred,bits=bits)
            difficulty={s:float(bits[k[splits['probe']]==s].mean()) for s in range(4)}
            for s in range(4):
                probes.append(dict(seed=seed,width=width,carry_chain=s,starting_bits=difficulty[s],n=c['probe_per_stratum']))
            write_csv(out/'probe_features.csv',probes)
            for q in mixtures:
                cell=next(r for r in cells if r['seed']==seed and r['width']==width and r['mixture']==q)
                cell.update(status='running',started_elapsed_seconds=time.monotonic()-started)
                save_json(out/'cells.json',cells)
                net.load_state_dict(initial)
                def on_step(step):
                    if step%100==0:
                        save_json(out/'heartbeat.json',dict(seed=seed,width=width,mixture=q,step=step,elapsed_seconds=time.monotonic()-started))
                        if time.monotonic()-started>c['max_wall_seconds']:
                            raise TimeoutError('Frozen whole-run budget reached')
                    if step not in c['checkpoints']:
                        return
                    for split in metrics:
                        ids=splits[split]
                        prediction,loss_bits=evaluate(net,x,y,ids)
                        np.savez_compressed(out/f'{split}_{seed}_{width}_{q:.2f}_{step}.npz',item_id=ids,prediction=prediction,bits=loss_bits)
                        for s in range(4):
                            mask=k[ids]==s
                            correct=int(np.sum(prediction[mask]==(a+b)[ids[mask]]))
                            metrics[split].append(dict(seed=seed,width=width,mixture=q,steps=step,carry_chain=s,parameter_count=parameter_count,stratum_fraction=float(train_counts(c,q)[s]/c['train_size']),starting_bits=difficulty[s],mean_bits=float(loss_bits[mask].mean()),correct=correct,n=int(mask.sum())))
                        write_csv(out/f'{split}_metrics.csv',metrics[split])
                advance(net,x,y,trainsets[q],c,torch.Generator().manual_seed(seed+10000),max(c['checkpoints']),on_step)
                torch.save(net.state_dict(),out/f'model_{seed}_{width}_{q:.2f}.pt')
                cell.update(status='complete',train_counts=train_counts(c,q).tolist(),parameter_count=parameter_count,elapsed_seconds=time.monotonic()-started-cell['started_elapsed_seconds'])
                save_json(out/'cells.json',cells)
                print(json.dumps({key:cell[key] for key in ['seed','width','mixture','status','elapsed_seconds']}),flush=True)
    runtime.update(completed_cells=sum(r['status']=='complete' for r in cells),elapsed_seconds=time.monotonic()-started,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save_json(out/'runtime.json',runtime)
    print(json.dumps({'status':'complete','elapsed_seconds':runtime['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    main()
