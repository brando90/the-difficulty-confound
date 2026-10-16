"""Answer-matched intervention. Frozen arithmetic is imported unchanged."""
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


def previous_path(c):
    p=ROOT/c['previous_split_relative']
    if not p.exists():p=ROOT.parents[1]/'03_answer_matched_alignment/expt_v1/results/heldout_items.npz'
    return p


def dataset(c):
    a, b = np.triu_indices(1000)
    k = carry_chain(a, b, 3)
    y = a + b
    counts = np.stack([np.bincount(y[(k == s) & (y < 1000)], minlength=1000) for s in (0, 2)], axis=1)
    answers = np.flatnonzero(counts.min(axis=1) >= c['minimum_pairs_per_answer_stratum'])
    previous=np.load(previous_path(c))
    excluded=np.concatenate([previous[n] for n in ('test','calibration','probe')])
    rng = np.random.default_rng(c['split_seed'])
    splits = {name: [] for name in ('test', 'calibration', 'probe')}
    pools = {}
    for answer in answers:
        for s in (0, 2):
            population=np.flatnonzero((y == answer) & (k == s))
            ids=rng.permutation(np.setdiff1d(population,excluded))
            off = 0
            for name in splits:
                n = c[name + '_per_answer_stratum']
                splits[name].extend(ids[off:off+n]); off += n
            chosen=ids[:off]
            pools[int(answer), s] = rng.permutation(np.setdiff1d(population,chosen))
    return a, b, k, answers, pools, {n: np.asarray(v) for n, v in splits.items()}


def training_ids(c, pools, seed, q):
    rng = np.random.default_rng(seed)
    chosen = []
    for (answer, s), ids in sorted(pools.items()):
        perm = rng.permutation(ids)
        n = round(c['train_per_answer'] * (q if s == 2 else 1-q))
        chosen.extend(perm[:n])
    chosen = np.asarray(chosen)
    return np.random.default_rng(seed+1000).permutation(chosen)


def check(c):
    a, b, k, answers, pools, splits = dataset(c)
    assert len(answers) == 117 and np.all(a <= b)
    sets = [set(v) for v in splits.values()] + [set(np.concatenate(list(pools.values())))]
    assert sum(map(len, sets)) == len(set.union(*sets))
    assert all(column_addition(int(x), int(y), 3) == x+y for x, y in zip(a, b))
    assert carry_chain(np.array([123,109,199,999]),np.array([234,1,1,1]),3).tolist() == [0,1,2,3]
    for name, ids in splits.items():
        assert len(ids) == 2*len(answers)*c[name+'_per_answer_stratum']
        for s in (0,2):
            vals, cnt = np.unique((a+b)[ids[k[ids]==s]], return_counts=True)
            assert np.array_equal(vals, answers) and np.all(cnt==c[name+'_per_answer_stratum'])
    for seed in c['seeds']:
        for q in c['train_mixtures']+c['heldout_mixtures']:
            ids = training_ids(c,pools,seed,q)
            assert len(set(ids)) == len(ids) == 4680
            assert not set(ids) & set.union(*sets[:3])
            vals, cnt = np.unique((a+b)[ids],return_counts=True)
            assert np.array_equal(vals,answers) and np.all(cnt==40)
            for s in (0,2):
                vals,cnt=np.unique((a+b)[ids[k[ids]==s]],return_counts=True)
                assert np.array_equal(vals,answers) and np.all(cnt==round(40*(q if s==2 else 1-q)))
    previous=np.load(previous_path(c))
    old_ids=set(np.concatenate([previous[n] for n in ('test','calibration','probe')]))
    assert all(not set(ids)&old_ids for ids in splits.values())
    assert set(c['train_widths']).isdisjoint(c['heldout_widths'])
    assert set(c['train_mixtures']).isdisjoint(c['heldout_mixtures'])
    return dict(oracle_pairs=500500, eligible_answers=len(answers), train_size=4680,
                test=len(splits['test']), calibration=len(splits['calibration']), probe=len(splits['probe']),
                split_overlap=0, previous_evaluation_overlap=0, exact_answer_matching=True)


@torch.no_grad()
def evaluate(net,x,y,ids):
    preds,bits=[],[]
    for off in range(0,len(ids),1024):
        ii=ids[off:off+1024];z=net(x[ii]).reshape(-1,4,10)
        preds.append((z.argmax(-1)*10**torch.arange(4)).sum(-1).numpy())
        bits.append(F.cross_entropy(z.reshape(-1,10),y[ii].reshape(-1),reduction='none').reshape(-1,4).sum(1).numpy()/math.log(2))
    return np.concatenate(preds),np.concatenate(bits)


def advance(net,x,y,ids,c,generator,steps,on_step=None):
    opt=torch.optim.AdamW(net.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    tx,ty=x[ids],y[ids]
    for step in range(1,steps+1):
        batch=torch.randint(len(tx),(c['batch_size'],),generator=generator)
        loss=F.cross_entropy(net(tx[batch]).reshape(-1,10),ty[batch].reshape(-1))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if on_step:on_step(step)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    c=json.loads((ROOT/'config.json').read_text());checks=check(c);print(json.dumps(checks),flush=True)
    if args.check_only:return
    out=ROOT/'results';out.mkdir(exist_ok=True)
    if (out/'runtime.json').exists():raise RuntimeError('Refusing to overwrite run')
    torch.set_num_threads(c['threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    runtime=dict(host=socket.gethostname(),pid=os.getpid(),device='cpu',threads=c['threads'],python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},config_sha256=sha(ROOT/'config.json'),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checks=checks,expected_cells=200,previous_split_sha256=sha(previous_path(c)))
    save_json(out/'runtime.json',runtime)
    a,b,k,answers,pools,splits=dataset(c)
    xd=np.concatenate([digits_of(a,3),digits_of(b,3)],1)
    x=F.one_hot(torch.from_numpy(xd).long(),10).float().reshape(-1,60)
    y=torch.from_numpy(digits_of(a+b,4)).long()
    np.savez_compressed(out/'heldout_items.npz',a=a,b=b,carry=k,answers=answers,**splits)
    write_csv(out/'answer_counts.csv',[dict(answer=int(ans),carry_chain=s,population=int(np.sum((a+b==ans)&(k==s))),test=8,calibration=4,probe=4,pool=len(pools[int(ans),s])) for ans in answers for s in (0,2)])
    widths=c['train_widths']+c['heldout_widths'];mixtures=c['train_mixtures']+c['heldout_mixtures']
    cells=[dict(seed=s,width=w,mixture=q,status='pending') for s in c['seeds'] for w in widths for q in mixtures]
    save_json(out/'cells.json',cells)
    metrics={name:[] for name in ('calibration','test')};probes=[];marginals=[]
    for seed in c['seeds']:
        trains={q:training_ids(c,pools,seed,q) for q in mixtures}
        np.savez_compressed(out/f'train_{seed}.npz',**{f'q{q:.2f}':ids for q,ids in trains.items()})
        for q,ids in trains.items():
            for column in range(6):
                hist=np.bincount(xd[ids,column],minlength=10)
                marginals.extend(dict(seed=seed,mixture=q,column=column,digit=d,count=int(hist[d]),n=len(ids)) for d in range(10))
        write_csv(out/'input_digit_marginals.csv',marginals)
        for width in widths:
            torch.manual_seed(seed);net=model(dict(hidden_width=width))
            advance(net,x,y,trains[.5],c,torch.Generator().manual_seed(seed+20000),c['warmup_steps'])
            initial={n:v.clone() for n,v in net.state_dict().items()};torch.save(initial,out/f'warm_{seed}_{width}.pt')
            count=sum(p.numel() for p in net.parameters());pred,bits=evaluate(net,x,y,splits['probe'])
            np.savez_compressed(out/f'probe_{seed}_{width}.npz',item_id=splits['probe'],prediction=pred,bits=bits)
            difficulty={s:float(bits[k[splits['probe']]==s].mean()) for s in (0,2)}
            for s in (0,2):probes.append(dict(seed=seed,width=width,carry_chain=s,starting_bits=difficulty[s],n=468))
            write_csv(out/'probe_features.csv',probes)
            for q in mixtures:
                cell=next(r for r in cells if (r['seed'],r['width'],r['mixture'])==(seed,width,q))
                cell.update(status='running',started_elapsed_seconds=time.monotonic()-started);save_json(out/'cells.json',cells)
                net.load_state_dict(initial)
                def on_step(step):
                    if step%100==0:
                        save_json(out/'heartbeat.json',dict(seed=seed,width=width,mixture=q,step=step,completed_cells=sum(r['status']=='complete' for r in cells),elapsed_seconds=time.monotonic()-started))
                        if time.monotonic()-started>c['max_wall_seconds']:raise TimeoutError('Frozen budget reached')
                    if step not in c['checkpoints']:return
                    for name in metrics:
                        ids=splits[name];prediction,loss=evaluate(net,x,y,ids)
                        np.savez_compressed(out/f'{name}_{seed}_{width}_{q:.2f}_{step}.npz',item_id=ids,prediction=prediction,bits=loss)
                        for s in (0,2):
                            mask=k[ids]==s
                            metrics[name].append(dict(seed=seed,width=width,mixture=q,steps=step,carry_chain=s,parameter_count=count,stratum_fraction=q if s==2 else 1-q,starting_bits=difficulty[s],mean_bits=float(loss[mask].mean()),correct=int(np.sum(prediction[mask]==(a+b)[ids[mask]])),n=int(mask.sum())))
                        write_csv(out/f'{name}_metrics.csv',metrics[name])
                advance(net,x,y,trains[q],c,torch.Generator().manual_seed(seed+10000),max(c['checkpoints']),on_step)
                torch.save(net.state_dict(),out/f'model_{seed}_{width}_{q:.2f}.pt')
                cell.update(status='complete',parameter_count=count,elapsed_seconds=time.monotonic()-started-cell['started_elapsed_seconds']);save_json(out/'cells.json',cells)
                print(json.dumps(cell),flush=True)
    runtime.update(completed_cells=200,elapsed_seconds=time.monotonic()-started,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save_json(out/'runtime.json',runtime)
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
