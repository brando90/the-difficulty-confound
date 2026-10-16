"""Independent checks of recorded outcomes, split isolation, forecasts and weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from addition import digits_of, model
from run import dataset, evaluate, train_counts, training_ids
from analyze import MODELS, KEYS, axis, matrix, key

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def numeric(rows):
    return [{k:float(v) for k,v in r.items()} for r in rows]


def verify(out,weights=True):
    c=json.loads((ROOT/'config.json').read_text())
    runtime=json.loads((out/'runtime.json').read_text())
    frozen=json.loads((out/'frozen_forecasts.json').read_text())
    assert runtime['completed_cells']==runtime['expected_cells']==200
    assert runtime['config_sha256']==frozen['config_sha256']==sha(ROOT/'config.json')
    for name,digest in runtime['source_sha256'].items():
        assert sha(ROOT/name)==digest,(name,'source changed')
    assert frozen['analysis_sha256']==sha(ROOT/'analyze.py')
    assert frozen['calibration_sha256']==sha(out/'calibration_metrics.csv')
    a,b,k,pools,splits=dataset(c)
    held=np.load(out/'heldout_items.npz')
    for name in splits:
        np.testing.assert_array_equal(held[name],splits[name])
    np.testing.assert_array_equal(held['a'],a)
    np.testing.assert_array_equal(held['b'],b)
    np.testing.assert_array_equal(held['carry'],k)
    excluded=np.concatenate(list(splits.values()))
    assert len(np.unique(excluded))==6000
    trainsets=0
    for seed in c['seeds']:
        stored=np.load(out/f'train_{seed}.npz')
        for q in c['train_mixtures']+c['heldout_mixtures']:
            ids=stored[f'q{q:.2f}']
            np.testing.assert_array_equal(ids,training_ids(c,pools,seed,q))
            assert len(np.unique(ids))==c['train_size'] and not np.isin(ids,excluded).any()
            np.testing.assert_array_equal(np.bincount(k[ids],minlength=4),train_counts(c,q))
            trainsets+=1
    cells=json.loads((out/'cells.json').read_text())
    expected={(s,w,q) for s in c['seeds'] for w in c['train_widths']+c['heldout_widths'] for q in c['train_mixtures']+c['heldout_mixtures']}
    assert {(r['seed'],r['width'],r['mixture']) for r in cells}==expected and len(cells)==200
    assert all(r['status']=='complete' for r in cells)
    probes=numeric(read(out/'probe_features.csv'))
    assert len(probes)==160
    probe_lookup={}
    for r in probes:
        seed,width,s=map(int,[r['seed'],r['width'],r['carry_chain']])
        raw=np.load(out/f'probe_{seed}_{width}.npz')
        np.testing.assert_array_equal(raw['item_id'],splits['probe'])
        mask=k[raw['item_id']]==s
        assert mask.sum()==250
        np.testing.assert_allclose(raw['bits'][mask].mean(),r['starting_bits'],rtol=0,atol=1e-12)
        probe_lookup[(seed,width,s)]=r['starting_bits']
    tables={};prediction_count=0
    for split in ['calibration','test']:
        table=numeric(read(out/f'{split}_metrics.csv'))
        assert len(table)==len(set(map(key,table)))==2400
        tables[split]={key(r):r for r in table}
        for seed,width,q in sorted(expected):
            for steps in c['checkpoints']:
                raw=np.load(out/f'{split}_{seed}_{width}_{q:.2f}_{steps}.npz')
                ids=raw['item_id'];np.testing.assert_array_equal(ids,splits[split])
                assert np.isfinite(raw['bits']).all() and (raw['bits']>=0).all()
                assert raw['prediction'].shape==raw['bits'].shape==ids.shape
                prediction_count+=len(ids)
                for s in range(4):
                    r=tables[split][(seed,width,q,steps,s)]
                    mask=k[ids]==s
                    assert r['n']==mask.sum()
                    assert r['correct']==sum(raw['prediction'][mask]==(a+b)[ids[mask]])
                    np.testing.assert_allclose(raw['bits'][mask].mean(),r['mean_bits'],rtol=0,atol=1e-12)
                    assert r['parameter_count']==width*width+102*width+40
                    assert r['stratum_fraction']==train_counts(c,q)[s]/c['train_size']
                    assert r['starting_bits']==probe_lookup[(seed,width,s)]
    forecast=read(out/'forecasts.csv');scored=read(out/'scored_forecasts.csv')
    assert len(forecast)==len(scored)==8*192*7
    assert len({tuple(r[x] for x in KEYS+['law']) for r in forecast})==len(forecast)
    assert sum(r['axis']=='joint' for r in forecast)==384*7
    for pred,score in zip(forecast,scored):
        kk=tuple(float(pred[x]) for x in KEYS)
        r=tables['calibration'][kk]
        assert axis(r,c)==pred['axis']!='fit'
        fitted=frozen['models'][str(int(r['seed']))][pred['law']]
        x,*_=matrix([r],pred['law'],fitted['beta'],fitted['gamma'])
        value=float((x@np.array(fitted['coefficients']))[0])
        np.testing.assert_allclose(value,float(pred['predicted_bits']),rtol=1e-10,atol=1e-10)
        assert float(score['observed_bits'])==tables['test'][kk]['mean_bits']
        np.testing.assert_allclose(float(score['error']),value-tables['test'][kk]['mean_bits'],rtol=1e-10,atol=1e-10)
    # Recompute the headline directly, without the fitting/scoring implementation.
    selected=[r for r in scored if r['axis']=='joint']
    mse={law:np.array([np.mean([float(r['error'])**2 for r in selected if int(float(r['seed']))==seed and r['law']==law]) for seed in c['seeds']]) for law in ['additive','full']}
    gain=1-np.sqrt(mse['full'].mean()/mse['additive'].mean())
    rng=np.random.default_rng(c['analysis_seed']);draws=rng.integers(0,8,(c['bootstrap_draws'],8))
    bootstrap=1-np.sqrt(mse['full'][draws].mean(1)/mse['additive'][draws].mean(1))
    summary=json.loads((out/'summary.json').read_text())
    np.testing.assert_allclose(gain,summary['primary']['relative_rmse_gain'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.quantile(bootstrap,[.025,.975]),[summary['primary']['ci95_lo'],summary['primary']['ci95_hi']],rtol=0,atol=1e-14)
    assert sha(out/'forecasts.csv')==summary['forecast_sha256']
    checked_weights=0
    max_bits_delta=0.0
    if weights:
        torch.set_num_threads(2)
        xd=np.concatenate([digits_of(a,3),digits_of(b,3)],axis=1)
        x=F.one_hot(torch.from_numpy(xd).long(),10).float().reshape(-1,60)
        y=torch.from_numpy(digits_of(a+b,4)).long()
        targets=[(f'warm_{s}_{w}.pt',f'probe_{s}_{w}.npz',w,'probe') for s in c['seeds'] for w in c['train_widths']+c['heldout_widths']]
        targets += [(f'model_{s}_{w}_{q:.2f}.pt',f'test_{s}_{w}_{q:.2f}_3000.npz',w,'test') for s,w,q in sorted(expected)]
        for checkpoint,rawfile,width,split in targets:
            net=model(dict(hidden_width=width));net.load_state_dict(torch.load(out/checkpoint,map_location='cpu',weights_only=True))
            pred,bits=evaluate(net,x,y,splits[split]);raw=np.load(out/rawfile)
            np.testing.assert_array_equal(pred,raw['prediction'])
            np.testing.assert_allclose(bits,raw['bits'],rtol=1e-4,atol=5e-5)
            max_bits_delta=max(max_bits_delta,float(np.max(abs(bits-raw['bits']))))
            checked_weights+=1
    return dict(status='PASS',complete_trajectories=200,warm_starts=40,seed_blocks=8,unique_training_manifests=trainsets,split_overlap=0,per_item_prediction_rows_verified=prediction_count,aggregate_outcomes_verified=4800,law_forecasts_verified=len(forecast),joint_targets_per_law=384,weights_replayed=checked_weights,max_cross_platform_bits_delta=max_bits_delta,bootstrap_reproduced=True,private_files_excluded=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--skip-weights',action='store_true');args=p.parse_args()
    result=verify(args.output,weights=not args.skip_weights)
    print(json.dumps(result,indent=2))
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    main()
