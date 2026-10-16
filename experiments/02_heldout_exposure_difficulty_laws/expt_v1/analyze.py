"""Freeze forecasts before opening test outcomes; report every frozen comparison."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

ROOT=Path(__file__).resolve().parent
MODELS=['resources','additive','exposure','difficulty','full','structural','full_structural']
KEYS=['seed','width','mixture','steps','carry_chain']


def read_csv(path):
    with Path(path).open() as f:
        return [{k:float(v) if k not in ['size','cond','exp','held_out'] else v for k,v in row.items()} for row in csv.DictReader(f)]


def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(r):
    return tuple(r[k] for k in KEYS)


def matrix(rows,name,beta,gamma):
    n=np.array([r['parameter_count']/10000 for r in rows])
    t=np.array([r['steps']/1000 for r in rows])
    q=np.array([r['stratum_fraction'] for r in rows])
    columns=[np.ones(len(rows)),n**-beta,t**-gamma]
    lower=[-np.inf,0,0];upper=[np.inf]*3;names=['a','b','c']
    if name!='resources':
        columns.append(q);lower.append(-np.inf);upper.append(0);names.append('q')
    if name in ['exposure','full','full_structural']:
        columns.append(np.log(q));lower.append(-np.inf);upper.append(0);names.append('log_q')
    if name in ['difficulty','full','full_structural']:
        columns.append(np.array([r['starting_bits'] for r in rows]));lower.append(0);upper.append(np.inf);names.append('starting_bits')
    if name in ['structural','full_structural']:
        for s in [1,2,3]:
            columns.append(np.array([r['carry_chain']==s for r in rows],float));lower.append(-np.inf);upper.append(np.inf);names.append(f'carry_{s}')
    return np.column_stack(columns),lower,upper,names


def fit(rows,name,c):
    y=np.array([r['mean_bits'] for r in rows])
    best=None
    for beta,gamma in itertools.product(c['exponents'],repeat=2):
        x,lo,hi,names=matrix(rows,name,beta,gamma)
        fitted=lsq_linear(x,y,bounds=(lo,hi),tol=1e-10,max_iter=1000)
        if not fitted.success:
            raise RuntimeError(f'Failed bounded least squares: {name}, {beta}, {gamma}')
        sse=float(np.sum((x@fitted.x-y)**2))
        if best is None or sse<best['training_sse']:
            best=dict(beta=beta,gamma=gamma,columns=names,coefficients=fitted.x.tolist(),training_sse=sse,n_training=len(rows))
    return best


def axis(r,c):
    model=r['width'] in c['heldout_widths']
    data=r['mixture'] in c['heldout_mixtures']
    return 'joint' if model and data else 'model_only' if model else 'mixture_only' if data else 'fit'


def freeze_forecasts(out,c):
    cal=read_csv(out/'calibration_metrics.csv')
    assert len(cal)==2400 and len(set(map(key,cal)))==2400
    # Outcome fields are removed from the forecast input schema.
    features=[{k:v for k,v in r.items() if k not in ['mean_bits','correct','n']} for r in cal]
    assert all(not {'mean_bits','correct','n'}&r.keys() for r in features)
    models={};forecasts=[]
    for seed in c['seeds']:
        fit_rows=[r for r in cal if r['seed']==seed and axis(r,c)=='fit']
        targets=[r for r in features if r['seed']==seed and axis(r,c)!='fit']
        assert len(fit_rows)==108 and len(targets)==192
        assert set(map(key,fit_rows)).isdisjoint(map(key,targets))
        models[str(seed)]={}
        for name in MODELS:
            fitted=fit(fit_rows,name,c)
            models[str(seed)][name]=fitted
            x,*_=matrix(targets,name,fitted['beta'],fitted['gamma'])
            for r,pred in zip(targets,x@np.array(fitted['coefficients'])):
                forecasts.append({**{k:r[k] for k in KEYS},'axis':axis(r,c),'law':name,'predicted_bits':float(pred)})
        # Nested forms must attain at least the baseline's training fit.
        for name in MODELS[2:]:
            assert models[str(seed)][name]['training_sse']<=models[str(seed)]['additive']['training_sse']+1e-6
    record={'config_sha256':sha(ROOT/'config.json'),'calibration_sha256':sha(out/'calibration_metrics.csv'),'analysis_sha256':sha(__file__),'models':models}
    if (out/'frozen_forecasts.json').exists():
        old=json.loads((out/'frozen_forecasts.json').read_text())
        assert old==record,'Frozen forecast mismatch: never overwrite a published forecast.'
    else:
        save(out/'frozen_forecasts.json',record)
        write_csv(out/'forecasts.csv',forecasts)
    # Reading final-test data is permitted only after these artifacts exist.
    return forecasts


def summarize(out,c,forecasts):
    test=read_csv(out/'test_metrics.csv')
    target={key(r):r for r in test}
    assert len(target)==2400
    rows=[{**r,'observed_bits':target[key(r)]['mean_bits'],'error':r['predicted_bits']-target[key(r)]['mean_bits']} for r in forecasts]
    write_csv(out/'scored_forecasts.csv',rows)
    rng=np.random.default_rng(c['analysis_seed'])
    draws=rng.integers(0,len(c['seeds']),(c['bootstrap_draws'],len(c['seeds'])))
    metrics=[];comparisons=[];seed_metrics=[]
    for split in ['joint','model_only','mixture_only','joint_width_96','joint_width_256']:
        relevant=[r for r in rows if r['axis']==split] if split not in ['joint_width_96','joint_width_256'] else [r for r in rows if r['axis']=='joint' and r['width']==int(split.rsplit('_',1)[1])]
        mses={}
        for name in MODELS:
            rr=[r for r in relevant if r['law']==name]
            errors=np.array([r['error'] for r in rr])
            mses[name]=np.array([np.mean([r['error']**2 for r in rr if r['seed']==seed]) for seed in c['seeds']])
            boot=np.sqrt(mses[name][draws].mean(1))
            metrics.append(dict(axis=split,law=name,n_targets=len(rr),n_seeds=len(c['seeds']),rmse=float(np.sqrt(np.mean(errors**2))),rmse_lo=float(np.quantile(boot,.025)),rmse_hi=float(np.quantile(boot,.975)),mae=float(np.mean(abs(errors))),bias=float(np.mean(errors)),prediction_mean=float(np.mean([r['predicted_bits'] for r in rr])),outcome_mean=float(np.mean([r['observed_bits'] for r in rr])),negative_predictions=sum(r['predicted_bits']<0 for r in rr)))
            for seed,mse in zip(c['seeds'],mses[name]):
                seed_metrics.append(dict(axis=split,law=name,seed=seed,mse=float(mse),rmse=float(np.sqrt(mse))))
        for base,new in [('additive','full'),('resources','additive'),('additive','exposure'),('additive','difficulty'),('exposure','full'),('difficulty','full'),('structural','full_structural')]:
            gains=1-np.sqrt(mses[new][draws].mean(1)/mses[base][draws].mean(1))
            comparisons.append(dict(axis=split,baseline=base,augmented=new,relative_rmse_gain=float(1-np.sqrt(mses[new].mean()/mses[base].mean())),ci95_lo=float(np.quantile(gains,.025)),ci95_hi=float(np.quantile(gains,.975)),p_value='n/a',resampling_unit='paired_seed_block',n_seeds=len(c['seeds'])))
    main=next(r for r in comparisons if r['axis']=='joint' and r['baseline']=='additive' and r['augmented']=='full')
    widths=[r for r in comparisons if r['axis'].startswith('joint_width') and r['baseline']=='additive' and r['augmented']=='full']
    passed=main['relative_rmse_gain']>=c['min_relative_rmse_gain'] and main['ci95_lo']>0 and all(r['relative_rmse_gain']>0 for r in widths)
    write_csv(out/'forecast_metrics.csv',metrics)
    write_csv(out/'forecast_comparisons.csv',comparisons)
    write_csv(out/'seed_forecast_metrics.csv',seed_metrics)
    save(out/'summary.json',dict(verdict='PASS' if passed else 'NOT_ESTABLISHED',criterion=dict(min_relative_rmse_gain=c['min_relative_rmse_gain'],positive_ci_lower=True,both_width_gains_positive=True),primary=main,widths=widths,forecast_sha256=sha(out/'forecasts.csv'),n_trajectories=200,n_seed_blocks=8,n_primary_targets=384,p_value='n/a'))


def math_fit(train,target,augmented,c):
    n=np.array([r['N']/1e9 for r in train]);alpha=np.array([r['alpha'] for r in train]);y=np.array([r['loss'] for r in train])
    logcount=np.log([r['n_examples'] for r in train]);center=float(logcount.mean());scale=float(logcount.std())
    best=None
    for beta in c['exponents']:
        x=np.column_stack([np.ones(len(train)),n**-beta,alpha])
        lo=[-np.inf,0,-np.inf];hi=[np.inf,np.inf,0]
        xt=np.column_stack([np.ones(len(target)),np.array([r['N']/1e9 for r in target])**-beta,np.array([r['alpha'] for r in target])])
        if augmented:
            assert scale>0
            x=np.column_stack([x,(logcount-center)/scale]);xt=np.column_stack([xt,(np.log([r['n_examples'] for r in target])-center)/scale]);lo.append(-np.inf);hi.append(np.inf)
        fitted=lsq_linear(x,y,bounds=(lo,hi),tol=1e-10,max_iter=1000)
        assert fitted.success
        sse=float(np.sum((x@fitted.x-y)**2))
        if best is None or sse<best['sse']:
            best=dict(sse=sse,beta=beta,coefficients=fitted.x.tolist(),center=center,scale=scale,predictions=(xt@fitted.x).tolist())
    return best


def actual_data(out,c):
    rows=[r for r in read_csv(ROOT/'source_data/math_grid.csv') if r['T']==21e6]
    metadata=json.loads((ROOT/'source_data/E5_decontam_verification.json').read_text())['subsets']
    for r in rows:
        r['n_examples']=metadata[r['cond']]['n_examples']
    assert len(rows)==24 and len({(r['size'],r['cond']) for r in rows})==24
    results=[];fits=[]
    for split in ['model_only','mixture_only','joint']:
        for target in rows:
            train=[r for r in rows if (split=='mixture_only' or r['size']!=target['size']) and (split=='model_only' or r['cond']!=target['cond'])]
            assert len(train)=={'model_only':18,'mixture_only':20,'joint':15}[split]
            for augmented in [False,True]:
                fitted=math_fit(train,[target],augmented,c)
                name='breadth' if augmented else 'additive'
                pred=fitted.pop('predictions')[0]
                results.append(dict(axis=split,size=target['size'],condition=target['cond'],law=name,prediction=pred,observed=target['loss'],error=pred-target['loss'],n_train=len(train)))
                fits.append(dict(axis=split,size=target['size'],condition=target['cond'],law=name,**fitted))
    rng=np.random.default_rng(c['analysis_seed'])
    sizes=sorted({r['size'] for r in rows});draws=rng.integers(0,len(sizes),(c['bootstrap_draws'],len(sizes)))
    summaries=[]
    for split in ['joint','model_only','mixture_only']:
        mse={}
        for name in ['additive','breadth']:
            rr=[r for r in results if r['axis']==split and r['law']==name]
            mse[name]=np.array([np.mean([r['error']**2 for r in rr if r['size']==size]) for size in sizes])
        gains=1-np.sqrt(mse['breadth'][draws].mean(1)/mse['additive'][draws].mean(1))
        summaries.append(dict(axis=split,baseline_rmse=float(np.sqrt(mse['additive'].mean())),breadth_rmse=float(np.sqrt(mse['breadth'].mean())),relative_rmse_gain=float(1-np.sqrt(mse['breadth'].mean()/mse['additive'].mean())),descriptive_ci95_lo=float(np.quantile(gains,.025)),descriptive_ci95_hi=float(np.quantile(gains,.975)),n_targets=24,n_size_blocks=4,p_value='n/a'))
    write_csv(out/'actual_math_predictions.csv',results)
    write_csv(out/'actual_math_comparisons.csv',summaries)
    save(out/'actual_math_fits.json',fits)
    save(out/'actual_math_metadata.json',dict(n_observations=24,alpha_log_breadth_correlation=float(np.corrcoef([r['alpha'] for r in rows],np.log([r['n_examples'] for r in rows]))[0,1]),sources={p.name:sha(p) for p in (ROOT/'source_data').iterdir()},limitation='Retrospective breadth proxy; four sizes; overlapping training folds; no independent starting difficulty or repeated exposure.'))


def self_test():
    # The larger design matrices contain the declared smaller ones; feature schema excludes targets.
    r=[dict(parameter_count=10000,steps=1000,stratum_fraction=.25,starting_bits=4,carry_chain=s) for s in range(4)]
    for name in MODELS:
        x,lo,hi,names=matrix(r,name,.5,.5)
        assert x.shape==(4,len(names)) and len(lo)==len(hi)==len(names)
        assert np.isfinite(x).all()
    c={'exponents':[.1,.5,1]}
    rr=[dict(parameter_count=n,steps=t,stratum_fraction=q,starting_bits=d,carry_chain=s,mean_bits=2+3*(n/10000)**-.5+4*(t/1000)**-.5-2*q-0.3*math.log(q)+.2*d) for n in [4000,10000,30000] for t in [300,1000,3000] for q in [.02,.25,.8] for s,d in enumerate([5,6,7,8])]
    f=fit(rr,'full',c)
    assert f['training_sse']<1e-12 and f['beta']==f['gamma']==.5
    print('Forecast design, bounds, and known-law recovery: PASS')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results');p.add_argument('--self-test',action='store_true');p.add_argument('--private-math-only',action='store_true');args=p.parse_args()
    if args.self_test:
        self_test();return
    c=json.loads((ROOT/'config.json').read_text());out=args.output
    if args.private_math_only:
        private=out/'private_math';private.mkdir(exist_ok=True,parents=True)
        actual_data(private,c);print('Private existing-data analysis complete.');return
    cells=json.loads((out/'cells.json').read_text())
    assert len(cells)==200 and all(r['status']=='complete' for r in cells),'Full frozen run must finish; no selective subset analysis.'
    forecasts=freeze_forecasts(out,c)
    summarize(out,c,forecasts)
    print('Frozen forecast scoring complete.',flush=True)

if __name__=='__main__':
    main()
