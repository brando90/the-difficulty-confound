"""AT-1346 part B: held-out error, separable vs alpha x K interaction law, and law-implied E per eps*; target-level bootstrap."""
import pandas as pd, numpy as np, json
rng=np.random.default_rng(13460)
g=pd.read_csv('gpu_all_slice.csv'); FLOOR=0.015; LN2=np.log(2)
EPS=[0.02,0.05,0.10]; SIZES=['S','M','L']
am=np.mean(np.log([.05,.15,.4,1.])); xm=np.mean(np.log([8,16,32,64,128]))
def prep(df,variant):
    d=df.copy()
    if variant=='drop': d=d[d.learn>FLOOR]
    else: d['learn']=d.learn.clip(lower=FLOOR)
    d['y']=np.log(d.learn); d['x']=np.log(d.K); d['a']=np.log(d.alpha); d['t']=np.log(d['T'])
    return d
def X(d,model,fe):
    c=[np.ones(len(d)),d.x,d.a,d.t]
    if model in('G1','FULL'): c.append(d.x*d.t)
    if model in('INT','FULL'): c.append(d.x*d.a)
    if fe:
        for s in ['M','L']: c.append((d['size']==s).astype(float))
    return np.column_stack(c)
def fit(d,model,fe): return np.linalg.lstsq(X(d,model,fe),d.y,rcond=None)[0]
def oof(d,model,fe,scheme):
    err=pd.Series(np.nan,index=d.index)
    key={'target':'tid' if 'tid' in d else 'target_id','level':'K','alpha':'alpha'}[scheme]
    for v in d[key].unique():
        te=d[key]==v; b=fit(d[~te],model,fe); err[te]=d.y[te]-X(d[te],model,fe)@b
    return err
def rel(ea,eb): return 1-np.sqrt(np.mean(ea**2))/np.sqrt(np.mean(eb**2))
res={}; tg=g.drop_duplicates('target_id')[['target_id','K']]; byl={k:list(v.target_id) for k,v in tg.groupby('K')}
def bootpick():
    p=[]
    for k,ts in byl.items(): p+=list(rng.choice(ts,size=len(ts),replace=True))
    return p
B=1000
for variant in ['drop','clip']:
    for scope in ['pooled']+SIZES:
        d0=prep(g if scope=='pooled' else g[g['size']==scope],variant); fe=scope=='pooled'
        for scheme in ['target','level','alpha']:
            E={m:oof(d0,m,fe,scheme) for m in ['SEP','G1','INT','FULL']}
            row={'n_rows':len(d0)}
            for m in E: row[f'rmse_log2_{m}']=float(np.sqrt(np.mean(E[m]**2))/LN2)
            if scheme=='level':
                for k in [32,8,128]:
                    msk=d0.K==k; row[f'rmse_log2_level{k}']={m:float(np.sqrt(np.mean(E[m][msk]**2))/LN2) for m in E}
            # target-level bootstrap on OOF errors
            G={t:ix for t,ix in d0.groupby('target_id').groups.items()}
            for pair in [('INT','SEP'),('FULL','SEP'),('FULL','G1')]:
                v=[]
                for _ in range(B):
                    idx=np.concatenate([G[t] for t in bootpick() if t in G]); v.append(rel(E[pair[0]][idx],E[pair[1]][idx]))
                row[f'reduction_{pair[0]}_vs_{pair[1]}']=dict(point=float(rel(E[pair[0]],E[pair[1]])),ci=[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))])
            res[f'{variant}|{scope}|{scheme}']=row
            print(variant,scope,scheme,json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items() if k.startswith('rmse_log2_') and not k.startswith('rmse_log2_level') or k=='n_rows'}),
                  {k:(round(v['point'],3),[round(c,3) for c in v['ci']]) for k,v in row.items() if k.startswith('reduction')},flush=True)
# --- refit coefficients + law-implied E at eps*, target-level bootstrap (variant drop)
def implied(b,eps,fe,model):
    # ln L = c + k x + bb a + dl x a - (g0 + g1 x) t   (+ size FE ignored: only slopes matter)
    c,k,bb,tt=b[0],b[1],b[2],b[3]; g1=-b[4] if model in('G1','FULL') else 0.0
    dl=b[-1 if not fe else -3] if model in('INT','FULL') else 0.0
    if model=='FULL': g1=-b[4]; dl=b[5]
    if model=='G1': g1=-b[4]; dl=0.0
    if model=='INT': dl=b[4]; g1=0.0
    g0=-tt
    def tstar(x,a,fe_shift=0.0): return (c+fe_shift+k*x+bb*a+dl*x*a-np.log(eps))/(g0+g1*x)
    h=1e-4; x,a=xm,am
    eK=(tstar(x+h,a)-tstar(x-h,a))/(2*h); ea=-(tstar(x,a+h)-tstar(x,a-h))/(2*h)
    return ea,eK,(ea/eK if eK>0 else np.nan)
coef={}
for scope in ['pooled']+SIZES:
    d0=prep(g if scope=='pooled' else g[g['size']==scope],'drop'); fe=scope=='pooled'
    G={t:x for t,x in d0.groupby('target_id')}
    for model in ['SEP','FULL']:
        b=fit(d0,model,fe)
        names=['c','kappa(x)','b(alpha)','-gamma0(t)']+(['x*t(-gamma1)','x*alpha(delta)'] if model=='FULL' else [])
        bs=[];Es={e:[] for e in EPS}
        for _ in range(B):
            dd=pd.concat([G[t] for t in bootpick()]); bb=fit(dd,model,fe); bs.append(bb[:len(names)])
            for e in EPS: Es[e].append(implied(bb,e,fe,model)[2])
        bs=np.array(bs)
        entry={'n_rows':len(d0),'coef':{n:dict(est=float(b[i]),ci=[float(np.percentile(bs[:,i],2.5)),float(np.percentile(bs[:,i],97.5))]) for i,n in enumerate(names)}}
        entry['E_law']={str(e):dict(est=float(implied(b,e,fe,model)[2]),ea=float(implied(b,e,fe,model)[0]),eK=float(implied(b,e,fe,model)[1]),ci=[float(np.nanpercentile(Es[e],2.5)),float(np.nanpercentile(Es[e],97.5))],frac_undefined=float(np.mean(~np.isfinite(Es[e])))) for e in EPS}
        if model=='SEP': entry['E_sep_closed_form_-b/kappa']=float(-b[2]/b[1])
        coef[f'{scope}|{model}']=entry
        print(scope,model,{n:round(entry['coef'][n]['est'],3) for n in names},{e:round(entry['E_law'][str(e)]['est'],3) for e in EPS},flush=True)
json.dump(dict(heldout=res,coef=coef),open('partB_law.json','w'),indent=1)
