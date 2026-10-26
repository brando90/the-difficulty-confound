"""AT-1346 CPU re-analysis of Laolu's 360 saved Exp 13-GPU runs (gpu_all_slice.csv = token-weighted 'all' slice of gpu_metrics.csv)."""
import pandas as pd, numpy as np, json, sys
from scipy.optimize import minimize
from scipy.stats import norm
rng=np.random.default_rng(1346)
g=pd.read_csv('gpu_all_slice.csv')
g['lvl']=g.K.astype(int); g['tid']=g.target_id
Kbits={8:568,16:None}
km=pd.read_csv('../at1342/gpu_metrics.csv',usecols=['K','K_bits']).drop_duplicates().set_index('K').K_bits.to_dict()
g['Kb']=g.K.map(km)
EPS=[0.02,0.05,0.10]; SIZES=['S','M','L']
# ---------- T* per cell with censoring status
def tstar(T,y,eps):
    o=np.argsort(T);T=np.array(T,float)[o];y=np.array(y,float)[o]
    if y[0]<=eps: return T[0],'left'
    for i in range(1,len(T)):
        if y[i]<=eps:
            f=(np.log(y[i-1])-np.log(eps))/(np.log(y[i-1])-np.log(y[i]))
            return float(np.exp(np.log(T[i-1])+f*(np.log(T[i])-np.log(T[i-1])))),'ok'
    return T[-1],'right'
def cells(df):
    rows=[]
    for (s,t,K,a,sz),x in df.groupby(['seed','tid','K','alpha','size']):
        for e in EPS:
            ts,st=tstar(x['T'],x.learn.clip(lower=1e-6),e); rows.append((s,t,K,x.Kb.iloc[0],a,sz,e,ts,st))
    return pd.DataFrame(rows,columns=['seed','tid','n','Kb','alpha','size','eps','Tstar','st'])
C=cells(g)
cen=C.groupby(['eps','size']).st.value_counts().unstack(fill_value=0)
print('CENSORING (cells=120 per size per eps)\n',cen.to_string())
# ---------- Tobit
def tobit(y,st,X):
    X=np.asarray(X,float);y=np.asarray(y,float)
    ex=(st=='ok');rc=(st=='right');lc=(st=='left')
    b0=np.linalg.lstsq(X,y,rcond=None)[0]; s0=np.log(max(np.std(y-X@b0),.05))
    def nll(p):
        b=p[:-1];ls=p[-1];s=np.exp(ls);m=X@b
        ll=np.zeros(len(y))
        ll[ex]=norm.logpdf(y[ex],m[ex],s)
        ll[rc]=norm.logsf(y[rc],m[rc],s)
        ll[lc]=norm.logcdf(y[lc],m[lc],s)
        return -ll.sum()
    r=minimize(nll,np.r_[b0,s0],method='BFGS')
    return r.x[:-1],r
def design(q,unit,inter,sizefe):
    x=np.log(q.n if unit=='n' else q.Kb); x=x-np.log(32 if unit=='n' else 1) if False else x
    a=np.log(q.alpha)
    cols=[np.ones(len(q)),a,x]
    if inter: cols.append(a*x)
    if sizefe:
        for s in ['M','L']: cols.append((q['size']==s).astype(float))
    return np.column_stack(cols)
def elast(b,q,unit,inter):
    # centred elasticities at geometric-mean alpha (over 4 levels) and geometric-mean complexity
    am=np.mean(np.log([.05,.15,.4,1.])); xm=np.log(q.n if unit=='n' else q.Kb).mean()
    ea=-(b[1]+(b[3]*xm if inter else 0)); eK=b[2]+(b[3]*am if inter else 0)
    return ea,eK
def fitE(q,unit='n',inter=False,sizefe=False):
    X=design(q,unit,inter,sizefe); b,r=tobit(np.log(q.Tstar),q.st.values,X); ea,eK=elast(b,q,unit,inter)
    return dict(ea=ea,eK=eK,E=ea/eK if eK>0 else np.nan,b=b)
# target-level bootstrap: resample the 2 targets per complexity level with replacement (all sizes, seeds, alphas, T of a target travel together)
tl=g.drop_duplicates('tid')[['tid','K']]; byl={k:list(v.tid) for k,v in tl.groupby('K')}
def boot_sample():
    pick=[]
    for k,ts in byl.items(): pick+=list(rng.choice(ts,size=len(ts),replace=True))
    return pick
def take(df,pick):
    G={t:x for t,x in df.groupby('tid')}
    return pd.concat([G[t] for t in pick])
B=int(sys.argv[1]) if len(sys.argv)>1 else 300
def job(args):
    e,sz,unit,seed=args
    rng=np.random.default_rng(seed)
    q=C[(C.eps==e)&((C['size']==sz) if sz!='pooled' else True)]
    pt=fitE(q,unit,False,sz=='pooled'); pi=fitE(q,unit,True,sz=='pooled')
    bs=[];bsi=[]
    G={t:x for t,x in q.groupby('tid')}
    for _ in range(B):
        pick=[]
        for k,ts in byl.items(): pick+=list(rng.choice(ts,size=len(ts),replace=True))
        qq=pd.concat([G[t] for t in pick])
        f=fitE(qq,unit,False,sz=='pooled'); fi=fitE(qq,unit,True,sz=='pooled')
        bs.append((f['ea'],f['eK'],f['E'])); bsi.append((fi['ea'],fi['eK'],fi['E'],fi['b'][3]))
    bs=np.array(bs);bsi=np.array(bsi)
    def ci(v): v=v[np.isfinite(v)]; return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] if len(v)>10 else [None,None]
    return f'{e}|{sz}|{unit}',dict(n_cells=len(q),right=int((q.st=='right').sum()),left=int((q.st=='left').sum()),B=B,
      ea=pt['ea'],ea_ci=ci(bs[:,0]),eK=pt['eK'],eK_ci=ci(bs[:,1]),E=pt['E'],E_ci=ci(bs[:,2]),E_boot_median=float(np.nanmedian(bs[:,2])),frac_eK_le0=float(np.mean(bs[:,1]<=0)),
      inter_ea=pi['ea'],inter_eK=pi['eK'],inter_E=pi['E'],inter_E_ci=ci(bsi[:,2]),inter_coef=float(pi['b'][3]),inter_coef_ci=ci(bsi[:,3]))
if __name__=='__main__':
    from multiprocessing import Pool
    jobs=[(e,sz,u,1000*i+7) for i,(e,sz,u) in enumerate([(e,sz,u) for e in EPS for sz in SIZES+['pooled'] for u in (['n','Kb'] if sz=='pooled' else ['n'])])]
    with Pool(4) as p: res=dict(p.map(job,jobs))
    json.dump(res,open('partA_E.json','w'),indent=1)
    print('done')
