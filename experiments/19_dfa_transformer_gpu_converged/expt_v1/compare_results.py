"""Compare two Exp 19 results dirs numerically: max abs/rel diff over every number in the JSONs and tstar.csv."""
import json,csv,sys,math
a,b=sys.argv[1],sys.argv[2]
def walk(x,y,p,acc):
    if isinstance(x,dict):
        if set(x)!=set(y): acc['struct'].append(p+' keys')
        for k in x:
            if k in y: walk(x[k],y[k],p+'/'+str(k),acc)
    elif isinstance(x,list):
        if len(x)!=len(y): acc['struct'].append(p+' len'); return
        for i,(u,v) in enumerate(zip(x,y)): walk(u,v,p+f'[{i}]',acc)
    elif isinstance(x,(int,float)) and not isinstance(x,bool):
        if x is None or y is None or isinstance(y,(str,bool)): acc['struct'].append(p); return
        if math.isnan(x) and math.isnan(y): return
        d=abs(x-y); r=d/max(abs(x),1e-12)
        if d>acc['abs'][0]: acc['abs']=(d,p)
        if r>acc['rel'][0] and abs(x)>1e-9: acc['rel']=(r,p)
    else:
        if x!=y: acc['struct'].append(p+f' {x!r}!={y!r}')
for f in ['analysis.json','summary.json','fits.json','forecasts_frozen.json','convergence_runs.json']:
    acc={'abs':(0,''),'rel':(0,''),'struct':[]}
    walk(json.load(open(f'{a}/{f}')),json.load(open(f'{b}/{f}')),'',acc)
    print(f, 'max_abs %.3g at %s' % acc['abs'], '| max_rel %.3g at %s' % acc['rel'], '| non-numeric diffs:', acc['struct'][:5])
A=list(csv.reader(open(f'{a}/tstar.csv'))); B=list(csv.reader(open(f'{b}/tstar.csv')))
m=0;s=0
for r1,r2 in zip(A[1:],B[1:]):
    for u,v in zip(r1,r2):
        try: m=max(m,abs(float(u)-float(v))/max(abs(float(u)),1e-12))
        except ValueError: s+= u!=v
print('tstar.csv rows',len(A),len(B),'max_rel %.3g'%m,'string diffs',s)
