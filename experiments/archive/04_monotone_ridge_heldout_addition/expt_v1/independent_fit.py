"""Independent small active-set enumeration for convex ridge verification."""
import itertools
import numpy as np

def reconstruct(rows,law,c):
    cols=['intercept','log_parameters','log_updates','two_carry','log_exposure']
    if law.endswith('difficulty'):cols.append('log_starting_bits')
    mapping={'log_parameters':'parameter_count','log_updates':'steps','log_exposure':'stratum_fraction','log_starting_bits':'starting_bits'}
    x=np.array([[1 if n=='intercept' else float(r['carry_chain'])==2 if n=='two_carry' else np.log(float(r[mapping[n]])) for n in cols] for r in rows],float)
    y=np.log([float(r['mean_bits']) for r in rows])
    if law.startswith('legacy'):return np.linalg.lstsq(x,y,rcond=None)[0],cols
    mu=np.mean(x,0);sd=np.std(x,0);mu[0]=0;sd[0]=1;z=(x-mu)/sd
    penalty=np.diag([0]+[np.sqrt(c['ridge_lambda'])]*(len(cols)-1))
    a=np.concatenate([z/np.sqrt(len(y)),penalty]);b=np.concatenate([y/np.sqrt(len(y)),np.zeros(len(cols))])
    choices=[[None] for _ in cols]
    if law.startswith('stable'):
        for n in ['log_parameters','log_updates','log_exposure']:choices[cols.index(n)]=[None,0.]
        if 'log_starting_bits' in cols:choices[-1]=[None,0.,sd[-1]*c['difficulty_elasticity_max']]
    feasible=[]
    for state in itertools.product(*choices):
        fixed=[i for i,v in enumerate(state) if v is not None];free=[i for i,v in enumerate(state) if v is None];coef=np.zeros(len(cols))
        for i in fixed:coef[i]=state[i]
        target=b-a[:,fixed]@coef[fixed];coef[free]=np.linalg.lstsq(a[:,free],target,rcond=None)[0]
        if law.startswith('stable'):
            if any(coef[cols.index(n)]>1e-10 for n in ['log_parameters','log_updates','log_exposure']):continue
            if 'log_starting_bits' in cols and not -1e-10<=coef[-1]<=sd[-1]*c['difficulty_elasticity_max']+1e-10:continue
        feasible.append((float(np.sum((a@coef-b)**2)),coef))
    assert feasible
    coef=min(feasible,key=lambda p:p[0])[1];raw=coef/sd;raw[0]-=np.sum(coef*mu/sd)
    return raw,cols
