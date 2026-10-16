"""Frozen log-loss laws. Fits see allowed calibration rows only."""
import numpy as np
from scipy.optimize import lsq_linear

LAWS=['legacy_base','legacy_difficulty','ridge_difficulty','stable_base','stable_difficulty']

def design(rows,law):
    names=['intercept','log_parameters','log_updates','two_carry','log_exposure']
    cols=[np.ones(len(rows)),np.log([r['parameter_count'] for r in rows]),np.log([r['steps'] for r in rows]),np.array([r['carry_chain']==2 for r in rows],float),np.log([r['stratum_fraction'] for r in rows])]
    if law.endswith('difficulty'):
        names.append('log_starting_bits');cols.append(np.log([r['starting_bits'] for r in rows]))
    return np.column_stack(cols),names

def fit_law(rows,law,c):
    x,names=design(rows,law);y=np.log([r['mean_bits'] for r in rows])
    if law.startswith('legacy'):
        beta,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
        return dict(columns=names,coefficients=beta.tolist(),rank=int(rank),regularized=False)
    center=x.mean(0);center[0]=0.;scale=x.std(0);scale[0]=1.;assert (scale>0).all()
    z=(x-center)/scale;penalty=np.eye(x.shape[1]);penalty[0,0]=0
    a=np.vstack([z/np.sqrt(len(x)),np.sqrt(c['ridge_lambda'])*penalty]);b=np.r_[y/np.sqrt(len(x)),np.zeros(x.shape[1])]
    lower=np.full(x.shape[1],-np.inf);upper=np.full(x.shape[1],np.inf)
    if law.startswith('stable'):
        for name in ['log_parameters','log_updates','log_exposure']:upper[names.index(name)]=0.
        if 'log_starting_bits' in names:
            j=names.index('log_starting_bits');lower[j]=0.;upper[j]=c['difficulty_elasticity_max']*scale[j]
        result=lsq_linear(a,b,bounds=(lower,upper),method='bvls',tol=1e-12,max_iter=1000)
        assert result.success,(result.status,result.message)
        coef=result.x
    else:coef=np.linalg.lstsq(a,b,rcond=None)[0]
    beta=coef/scale;beta[0]-=np.sum(coef*center/scale)
    assert np.isfinite(beta).all()
    return dict(columns=names,coefficients=beta.tolist(),rank=int(np.linalg.matrix_rank(x)),regularized=True,feature_center=center.tolist(),feature_scale=scale.tolist(),standardized_coefficients=coef.tolist(),ridge_lambda=c['ridge_lambda'],objective=float(np.mean((z@coef-y)**2)+c['ridge_lambda']*np.sum(coef[1:]**2)))
