"""Numerical checks of the exact oracle and classical switching-mixture identity."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp
from headroom import make_task, sample, finite_prior_reference

def fixed_share(probs, y, alpha):
    """Two experts; alpha is probability of switching to the OTHER expert."""
    logw = np.log([.5,.5]); predictions=[]; losses=[]
    lt = np.array([[np.log1p(-alpha),np.log(alpha)],
                   [np.log(alpha),np.log1p(-alpha)]]) if alpha else None
    for p, label in zip(probs,y):
        predictions.append(float(np.exp(logw) @ p))
        ll = np.log(p if label else 1-p)
        evidence = logsumexp(logw+ll)
        losses.append(-evidence)
        post = logw+ll-evidence
        logw = logsumexp(post[:,None]+lt,axis=0) if alpha else post
    return np.array(predictions),np.array(losses)

def check():
    rng=np.random.default_rng(4251); max_err=0.; min_slack=float('inf')
    for trial in range(24):
        T=8; alpha=1/T
        ps=rng.uniform(.01,.99,size=(T,2)); y=rng.integers(0,2,size=T)
        _,loss=fixed_share(ps,y,alpha); L=loss.sum()
        e=-np.log(np.where(y[:,None],ps,1-ps))
        paths=[]
        for path in itertools.product([0,1],repeat=T):
            s=np.sum(np.diff(path)!=0)
            penalty=np.log(2)+s*np.log(1/alpha)+(T-1-s)*np.log(1/(1-alpha))
            pathloss=sum(e[t,k] for t,k in enumerate(path))
            paths.append(-pathloss-penalty)
            min_slack=min(min_slack,pathloss+penalty-L)
        max_err=max(max_err,abs(L+logsumexp(paths)))
    assert max_err<1e-10 and min_slack>=-1e-10
    # Exact known-mechanism Bayes posterior from enumerated likelihoods.
    oracle_err=0.
    for value in (False,True):
        for u,m1,m2,s in itertools.product([-1,1],[0,1],[0,1],[-1,1]):
            f=s*(2*m1-1)*(2*m2-1)*(u if value else 1)
            prior=expit(.8*u)
            l1=(1+.8*f)/4; l0=(1-.8*f)/4
            direct=prior*l1/(prior*l1+(1-prior)*l0)
            formula=expit(.8*u+np.log1p(.8*f)-np.log1p(-.8*f))
            oracle_err=max(oracle_err,abs(direct-formula))
    assert oracle_err<1e-12
    # Same-label finite-prior formula against explicit joint enumeration.
    task=make_task('joint_pair',rng); ctx=sample(task,7,rng); qry=sample(task,20,rng)
    candidates=[(ix,s) for ix in itertools.combinations(range(4),2) for s in (-1,1)]
    likelihood=[]; query_pos=[]; query_neg=[]
    for ix,s in candidates:
        fc=s*np.prod(2*ctx['m'][:,ix]-1,axis=1)
        fq=s*np.prod(2*qry['m'][:,ix]-1,axis=1)
        likelihood.append(np.prod(1+.8*(2*ctx['y']-1)*fc))
        query_pos.append(1+.8*fq); query_neg.append(1-.8*fq)
    w=np.array(likelihood); w=w/w.sum()
    pos=qry['p0']*(w@np.array(query_pos)); neg=(1-qry['p0'])*(w@np.array(query_neg))
    prior_err=float(np.max(np.abs(pos/(pos+neg)-finite_prior_reference(ctx,qry,task))))
    assert prior_err<1e-12
    # Illustrative switch, not evidence of performance on a real task.
    T=256; y=np.r_[np.zeros(128,dtype=int),np.ones(128,dtype=int)]
    ps=np.column_stack([np.full(T,.5),np.full(T,.9)])
    _,fs=fixed_share(ps,y,1/T); _,static=fixed_share(ps,y,0)
    base=float(T*np.log(2)); switch=float(128*np.log(2)-128*np.log(.9))
    assert fs.sum()-base<=np.log(2)+1+1e-10
    result=dict(checked_sequences=24, paths_per_sequence=256,
        mixture_identity_max_error=max_err,minimum_path_bound_slack=min_slack,
        analytic_oracle_max_error=oracle_err,finite_prior_max_error=prior_err,
        illustrative_switch=dict(rounds=T,shift_round=129,baseline_loss=base,
            static_mixture_loss=float(static.sum()),fixed_share_loss=float(fs.sum()),
            switching_comparator_loss=switch,
            baseline_regret_bound=float(np.log(2)+(T-1)*np.log(T/(T-1))),
            one_switch_regret_bound=float(np.log(2)+np.log(T)+(T-2)*np.log(T/(T-1)))))
    Path('math_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': check()
