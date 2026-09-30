"""CPU headroom diagnostic. Analytic base probabilities, NOT TFM predictions.

Fresh-target contexts give the shift boundary to every learner. Rolling contexts
contain old and new rows and give no boundary. The finite-prior reference assumes
a stationary context and knows the mechanism family and gamma. It is an optimistic
same-label reference only for fresh-target contexts; it is not a novel method.
"""
import argparse
import csv
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import expit, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

FAMILIES = {
    'label_only': (1, 1, False),
    'joint_pair': (4, 2, False),
    'value_dependent': (8, 1, True),
    'sparse_pair': (16, 2, False),
}

def make_task(family, rng, gamma=.8, beta=.8):
    d, degree, value = FAMILIES[family]
    active = tuple(sorted(rng.choice(d, degree, replace=False).tolist()))
    return dict(family=family, d=d, degree=degree, value=value, active=active,
                sign=int(rng.choice([-1, 1])), gamma=gamma, beta=beta)

def sample(task, n, rng, sign=None):
    u = rng.choice([-1., 1.], size=n)
    p0 = expit(task['beta'] * u)
    y = (rng.random(n) < p0).astype(int)
    m = rng.integers(0, 2, size=(n, task['d']))
    active = task['active']
    g = np.prod(2 * m[:, active[:-1]] - 1, axis=1)
    if task['value']:
        g = g * u
    s = task['sign'] if sign is None else sign
    p_last = (1 + task['gamma'] * s * (2*y-1) * g) / 2
    m[:, active[-1]] = rng.random(n) < p_last
    f = s * np.prod(2*m[:, active]-1, axis=1)
    if task['value']:
        f = f * u
    p = expit(task['beta'] * u + np.log1p(task['gamma']*f)
              - np.log1p(-task['gamma']*f))
    x = rng.normal(size=m.shape)
    x[m == 1] = np.nan
    return dict(u=u, m=m, y=y, p0=p0, oracle=p, x=x)

def subset(a, sel):
    return {k: v[sel] for k, v in a.items()}

def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}

def expected_nll(p, true_p):
    # Integrate out the query label; Monte Carlo only over independent U,M.
    p = np.clip(p, 1e-6, 1-1e-6)
    return float(np.mean(-true_p*np.log(p) - (1-true_p)*np.log1p(-p)))

def features(a, pair=False):
    z = 2*a['m']-1
    parts = [a['u'][:, None], z]
    if pair:
        ij = list(itertools.combinations(range(z.shape[1]), 2))
        zz = np.column_stack([z[:, i]*z[:, j] for i,j in ij]) if ij else np.empty((len(z),0))
        parts += [zz, a['u'][:, None]*z, a['u'][:, None]*zz]
    return np.column_stack(parts).astype(float)

def mask_nb(context, query, strength=1.):
    rates = []
    for c in (0, 1):
        rows = context['m'][context['y'] == c]
        rates.append((rows.sum(axis=0) + strength)/(len(rows)+2*strength))
    r0, r1 = rates
    m = query['m']
    ratio = m @ np.log(r1/r0) + (1-m) @ np.log((1-r1)/(1-r0))
    # Exact correction only in the stated conditional-independence setting.
    # p0 is analytic and mask-blind, so this test does not double-count masks.
    return expit(np.log(query['p0']/(1-query['p0'])) + ratio)

def finite_prior_reference(context, query, task):
    """Exact finite-family Bayes reference, unknown active indices and sign.

    Known family, gamma and full-data beta; uniform prior on active subsets/sign.
    This reference is given structural knowledge that generic baselines lack.
    It is NOT a fair generic adapter competitor or evidence an adapter will learn.
    """
    candidates = list(itertools.combinations(range(task['d']), task['degree']))
    def basis(a):
        b = np.column_stack([np.prod(2*a['m'][:, ix]-1, axis=1) for ix in candidates])
        if task['value']:
            b = b * a['u'][:, None]
        return np.column_stack([b, -b])
    bc, bq = basis(context), basis(query)
    ll = np.log1p(task['gamma'] * (2*context['y']-1)[:, None]*bc).sum(axis=0)
    w = np.exp(ll-logsumexp(ll))
    mu = bq @ w
    return expit(np.log(query['p0']/(1-query['p0'])) + np.log1p(task['gamma']*mu)
                 - np.log1p(-task['gamma']*mu))

def sparse_likelihood_mixture(context, query):
    """Simple finite Bayesian mixture; no family, active indices or gamma input.

    Prior: mass 1/3 on no mask signal; remaining mass uniform across admissible
    degrees 1/2, with/without U interaction, gamma=.4/.8 and either sign. Within
    each degree, uniform over feature subsets. This is a deliberately strong
    generator-informed baseline, not a proposed new architecture.
    """
    d=context['m'].shape[1]; degrees=[k for k in (1,2) if k<=d]
    logs=[np.log(1/3)]; bq_columns=[np.zeros(len(query['u']))]
    for degree in degrees:
        subsets=list(itertools.combinations(range(d),degree))
        bc=np.column_stack([np.prod(2*context['m'][:,ix]-1,axis=1) for ix in subsets])
        bq=np.column_stack([np.prod(2*query['m'][:,ix]-1,axis=1) for ix in subsets])
        for value in (False,True):
            cc=bc*context['u'][:,None] if value else bc
            qq=bq*query['u'][:,None] if value else bq
            for gamma in (.4,.8):
                for sign in (-1,1):
                    prior=(2/3)/(len(degrees)*2*2*2*len(subsets))
                    ll=np.log1p(gamma*sign*(2*context['y']-1)[:,None]*cc).sum(axis=0)
                    logs.extend(np.log(prior)+ll)
                    bq_columns.extend((gamma*sign*qq).T)
    w=np.exp(np.array(logs)-logsumexp(logs))
    signal=w@np.array(bq_columns)
    return expit(np.log(query['p0']/(1-query['p0']))+np.log1p(signal)-np.log1p(-signal))

def predict_methods(context, query, task):
    out = {'analytic_base': query['p0'], 'mechanism_oracle': query['oracle'],
           'finite_prior_reference': finite_prior_reference(context, query, task)}
    for window in (32,128):
        cc=subset(context,slice(-window,None))
        out[f'sparse_likelihood_w{window}']=sparse_likelihood_mixture(cc,query)
        for strength in (1., 8.):
            out[f'mask_nb_a{strength:g}_w{window}'] = mask_nb(cc, query, strength)
        for c in (.3,3.,30.):
            name=f'lr_sparse_interactions_C{c:g}_w{window}'
            if len(np.unique(cc['y']))<2:
                out[name]=query['p0'];continue
            model=LogisticRegression(C=c,solver='liblinear',l1_ratio=1.,max_iter=1000,tol=1e-6,random_state=0)
            model.fit(features(cc,True),cc['y'])
            out[name]=model.predict_proba(features(query,True))[:,1]
    if len(np.unique(context['y'])) < 2:
        for pair in (False, True):
            for c in (.03, .3, 3.):
                out[f'lr_{"interactions" if pair else "main"}_C{c:g}'] = query['p0']
        out['hgb_native'] = query['p0']
        return out
    for pair in (False, True):
        xc, xq = features(context, pair), features(query, pair)
        for c in (.03, .3, 3.):
            model = LogisticRegression(C=c, solver='lbfgs', max_iter=500, tol=1e-6)
            model.fit(xc, context['y'])
            out[f'lr_{"interactions" if pair else "main"}_C{c:g}'] = model.predict_proba(xq)[:,1]
    xc = np.column_stack([context['u'], context['x']])
    xq = np.column_stack([query['u'], query['x']])
    model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7,
        min_samples_leaf=5, l2_regularization=1., early_stopping=False, random_state=0)
    model.fit(xc, context['y'])
    out['hgb_native'] = model.predict_proba(xq)[:,1]
    return out

def run(args):
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for split, offset, ntasks in [('development', 10000, args.dev_tasks), ('evaluation', args.eval_offset, args.eval_tasks)]:
        for family in FAMILIES:
            for i in range(ntasks):
                seed = offset+i
                # Identical RNG construction across reruns, independent task identities per family.
                rng = np.random.default_rng(np.random.SeedSequence([seed, list(FAMILIES).index(family)]))
                task = make_task(family, rng)
                old = sample(task, 128, rng, sign=-task['sign'])
                new = sample(task, 128, rng)
                query = sample(task, args.queries, rng)
                for n in (8, 32, 128):
                    fresh = subset(new, slice(0,n))
                    rolling = subset(concat(old, fresh), slice(-128,None))
                    for protocol, context in [('fresh_target',fresh), ('rolling_128',rolling)]:
                        predictions = predict_methods(context, query, task)
                        for method, p in predictions.items():
                            rows.append(dict(split=split, family=family, seed=seed,
                                protocol=protocol, new_labels=n, context_size=len(context['y']),
                                method=method, expected_nll=expected_nll(p,query['oracle'])))
            print(f'{split} {family}: {ntasks} tasks complete; {time.time()-started:.1f}s', flush=True)
    with (outdir/'task_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for family, protocol, n in itertools.product(FAMILIES, ['fresh_target','rolling_128'],[8,32,128]):
        dev=[r for r in rows if r['split']=='development' and r['family']==family and r['protocol']==protocol and r['new_labels']==n]
        simple=sorted({r['method'] for r in dev} - {'mechanism_oracle','finite_prior_reference'})
        # Per-cell development selection, with no final evaluation outcomes used.
        # n is known target-label budget in fresh_target. For rolling_128, this
        # selection is a diagnostic envelope indexed by evaluator time-since-shift,
        # NOT a deployable unknown-shift model selector.
        chosen=min(simple,key=lambda m:np.mean([r['expected_nll'] for r in dev if r['method']==m]))
        ev=[r for r in rows if r['split']=='evaluation' and r['family']==family and r['protocol']==protocol and r['new_labels']==n]
        bymethod={m:np.array([r['expected_nll'] for r in ev if r['method']==m]) for m in {r['method'] for r in ev}}
        diff=bymethod[chosen]-bymethod['mechanism_oracle']
        prior_diff=bymethod[chosen]-bymethod['finite_prior_reference']
        record=dict(family=family,protocol=protocol,new_labels=n,selected_simple=chosen,
            base_nll=float(bymethod['analytic_base'].mean()),simple_nll=float(bymethod[chosen].mean()),
            oracle_nll=float(bymethod['mechanism_oracle'].mean()),
            prior_reference_nll=float(bymethod['finite_prior_reference'].mean()),
            oracle_gap=float(diff.mean()), oracle_gap_se=float(diff.std(ddof=1)/np.sqrt(len(diff))),
            same_label_prior_gap=float(prior_diff.mean()),
            same_label_prior_gap_se=float(prior_diff.std(ddof=1)/np.sqrt(len(diff))))
        summary.append(record)
    metadata=dict(arguments=vars(args),elapsed_seconds=time.time()-started,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        gamma=.8,beta=.8, masks='1 means missing', base='analytic sigmoid(0.8 U), not a TFM',
        evaluation='expected query log-loss integrates over true conditional label probability',
        limits=['high missingness, high signal, binary tasks only',
                'fresh_target gives an oracle segment boundary',
                'rolling_128 uses evaluator-indexed development method selection; not an online selector',
                'finite_prior_reference knows family and gamma; not a learned adapter',
                'TFM, XGBoost and Drift-Resilient TabPFN not run'])
    (outdir/'summary.json').write_text(json.dumps({'metadata':metadata,'summary':summary},indent=2))
    with (outdir/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='results')
    parser.add_argument('--dev-tasks',type=int,default=12); parser.add_argument('--eval-tasks',type=int,default=48)
    parser.add_argument('--queries',type=int,default=2048)
    parser.add_argument('--eval-offset',type=int,default=30000)
    args=parser.parse_args()
    with threadpool_limits(limits=1): run(args)
