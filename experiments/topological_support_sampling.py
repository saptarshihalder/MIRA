"""Label-blind H0/MST support selection; source geometry only, no model scores."""
import csv, hashlib, io, json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
import wine_acquisition_prepare as prep

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/topological_support_sampling_v1'

def select_h0(x,budget=64,components=8):
    """Cut the longest H0 merge edges, then cover each resulting component."""
    x=np.asarray(x,dtype=float);assert np.isfinite(x).all() and 1<=budget<=len(x)
    unique,inverse=np.unique(x,axis=0,return_inverse=True)
    distance=cdist(unique,unique);tree=minimum_spanning_tree(distance).tocoo()
    edges=sorted(zip(tree.data.tolist(),tree.row.tolist(),tree.col.tolist()))
    groups=min(components,budget,len(unique));parent=list(range(len(unique)))
    def root(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for _,a,b in edges[:max(0,len(unique)-groups)]:parent[root(a)]=root(b)
    assignment=np.array([root(int(a)) for a in inverse]);labels=np.unique(assignment)
    pools=[np.where(assignment==g)[0].tolist() for g in labels]
    d=cdist(x,x);chosen=[]
    for pool in pools:
        medoid=pool[int(np.argmin(d[np.ix_(pool,pool)].sum(1)))];chosen.append(medoid);pool.remove(medoid)
    while len(chosen)<budget:
        for pool in pools:
            if not pool:continue
            j=pool[int(np.argmax(d[np.ix_(pool,chosen)].min(1)))];chosen.append(j);pool.remove(j)
            if len(chosen)==budget:break
    return np.array(chosen),dict(h0_death_lengths=[e[0] for e in edges],components=groups,unique_points=len(unique))

def main():
    OUT.mkdir(parents=True,exist_ok=False);records=[]
    for color in ('red','white'):
        raw=ROOT/f'data/raw/wine_acquisition/winequality-{color}.csv'
        audit=json.loads((ROOT/'artifacts/manifests/wine_acquisition_data_audit.json').read_text())
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==audit['sources'][color]['sha256']
        rows=list(csv.reader(io.StringIO(raw.read_text()),delimiter=';'))[1:]
        # The quality column is never parsed or passed to the selection routine.
        x=np.array([[float(v) for v in row[:-1]] for row in rows])
        fit=np.array([prep.partition_for(prep.sha256(prep.canonical_features(v)))==0 for v in x])
        ids=np.flatnonzero(fit);source=x[fit]
        med=np.median(source,axis=0);q=np.quantile(source,[.25,.75],axis=0);scale=np.where(q[1]>q[0],q[1]-q[0],1.)
        for seed in (816001,816002,816003):
            rng=np.random.default_rng(seed);pool=rng.choice(len(source),256,replace=False)
            z=np.clip((source[pool]-med)/scale,-5,5)
            selected,meta=select_h0(z);uniform=rng.choice(len(z),64,replace=False)
            assert len(np.unique(selected))==len(np.unique(uniform))==64
            ds=cdist(z,z[selected]).min(1);du=cdist(z,z[uniform]).min(1)
            name=f'{color}_{seed}.npz'
            np.savez_compressed(OUT/name,pool_source_lines=ids[pool]+2,topological_indices=selected,uniform_indices=uniform,median=med,scale=scale)
            records.append(dict(color=color,seed=seed,pool_rows=256,selected_rows_per_method=64,source_feature_rows_for_scaling=len(source),topological_mean_cover=float(ds.mean()),uniform_mean_cover=float(du.mean()),topological_max_cover=float(ds.max()),uniform_max_cover=float(du.max()),**meta))
    # Synthetic connectivity fixture checks the H0 split, without labels.
    fixture=np.r_[np.arange(10)[:,None]*.01,10+np.arange(10)[:,None]*.01]
    chosen,_=select_h0(fixture,4,2);assert (chosen<10).any() and (chosen>=10).any()
    report=dict(records=records,source_only=True,labels_used_by_sampler=False,prediction_evaluations=0,new_model_fits=0,cloud_calls=0,scope='H0 connectivity coverage only; no predictive, novelty or venue-readiness claim')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('H0 source sampler verified: six 256-row pools; 64 selections per method; no labels, model fits or prediction scores.')

if __name__=='__main__':main()
