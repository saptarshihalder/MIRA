"""Precommitted finite annual-cohort diagnostic. CPU only; not a novel model."""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import expit
from xgboost import XGBClassifier
import headroom_run as h

ROOT=h.ROOT
INPUT=ROOT/'artifacts/reports/annual_rl_inputs'
OUT=ROOT/'artifacts/reports/annual_rl_v1'
FREEZE=ROOT/'artifacts/manifests/annual_rl_freeze.json'

def save(path, obj):
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')

def prepare():
    INPUT.mkdir(parents=True,exist_ok=False)
    rows=[]; paths=[Path(__file__), ROOT/'experiments/crossfit_v1/rl_run.py', ROOT/'experiments/crossfit_v1/annual_headroom.py', ROOT/'experiments/crossfit_v1/headroom_run.py',
        ROOT/'docs/ANNUAL_RL_PROTOCOL.md',ROOT/'configs/large_native_v3.json',
        ROOT/'artifacts/manifests/unscored_cohort_access_audit.json']
    for spec in json.loads((ROOT/'configs/large_native_v3.json').read_text())['tasks']:
        task=spec['name']; raw=ROOT/'artifacts/runs/large_native_data'/spec['npz']
        assert h.sha(raw)==spec['npz_sha256']; paths.append(raw)
        folder=ROOT/f'artifacts/reports/large_native_a100_v3/{task}_seed171001'
        boundary=folder/'boundaries_and_meta.npz'; backbone=folder/'source_backbone.json'
        paths.extend([boundary,backbone])
        with np.load(raw) as f: a={k:f[k] for k in f.files}
        key=a['year'].astype(np.int64)*2**40+a['row_id']; order=np.argsort(key)
        with np.load(boundary) as b: ids=b['source_fit_ids']
        ix=np.searchsorted(key[order],ids[:,0]*2**40+ids[:,1]);fit=order[ix]
        assert np.array_equal(key[fit],ids[:,0]*2**40+ids[:,1])
        assert (a['year'][fit]==2024).all()
        source=a['x'][fit].astype(np.float32); encoded=source.copy();cat=[];vocab={}
        for j in range(source.shape[1]):
            unique=np.unique(source[np.isfinite(source[:,j]),j]);c=len(unique)<=32;cat.append(c)
            if c:
                vocab[j]=unique
                encoded[:,j]=np.where(np.isnan(source[:,j]),np.nan,np.searchsorted(unique,source[:,j]))
        source_file=INPUT/f'{task}_source.npz'
        np.savez_compressed(source_file,source_x=encoded,categorical=cat,source_ids=ids)
        paths.append(source_file)
        eligible=[int(g) for g in sorted(np.unique(a['groups'][(a['year']==2025)&(a['groups']%5<3)]))
                  if np.sum((a['year']==2025)&(a['groups']==g))>=1536][5:10]
        assert len(eligible)==5
        for group in eligible:
            ix=np.flatnonzero((a['year']==2025)&(a['groups']==group))
            ix=np.array(sorted(ix,key=lambda i:hashlib.sha256(f'annual-rl-v1:2025:{a["row_id"][i]}'.encode()).digest())[:1536])
            x=a['x'][ix].astype(np.float32);xe=x.copy()
            for j,unique in vocab.items():
                column=x[:,j];code=np.searchsorted(unique,column)
                if len(unique): known=(code<len(unique))&(unique[np.minimum(code,len(unique)-1)]==column)
                else: known=np.zeros(len(x),dtype=bool)
                xe[:,j]=np.where(np.isnan(column),np.nan,np.where(known,code,126))
            file=INPUT/f'{task}_group{group}.npz';rowids=np.column_stack((a['year'][ix],a['row_id'][ix]))
            assert len(set(map(tuple,rowids)))==1536
            np.savez_compressed(file,x=x,encoded_x=xe,y=a['y'][ix],ids=rowids)
            paths.append(file)
            rows.append(dict(task=task,group=group,input=str(file.relative_to(ROOT)).replace('\\','/'),
                source=str(source_file.relative_to(ROOT)).replace('\\','/'),backbone=str(backbone.relative_to(ROOT)).replace('\\','/')))
    save(INPUT/'index.json',rows);paths.append(INPUT/'index.json')
    finish_inputs(rows,paths)
    save(FREEZE,{str(p.relative_to(ROOT)).replace('\\','/'):h.sha(p) for p in paths})
    print(json.dumps(dict(prepared=len(rows),scores_computed=0)))


def finish_inputs(rows,paths):
    old=json.loads((ROOT/'artifacts/reports/annual_headroom_inputs/index.json').read_text())
    prior={}
    for spec in old:
        with np.load(ROOT/spec['input']) as a:prior.setdefault(spec['task'],set()).update(map(tuple,a['ids']))
    for spec in rows:
        with np.load(ROOT/spec['input']) as f:a={k:f[k] for k in f.files}
        with np.load(ROOT/spec['source']) as f:source=f['source_x'];cat=f['categorical']
        assert not set(map(tuple,a['ids']))&prior[spec['task']]
        base=XGBClassifier();base.load_model(ROOT/spec['backbone']);base.set_params(device='cpu',n_jobs=2)
        p=base.predict_proba(a['x'])[:,1].clip(1e-6,1-1e-6).astype(float);z=np.log(p/(1-p))
        d=sparse.csr_matrix(np.column_stack((np.ones(len(z)),z)))
        w,certificate=h.fit(d[:384],z[:384],a['y'][:384],.001);anchor=z+d@w
        features=[]
        for j,categorical in enumerate(cat):
            col=a['encoded_x'][:,j]
            if categorical:
                code=np.where(np.isnan(col),33,np.where(col==126,32,col)).astype(int)
                assert ((code>=0)&(code<34)).all();features.append(np.eye(34,dtype=np.float32)[code])
            else:
                mean=np.nanmean(source[:,j]);scale=np.nanstd(source[:,j]);mean=0 if not np.isfinite(mean) else mean;scale=max(scale,1e-8) if np.isfinite(scale) else 1
                features.append(np.column_stack((np.clip(np.nan_to_num((col-mean)/scale,nan=0),-10,10),np.isnan(col))))
        features.append(np.column_stack((z,anchor)))
        a.update(features=np.concatenate(features,axis=1).astype(np.float32),z=z,anchor=anchor,anchor_weights=w,anchor_gradient_max=certificate['gradient_max'])
        assert np.isfinite(a['features']).all()
        np.savez_compressed(ROOT/spec['input'],**a)

if __name__=='__main__':prepare()
