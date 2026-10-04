"""Precommitted finite annual-cohort diagnostic. CPU only; not a novel model."""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import expit
from xgboost import XGBClassifier
import headroom_run as h

ROOT=h.ROOT
INPUT=ROOT/'artifacts/reports/annual_headroom_inputs'
OUT=ROOT/'artifacts/reports/annual_headroom_v1'
FREEZE=ROOT/'artifacts/manifests/annual_headroom_freeze.json'

def save(path, obj):
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')

def prepare():
    INPUT.mkdir(parents=True,exist_ok=False)
    rows=[]; paths=[Path(__file__), ROOT/'experiments/crossfit_v1/headroom_run.py',
        ROOT/'docs/ANNUAL_HEADROOM_PROTOCOL.md',ROOT/'configs/large_native_v3.json',
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
                  if np.sum((a['year']==2025)&(a['groups']==g))>=1536][:5]
        assert len(eligible)==5
        for group in eligible:
            ix=np.flatnonzero((a['year']==2025)&(a['groups']==group))
            ix=np.array(sorted(ix,key=lambda i:hashlib.sha256(f'annual-headroom-v1:2025:{a["row_id"][i]}'.encode()).digest())[:1536])
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
    save(FREEZE,{str(p.relative_to(ROOT)).replace('\\','/'):h.sha(p) for p in paths})
    print(json.dumps(dict(prepared=len(rows),scores_computed=0)))

def checked_inputs():
    for name,digest in json.loads(FREEZE.read_text()).items():assert h.sha(ROOT/name)==digest,name
    return json.loads((INPUT/'index.json').read_text())

def platt_design(z):return sparse.csr_matrix(np.column_stack((np.ones(len(z)),z)))

def tree_model(depth, penalty, seed):
    return XGBClassifier(device='cpu',tree_method='hist',n_estimators=100,max_depth=depth,
        learning_rate=.05,min_child_weight=5,reg_lambda=penalty,n_jobs=2,random_state=seed)

def run():
    specs=checked_inputs();OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    for episode,spec in enumerate(specs):
        assert time.monotonic()-start<120,'Finite CPU time limit reached'
        with np.load(ROOT/spec['input']) as a:x=a['x'];xe=a['encoded_x'];y=a['y'];ids=a['ids']
        with np.load(ROOT/spec['source']) as a:source=a['source_x'];cat=a['categorical']
        base=XGBClassifier();base.load_model(ROOT/spec['backbone']);base.set_params(device='cpu',n_jobs=2)
        prob=base.predict_proba(x)[:,1].clip(1e-6,1-1e-6).astype(float);z=np.log(prob/(1-prob))
        pdesign=platt_design(z);additive=h.design(xe,z,source,cat);sy=y[:512]
        rng=np.random.default_rng(884001+episode);fold=np.zeros(512,dtype=int)
        for value in (0,1):
            ix=rng.permutation(np.flatnonzero(sy==value));assert len(ix)>=3;fold[ix]=np.arange(len(ix))%3
        predictions={'frozen':prob[512:]};weights={};searches={};certificates={}
        for name,design,penalties in [('platt',pdesign,(.001,.01,.1,1.)),('additive',additive,(.01,.1,1.,10.))]:
            losses=[];certs=[]
            for ridge in penalties:
                oof=np.zeros(512)
                for f in range(3):
                    train=np.flatnonzero(fold!=f);val=np.flatnonzero(fold==f)
                    w,c=h.fit(design[train],z[train],sy[train],ridge);certs.append(c);oof[val]=expit(z[val]+design[val]@w)
                losses.append(h.nll(oof,sy))
            chosen=int(np.argmin(losses));w,c=h.fit(design[:512],z[:512],sy,penalties[chosen]);weights[name]=w
            predictions[name]=expit(z[512:]+design[512:]@w);certificates[name]=c
            searches[name]=dict(penalties=penalties,support_oof_nll=losses,chosen=chosen,certificates=certs)
        anchors=[];anchorcerts=[]
        for f in range(3):
            train=np.flatnonzero(fold!=f);w,c=h.fit(pdesign[train],z[train],sy[train],.001)
            anchors.append(z[:512]+pdesign[:512]@w);anchorcerts.append(c)
        w,c=h.fit(pdesign[:512],z[:512],sy,.001);weights['tree_anchor']=w;certificates['tree_anchor']=c
        anchor=z+pdesign@w;features=np.column_stack((x,z)).astype(np.float32);settings=[(d,l) for d in (2,3) for l in (10,100)];losses=[]
        for depth,penalty in settings:
            oof=np.zeros(512)
            for f in range(3):
                train=np.flatnonzero(fold!=f);val=np.flatnonzero(fold==f);tree=tree_model(depth,penalty,884001+episode)
                tree.fit(features[train],sy[train],base_margin=anchors[f][train]);oof[val]=tree.predict_proba(features[val],base_margin=anchors[f][val])[:,1]
            losses.append(h.nll(oof,sy))
        chosen=int(np.argmin(losses));tree=tree_model(*settings[chosen],884001+episode)
        tree.fit(features[:512],sy,base_margin=anchor[:512]);predictions['conditional_tree']=tree.predict_proba(features[512:],base_margin=anchor[512:])[:,1].astype(float)
        searches['conditional_tree']=dict(settings=settings,support_oof_nll=losses,chosen=chosen,anchor_certificates=anchorcerts)
        tree.save_model(OUT/f'episode{episode}_tree.json');np.savez_compressed(OUT/f'episode{episode}_weights.npz',**weights)
        np.savez_compressed(OUT/f'episode{episode}_predictions.npz',**predictions,labels=y[512:],ids=ids,z=z,fold=fold)
        metrics={k:dict(nll=h.nll(p,y[512:]),brier=float(np.mean((p-y[512:])**2))) for k,p in predictions.items()}
        rows.append(dict(episode=episode,**spec,metrics=metrics,searches=searches,certificates=certificates))
        save(OUT/'partial.json',rows)
    summary={}
    for task in sorted({r['task'] for r in rows}):
        subset=[r for r in rows if r['task']==task];means={k:float(np.mean([r['metrics'][k]['nll'] for r in subset])) for k in predictions}
        gains={k:dict(mean=means[k]-means['conditional_tree'],positive=sum(r['metrics'][k]['nll']>r['metrics']['conditional_tree']['nll'] for r in subset)) for k in ('frozen','platt','additive')}
        summary[task]=dict(means=means,gains=gains,gate=all(g['mean']>=.003 and g['positive']>=4 for g in gains.values()))
    general=summary['road_ksi']['gate'] and any(summary[k]['gate'] for k in summary if k.startswith('brfss'))
    report=dict(rows=rows,summary=summary,general_successor_gate=general,seconds=time.monotonic()-start,cloud_calls=0)
    save(OUT/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__=='__main__': {'prepare':prepare,'run':run}[sys.argv[1]]()
