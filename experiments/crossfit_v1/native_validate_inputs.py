"""Independent raw-row replay of all prepared native inputs before training."""
import json
import numpy as np
from native_prepare import ROOT,OUT,sha
from xgboost import XGBClassifier

def main():
    config=json.loads((ROOT/'configs/large_native_v3.json').read_text());files=0;max_values=max_logits=max_probability=0.
    for spec in config['tasks']:
        name=spec['name'];raw=ROOT/'artifacts/runs/large_native_data'/spec['npz'];assert sha(raw)==spec['npz_sha256']
        with np.load(raw) as a:d={k:a[k].copy() for k in ('x','y','year','row_id','groups')}
        ids=np.column_stack((d['year'],d['row_id']));keys=d['year'].astype(np.int64)*2**40+d['row_id'];assert len(np.unique(keys))==len(keys)
        order=np.argsort(keys);lookup=keys[order]
        def join(v):
            wanted=v[:,0]*2**40+v[:,1];ix=np.searchsorted(lookup,wanted);assert (ix<len(lookup)).all();assert np.array_equal(lookup[ix],wanted);return order[ix]
        folder=ROOT/f'artifacts/runs/large_native_a100_v3/results/{name}_seed171001'
        b=np.load(folder/'boundaries_and_meta.npz');selected=[spec['columns'].index(v) for v in json.loads((folder/'report.json').read_text())['selected_columns']]
        fit=join(b['source_fit_ids']);mean=np.nanmean(d['x'][fit][:,selected],0);scale=np.maximum(np.nanstd(d['x'][fit][:,selected],0),1e-8)
        model=XGBClassifier();model.load_model(folder/'source_backbone.json');model.set_params(device='cpu',n_jobs=2)
        union=set(map(tuple,np.r_[b['source_fit_ids'],b['source_context_ids'],b['meta_support_ids'].reshape(-1,2),b['meta_query_ids'].reshape(-1,2)].tolist()))
        paths=sorted((OUT/'training').glob(name+'_*.npz'))+sorted((OUT/'development').glob(name+'_*.npz'))
        for p in paths:
            with np.load(p) as e:
                assert np.array_equal(e['source_ids'],b['source_context_ids'])
                for prefix in ('source','target','query'):
                    v=e[prefix+'_ids'];assert len(np.unique(v,axis=0))==len(v);ix=join(v);x=d['x'][ix][:,selected];mask=np.isnan(x)
                    assert np.array_equal(mask,e[prefix+'_mask']) and np.array_equal(d['y'][ix],e[prefix+'_y'])
                    error=float(np.max(np.abs(np.nan_to_num((x-mean)/scale,nan=0.)-e[prefix+'_x'])));max_values=max(max_values,error);assert error<1e-6
                    prob=model.predict_proba(d['x'][ix])[:,1].clip(1e-6,1-1e-6);z=np.log(prob/(1-prob))
                    error=float(np.max(np.abs(z-e[prefix+'_logit'])));max_logits=max(max_logits,error);assert error<1e-5
                support=set(map(tuple,e['target_ids'].tolist()));query=set(map(tuple,e['query_ids'].tolist()));assert not support&query
                ix=join(np.r_[e['target_ids'],e['query_ids']])
                if p.parent.name=='development':
                    assert not (support|query)&union;assert (d['year'][ix]==2025).all() and (d['groups'][ix]%5==4).all();assert len(np.unique(d['groups'][ix]))==1
                else:
                    i=int(p.stem.rsplit('_',1)[1]);assert np.array_equal(e['query_ids'],b['meta_query_ids'][i])
                    prob=1/(1+np.exp(-e['query_logit']));error=float(np.max(np.abs(prob-b['meta_experts'][i,:,0])));max_probability=max(max_probability,error);assert error<1e-6
                files+=1
    assert files==122
    report=dict(passed=True,raw_identity_uniqueness=True,raw_replayed_episode_files=files,max_feature_replay_error=max_values,max_logit_replay_error=max_logits,max_training_frozen_probability_error=max_probability,development_source_meta_disjoint=True)
    (OUT/'raw_replay_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
