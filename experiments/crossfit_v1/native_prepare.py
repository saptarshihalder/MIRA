"""Recover genuine source-only native meta episodes using frozen CUDA backbones."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/reports/native_meta_inputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'training').mkdir();(OUT/'development').mkdir()
    config=json.loads((ROOT/'configs/large_native_v3.json').read_text());reports=[]
    codes=np.array([c for c in itertools.product((0,1),repeat=4) if any(c)])
    singles=[int(np.flatnonzero((codes==np.eye(4,dtype=int)[j]).all(1))[0]) for j in range(4)]
    for spec in config['tasks']:
        task=spec['name'];raw=ROOT/'artifacts/runs/large_native_data'/spec['npz'];assert sha(raw)==spec['npz_sha256']
        folder=ROOT/f'artifacts/runs/large_native_a100_v3/results/{task}_seed171001'
        old=json.loads((folder/'report.json').read_text());boundary=np.load(folder/'boundaries_and_meta.npz')
        with np.load(raw) as a:data={k:a[k].copy() for k in ('x','y','groups','year','row_id')}
        assert np.all((data['row_id']>=0)&(data['row_id']<2**40))
        key=data['year'].astype(np.int64)*2**40+data['row_id'];order=np.argsort(key);sorted_key=key[order]
        def join(ids):
            wanted=ids[:,0]*2**40+ids[:,1];ix=np.searchsorted(sorted_key,wanted);assert np.array_equal(sorted_key[ix],wanted);return order[ix]
        fit=join(boundary['source_fit_ids']);selected=[spec['columns'].index(c) for c in old['selected_columns']]
        missing=np.isnan(data['x'][fit]).mean(0)
        expected=sorted(np.flatnonzero((missing>0)&(missing<1)),key=lambda j:(-missing[j],spec['columns'][j]))[:4]
        assert selected==expected
        mean=np.nanmean(data['x'][fit][:,selected],0);scale=np.maximum(np.nanstd(data['x'][fit][:,selected],0),1e-8)
        backbone=XGBClassifier();backbone.load_model(folder/'source_backbone.json');backbone.set_params(device='cpu',n_jobs=2)
        def observed(ids):
            rows=join(ids);x=data['x'][rows][:,selected];mask=np.isnan(x);value=np.nan_to_num((x-mean)/scale,nan=0.)
            probability=backbone.predict_proba(data['x'][rows])[:,1].clip(1e-6,1-1e-6);z=np.log(probability/(1-probability)).astype(np.float32)
            return value.astype(np.float32),mask,z,data['y'][rows].copy(),rows
        sid=boundary['source_context_ids'];sx,sm,sz,sy,_=observed(sid);source_f=boundary['source_f']
        assert np.array_equal(sy,boundary['source_y']);assert np.allclose(sx,source_f[:,3:7],atol=1e-6,rtol=0)
        assert np.array_equal(sm,((1-source_f[:,7+np.array(singles)])/2).astype(bool));assert np.allclose(sz,source_f[:,-1],atol=1e-5,rtol=0)
        source=set(map(tuple,np.concatenate((boundary['source_fit_ids'],sid)).tolist()));meta=set();max_logit_error=0.
        for i,(tid,qid) in enumerate(zip(boundary['meta_support_ids'],boundary['meta_query_ids'])):
            tx,tm,tz,ty,tr=observed(tid);qx,qm,qz,qy,qr=observed(qid)
            assert np.array_equal(qy,boundary['meta_labels'][i])
            assert (data['year'][np.r_[tr,qr]]==2024).all() and (data['groups'][np.r_[tr,qr]]%5<3).all()
            current=set(map(tuple,np.concatenate((tid,qid)).tolist()));assert len(current)==640 and not current&source and not current&meta;meta|=current
            np.savez_compressed(OUT/'training'/f'{task}_{i}.npz',query_x=qx,query_mask=qm,query_logit=qz,query_y=qy,query_ids=qid,source_x=sx,source_mask=sm,source_logit=sz,source_y=sy,source_ids=sid,target_x=tx,target_mask=tm,target_logit=tz,target_y=ty,target_ids=tid)
        for file in sorted((folder/'predictions').glob('development*.npz')):
            original=ROOT/f'artifacts/reports/crossfit_extension/predictions/native_{task}_171001_{file.stem}.npz'
            with np.load(original) as a:e={k:a[k].copy() for k in a.files if k in ('labels','source_ids','target_ids','query_ids') or k.startswith(('query_','target_','source_'))}
            _,_,new_z,new_y,_=observed(e['query_ids']);assert np.array_equal(new_y,e['labels'])
            max_logit_error=max(max_logit_error,float(np.max(np.abs(new_z-e['query_logit']))));assert max_logit_error<1e-5
            ids=set(map(tuple,np.concatenate((e['target_ids'],e['query_ids'])).tolist()));assert not ids&source and not ids&meta
            e['query_y']=e.pop('labels');np.savez_compressed(OUT/'development'/f'{task}_{file.stem}.npz',**e)
        reports.append(dict(task=task,family='brfss' if task.startswith('brfss') else 'road',training_episodes=32,development_episodes=len(list((folder/'predictions').glob('development*.npz'))),raw_sha256=sha(raw),boundary_sha256=sha(folder/'boundaries_and_meta.npz'),backbone_sha256=sha(folder/'source_backbone.json'),selected_columns=old['selected_columns'],max_cpu_gpu_logit_replay_error=max_logit_error,source_fit_labels=len(fit),source_context_labels=len(sid)))
    (OUT/'report.json').write_text(json.dumps(dict(tasks=reports,scope='2024 source-only native meta labels; used2025 development',xgboost='3.4.1',new_cloud_calls=0),indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(reports))

if __name__=='__main__':main()
