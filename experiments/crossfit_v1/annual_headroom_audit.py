"""Reload models and independently check scores, identities, means and gates."""
import json
import numpy as np
from scipy.special import expit
from xgboost import XGBClassifier
import annual_headroom as r

def main():
    specs=r.checked_inputs();report=json.loads((r.OUT/'report.json').read_text())
    assert len(report['rows'])==len(specs)==15
    arrays=0;max_score=max_replay=0.;seen={};family_ids={};cache={}
    config=json.loads((r.ROOT/'configs/large_native_v3.json').read_text())
    for spec in config['tasks']:
        with np.load(r.ROOT/'artifacts/runs/large_native_data'/spec['npz']) as a:
            key=a['year'].astype(np.int64)*2**40+a['row_id'];order=np.argsort(key)
            cache[spec['name']]=({k:a[k] for k in a.files},key,order)
    for row,spec in zip(report['rows'],specs):
        i=row['episode'];task=row['task'];assert all(row[k]==v for k,v in spec.items())
        with np.load(r.ROOT/spec['input']) as a,np.load(r.ROOT/spec['source']) as src,np.load(r.OUT/f'episode{i}_predictions.npz') as p,np.load(r.OUT/f'episode{i}_weights.npz') as w:
            ids=a['ids'];identities=set(map(tuple,ids));assert len(identities)==1536
            assert not identities&seen.setdefault(task,set());seen[task]|=identities
            family='brfss' if task.startswith('brfss') else 'road';family_ids.setdefault(family,set()).update(identities)
            assert not identities&set(map(tuple,src['source_ids']))
            assert (ids[:,0]==2025).all() and np.array_equal(ids,p['ids']) and np.array_equal(p['labels'],a['y'][512:])
            raw,key,order=cache[task];qkey=ids[:,0]*2**40+ids[:,1];ix=order[np.searchsorted(key[order],qkey)]
            assert np.array_equal(key[ix],qkey) and (raw['groups'][ix]==spec['group']).all() and spec['group']%5<3
            assert np.array_equal(raw['x'][ix],a['x'],equal_nan=True) and np.array_equal(raw['y'][ix],a['y'])
            base=XGBClassifier();base.load_model(r.ROOT/spec['backbone']);base.set_params(device='cpu',n_jobs=2)
            prob=base.predict_proba(a['x'])[:,1].clip(1e-6,1-1e-6).astype(float);z=np.log(prob/(1-prob));assert np.array_equal(z,p['z'])
            max_replay=max(max_replay,float(np.max(np.abs(prob[512:]-p['frozen']))))
            pd=r.platt_design(z);designs={'platt':pd,'additive':r.h.design(a['encoded_x'],z,src['source_x'],src['categorical']),'tree_anchor':pd}
            for name,d in designs.items():
                ridge=row['certificates'][name]['ridge'];gradient=np.asarray(d[:512].T@(expit(z[:512]+d[:512]@w[name])-a['y'][:512])).ravel()/512+ridge*w[name]
                assert np.max(np.abs(gradient))<=1e-6
                if name!='tree_anchor':max_replay=max(max_replay,float(np.max(np.abs(expit(z[512:]+d[512:]@w[name])-p[name]))))
            for name,search in row['searches'].items():
                assert search['chosen']==int(np.argmin(search['support_oof_nll'])) and len(search['support_oof_nll'])==4
                certs=search.get('certificates',search.get('anchor_certificates',[]))
                assert len(certs)==(3 if name=='conditional_tree' else 12) and all(c['gradient_max']<=1e-6 for c in certs)
            assert set(p['fold'])=={0,1,2} and len(p['fold'])==512
            tree=XGBClassifier();tree.load_model(r.OUT/f'episode{i}_tree.json');tree.set_params(device='cpu',n_jobs=2)
            features=np.column_stack((a['x'],z)).astype(np.float32);anchor=z+pd@w['tree_anchor']
            prediction=tree.predict_proba(features[512:],base_margin=anchor[512:])[:,1].astype(float)
            max_replay=max(max_replay,float(np.max(np.abs(prediction-p['conditional_tree']))))
            for name,m in row['metrics'].items():
                y=p['labels'];prob=p[name].astype(float);clipped=np.clip(prob,1e-7,1-1e-7)
                score=np.mean(np.where(y==1,-np.log(clipped),-np.log1p(-clipped)))
                max_score=max(max_score,abs(float(score)-m['nll']))
                assert abs(float(np.mean((prob-y)**2))-m['brier'])<1e-12;arrays+=1
    for task,summary in report['summary'].items():
        rows=[x for x in report['rows'] if x['task']==task];assert len(rows)==5
        means={k:float(np.mean([x['metrics'][k]['nll'] for x in rows])) for k in summary['means']}
        assert all(abs(means[k]-v)<1e-12 for k,v in summary['means'].items())
        gate=all(means[k]-means['conditional_tree']>=.003 and sum(x['metrics'][k]['nll']>x['metrics']['conditional_tree']['nll'] for x in rows)>=4 for k in ('frozen','platt','additive'))
        assert gate==summary['gate']
    general=report['summary']['road_ksi']['gate'] and any(s['gate'] for t,s in report['summary'].items() if t.startswith('brfss'))
    assert general==report['general_successor_gate'] and max_replay==0 and max_score<1e-12
    audit=dict(passed=True,score_arrays=arrays,max_score_error=max_score,max_model_replay_error=max_replay,
        endpoint_rows=sum(len(v) for v in seen.values()),unique_rows_by_family={k:len(v) for k,v in family_ids.items()},
        raw_input_replay=True,source_target_disjoint=True,convex_stationarity=True,general_successor_gate=general,auditor_sha256=r.h.sha(r.Path(__file__)))
    r.save(r.OUT/'audit.json',audit)
    r.save(r.OUT/'manifest.json',{p.name:r.h.sha(p) for p in sorted(r.OUT.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(audit))

if __name__=='__main__':main()
