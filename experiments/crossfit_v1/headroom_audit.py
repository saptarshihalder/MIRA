"""Independent metric, identity, stationarity and model-replay checks."""
import json
import numpy as np
from scipy import sparse
from scipy.special import expit
from xgboost import XGBClassifier
import headroom_run as r

def main():
    freeze=json.loads((r.ROOT/'artifacts/manifests/unscored_headroom_freeze.json').read_text())
    for name,h in freeze.items():assert r.sha(r.ROOT/name)==h
    report=json.loads((r.OUT/'report.json').read_text());seen=set();max_score=max_replay=0.;arrays=0
    for row in report['rows']:
        i=row['episode']
        with np.load(r.INPUT/f'episode{i}.npz') as a,np.load(r.OUT/f'episode{i}_predictions.npz') as p,np.load(r.OUT/f'episode{i}_linear_weights.npz') as weights:
            ids=a['patient_ids'];assert not seen&set(ids);seen.update(ids);assert np.array_equal(ids,p['patient_ids'])
            assert np.array_equal(a['representative_indices'],np.arange(47114+i*1536,47114+(i+1)*1536))
            assert np.array_equal(p['labels'],a['y'][512:]);assert np.array_equal(a['representative_indices'],p['representative_indices'])
            z=p['frozen_logits'];y=a['y'][512:]
            backbone=XGBClassifier();backbone.load_model(r.ROOT/'artifacts/reports/policy_native_gpu_v1/source_backbone.json');backbone.set_params(device='cpu',enable_categorical=True,n_jobs=2)
            base=backbone.predict_proba(a['x'])[:,1].clip(1e-6,1-1e-6).astype(float)
            assert np.allclose(z,np.log(base/(1-base)),atol=0,rtol=0)
            max_replay=max(max_replay,float(np.max(np.abs(base[512:]-p['frozen']))))
            for name in ('platt','additive'):
                d=sparse.csr_matrix(np.column_stack((np.ones(len(z)),z))) if name=='platt' else r.design(a['x'],z,a['source_x'],a['categorical'])
                w=weights[name];ridge=row['final_fit_certificates'][name]['ridge'];gradient=np.asarray(d[:512].T@(expit(z[:512]+d[:512]@w)-a['y'][:512])).ravel()/512+ridge*w
                assert np.max(np.abs(gradient))<=1e-6
                max_replay=max(max_replay,float(np.max(np.abs(expit(z[512:]+d[512:]@w)-p[name]))))
                search=row['searches'][name];assert search['chosen']==int(np.argmin(search['support_oof_nll'])) and search['search_folds']==12
                assert all(c['gradient_max']<=1e-6 for c in search['certificates'])
            tree=XGBClassifier();tree.load_model(r.OUT/f'episode{i}_conditional_tree.json');tree.set_params(device='cpu',enable_categorical=True,n_jobs=2)
            q=np.column_stack((a['x'],z)).astype(np.float32)[512:];pred=tree.predict_proba(q,base_margin=z[512:])[:,1]
            max_replay=max(max_replay,float(np.max(np.abs(pred-p['conditional_tree']))))
            for name,m in row['metrics'].items():
                prob=p[name].astype(float);clipped=np.clip(prob,1e-7,1-1e-7)
                score=float(np.mean(np.where(y==1,-np.log(clipped),-np.log1p(-clipped))))
                max_score=max(max_score,abs(score-m['nll']));assert abs(float(np.mean((prob-y)**2))-m['brier'])<1e-12;arrays+=1
    assert len(seen)==7680 and len(report['rows'])==5 and max_score<1e-7 and max_replay==0, (len(seen),max_score,max_replay)
    means={k:float(np.mean([x['metrics'][k]['nll'] for x in report['rows']])) for k in report['means']}
    for k,v in means.items():assert abs(v-report['means'][k])<1e-12
    gate=all(means[k]-means['conditional_tree']>=.003 and sum(x['metrics'][k]['nll']>x['metrics']['conditional_tree']['nll'] for x in report['rows'])>=4 for k in ('frozen','platt','additive'))
    assert gate==report['headroom_gate']
    audit=dict(passed=True,patients=7680,episodes=5,score_arrays=arrays,max_score_error=max_score,score_tolerance=1e-7,score_precision_note="Published conditional-tree NLL used float32 logs; independent scoring uses float64",max_model_replay_error=max_replay,linear_stationarity_verified=True,headroom_gate=gate,auditor_sha256=r.sha(r.Path(__file__)))
    (r.OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (r.OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(r.OUT)).replace('\\','/'):r.sha(p) for p in sorted(r.OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'},indent=2)+'\n');print(json.dumps(audit))

if __name__=='__main__':main()
