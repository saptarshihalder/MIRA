"""Recompute all pilot scores/intervals and replay every saved model."""
import json,hashlib
import numpy as np
import torch
from scipy.stats import t
import anchored_run as r

def main():
    torch.set_num_threads(2);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    frozen=json.loads((r.x.ROOT/'artifacts/manifests/anchored_freeze.json').read_text())
    for name,h in frozen.items():assert sha(r.x.ROOT/name)==h
    plan=json.loads((r.x.ROOT/'artifacts/reports/crossfit_v1_gpu/report.json').read_text())['plan']
    training=json.loads((r.OUT/'training.json').read_text());models={};count=replays=0;error=0.
    for name in ('query','support_head','task_scalar','anchor'):
        m=r.make(name,plan)
        if name!='anchor':
            file=r.OUT/(name+'_weights.npz');assert sha(file)==training[name]['checkpoint_sha256']
            assert training[name]['parameters']==370 and training[name]['updates']==500 and training[name]['max_gradient']>0
            with np.load(file) as a:m.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in a.files})
        models[name]=m.eval()
    for kind in ('synthetic','native'):
        report=json.loads((r.OUT/(kind+'.json')).read_text())
        for row in report['rows']:
            with np.load(r.OUT/row['prediction_file']) as a:
                probabilities={k:a[k] for k in row['metrics'] if k in a.files}
                if kind=='native':
                    original=r.x.ROOT/f'artifacts/runs/large_native_a100_v3/results/{row["task"]}_seed{row["seed"]}/predictions/{row["group"]}.npz'
                    assert sha(original)==row['input_sha256']
                    with np.load(original) as b:
                        assert np.array_equal(a['labels'],b['labels'])
                        probabilities.update({k:b[k] for k in row['metrics'] if k not in probabilities})
                for k,p in probabilities.items():
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();p=p.clip(1e-7,1-1e-7)
                    score=float(np.mean(np.where(a['labels']==1,-np.log(p),-np.log1p(-p))))
                    assert abs(score-row['metrics'][k]['nll'])<1e-12;count+=1
                inputs=[torch.as_tensor(a[k][None],dtype=torch.bool if k.endswith('_mask') else torch.float32) for k in r.x.run.v1.INPUT_KEYS]
                for name,m in models.items():
                    with torch.no_grad():out=m.predict(m.prepare(inputs))
                    error=max(error,float(np.max(np.abs(out['hard'][0].numpy()-a[name]))));replays+=1
                    assert out['accepted'][:,0].tolist()==row['decisions'][name]
        if kind=='synthetic':
            assert len(report['rows'])==160 and {v['seed'] for v in report['rows']}==set(range(710001,710021))
            for regime,c in report['contrasts'].items():
                for k,v in c.items():
                    effects=[np.mean([x['metrics'][k]['nll']-x['metrics']['query']['nll'] for x in report['rows'] if x['regime']==regime and x['seed']==seed]) for seed in range(710001,710021)]
                    radius=t.ppf(1-.05/6,19)*np.std(effects,ddof=1)/np.sqrt(20)
                    assert abs(np.mean(effects)-radius-v['family_lower'])<1e-12
                    assert abs(np.mean(effects)+radius-v['family_upper'])<1e-12
        else:
            assert len(report['rows'])==78
            for task,means in report['means'].items():
                for k,v in means.items():assert abs(v-np.mean([x['metrics'][k]['nll'] for x in report['rows'] if x['task']==task]))<1e-12
    assert error==0
    result=dict(passed=True,source_hashes=len(frozen),independent_score_arrays=count,checkpoint_replays=replays,max_reload_error=error,primary_intervals_verified=True,task_means_verified=True)
    (r.OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (r.OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(r.OUT)).replace('\\','/'):sha(p) for p in sorted(r.OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
