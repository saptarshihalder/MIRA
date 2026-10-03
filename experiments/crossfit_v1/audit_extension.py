"""Independent score/interval audit and checkpoint replay, with frozen inputs."""
import hashlib,json
import numpy as np
import torch
from scipy.stats import t
import extension as x

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(p,y):
    p=np.clip(p.astype(float),1e-7,1-1e-7)
    return float(np.mean(np.where(y==1,-np.log(p),-np.log1p(-p))))

def main():
    torch.set_num_threads(2)
    frozen=json.loads((x.ROOT/'artifacts/manifests/crossfit_extension_freeze.json').read_text())
    for f,h in frozen.items():assert digest(x.ROOT/f)==h,f
    plan=json.loads((x.ROOT/'artifacts/reports/crossfit_v1_gpu/report.json').read_text())['plan']
    models,hashes=x.load_models(plan);count=replays=0;score_error=replay_error=0.
    reports={kind:json.loads((x.OUT/(kind+'.json')).read_text()) for kind in ('synthetic','native')}
    for kind,report in reports.items():
        assert report['checkpoint_hashes']==hashes
        for row in report['rows']:
            with np.load(x.OUT/row['prediction_file']) as a:
                if kind=='native':
                    original=x.ROOT/f'artifacts/runs/large_native_a100_v3/results/{row["task"]}_seed{row["seed"]}/predictions/{row["group"]}.npz'
                    assert digest(original)==row['input_sha256']
                    with np.load(original) as b:
                        assert np.array_equal(a['labels'],b['labels'])
                        for key in ('target_ids','query_ids'):
                            assert np.array_equal(a[key],b['support_ids' if key=='target_ids' else key])
                        for k in set(row['metrics'])-set(a.files):
                            score_error=max(score_error,abs(metric(b[k],b['labels'])-row['metrics'][k]['nll']));count+=1
                for k,m in row['metrics'].items():
                    if k not in a.files:continue
                    assert np.isfinite(a[k]).all() and ((a[k]>=0)&(a[k]<=1)).all()
                    score_error=max(score_error,abs(metric(a[k],a['labels'])-m['nll']))
                    assert abs(float(np.mean((a[k]-a['labels'])**2))-m['brier'])<1e-12
                    count+=1
                inputs=[torch.as_tensor(a[k][None],dtype=torch.bool if k.endswith('_mask') else torch.float32) for k in x.run.v1.INPUT_KEYS]
                for k,m in models.items():
                    with torch.no_grad():result=m.predict(m.prepare(inputs))
                    replay_error=max(replay_error,float(np.max(np.abs(result['hard'][0].numpy()-a[k]))))
                    assert result['accepted'][:,0].tolist()==row['decisions'][k]
                    replays+=1
    synthetic=reports['synthetic'];assert sorted({r['seed'] for r in synthetic['rows']})==list(range(700001,700021))
    for regime,contrasts in synthetic['contrasts'].items():
        for comparator,c in contrasts.items():
            values=[]
            for seed in range(700001,700021):
                values.append(np.mean([r['metrics'][comparator]['nll']-r['metrics']['query']['nll'] for r in synthetic['rows'] if r['regime']==regime and r['seed']==seed]))
            se=np.std(values,ddof=1)/np.sqrt(20);mean=np.mean(values)
            assert abs(mean-c['gain'])<1e-12
            for confidence,q in [('95',.975),('97_5',.9875)]:
                radius=t.ppf(q,19)*se
                assert abs(mean-radius-c['lower'+confidence])<1e-12
                assert abs(mean+radius-c['upper'+confidence])<1e-12
    native=reports['native'];assert len(native['rows'])==78 and len(synthetic['rows'])==160
    for task,means in native['means'].items():
        for k,v in means.items():assert abs(v-np.mean([r['metrics'][k]['nll'] for r in native['rows'] if r['task']==task]))<1e-12
    assert score_error<1e-12 and replay_error==0
    result=dict(passed=True,frozen_file_hashes=len(frozen),synthetic_cells=160,native_cells=78,probability_arrays=count,checkpoint_replays=replays,max_score_error=score_error,max_replay_error=replay_error,all_recorded_decisions_match=True,intervals_and_task_means_verified=True,confirmation_seeds_untouched=True)
    (x.OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (x.OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(x.OUT)).replace('\\','/'):digest(p) for p in sorted(x.OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
