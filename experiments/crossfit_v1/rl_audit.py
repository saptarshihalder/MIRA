"""Independent saved-prediction and checkpoint audit; no experimental fitting."""
import json
import numpy as np
import torch
import rl_run as r

def main():
    torch.set_num_threads(1)
    for path,digest in json.loads(r.FREEZE.read_text()).items():assert r.sha(r.ROOT/path)==digest
    specs=json.loads((r.INPUT/'index.json').read_text());report=json.loads((r.OUT/'report.json').read_text())
    baseline_dir=r.ROOT/'artifacts/reports/annual_rl_baselines'
    baseline=json.loads((baseline_dir/'report.json').read_text())
    assert json.loads((baseline_dir/'audit.json').read_text())['passed']
    assert len(report['rows'])==135
    max_score=max_replay=0.;arrays=0;initial={};params={};seen={};family_ids={}
    for row in report['rows']:
        i=row['episode'];seed=row['seed'];mode=row['mode'];name=f'episode{i}_{seed}_{mode}';spec=specs[i]
        assert mode in r.MODES and seed in r.SEEDS and row['task']==spec['task'] and row['group']==spec['group']
        with np.load(r.ROOT/spec['input']) as a,np.load(r.OUT/f'{name}_predictions.npz') as p,np.load(r.OUT/f'{name}_weights.npz') as weights,np.load(baseline_dir/f'episode{i}_predictions.npz') as b:
            assert r.sha(r.OUT/f'{name}_weights.npz')==row['weights_sha256']
            assert np.array_equal(p['ids'],a['ids']) and np.array_equal(p['labels'],a['y'][512:])
            assert np.array_equal(b['ids'],a['ids']) and np.array_equal(b['z'],a['z'])
            if seed==r.SEEDS[0] and mode==r.MODES[0]:
                identities=set(map(tuple,a['ids']));assert len(identities)==1536
                assert not identities&seen.setdefault(spec['task'],set());seen[spec['task']]|=identities
                family='brfss' if spec['task'].startswith('brfss') else 'road';family_ids.setdefault(family,set()).update(identities)
            model=r.Actor(a['features'].shape[1]);model.load_state_dict({k:torch.tensor(weights[k]) for k in weights.files})
            assert r.sha_state(model)!=row['training']['initial_sha256']
            key=(i,seed);initial.setdefault(key,row['training']['initial_sha256']);assert initial[key]==row['training']['initial_sha256']
            params.setdefault(i,row['training']['parameters']);assert params[i]==sum(x.numel() for x in model.parameters())==row['training']['parameters']
            assert [x['step'] for x in row['training']['trace']]==[50,100,150,200,250,300]
            raw,policy=r.predict(model,a['features'][512:],a['anchor'][512:]);val,_=r.predict(model,a['features'][384:512],a['anchor'][384:512])
            anchor=1/(1+np.exp(-a['anchor']));alphas=(0.,.25,.5,1.)
            # Formula-based support selection and metrics, separately from runner nll().
            def loss(prob,y):
                prob=np.clip(np.asarray(prob,dtype=float),1e-7,1-1e-7)
                return float(np.mean(np.where(y==1,-np.log(prob),-np.log1p(-prob))))
            losses=[loss(anchor[384:512]+v*(val-anchor[384:512]),a['y'][384:512]) for v in alphas]
            assert alphas[int(np.argmin(losses))]==row['alpha'] and np.allclose(losses,row['selection_losses'],atol=1e-12,rtol=0)
            chosen=anchor[512:]+row['alpha']*(raw-anchor[512:])
            for actual,expected in [(raw,p['raw']),(policy,p['policy']),(chosen,p['selected']),(val,p['selection_raw']),(anchor[512:],p['anchor'])]:
                max_replay=max(max_replay,float(np.max(np.abs(actual-expected))))
            assert np.all(policy>=0) and np.allclose(policy.sum(1),1,atol=1e-6,rtol=0)
            for key,pred in [('nll',chosen),('raw_nll',raw),('anchor_nll',anchor[512:])]:
                max_score=max(max_score,abs(loss(pred,a['y'][512:])-row[key]));arrays+=1
            assert abs(float(np.mean((chosen-a['y'][512:])**2))-row['brier'])<1e-12
            residual=anchor[:384]-a['y'][:384]
            gradient=np.array([residual.mean(),(residual*a['z'][:384]).mean()])+.001*a['anchor_weights']
            assert np.max(np.abs(gradient))<=1e-6
    assert max_score<1e-12 and max_replay==0
    summary={}
    for task in sorted({s['task'] for s in specs}):
        indices=[i for i,s in enumerate(specs) if s['task']==task];assert len(indices)==5
        groups=[]
        for i in indices:
            neural={mode:float(np.mean([x['nll'] for x in report['rows'] if x['episode']==i and x['mode']==mode])) for mode in r.MODES}
            base={k:v['nll'] for k,v in baseline['rows'][i]['metrics'].items()}
            groups.append(dict(episode=i,group=specs[i]['group'],means={**base,**neural}))
        names=list(groups[0]['means']);means={k:float(np.mean([g['means'][k] for g in groups])) for k in names}
        comparisons={k:dict(gain=means[k]-means['reinforce'],positive_groups=sum(g['means'][k]>g['means']['reinforce'] for g in groups)) for k in names if k!='reinforce'}
        raw_means={m:float(np.mean([x['raw_nll'] for x in report['rows'] if x['task']==task and x['mode']==m])) for m in r.MODES}
        accepted={m:sum(x['alpha']>0 for x in report['rows'] if x['task']==task and x['mode']==m) for m in r.MODES}
        by_seed={str(seed):{m:float(np.mean([x['nll'] for x in report['rows'] if x['task']==task and x['mode']==m and x['seed']==seed])) for m in r.MODES} for seed in r.SEEDS}
        summary[task]=dict(means=means,raw_means=raw_means,accepted_of_15=accepted,by_seed=by_seed,comparisons=comparisons,
            gate=all(c['gain']>=.003 and c['positive_groups']>=4 for c in comparisons.values()),groups=groups)
    gate=summary['road_ksi']['gate'] and any(s['gate'] for task,s in summary.items() if task.startswith('brfss'))
    r.save(r.OUT/'summary.json',dict(tasks=summary,general_rl_gate=gate))
    audit=dict(passed=True,checkpoints=135,metric_arrays=arrays,max_score_error=max_score,max_model_replay_error=max_replay,
        matching_initializations=45,source_only_anchor_stationarity=True,selection_replayed=True,endpoint_rows=sum(len(v) for v in seen.values()),
        unique_rows_by_family={k:len(v) for k,v in family_ids.items()},general_rl_gate=gate,auditor_sha256=r.sha(r.Path(__file__)))
    r.save(r.OUT/'audit.json',audit)
    r.save(r.OUT/'manifest.json',{p.name:r.sha(p) for p in sorted(r.OUT.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(dict(audit=audit,tasks={k:{x:v[x] for x in ('means','raw_means','accepted_of_15','gate')} for k,v in summary.items()})))

if __name__=='__main__':main()
