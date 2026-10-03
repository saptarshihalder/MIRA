"""One fixed replication and saved native transfer; no model selection."""
import itertools, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import t
import run
from support_head import SupportHead
from matched_control import equalize_modes
from recover_evaluation import stable_platt

ROOT=run.ROOT
OUT=ROOT/'artifacts/reports/crossfit_extension'

def load_models(plan):
    models={}
    files={name:ROOT/f'artifacts/reports/crossfit_v1_gpu/{name}_weights.npz' for name in run.NAMES}
    files.update(task_scalar=ROOT/'artifacts/reports/crossfit_matched_control/weights.npz',
                 support_head=ROOT/'artifacts/reports/crossfit_support_control/weights.npz')
    for name, file in files.items():
        model=SupportHead('query') if name=='support_head' else run.make_model(name if name in run.NAMES else 'query',plan,'cpu')
        if name=='task_scalar':model.network.register_forward_pre_hook(equalize_modes)
        with np.load(file) as a:model.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in a.files})
        models[name]=model.eval()
    return models,{name:run.v1.checksum(p) for name,p in files.items()}

def evaluate(e,models,plan,path):
    predictions,certificates=run.v1.controls(e,plan)
    assert certificates['support_logistic'].get('converged',True)
    decisions={}; inputs=run.v1.tensor_input([e],'cpu')
    for name,model in models.items():
        with torch.no_grad():r=model.predict(model.prepare(inputs))
        predictions[name]=r['hard'][0].numpy();decisions[name]=r['accepted'][:,0].tolist()
    guarded=[]
    for parity in (0,1):
        fit=np.arange(parity,len(e['target_y']),2);val=np.arange(1-parity,len(e['target_y']),2)
        inner={**e,**{k:e[k][fit] for k in ('target_x','target_mask','target_y','target_logit')},
               **{k:np.concatenate((e['target_'+k[6:]][val],e[k])) for k in ('query_x','query_mask','query_logit')}}
        pred,audit=run.v1.controls(inner,plan);assert audit['support_logistic'].get('converged',True)
        n=len(val);y=e['target_y'][val];z=e['target_logit'][val];p=pred['support_logistic'][:n].clip(1e-8,1-1e-8)
        gain=np.logaddexp(0,z)-y*z+y*np.log(p)+(1-y)*np.log1p(-p)
        accepted=gain.mean()>1.645*gain.std(ddof=1)/np.sqrt(n)
        guarded.append(pred['support_logistic'][n:] if accepted else predictions['frozen'])
    predictions['guarded_logistic']=np.mean(guarded,0)
    np.savez_compressed(path,**predictions,labels=e['query_y'],**{k:e[k] for k in run.v1.INPUT_KEYS},
                        **{k:e[k] for k in ('source_ids','target_ids','query_ids') if k in e})
    return dict(metrics={k:run.v1.metrics(v,e['query_y']) for k,v in predictions.items()},decisions=decisions,
                prediction_file=str(path.relative_to(OUT)).replace('\\','/'),control_certificates=certificates)

def summary(rows,seeds):
    result={}
    for regime in run.v1.REGIMES:
        subset=[r for r in rows if r['regime']==regime];contrasts={}
        for name in subset[0]['metrics']:
            effect=np.array([np.mean([r['metrics'][name]['nll']-r['metrics']['query']['nll'] for r in subset if r['seed']==seed]) for seed in seeds])
            radius=t.ppf(.975,19)*effect.std(ddof=1)/np.sqrt(20)
            family_radius=t.ppf(.9875,19)*effect.std(ddof=1)/np.sqrt(20)
            contrasts[name]=dict(gain=float(effect.mean()),lower95=float(effect.mean()-radius),upper95=float(effect.mean()+radius),lower97_5=float(effect.mean()-family_radius),upper97_5=float(effect.mean()+family_radius))
        result[regime]=contrasts
    return result

def native_episodes():
    codes=np.array([a for a in itertools.product((0,1),repeat=4) if any(a)])
    singleton=[int(np.flatnonzero((codes==np.eye(4,dtype=int)[j]).all(1))[0]) for j in range(4)]
    def unpack(f):
        mask=((1-f[:,7+np.array(singleton)])/2).astype(bool)
        assert np.isin(f[:,7:22],[-1,1]).all()
        assert np.array_equal(1-2*((mask.astype(int)@codes.T)%2),f[:,7:22])
        return f[:,3:7].copy(),mask,f[:,-1].copy()
    def ids(a):return set(map(tuple,a.tolist()))
    base=ROOT/'artifacts/runs/large_native_a100_v3/results'
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():continue
        with np.load(folder/'boundaries_and_meta.npz') as boundary:
            sx,sm,sz=unpack(boundary['source_f']);sy=boundary['source_y'].copy();sid=boundary['source_context_ids'].copy()
            source=ids(boundary['source_fit_ids'])|ids(sid)
            for file in sorted((folder/'predictions').glob('development*.npz')):
                with np.load(file) as a:
                    tx,tm,tz=unpack(a['support_f']);qx,qm,qz=unpack(a['query_f'])
                    support,query=ids(a['support_ids']),ids(a['query_ids'])
                    assert not source&support and not source&query and not support&query
                    e=dict(source_x=sx,source_mask=sm,source_logit=sz,source_y=sy,source_ids=sid,
                           target_x=tx,target_mask=tm,target_logit=tz,target_y=a['support_y'].copy(),target_ids=a['support_ids'].copy(),
                           query_x=qx,query_mask=qm,query_logit=qz,query_y=a['labels'].copy(),query_ids=a['query_ids'].copy())
                    task,seed=folder.name.rsplit('_seed',1)
                    extra={k:a[k].copy() for k in ('native_linear_converged','global_platt','global_simplex','support_platt')}
                    yield task,int(seed),file.stem,e,extra,run.v1.checksum(file),run.v1.checksum(folder/'boundaries_and_meta.npz')

def main():
    torch.set_num_threads(2);start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'predictions').mkdir();plan=json.loads((ROOT/'artifacts/reports/crossfit_v1_gpu/report.json').read_text())['plan']
    models,hashes=load_models(plan);run.v1.target_platt=stable_platt
    seeds=list(range(700001,700021));rows=[]
    for seed in seeds:
        for width in (6,10):
            for e in run.v1.world(seed,width,plan):
                row=evaluate(e,models,plan,OUT/'predictions'/f'synthetic_{seed}_{width}_{e["regime"]}.npz')
                rows.append(dict(row,seed=seed,width=width,regime=e['regime']))
    contrasts=summary(rows,seeds)
    gate=all(contrasts['sign_flip'][k]['gain']>=.003 and contrasts['sign_flip'][k]['lower97_5']>0 for k in ('task_scalar','support_head'))
    robust=contrasts['sign_flip']['frozen']['gain']>=.003 and contrasts['sign_flip']['frozen']['lower95']>0 and all(-contrasts[k]['frozen']['lower95']<=.001 for k in ('no_shift','ignorable_shift'))
    result=dict(scope='One fixed independent development replication, not confirmation',checkpoint_hashes=hashes,rows=rows,contrasts=contrasts,architecture_gate=gate,robustness_gate=robust)
    (OUT/'synthetic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(synthetic_architecture_gate=gate,synthetic_robustness_gate=robust,sign_flip=contrasts['sign_flip'])),flush=True)
    native=[]
    for task,seed,group,e,extra,original_hash,boundary_hash in native_episodes():
        row=evaluate(e,models,plan,OUT/'predictions'/f'native_{task}_{seed}_{group}.npz')
        row['metrics'].update({k:run.v1.metrics(v,e['query_y']) for k,v in extra.items()})
        native.append(dict(row,task=task,seed=seed,group=group,input_sha256=original_hash,boundary_sha256=boundary_hash,queries=len(e['query_y'])))
    means={task:{k:float(np.mean([r['metrics'][k]['nll'] for r in native if r['task']==task])) for k in native[0]['metrics']} for task in sorted({r['task'] for r in native})}
    native_gate=all(v['query']<v[k] for v in means.values() for k in ('frozen','guarded_logistic','support_head','task_scalar'))
    result=dict(scope='Used native development panels; three seeds share rows/groups, BRFSS tasks share survey; no independent dataset-level interval',rows=native,means=means,transfer_gate=native_gate,seconds=time.monotonic()-start,checkpoint_hashes=hashes)
    (OUT/'native.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(native_gate=native_gate,means=means,seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
