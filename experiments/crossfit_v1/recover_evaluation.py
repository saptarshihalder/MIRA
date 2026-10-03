"""CPU-only completion from saved CUDA checkpoints; no retraining or cloud calls."""
import json,shutil,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
import run

ROOT=run.ROOT

def stable_platt(episode,ridge):
    z=episode['target_logit'].astype(float);y=episode['target_y'].astype(float)
    a=np.column_stack((z,np.ones(len(z))));w=np.zeros(2)
    def objective(w):
        score=z+a@w
        return np.mean(np.logaddexp(0,score)-y*score)+.5*ridge*(w@w)
    for iteration in range(100):
        p=expit(z+a@w);g=a.T@(p-y)/len(y)+ridge*w
        if np.max(np.abs(g))<1e-10:break
        h=a.T@(a*(p*(1-p))[:,None])/len(y)+ridge*np.eye(2)
        direction=np.linalg.solve(h,g);before=objective(w);alpha=1.
        for _ in range(30):
            candidate=w-alpha*direction
            if objective(candidate)<=before-1e-4*alpha*(g@direction)+1e-14:break
            alpha*=.5
        w=candidate
    g=a.T@(expit(z+a@w)-y)/len(y)+ridge*w
    if np.max(np.abs(g))>=1e-8:raise RuntimeError('Platt gradient certificate failed')
    return expit((1+w[0])*episode['query_logit']+w[1]),dict(success=True,method='damped exact Newton, same objective/ridge',gradient_max=float(np.abs(g).max()),objective_gap_upper=float(g@g/(2*ridge)),iterations=iteration+1,weights=w.tolist(),ridge=ridge)

def main():
    torch.set_num_threads(2);started=time.monotonic()
    config=ROOT/'configs/crossfit_v1.json';plan={**run.v1.DEFAULTS,**json.loads(config.read_text())}
    for name,digest in plan['file_sha256'].items():assert run.v1.checksum(ROOT/name)==digest,name
    source=ROOT/'artifacts/runs/crossfit_v1_gpu/results';out=ROOT/'artifacts/runs/crossfit_v1_recovered';out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    run.v1.target_platt=stable_platt
    models={};traces={}
    for name in run.NAMES:
        model=run.make_model(name,plan,'cpu')
        initial={k:v.clone() for k,v in model.state_dict().items()}
        checkpoint=source/(name+'_weights.npz');shutil.copy2(checkpoint,out/checkpoint.name)
        with np.load(checkpoint,allow_pickle=False) as state:model.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files})
        changed=sum(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
        assert changed>0
        traces[name]=dict(updates=500,update_evidence='Frozen worker command has no smoke flag; checkpoint written only after500updates and nonzero-gradient assertion. Per-update trace lost at late evaluation abort.',max_gradient=None,nonzero_gradient_assertion_passed=True,changed_state_arrays=changed,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checkpoint_sha256=run.v1.checksum(checkpoint));models[name]=model.eval()
    results=[];max_increase=-float('inf');original_files=0;max_old_difference=0.;max_neural_difference=0.;max_platt_gradient=0.
    for seed in plan['development_seeds']:
        for width in plan['development_widths']:
            for e in run.v1.world(seed,width,plan):
                predictions,controls=run.v1.controls(e,plan);decisions={};max_platt_gradient=max(max_platt_gradient,controls['target_platt']['gradient_max'])
                inputs=run.v1.tensor_input([e],'cpu')
                for name,model in models.items():
                    with torch.no_grad():output=model.predict(model.prepare(inputs),True)
                    predictions[name]=output['hard'][0].numpy();predictions[name+'_unguarded']=output['unguarded'][0].numpy()
                    decisions[name]=dict(accepted=output['accepted'][:,0].tolist(),gains=output['gains'][:,0].tolist(),ses=output['ses'][:,0].tolist())
                    max_increase=max(max_increase,max(float((o[1:]-o[:-1]).max()) for o in output['objectives']))
                paired=[];accepted=[]
                for parity in (0,1):
                    fit=np.arange(parity,len(e['target_y']),2);val=np.arange(1-parity,len(e['target_y']),2)
                    inner={**e,**{k:e[k][fit] for k in ('target_x','target_mask','target_y','target_logit')},**{k:np.concatenate((e['target_'+k[6:]][val],e[k])) for k in ('query_x','query_mask','query_logit')}}
                    pp,ca=run.v1.controls(inner,plan);assert ca['support_logistic'].get('converged',True)
                    p=pp['support_logistic'];n=len(val);y=e['target_y'][val];z=e['target_logit'][val]
                    v=p[:n].clip(1e-8,1-1e-8);gain=np.logaddexp(0,z)-y*z+y*np.log(v)+(1-y)*np.log1p(-v)
                    accept=bool(gain.mean()>1.645*gain.std(ddof=1)/n**.5)
                    paired.append(p[n:] if accept else predictions['frozen']);accepted.append(accept)
                predictions['guarded_logistic']=np.mean(paired,axis=0);decisions['guarded_logistic']=dict(accepted=accepted)
                filename=f"{seed}_{width}_{e['regime']}.npz";old=source/'predictions'/filename
                if old.exists():
                    with np.load(old) as saved:
                        for name,p in predictions.items():
                            difference=float(np.abs(p-saved[name]).max());max_old_difference=max(max_old_difference,difference)
                            if name in (*run.NAMES,*(n+'_unguarded' for n in run.NAMES)):max_neural_difference=max(max_neural_difference,difference)
                    original_files+=1
                np.savez_compressed(out/'predictions'/filename,**predictions,labels=e['query_y'],**{k:e[k] for k in run.v1.INPUT_KEYS},**{k:e[k] for k in ('source_fit_ids','source_ids','target_ids','query_ids')})
                results.append(dict(seed=seed,width=width,regime=e['regime'],decisions=decisions,control_audits=controls,metrics={name:run.v1.metrics(p,e['query_y']) for name,p in predictions.items()}))
    assert original_files==132 and max_neural_difference<1e-8 and max_old_difference<1e-5 and max_increase<=1e-8
    means={regime:{name:float(np.mean([r['metrics'][name]['nll'] for r in results if r['regime']==regime])) for name in results[0]['metrics']} for regime in run.v1.REGIMES}
    report=dict(plan=plan,models=traces,results=results,means=means,device='cuda',evaluation_device='cpu_recovery',gpu='Modal A100-40GB provision; CUDA enforced by frozen worker; exact device-name log lost at evaluation abort',max_support_objective_increase=max_increase,seconds=time.monotonic()-started,protocol_sha256=run.v1.checksum(config),model_sha256=run.v1.checksum(Path(__file__).with_name('model.py')),runner_sha256=run.v1.checksum(Path(__file__).with_name('run.py')),recovery=dict(failure='Target Platt L-BFGS-B ABNORMAL after132prediction files',retrained=False,remote_retry=False,source_checkpoint_hashes={name:v['checkpoint_sha256'] for name,v in traces.items()},original_files_replayed=original_files,max_old_probability_difference=max_old_difference,max_gpu_cpu_neural_difference=max_neural_difference,max_platt_gradient=max_platt_gradient,repair='Same convex Platt objective/ridge, damped Newton with gradient certificate; all160cells retained; no model/hyperparameter/seed selection'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(means=means,recovery=report['recovery'],seconds=report['seconds'])))

if __name__=='__main__':main()
