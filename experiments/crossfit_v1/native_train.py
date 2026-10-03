"""Finite native family-transfer meta-training, with matched capacity and costs."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import extension as x
from support_head import SupportHead
from matched_control import equalize_modes

ROOT=x.ROOT;INPUT=ROOT/'artifacts/reports/native_meta_inputs';OUT=ROOT/'artifacts/reports/native_meta_v1'
SEEDS=(760001,760002,760003);NAMES=('query','task_scalar','support_head')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def family(task):return 'brfss' if task.startswith('brfss') else 'road'
def model(name,seed):
    torch.manual_seed(seed);m=SupportHead('query') if name=='support_head' else x.run.make_model('query',dict(training_seed=seed),'cpu')
    if name=='task_scalar':m.network.register_forward_pre_hook(equalize_modes)
    return m
def read(p):
    with np.load(p) as a:return {k:a[k].copy() for k in a.files}

def main():
    torch.set_num_threads(2);started=time.monotonic();OUT.mkdir(parents=True,exist_ok=False);(OUT/'predictions').mkdir();x.OUT=OUT;x.run.v1.target_platt=x.stable_platt
    freeze=json.loads((ROOT/'artifacts/manifests/native_meta_freeze.json').read_text())
    for f,h in freeze.items():assert sha(ROOT/f)==h,f
    plan=json.loads((ROOT/'artifacts/reports/crossfit_v1_gpu/report.json').read_text())['plan']
    files=sorted((INPUT/'training').glob('*.npz'));development=sorted((INPUT/'development').glob('*.npz'));traces={};rows=[]
    for source in ('brfss','road'):
        chosen=[p for p in files if family(p.stem)==source];episodes=[read(p) for p in chosen]
        assert len(chosen)==(64 if source=='brfss' else 32)
        assert sum(family(p.stem)!=source for p in development)==(6 if source=='brfss' else 20)
        with torch.no_grad():parts=model('query',SEEDS[0]).prepare(x.run.v1.tensor_input(episodes,'cpu'))
        labels=torch.tensor(np.stack([e['query_y'] for e in episodes]),dtype=torch.float64)
        for seed in SEEDS:
            models={}
            for name in NAMES:
                m=model(name,seed);assert sum(p.numel() for p in m.parameters() if p.requires_grad)==370
                optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.003);rng=np.random.default_rng(seed);history=[];gradient=0.
                for update in range(500):
                    idx=torch.tensor(rng.choice(len(episodes),8,replace=False));r=m.predict([{k:v[idx] for k,v in p.items()} for p in parts])
                    loss=sum(torch.nn.functional.binary_cross_entropy(r[k].clamp(1e-8,1-1e-8),labels[idx]) for k in ('soft','unguarded'))/2
                    optimizer.zero_grad();loss.backward()
                    for p in m.parameters():
                        if p.grad is not None:assert torch.isfinite(p.grad).all();gradient=max(gradient,p.grad.abs().max().item())
                    optimizer.step()
                    if update%50==0 or update==499:history.append(dict(update=update+1,loss=loss.item()))
                assert gradient>0
                key=f'{source}_{seed}_{name}';file=OUT/(key+'_weights.npz');np.savez_compressed(file,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
                traces[key]=dict(source_family=source,initialization=seed,variant=name,updates=500,parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),training_episodes=len(episodes),training_input_files=[str(p.relative_to(ROOT)).replace('\\','/') for p in chosen],max_gradient=gradient,trace=history,checkpoint_sha256=sha(file))
                (OUT/'training.json').write_text(json.dumps(traces,indent=2)+'\n');models[name]=m.eval()
            for p in development:
                if family(p.stem)==source:continue
                task,group=p.stem.split('_development_');assert family(task)!=source
                e=read(p);r=x.evaluate(e,models,plan,OUT/'predictions'/f'{source}_{seed}_{p.stem}.npz')
                old=ROOT/f'artifacts/runs/large_native_a100_v3/results/{task}_seed171001/predictions/development_{group}.npz'
                with np.load(old) as a:
                    assert np.array_equal(a['labels'],e['query_y']) and np.array_equal(a['query_ids'],e['query_ids'])
                    extra={k:a[k].copy() for k in ('native_linear_converged','global_platt','global_simplex')}
                r['metrics'].update({k:x.run.v1.metrics(v,e['query_y']) for k,v in extra.items()})
                output=OUT/r['prediction_file'];saved=read(output);np.savez_compressed(output,**saved,**extra)
                rows.append(dict(r,source_family=source,target_family=family(task),initialization=seed,task=task,group=group,input_file=str(p.relative_to(ROOT)).replace('\\','/'),input_sha256=sha(p)))
                (OUT/'partial_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(dict(source_family=source,initialization=seed,trained_models=3,evaluated_cells=sum(r['source_family']==source and r['initialization']==seed for r in rows))),flush=True)
    assert len(rows)==78
    means={task:{k:float(np.mean([r['metrics'][k]['nll'] for r in rows if r['task']==task])) for k in rows[0]['metrics']} for task in sorted({r['task'] for r in rows})}
    individual={task:{str(seed):float(np.mean([r['metrics']['query']['nll'] for r in rows if r['task']==task and r['initialization']==seed])) for seed in SEEDS} for task in means}
    gains={task:{k:v[k]-v['query'] for k in v} for task,v in means.items()}
    gate=all(v['query']<v[k] for v in means.values() for k in ('frozen','task_scalar','support_head','guarded_logistic','target_platt')) and all(gains[task][k]>=.001 for task in gains for k in ('frozen','task_scalar','support_head'))
    report=dict(scope='Two dataset-family meta-learner holdouts, USED development cohorts; target backbones/source context and support remain labeled',rows=rows,means=means,gains=gains,initialization_nll=individual,development_gate=gate,seconds=time.monotonic()-started,cloud_calls=0,device='cpu',freeze_sha256=sha(ROOT/'artifacts/manifests/native_meta_freeze.json'))
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(gate=gate,means=means,initialization_nll=individual,seconds=report['seconds'])),flush=True)

if __name__=='__main__':main()
