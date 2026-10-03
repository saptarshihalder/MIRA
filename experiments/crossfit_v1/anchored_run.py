"""One bounded CPU pilot of calibration-anchored trainable innovations."""
import json,time
import numpy as np
import torch
from scipy.stats import t
import extension as x
from anchored_model import AnchoredQuery,AnchoredSupport,AnchorOnly
from matched_control import equalize_modes

OUT=x.ROOT/'artifacts/reports/anchored_v1'

def make(name,plan):
    torch.manual_seed(plan['training_seed'])
    model=(AnchoredSupport if name=='support_head' else AnchorOnly if name=='anchor' else AnchoredQuery)('query')
    if name=='task_scalar':model.network.register_forward_pre_hook(equalize_modes)
    return model

def main():
    torch.set_num_threads(2);start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False);(OUT/'predictions').mkdir();x.OUT=OUT
    plan=json.loads((x.ROOT/'artifacts/reports/crossfit_v1_gpu/report.json').read_text())['plan'];x.run.v1.target_platt=x.stable_platt
    episodes=[e for seed in plan['train_seeds'] for width in plan['train_widths'] for e in x.run.v1.world(seed,width,plan,training=True)]
    cache={};helper=make('query',plan)
    for width in plan['train_widths']:
        rows=[e for e in episodes if e['width']==width]
        with torch.no_grad():parts=helper.prepare(x.run.v1.tensor_input(rows,'cpu'))
        cache[width]=(parts,torch.tensor(np.stack([e['query_y'] for e in rows]),dtype=torch.float64))
    models={};training={}
    for name in ('query','task_scalar','support_head'):
        m=make(name,plan);optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.003);rng=np.random.default_rng(plan['training_seed']);trace=[];gradient=0.
        for update in range(500):
            width=plan['train_widths'][int(rng.integers(len(plan['train_widths'])))];parts,y=cache[width];idx=torch.tensor(rng.choice(len(y),8,replace=False))
            result=m.predict([{k:v[idx] for k,v in p.items()} for p in parts]);loss=sum(torch.nn.functional.binary_cross_entropy(result[k].clamp(1e-8,1-1e-8),y[idx]) for k in ('soft','unguarded'))/2
            optimizer.zero_grad();loss.backward()
            for p in m.parameters():
                if p.grad is not None:assert torch.isfinite(p.grad).all();gradient=max(gradient,p.grad.abs().max().item())
            optimizer.step()
            if update%50==0 or update==499:trace.append(dict(update=update+1,loss=loss.item()))
        assert gradient>0
        file=OUT/(name+'_weights.npz');np.savez_compressed(file,**{k:v.detach().numpy() for k,v in m.state_dict().items()})
        training[name]=dict(updates=500,parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),trace=trace,max_gradient=gradient,checkpoint_sha256=x.run.v1.checksum(file))
        (OUT/'training.json').write_text(json.dumps(training,indent=2)+'\n');models[name]=m.eval()
    models['anchor']=make('anchor',plan).eval();rows=[];seeds=list(range(710001,710021))
    for seed in seeds:
        for width in (6,10):
            for e in x.run.v1.world(seed,width,plan):
                r=x.evaluate(e,models,plan,OUT/'predictions'/f'synthetic_{seed}_{width}_{e["regime"]}.npz');rows.append(dict(r,seed=seed,width=width,regime=e['regime']))
    contrasts=x.summary(rows,seeds)
    for regime,c in contrasts.items():
        for comparator,v in c.items():
            values=[np.mean([r['metrics'][comparator]['nll']-r['metrics']['query']['nll'] for r in rows if r['regime']==regime and r['seed']==seed]) for seed in seeds]
            radius=t.ppf(1-.05/6,19)*np.std(values,ddof=1)/np.sqrt(20);v['family_lower']=float(np.mean(values)-radius);v['family_upper']=float(np.mean(values)+radius)
    gate=all(contrasts['sign_flip'][k]['gain']>=.003 and contrasts['sign_flip'][k]['family_lower']>0 for k in ('task_scalar','support_head','anchor'))
    robust=contrasts['sign_flip']['frozen']['gain']>=.003 and contrasts['sign_flip']['frozen']['lower95']>0 and all(-contrasts[k]['frozen']['lower95']<=.001 for k in ('no_shift','ignorable_shift'))
    (OUT/'synthetic.json').write_text(json.dumps(dict(rows=rows,contrasts=contrasts,architecture_gate=gate,robustness_gate=robust),indent=2)+'\n')
    print(json.dumps(dict(architecture_gate=gate,robustness_gate=robust,sign_flip=contrasts['sign_flip'])),flush=True)
    native=[]
    for task,seed,group,e,extra,input_hash,boundary_hash in x.native_episodes():
        r=x.evaluate(e,models,plan,OUT/'predictions'/f'native_{task}_{seed}_{group}.npz');r['metrics'].update({k:x.run.v1.metrics(v,e['query_y']) for k,v in extra.items()});native.append(dict(r,task=task,seed=seed,group=group,input_sha256=input_hash,boundary_sha256=boundary_hash))
    means={task:{k:float(np.mean([r['metrics'][k]['nll'] for r in native if r['task']==task])) for k in native[0]['metrics']} for task in sorted({r['task'] for r in native})}
    gate=all(v['query']<v[k] for v in means.values() for k in ('frozen','task_scalar','support_head','anchor'))
    (OUT/'native.json').write_text(json.dumps(dict(rows=native,means=means,transfer_gate=gate,seconds=time.monotonic()-start),indent=2)+'\n');print(json.dumps(dict(native_gate=gate,means=means,seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
