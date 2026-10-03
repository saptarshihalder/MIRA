"""Fresh development for trained query-dependent corrections and fixed verification."""
import argparse,importlib.util,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from model import CrossfitOperator

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('frozen_generator',ROOT/'experiments/bridge_v1/run.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
NAMES=('query','no_query','scalar')

def make_model(name,plan,device):
    torch.manual_seed(plan['training_seed']);return CrossfitOperator(name).to(device)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--out');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    plan={**v1.DEFAULTS,**json.loads(Path(args.config).read_text())}
    v1.validate_plan(plan)
    if args.smoke:plan.update(train_seeds=[690001,690002],development_seeds=[690101],train_widths=[6],development_widths=[6],updates=2)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    if set(plan['train_seeds']+plan['development_seeds'])&set(range(98000,98020)):raise ValueError('Confirmation protected')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.monotonic()
    episodes=[e for seed in plan['train_seeds'] for width in plan['train_widths'] for e in v1.world(seed,width,plan,training=True)]
    helper=make_model('query',plan,args.device);cache={}
    for width in plan['train_widths']:
        rows=[e for e in episodes if e['width']==width]
        with torch.no_grad():parts=helper.prepare(v1.tensor_input(rows,args.device))
        cache[width]=(parts,torch.tensor(np.stack([e['query_y'] for e in rows]),device=args.device,dtype=torch.float64))
    models={};traces={}
    for name in NAMES:
        model=make_model(name,plan,args.device);optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate']);rng=np.random.default_rng(plan['training_seed']);history=[];gradient_max=0.
        for update in range(plan['updates']):
            width=plan['train_widths'][int(rng.integers(len(plan['train_widths'])))];parts,y=cache[width]
            index=torch.tensor(rng.choice(len(y),plan['batch_size'],replace=False),device=args.device)
            output=model.predict([{k:value[index] for k,value in p.items()} for p in parts])
            loss=sum(torch.nn.functional.binary_cross_entropy(output[k].clamp(1e-8,1-1e-8),y[index]) for k in ('soft','unguarded'))/2
            optimizer.zero_grad();loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite gradient')
                    gradient_max=max(gradient_max,float(p.grad.abs().max().detach().cpu()))
            optimizer.step()
            if update%50==0 or update+1==plan['updates']:history.append(dict(update=update+1,loss=float(loss.detach().cpu())))
        if not gradient_max>0:raise RuntimeError('No training gradient')
        np.savez_compressed(out/(name+'_weights.npz'),**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        traces[name]=dict(updates=plan['updates'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),max_gradient=gradient_max,trace=history);models[name]=model.eval()
    (out/'predictions').mkdir();results=[];max_increase=-float('inf')
    for seed in plan['development_seeds']:
        for width in plan['development_widths']:
            for e in v1.world(seed,width,plan):
                predictions,controls=v1.controls(e,plan);decisions={}
                inputs=v1.tensor_input([e],args.device)
                for name,model in models.items():
                    with torch.no_grad():output=model.predict(model.prepare(inputs),True)
                    predictions[name]=output['hard'][0].cpu().numpy();predictions[name+'_unguarded']=output['unguarded'][0].cpu().numpy()
                    decisions[name]=dict(accepted=output['accepted'][:,0].cpu().tolist(),gains=output['gains'][:,0].cpu().tolist(),ses=output['ses'][:,0].cpu().tolist())
                    max_increase=max(max_increase,max(float((x[1:]-x[:-1]).max().cpu()) for x in output['objectives']))
                # Same split/verification rule applied to an ordinary interaction-logistic control.
                paired=[];accepted=[]
                for parity in (0,1):
                    fit=np.arange(parity,len(e['target_y']),2);val=np.arange(1-parity,len(e['target_y']),2)
                    inner={**e,**{k:e[k][fit] for k in ('target_x','target_mask','target_y','target_logit')},**{k:np.concatenate((e['target_'+k[6:]][val],e[k])) for k in ('query_x','query_mask','query_logit')}}
                    inner_preds,_=v1.controls(inner,plan);prob=inner_preds['support_logistic'];n=len(val);label=e['target_y'][val]
                    base=np.logaddexp(0,e['target_logit'][val])-label*e['target_logit'][val]
                    vp=prob[:n].clip(1e-8,1-1e-8);gain=base+label*np.log(vp)+(1-label)*np.log1p(-vp)
                    accept=bool(gain.mean()>1.645*gain.std(ddof=1)/n**.5)
                    paired.append(prob[n:] if accept else predictions['frozen']);accepted.append(accept)
                predictions['guarded_logistic']=np.mean(paired,axis=0);decisions['guarded_logistic']=dict(accepted=accepted)
                np.savez_compressed(out/'predictions'/f"{seed}_{width}_{e['regime']}.npz",**predictions,labels=e['query_y'],**{k:e[k] for k in v1.INPUT_KEYS},**{k:e[k] for k in ('source_fit_ids','source_ids','target_ids','query_ids')})
                results.append(dict(seed=seed,width=width,regime=e['regime'],decisions=decisions,control_audits=controls,metrics={name:v1.metrics(p,e['query_y']) for name,p in predictions.items()}))
    if max_increase>1e-8:raise RuntimeError('Inner support descent violation')
    means={regime:{name:float(np.mean([r['metrics'][name]['nll'] for r in results if r['regime']==regime])) for name in results[0]['metrics']} for regime in v1.REGIMES}
    report=dict(plan=plan,models=traces,results=results,means=means,scope='Fresh adaptive development; guard is nominal heuristic, not a distribution-free risk guarantee',device=args.device,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,max_support_objective_increase=max_increase,seconds=time.monotonic()-start,protocol_sha256=v1.checksum(args.config),model_sha256=v1.checksum(Path(__file__).with_name('model.py')),runner_sha256=v1.checksum(__file__))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(means=means,gpu=report['gpu'],seconds=report['seconds'])))

if __name__=='__main__':main()
