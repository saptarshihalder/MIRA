"""One prespecified diagnostic pilot; no confirmation and no best-epoch selection."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import argparse
import numpy as np
import torch
from model import SpectralFilter

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('frozen_world',ROOT/'experiments/bridge_v1/run.py')
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--out');parser.add_argument('--device',default='cuda');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();plan=json.loads(Path(args.config).read_text());plan={**v1.DEFAULTS,**plan}
    if args.smoke:plan.update(train_seeds=[590001,590002],development_seeds=[590101],train_widths=[6],development_widths=[6],updates=2)
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('Real CUDA required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.monotonic()
    training=[e for seed in plan['train_seeds'] for width in plan['train_widths'] for e in v1.world(seed,width,plan,training=True)]
    models={};traces={}
    for kind in ('learned','global','full'):
        torch.manual_seed(plan['training_seed']);model=SpectralFilter(kind).to(args.device)
        cache={}
        for width in plan['train_widths']:
            rows=[e for e in training if e['width']==width]
            with torch.no_grad():prepared=model.prepare(v1.tensor_input(rows,args.device))
            cache[width]=(prepared,torch.tensor(np.stack([e['query_y'] for e in rows]),device=args.device,dtype=torch.float64))
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.003) if kind!='full' else None
        rng=np.random.default_rng(plan['training_seed']);history=[]
        for update in range(plan['updates'] if optimizer else 0):
            width=plan['train_widths'][int(rng.integers(len(plan['train_widths'])))];p,y=cache[width]
            index=torch.tensor(rng.choice(len(y),size=8,replace=False),device=args.device)
            logits=model.solve({k:x[index] for k,x in p.items()})
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[index])
            optimizer.zero_grad();loss.backward();optimizer.step()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            if update%50==0 or update==plan['updates']-1:history.append(dict(update=update+1,loss=float(loss.detach().cpu())))
        np.savez_compressed(out/(kind+'_weights.npz'),**{k:x.detach().cpu().numpy() for k,x in model.state_dict().items()})
        models[kind]=model.eval();traces[kind]=dict(updates=plan['updates'] if optimizer else 0,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),trace=history)
    (out/'predictions').mkdir();rows=[];largest_increase=-float('inf')
    for seed in plan['development_seeds']:
        for width in plan['development_widths']:
            for e in v1.world(seed,width,plan):
                predictions,audit=v1.controls(e,plan)
                for kind,model in models.items():
                    with torch.no_grad():logits,objectives=model.solve(model.prepare(v1.tensor_input([e],args.device)),True)
                    largest_increase=max(largest_increase,float((objectives[1:]-objectives[:-1]).max().cpu()))
                    predictions[kind]=logits.sigmoid()[0].cpu().numpy()
                np.savez_compressed(out/'predictions'/f"{seed}_{width}_{e['regime']}.npz",**predictions,labels=e['query_y'],**{k:e[k] for k in v1.INPUT_KEYS})
                rows.append(dict(seed=seed,width=width,regime=e['regime'],control_audits=audit,metrics={k:v1.metrics(p,e['query_y']) for k,p in predictions.items()}))
    means={regime:{k:float(np.mean([r['metrics'][k]['nll'] for r in rows if r['regime']==regime])) for k in rows[0]['metrics']} for regime in v1.REGIMES}
    if largest_increase>1e-8:raise ValueError('Support objective descent violated')
    report=dict(scope='Fresh synthetic development pilot; no confirmation, no universal safety or novelty claim',gate_pass=None,plan=plan,training=traces,means=means,rows=rows,max_support_objective_increase=largest_increase,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,device=args.device,seconds=time.monotonic()-start,protocol_sha256=v1.checksum(args.config),model_sha256=v1.checksum(Path(__file__).with_name('model.py')),source_sha256=v1.checksum(__file__))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(gpu=report['gpu'],means=means,max_support_objective_increase=largest_increase,seconds=report['seconds'])))


if __name__=='__main__':main()
