"""Colab/local-GPU mechanism pilot. NOT executed in the authoring runtime.

Paired versions, NaN-native vs explicit indicators and shuffled-indicator control.
The known posterior is from the generator, never a TFM posterior times P(M|Y).
No neural weights are trained. XGBoost does fit trees on each labeled context.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from headroom import make_task, sample, expected_nll

def transform_pair(c, q, mode, seed):
    xc=np.column_stack([c['u'],c['x']]); xq=np.column_stack([q['u'],q['x']])
    if mode.startswith('imputed'):
        # Training-only means, deterministic zero for columns empty in context.
        observed=np.isfinite(xc); count=observed.sum(axis=0)
        means=np.divide(np.nansum(xc,axis=0), count, out=np.zeros(xc.shape[1]), where=count>0)
        xc=np.where(np.isnan(xc),means,xc); xq=np.where(np.isnan(xq),means,xq)
    if mode.endswith('indicators'):
        xc=np.column_stack([xc,c['m']]); xq=np.column_stack([xq,q['m']])
    elif mode=='native_shuffled':
        rng=np.random.default_rng(seed)
        # Independently permute each column within context/query; never query labels.
        mc=np.column_stack([rng.permutation(v) for v in c['m'].T])
        mq=np.column_stack([rng.permutation(v) for v in q['m'].T])
        xc=np.column_stack([xc,mc]); xq=np.column_stack([xq,mq])
    return xc.astype(np.float32),xq.astype(np.float32)

def factory(name,args,seed):
    if name.startswith('tabpfn:'):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        version=ModelVersion(name.split(':',1)[1])
        return TabPFNClassifier.create_default_for_version(version,device=args.device,
            n_estimators=args.ensembles,random_state=seed)
    if name=='tabicl:v2':
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=args.device,n_estimators=args.ensembles,
            checkpoint_version='tabicl-classifier-v2-20260212.ckpt',random_state=seed)
    if name=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,
            reg_lambda=1.,tree_method='hist',random_state=seed,n_jobs=2,
            eval_metric='logloss')
    raise ValueError(f'Unsupported model: {name}; no silent version fallback.')

def main(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; errors=[]; params={}; blocked_models=set()
    versions={}
    for package in ['numpy','scipy','scikit-learn','torch','tabpfn','tabicl','xgboost']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='not installed'
    (out/'environment.json').write_text(json.dumps(dict(args=vars(args),packages=versions,
        python=platform.python_version(),status='running'),indent=2))
    try:
        freeze=subprocess.run(['python','-m','pip','freeze'],capture_output=True,text=True,check=True)
        (out/'requirements-lock.txt').write_text(freeze.stdout)
    except Exception as e:
        errors.append(dict(stage='pip_freeze',error=repr(e)))
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        for gamma in args.gammas:
            rng=np.random.default_rng(seed)
            task=make_task('label_only',rng,gamma=gamma,beta=.8)
            # Embed one informative mask in 8 columns. Family metadata deliberately
            # stays label_only; no method receives the active feature identity.
            task['d']=8; task['active']=(int(rng.integers(8)),)
            c=sample(task,args.context,rng); q=sample(task,args.queries,rng)
            np.savez_compressed(out/f'data_seed{seed}_g{gamma:g}.npz',
                xc=c['x'],uc=c['u'],mc=c['m'],yc=c['y'],xq=q['x'],uq=q['u'],
                mq=q['m'],yq=q['y'],oracle=q['oracle'],base=q['p0'])
            base_loss=expected_nll(q['p0'],q['oracle']); oracle_loss=expected_nll(q['oracle'],q['oracle'])
            for name in args.models:
                if name in blocked_models: continue
                for mode in args.modes:
                    started=time.time(); model=None
                    try:
                        xc,xq=transform_pair(c,q,mode,seed+731)
                        model=factory(name,args,seed)
                        params[name]={k:repr(v) for k,v in model.get_params(deep=False).items()}
                        model.fit(xc,c['y']); p=model.predict_proba(xq)[:,1]
                        if len(p)!=len(q['y']) or not np.isfinite(p).all():
                            raise ValueError('Invalid probability output')
                        loss=expected_nll(p,q['oracle'])
                        empirical=float(np.mean(-q['y']*np.log(np.clip(p,1e-6,1-1e-6))
                            -(1-q['y'])*np.log1p(-np.clip(p,1e-6,1-1e-6))))
                        gap=base_loss-oracle_loss
                        row=dict(seed=seed,gamma=gamma,model=name,mode=mode,
                            expected_nll=loss,empirical_nll=empirical,analytic_base_nll=base_loss,
                            oracle_nll=oracle_loss,
                            fraction_analytic_gain=(base_loss-loss)/gap if gap>1e-4 else None,
                            seconds=time.time()-started)
                        rows.append(row)
                        key=name.replace(':','_').replace('.','_')+'_'+mode
                        np.savez_compressed(out/f'pred_{key}_seed{seed}_g{gamma:g}.npz',probability=p)
                        print(row,flush=True)
                    except Exception as e:
                        entry=dict(seed=seed,gamma=gamma,model=name,mode=mode,error=repr(e))
                        errors.append(entry); blocked_models.add(name)
                        print('FAILED; remaining runs for this model will be skipped',entry,flush=True)
                    finally:
                        del model; gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                        except ImportError: pass
                    # Write incrementally so a Colab timeout preserves completed work.
                    if rows:
                        with (out/'results.csv').open('w',newline='') as f:
                            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                    (out/'errors.json').write_text(json.dumps(errors,indent=2))
                    (out/'model_parameters.json').write_text(json.dumps(params,indent=2))
                    if name in blocked_models: break
    # Paired native-to-indicator comparison; keep negative gains, no clipping.
    paired=[]
    for seed in range(args.seed_start,args.seed_start+args.seeds):
        for gamma in args.gammas:
            for name in args.models:
                cell={r['mode']:r for r in rows if r['seed']==seed and r['gamma']==gamma and r['model']==name}
                if 'native' in cell and 'native_indicators' in cell:
                    a,b=cell['native'],cell['native_indicators']; headroom=a['expected_nll']-a['oracle_nll']
                    gain=a['expected_nll']-b['expected_nll']
                    paired.append(dict(seed=seed,gamma=gamma,model=name,gain_nats=gain,
                        native_to_oracle_gap=headroom,
                        fraction_native_gap=gain/headroom if headroom>1e-4 else None))
    (out/'paired_comparisons.json').write_text(json.dumps(paired,indent=2))
    # Hash all materialized checkpoints under explicitly supplied cache directories.
    hashes={}
    for cache in args.checkpoint_dirs:
        for path in Path(cache).rglob('*'):
            if path.is_file() and path.suffix in ['.ckpt','.pt','.pth','.safetensors']:
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(2**20),b''):h.update(block)
                hashes[str(path)]=h.hexdigest()
    (out/'checkpoint_sha256.json').write_text(json.dumps(hashes,indent=2))
    (out/'environment.json').write_text(json.dumps(dict(args=vars(args),packages=versions,
        python=platform.python_version(),completed=len(rows),failed=len(errors),
        status='complete' if not errors else 'incomplete; inspect errors.json'),indent=2))
    if errors: raise SystemExit('One or more runs failed; inspect errors.json. No substitute model was used.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',default=['tabpfn:v2','tabicl:v2','xgboost'])
    p.add_argument('--modes',nargs='+',choices=['native','native_indicators','native_shuffled','imputed','imputed_indicators'],
        default=['native','native_indicators','native_shuffled'])
    p.add_argument('--gammas',type=float,nargs='+',default=[0.,.25,.5,.75,.9])
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--seed-start',type=int,default=40000)
    p.add_argument('--context',type=int,default=256);p.add_argument('--queries',type=int,default=1024)
    p.add_argument('--ensembles',type=int,default=4);p.add_argument('--device',default='cuda')
    p.add_argument('--checkpoint-dirs',nargs='*',default=[])
    p.add_argument('--out',default='tfm_results')
    main(p.parse_args())
