"""One user-authorized CUDA engineering/native-training run; no confirmation claim."""
import argparse
import json
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import torch
from torch import nn
from xgboost import XGBClassifier
import xgboost
import native as n
import mixture as m
import pilot as p

ROOT=p.ROOT


def prediction(model,task):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(task['context'],device='cuda')[None],
                     torch.as_tensor(task['probability'],device='cuda')[None])[0].cpu().numpy().astype(float)


def execute(plan,out):
    start=time.monotonic()
    if not torch.cuda.is_available(): raise RuntimeError('Real CUDA is required')
    torch.set_num_threads(2)
    raw=ROOT/'artifacts/runs/diabetes_native_raw/diabetic_data.csv'
    if p.checksum(raw)!=plan['raw_csv_sha256']: raise ValueError('Raw source hash mismatch')
    rows,columns=n.load(raw)
    count=len(rows)
    source=np.arange(int(count*.6))
    validation=np.arange(int(count*.6),int(count*.8))
    development=np.arange(int(count*.8),count)
    fit=source[:plan['backbone_labels']]
    context=source[plan['backbone_labels']:plan['backbone_labels']+plan['source_labels']]
    x,categorical=n.encode(rows,columns,fit)
    y=np.array([int(row['readmitted']=='<30') for row in rows])
    backbone=XGBClassifier(device='cuda',tree_method='hist',n_estimators=100,max_depth=4,learning_rate=.05,
        reg_lambda=1.,n_jobs=2,random_state=1951,enable_categorical=True,feature_types=['c' if flag else 'q' for flag in categorical])
    backbone.fit(x[fit],y[fit])
    config=json.loads(backbone.get_booster().save_config())
    if not config['learner']['generic_param']['device'].startswith('cuda'): raise RuntimeError('XGBoost did not train on CUDA')
    backbone.save_model(out/'source_backbone.json')
    missing=np.isnan(x[fit]).mean(0)
    selected=sorted([j for j in range(len(columns)) if 0<missing[j]<1],key=lambda j:(-missing[j],columns[j]))[:4]
    mean=np.nanmean(x[fit][:,selected],axis=0)
    scale=np.maximum(np.nanstd(x[fit][:,selected],axis=0),1e-8)
    source_f=n.correction_features(x[context],selected,backbone.predict_proba(x[context])[:,1],mean,scale)
    def tasks(pool,episodes):
        width=plan['target_labels']+plan['queries']
        if len(pool)<width*episodes: raise ValueError('Insufficient disjoint patient rows')
        result=[]
        for episode in range(episodes):
            ids=pool[episode*width:(episode+1)*width]
            support,query=ids[:plan['target_labels']],ids[plan['target_labels']:]
            f=n.correction_features(x[ids],selected,backbone.predict_proba(x[ids])[:,1],mean,scale)
            e=dict(source_f=source_f,source_y=y[context],support_f=f[:plan['target_labels']],support_y=y[support],
                query_f=f[plan['target_labels']:],query_y=y[query],oracle=None,seed=episode,regime='native')
            task=m.prepare(e,plan)
            task.update(support_ids=support,query_ids=query,source_ids=context,support_f=e['support_f'],support_y=e['support_y'],query_f=e['query_f'])
            result.append(task)
        return result
    training=tasks(source[plan['backbone_labels']+plan['source_labels']:],plan['train_episodes'])
    val=tasks(validation,plan['validation_episodes'])
    dev=tasks(development,plan['development_episodes'])
    pretrained=ROOT/'artifacts/reports/policy_mixture_v1/neural_mixture.npz'
    if p.checksum(pretrained)!=plan['pretrained_sha256']: raise ValueError('Pretrained checkpoint mismatch')
    tensors=[torch.as_tensor(np.stack([e[key] for e in training]),device='cuda') for key in ('context','probability','labels')]
    models={}
    traces={}
    for name in ('pretrained_fixed','native_finetuned','native_scratch','native_linear'):
        torch.manual_seed(plan['training_seed'])
        model=m.Mixture(linear=name=='native_linear').cuda()
        if name in ('pretrained_fixed','native_finetuned'):
            with np.load(pretrained) as state:
                model.load_state_dict({k:torch.from_numpy(state[k].copy()).cuda() for k in state.files})
        history=[]
        updates=0
        if name!='pretrained_fixed':
            optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
            rng=np.random.default_rng(plan['training_seed'])
            for epoch in range(plan['epochs']):
                order=rng.permutation(len(training))
                loss_values=[]
                for step in range(0,len(order),plan['batch_size']):
                    index=torch.as_tensor(order[step:step+plan['batch_size']],device='cuda')
                    probability=model(tensors[0][index],tensors[1][index]).clamp(1e-7,1-1e-7)
                    loss=nn.functional.binary_cross_entropy(probability,tensors[2][index])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_values.append(float(loss.detach().cpu()))
                    updates+=1
                history.append(float(np.mean(loss_values)))
        torch.cuda.synchronize()
        np.savez_compressed(out/(name+'_weights.npz'),**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        traces[name]=dict(parameters=sum(t.numel() for t in model.parameters()),parameter_device=str(next(model.parameters()).device),
            updates=updates,train_nll=history,validation_nll=float(np.mean([p.bce(prediction(model,e),e['labels']) for e in val])),
            checkpoint_sha256=p.checksum(out/(name+'_weights.npz')))
        models[name]=model
    # Guard receives native validation labels too. No development-label model or guard selection.
    grid=plan['guard_grid']
    guard_scores={str(threshold):float(np.mean([p.bce(e['probability'][:,e['moment_choice']] if e['moment_cv_gain']>threshold else e['probability'][:,0],e['labels']) for e in val])) for threshold in grid}
    threshold=float(min(guard_scores,key=guard_scores.get))
    results=[]
    (out/'predictions').mkdir()
    for e in dev:
        predictions={name:prediction(model,e) for name,model in models.items()}
        predictions.update(frozen=e['probability'][:,0],moment=e['probability'][:,e['moment_choice']],
            cv_select=e['probability'][:,e['cv_choice']],
            native_guard=e['probability'][:,e['moment_choice']] if e['moment_cv_gain']>threshold else e['probability'][:,0])
        np.savez_compressed(out/'predictions'/f"episode{e['seed']:02}.npz",**predictions,labels=e['labels'],
            context=e['context'],experts=e['probability'],support_ids=e['support_ids'],query_ids=e['query_ids'],source_ids=context,
            support_f=e['support_f'],support_y=e['support_y'],query_f=e['query_f'],source_f=source_f,source_y=y[context])
        results.append(dict(episode=e['seed'],nll={name:p.bce(prob,e['labels']) for name,prob in predictions.items()}))
    primary='native_finetuned'
    gains={name:p.interval([r['nll'][name]-r['nll'][primary] for r in results]) for name in results[0]['nll'] if name!=primary}
    report=dict(protocol_sha256=p.checksum(ROOT/'configs/policy_gpu_native_v1.json'),source_sha256=p.checksum(__file__),
        raw_csv_sha256=p.checksum(raw),pretrained_sha256=p.checksum(pretrained),
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,xgboost=xgboost.__version__,python=platform.python_version(),
        source_backbone_device=config['learner']['generic_param']['device'],source_backbone_sha256=p.checksum(out/'source_backbone.json'),
        device='cuda',training_models=traces,native_patient_rows=count,selected_columns=[columns[j] for j in selected],
        native_train_episodes=len(training),native_validation_episodes=len(val),native_development_episodes=len(dev),
        guard_threshold=threshold,guard_validation_scores=guard_scores,gains=gains,results=results,seconds=time.monotonic()-start,
        patient_boundaries_disjoint=True,source_fit_ids=fit.tolist(),source_context_ids=context.tolist(),
        native_training_ids=np.concatenate([np.r_[e['support_ids'],e['query_ids']] for e in training]).tolist(),
        native_validation_ids=np.concatenate([np.r_[e['support_ids'],e['query_ids']] for e in val]).tolist(),
        scope='One-dataset retrospective CUDA engineering/development; source backbone and three correction models trained on real GPU; no confirmation, identified policy shift, safety, clinical or venue-readiness claim')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gpu=report['gpu'],seconds=report['seconds'],gains=gains)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists(): raise ValueError('Preserve old output; use a fresh child directory')
    out.mkdir(parents=True)
    plan=json.loads((ROOT/'configs/policy_gpu_native_v1.json').read_text())
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try: execute(plan,out)
        finally:
            (out/'warnings.json').write_text(json.dumps([str(w.message) for w in recorded],indent=2))
