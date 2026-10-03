"""Frozen synthetic learner on a larger native dataset; retrospective development."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
import mixture as m
import pilot as p

ROOT=p.ROOT
NUMERIC={'time_in_hospital','num_lab_procedures','num_procedures','num_medications',
         'number_outpatient','number_emergency','number_inpatient','number_diagnoses'}


def key(text):
    return hashlib.sha256(('mira-native1901:'+text).encode()).hexdigest()


def load(path):
    with path.open(newline='',encoding='utf-8-sig') as source:
        reader=csv.DictReader(source)
        columns=[c for c in reader.fieldnames if c not in ('encounter_id','patient_nbr','readmitted')]
        patients={}
        for row in reader:
            if row['discharge_disposition_id'] in {'11','13','14','19','20','21'}:
                continue
            patient=row['patient_nbr']
            if patient not in patients or key(row['encounter_id'])<key(patients[patient]['encounter_id']):
                patients[patient]=row
    rows=sorted(patients.values(),key=lambda row:key(row['patient_nbr']))
    return rows,columns


def encode(rows,columns,training):
    x=np.full((len(rows),len(columns)),np.nan)
    category=[]
    for j,column in enumerate(columns):
        missing=lambda value:value in ('?','', 'NaN') or (column in ('max_glu_serum','A1Cresult') and value.lower()=='none')
        if column in NUMERIC:
            x[:,j]=[np.nan if missing(row[column]) else float(row[column]) for row in rows]
            category.append(False)
        else:
            counts={}
            for i in training:
                value=rows[i][column]
                if not missing(value): counts[value]=counts.get(value,0)+1
            vocabulary={value:k for k,value in enumerate(sorted(counts,key=lambda value:(-counts[value],value))[:126])}
            for i,row in enumerate(rows):
                if not missing(row[column]): x[i,j]=vocabulary.get(row[column],126)
            category.append(True)
    return x,category


def correction_features(x,selected,probability,mean,scale):
    mask=np.isnan(x[:,selected]).astype(int)
    z=logit(np.clip(probability,1e-6,1-1e-6))
    values=np.nan_to_num((x[:,selected]-mean)/scale,nan=0.)
    parity=1.-2.*((mask@p.CODES.T)%2)
    return np.column_stack((z,np.tanh(z),np.clip(z*z-1.,-3,5),values,parity,z)).astype(np.float32)


def run(csv_path,out):
    plan=json.loads((ROOT/'configs/policy_native_v1.json').read_text())
    if p.checksum(csv_path)!=plan['raw_csv_sha256']: raise ValueError('Official raw source changed')
    torch.set_num_threads(2)
    started=time.monotonic()
    rows,columns=load(csv_path)
    # Sorted by label-independent cryptographic patient key; IDs are never inputs.
    count=len(rows)
    training=np.arange(int(count*.6))
    development=np.arange(int(count*.8),count)
    fit=training[:plan['backbone_labels']]
    context=training[plan['backbone_labels']:plan['backbone_labels']+plan['source_labels']]
    need=plan['episodes']*(plan['target_labels']+plan['queries'])
    if len(development)<need: raise ValueError('Insufficient patient-disjoint development rows')
    x,categorical=encode(rows,columns,fit)
    y=np.array([int(row['readmitted']=='<30') for row in rows])
    backbone=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.05,
        l2_regularization=1.,categorical_features=categorical,early_stopping=False,random_state=1901)
    backbone.fit(x[fit],y[fit])
    missing=np.isnan(x[fit]).mean(0)
    candidates=[j for j in range(len(columns)) if 0<missing[j]<1]
    selected=sorted(candidates,key=lambda j:(-missing[j],columns[j]))[:4]
    if len(selected)!=4: raise ValueError('Need four nonconstant native missingness columns')
    mean=np.nanmean(x[fit][:,selected],axis=0)
    scale=np.nanstd(x[fit][:,selected],axis=0)
    scale=np.where(scale>1e-8,scale,1.)
    source_probability=backbone.predict_proba(x[context])[:,1]
    source_f=correction_features(x[context],selected,source_probability,mean,scale)
    weights=ROOT/'artifacts/reports/policy_mixture_v1/neural_mixture.npz'
    model=m.Mixture()
    with np.load(weights) as state:
        model.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files})
    mixture_meta=json.loads((ROOT/'artifacts/reports/policy_mixture_v1/training.json').read_text())
    if p.checksum(weights)!=mixture_meta['models']['neural_mixture']['sha256']: raise ValueError('Checkpoint changed')
    results=[]
    (out/'predictions').mkdir(parents=True)
    width=plan['target_labels']+plan['queries']
    for episode in range(plan['episodes']):
        ids=development[episode*width:(episode+1)*width]
        support,query=ids[:plan['target_labels']],ids[plan['target_labels']:]
        probability=backbone.predict_proba(x[ids])[:,1]
        f=correction_features(x[ids],selected,probability,mean,scale)
        e=dict(source_f=source_f,source_y=y[context],support_f=f[:plan['target_labels']],support_y=y[support],
               query_f=f[plan['target_labels']:],query_y=y[query],oracle=None,seed=episode,regime='native')
        task=m.prepare(e,plan)
        predictions=p.simple_predictions(e,plan)
        predictions['neural_mixture']=m.predict(model,task['context'],task['probability'])
        predictions['calibrated_guard']=task['probability'][:,task['moment_choice']] if task['moment_cv_gain']>mixture_meta['guard_threshold'] else task['probability'][:,0]
        predictions['cv_select']=task['probability'][:,task['cv_choice']]
        np.savez_compressed(out/'predictions'/f'episode{episode:02}.npz',**predictions,labels=y[query],
            context=task['context'],experts=task['probability'],support_f=e['support_f'],support_y=e['support_y'],
            query_f=e['query_f'],source_f=source_f,source_y=y[context],support_ids=support,query_ids=query,source_ids=context)
        results.append(dict(episode=episode,nll={name:p.bce(pr,y[query]) for name,pr in predictions.items()}))
    gains={name:p.interval([r['nll'][name]-r['nll']['neural_mixture'] for r in results]) for name in results[0]['nll'] if name!='neural_mixture'}
    report=dict(source_sha256=p.checksum(__file__),protocol_sha256=p.checksum(ROOT/'configs/policy_native_v1.json'),
        raw_csv_sha256=p.checksum(csv_path),checkpoint_sha256=p.checksum(weights),native_patient_rows=count,
        selected_columns=[columns[j] for j in selected],backbone_labels=len(fit),source_labels=len(context),
        episodes=len(results),results=results,gains=gains,seconds=time.monotonic()-started,device='cpu',
        eligibility='exclude expired/hospice disposition11/13/14/19/20/21; one hash-selected encounter perpatient',
        scope='Single-dataset retrospective development; patient-disjoint target episodes conditional on shared fitted source backbone; no verified time/site/policy shift or clinical effectiveness')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','source_sha256','protocol_sha256')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',required=True)
    args=parser.parse_args()
    out=ROOT/'artifacts/reports/policy_native_v1'
    if out.exists(): raise ValueError('Preserve prior output')
    run(Path(args.csv),out)
