"""Finite conventional conditional-headroom screen; no neural-model claim."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[2];INPUT=ROOT/'artifacts/reports/unscored_headroom_inputs';OUT=ROOT/'artifacts/reports/unscored_headroom_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nll(p,y):
    p=np.clip(p,1e-7,1-1e-7);return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
def fit(a,z,y,ridge):
    def loss(w):
        score=z+a@w;prob=expit(score)
        return float(np.mean(np.logaddexp(0,score)-y*score)+.5*ridge*(w@w)),np.asarray(a.T@(prob-y)).ravel()/len(y)+ridge*w
    result=minimize(loss,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-7,ftol=1e-14,maxls=30))
    gradient=float(np.max(np.abs(loss(result.x)[1])))
    if not result.success or gradient>1e-6:raise RuntimeError(f'Convex control stationarity failure {result.message}: {gradient}')
    return result.x,dict(iterations=result.nit,gradient_max=gradient,ridge=ridge)
def design(x,z,source,cat):
    parts=[sparse.csr_matrix(np.column_stack((np.ones(len(x)),z)))]
    for j,categorical in enumerate(cat):
        col=x[:,j]
        if categorical:
            codes=np.where(np.isnan(col),127,col).astype(int);assert ((codes>=0)&(codes<=127)).all()
            parts.append(sparse.csr_matrix((np.ones(len(x)),(np.arange(len(x)),codes)),shape=(len(x),128)))
        else:
            center=np.nanmean(source[:,j]);scale=np.nanstd(source[:,j]);center=0. if not np.isfinite(center) else center;scale=max(scale,1e-8) if np.isfinite(scale) else 1.
            parts.append(sparse.csr_matrix(np.column_stack((np.nan_to_num((col-center)/scale,nan=0.),np.isnan(col).astype(float)))))
    return sparse.hstack(parts,format='csr')

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False);rows=[]
    frozen=json.loads((ROOT/'artifacts/manifests/unscored_headroom_freeze.json').read_text())
    for name,h in frozen.items():assert sha(ROOT/name)==h
    backbone_file=ROOT/'artifacts/reports/policy_native_gpu_v1/source_backbone.json';backbone=XGBClassifier();backbone.load_model(backbone_file);backbone.set_params(device='cpu',n_jobs=2,enable_categorical=True)
    for episode in range(5):
        file=INPUT/f'episode{episode}.npz'
        with np.load(file) as a:x=a['x'];y=a['y'];cat=a['categorical'];source=a['source_x'];patient_ids=a['patient_ids'];indices=a['representative_indices']
        probability=backbone.predict_proba(x)[:,1].clip(1e-6,1-1e-6).astype(float);z=np.log(probability/(1-probability));sx=x[:512];sy=y[:512];sz=z[:512];qx=x[512:];qy=y[512:];qz=z[512:]
        rng=np.random.default_rng(773001+episode);fold=np.zeros(512,dtype=int)
        for value in (0,1):
            ids=rng.permutation(np.flatnonzero(sy==value));assert len(ids)>=3;fold[ids]=np.arange(len(ids))%3
        predictions=dict(frozen=probability[512:]);audits={};searches={};weights={}
        for name,penalties in [('platt',(.001,.01,.1,1.)),('additive',(.01,.1,1.,10.))]:
            a=sparse.csr_matrix(np.column_stack((np.ones(len(x)),z))) if name=='platt' else design(x,z,source,cat)
            losses=[];certificates=[]
            for ridge in penalties:
                oof=np.zeros(512)
                for f in range(3):
                    train=np.flatnonzero(fold!=f);val=np.flatnonzero(fold==f);w,cert=fit(a[train],sz[train],sy[train],ridge);oof[val]=expit(sz[val]+a[val]@w);certificates.append(cert)
                losses.append(nll(oof,sy))
            chosen=int(np.argmin(losses));w,cert=fit(a[:512],sz,sy,penalties[chosen]);predictions[name]=expit(qz+a[512:]@w);weights[name]=w
            audits[name]=cert;searches[name]=dict(penalties=list(penalties),support_oof_nll=losses,chosen=chosen,search_folds=12,certificates=certificates)
        features=np.column_stack((x,z)).astype(np.float32)
        correction=XGBClassifier(device='cpu',tree_method='hist',n_estimators=100,max_depth=2,learning_rate=.05,min_child_weight=5,reg_lambda=10,n_jobs=2,random_state=773001+episode,enable_categorical=True,feature_types=['c' if v else 'q' for v in cat]+['q'])
        correction.fit(features[:512],sy,base_margin=sz);predictions['conditional_tree']=correction.predict_proba(features[512:],base_margin=qz)[:,1]
        correction.save_model(OUT/f'episode{episode}_conditional_tree.json');np.savez_compressed(OUT/f'episode{episode}_linear_weights.npz',**weights)
        np.savez_compressed(OUT/f'episode{episode}_predictions.npz',**predictions,labels=qy,frozen_logits=z,patient_ids=patient_ids,representative_indices=indices,support_folds=fold)
        metrics={k:dict(nll=nll(p,qy),brier=float(np.mean((p-qy)**2)),calibration_mean_gap=float(np.mean(p)-np.mean(qy))) for k,p in predictions.items()}
        rows.append(dict(episode=episode,input_sha256=sha(file),metrics=metrics,searches=searches,final_fit_certificates=audits,tree_sha256=sha(OUT/f'episode{episode}_conditional_tree.json')))
        (OUT/'partial_rows.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(episode=episode,nll={k:v['nll'] for k,v in metrics.items()})),flush=True)
    means={k:float(np.mean([r['metrics'][k]['nll'] for r in rows])) for k in rows[0]['metrics']}
    gains={k:dict(mean=means[k]-means['conditional_tree'],positive_episodes=sum(r['metrics'][k]['nll']>r['metrics']['conditional_tree']['nll'] for r in rows)) for k in ('frozen','platt','additive')}
    gate=all(v['mean']>=.003 and v['positive_episodes']>=4 for v in gains.values())
    (OUT/'report.json').write_text(json.dumps(dict(rows=rows,means=means,gains=gains,headroom_gate=gate,scope='Previously unscored recorded-workflow cohort, now used development; conventional diagnostic models only',seconds=time.monotonic()-start,backbone_sha256=sha(backbone_file),cloud_calls=0),indent=2)+'\n');print(json.dumps(dict(gate=gate,means=means,gains=gains,seconds=time.monotonic()-start)))

if __name__=='__main__':main()
