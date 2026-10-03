"""Post-hoc support-holdout diagnosis on consumed development worlds."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import t
import run
from model import SpectralFilter

ROOT=run.ROOT


def crossfit(model, inputs, threshold=1.645):
    qx,qm,qz,sx,sm,sy,sz,tx,tm,ty,tz=inputs
    parts=[];logs=[]
    for parity in (0,1):
        fit=torch.arange(parity,tx.shape[1],2,device=tx.device)
        val=torch.arange(1-parity,tx.shape[1],2,device=tx.device)
        validation_count=len(val)
        joined=[torch.cat((tx[:,val],qx),1),torch.cat((tm[:,val],qm),1),torch.cat((tz[:,val],qz),1),sx,sm,sy,sz,tx[:,fit],tm[:,fit],ty[:,fit],tz[:,fit]]
        logits=model(*joined)
        v=logits[:,:validation_count]
        y=ty[:,val].double();base=tz[:,val].double()
        gain=(torch.nn.functional.softplus(base)-y*base)-(torch.nn.functional.softplus(v)-y*v)
        mean=gain.mean(1);se=gain.std(1,unbiased=True)/validation_count**.5
        accept=mean>threshold*se
        pred=logits[:,validation_count:].sigmoid()
        parts.append(torch.where(accept[:,None],pred,qz.sigmoid()))
        logs.append(dict(gain=float(mean.item()),se=float(se.item()),accepted=bool(accept.item())))
    return torch.stack(parts).mean(0),logs


def main():
    torch.set_num_threads(2)
    folder=ROOT/'artifacts/reports/spectral_v1_gpu'
    report=json.loads((folder/'report.json').read_text())
    model=SpectralFilter('learned').eval()
    with np.load(folder/'learned_weights.npz') as state:model.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files})
    rows=[]
    for file in sorted((folder/'predictions').glob('*.npz')):
        with np.load(file) as saved:
            inputs=[torch.tensor(saved[k][None],dtype=torch.bool if k.endswith('_mask') else torch.float32) for k in run.v1.INPUT_KEYS]
            seed,width,regime=file.stem.split('_',2)
            with torch.no_grad():pred,logs=crossfit(model,inputs)
            rows.append(dict(seed=int(seed),width=int(width),regime=regime,folds=logs,nll=run.v1.metrics(pred[0].numpy(),saved['labels'])['nll'],frozen=run.v1.metrics(saved['frozen'],saved['labels'])['nll'],original=run.v1.metrics(saved['learned'],saved['labels'])['nll']))
    summary={}
    for regime in run.v1.REGIMES:
        selected=[r for r in rows if r['regime']==regime]
        effect=np.array([np.mean([r['frozen']-r['nll'] for r in selected if r['seed']==seed]) for seed in sorted({r['seed'] for r in selected})])
        radius=t.ppf(.975,len(effect)-1)*effect.std(ddof=1)/len(effect)**.5
        summary[regime]=dict(nll=float(np.mean([r['nll'] for r in selected])),frozen=float(np.mean([r['frozen'] for r in selected])),original=float(np.mean([r['original'] for r in selected])),gain=float(effect.mean()),lower95=float(effect.mean()-radius),upper95=float(effect.mean()+radius),accepted_folds=sum(f['accepted'] for r in selected for f in r['folds']),total_folds=2*len(selected))
    out=ROOT/'artifacts/reports/crossfit_diagnosis';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(dict(scope='Used-panel post-hoc CPU diagnosis; nominal t-rule, not distribution-free safety; no query labels enter acceptance',threshold=1.645,rows=rows,summary=summary),indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
