"""Finite support-only contextual-bandit comparison; no query feedback."""
import json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'artifacts/reports/annual_rl_inputs'
OUT=ROOT/'artifacts/reports/annual_rl_v1'
FREEZE=ROOT/'artifacts/manifests/annual_rl_freeze.json'
ACTIONS=torch.tensor([-1.,-.5,-.25,0.,.25,.5,1.])
SEEDS=(991001,991002,991003)
MODES=('reinforce','exact_pg','supervised')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
def nll(p,y):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))

class Actor(nn.Module):
    def __init__(self,width):
        super().__init__();self.network=nn.Sequential(nn.Linear(width,16),nn.Tanh(),nn.Linear(16,7))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():self.network[-1].bias[3]=2.
    def forward(self,x):return self.network(x).softmax(-1)

def train(features,anchor,labels,seed,mode):
    # The API receives only fitting rows; selection and query labels are excluded.
    assert len(labels)==384 and len(features)==len(anchor)==len(labels)
    torch.manual_seed(seed);model=Actor(features.shape[1]);initial=sha_state(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);sample_rng=torch.Generator().manual_seed(seed+100000)
    x=torch.tensor(features,dtype=torch.float32);z=torch.tensor(anchor,dtype=torch.float32);y=torch.tensor(labels,dtype=torch.float32)
    trace=[];start=time.monotonic()
    for step in range(300):
        ix=rng.integers(0,384,128);p=model(x[ix]);zz=z[ix,None]+ACTIONS[None,:]
        losses=F.binary_cross_entropy_with_logits(zz,y[ix,None].expand(-1,7),reduction='none')
        reward=(F.binary_cross_entropy_with_logits(z[ix],y[ix],reduction='none')[:,None]-losses-.01*ACTIONS.square()).detach()
        entropy=-(p*p.clamp_min(1e-8).log()).sum(-1).mean()
        if mode=='reinforce':
            action=torch.multinomial(p,1,generator=sample_rng).squeeze(1)
            objective=-(p.gather(1,action[:,None]).clamp_min(1e-8).log().squeeze(1)*reward.gather(1,action[:,None]).squeeze(1)).mean()
        elif mode=='exact_pg':objective=-(p*reward).sum(-1).mean()
        else:
            mixture=(p*zz.sigmoid()).sum(-1)
            objective=F.binary_cross_entropy(mixture.clamp(1e-7,1-1e-7),y[ix])+.01*(p*ACTIONS.square()).sum(-1).mean()
        loss=objective-.001*entropy;assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%50==0:
            with torch.no_grad():
                pp=model(x);mixture=(pp*(z[:,None]+ACTIONS).sigmoid()).sum(-1)
            trace.append(dict(step=step+1,train_nll=nll(mixture.numpy(),labels),entropy=float(entropy.detach()),gradient_norm=float(grad)))
    return model,dict(trace=trace,initial_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start)

def sha_state(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()

def predict(model,features,anchor):
    with torch.no_grad():
        weights=model(torch.tensor(features,dtype=torch.float32));z=torch.tensor(anchor,dtype=torch.float32)
        prob=(weights*(z[:,None]+ACTIONS).sigmoid()).sum(-1)
    return prob.numpy().astype(float),weights.numpy()

def main():
    torch.set_num_threads(1)
    for name,digest in json.loads(FREEZE.read_text()).items():assert sha(ROOT/name)==digest,name
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    for episode,spec in enumerate(json.loads((INPUT/'index.json').read_text())):
        with np.load(ROOT/spec['input']) as a:x=a['features'];z=a['anchor'];y=a['y'];ids=a['ids']
        anchor=1/(1+np.exp(-z));query=y[512:]
        for seed in SEEDS:
            for mode in MODES:
                assert time.monotonic()-start<600,'Neural fitting wall limit reached'
                model,training=train(x[:384],z[:384],y[:384],seed,mode)
                selection,_=predict(model,x[384:512],z[384:512]);alphas=(0.,.25,.5,1.)
                losses=[nll(anchor[384:512]+alpha*(selection-anchor[384:512]),y[384:512]) for alpha in alphas]
                alpha=alphas[int(np.argmin(losses))]
                raw,policy=predict(model,x[512:],z[512:]);pred=anchor[512:]+alpha*(raw-anchor[512:])
                name=f'episode{episode}_{seed}_{mode}';weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
                np.savez_compressed(OUT/f'{name}_weights.npz',**weights)
                np.savez_compressed(OUT/f'{name}_predictions.npz',selected=pred,raw=raw,anchor=anchor[512:],policy=policy,labels=query,ids=ids,selection_raw=selection)
                rows.append(dict(episode=episode,task=spec['task'],group=spec['group'],seed=seed,mode=mode,alpha=alpha,
                    selection_losses=losses,training=training,nll=nll(pred,query),raw_nll=nll(raw,query),anchor_nll=nll(anchor[512:],query),
                    brier=float(np.mean((pred-query)**2)),weights_sha256=sha(OUT/f'{name}_weights.npz')))
                save(OUT/'partial.json',rows)
        print(json.dumps(dict(episode=episode,task=spec['task'],group=spec['group'],checkpoints=len(rows))),flush=True)
    save(OUT/'report.json',dict(rows=rows,seconds=time.monotonic()-start,cloud_calls=0,device='cpu',updates_per_model=300))

if __name__=='__main__':main()
