"""Finite synthetic feasibility probe, not a novelty or native-data claim."""
import itertools,json,hashlib,time,functools
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/reports/acquisition_repair_probe'
D=4;ALL=15;SEEDS=(812001,812002,812003)
TRAIN_A=(15,14,13,11,7);TEST_A=(3,5,6,9,10,12)
POLICIES=((.02,.02,.4,.4),(.4,.4,.02,.4),(.4,.4,.4,.02),(.12,.12,.12,.12))
TEST_COSTS=((.01,.2,.05,.4),(.2,.01,.4,.05),(.3,.3,.01,.2),(.08,.15,.2,.01))
MODES=('repair','direct','blind')
STATES=list(itertools.product((-1,0,1),repeat=D))
JOINT=[]
for x0,x1,y,x2,x3 in itertools.product((0,1),repeat=5):
    probability=.25*(.95 if y==x0^x1 else .05)*(.8 if x2==y else .2)*(.65 if x3==y else .35)
    JOINT.append(((x0,x1,x2,x3),y,probability))

@functools.lru_cache(None)
def conditional(s):
    rows=[(x,y,p) for x,y,p in JOINT if all(v<0 or x[j]==v for j,v in enumerate(s))]
    total=sum(p for _,_,p in rows);return [(x,y,p/total) for x,y,p in rows]

@functools.lru_cache(None)
def entropy(s):
    p=sum(y*w for _,y,w in conditional(s));return float(-p*np.log(p)-(1-p)*np.log1p(-p))

def available(s,A):return [j for j,v in enumerate(s) if v<0 and A&(1<<j)]

@functools.lru_cache(None)
def branches(s,a):
    result=[]
    for v in (0,1):
        prob=sum(w for x,_,w in conditional(s) if x[a]==v);next_state=list(s);next_state[a]=v
        result.append((tuple(next_state),prob))
    return result

@functools.lru_cache(None)
def optimal(s,A,b,c):
    candidates=[(entropy(s),-1)]
    if b:
        for a in available(s,A):candidates.append((c[a]+sum(p*optimal(t,A,b-1,c)[0] for t,p in branches(s,a)),a))
    return min(candidates)

@functools.lru_cache(None)
def successor(s,A,b,k,a):
    if a==-1:return (entropy(s),0.,0.,0.,0.)
    value=np.zeros(5);value[a+1]=1.
    for t,p in branches(s,a):
        action=optimal(t,A,b-1,POLICIES[k])[1]
        value+=p*np.array(successor(t,A,b-1,k,action))
    return tuple(value)

def features(s,A,b,k,a):
    base=np.array(successor(s,ALL,b,k,a))
    feature=np.r_[np.maximum(s,0),np.array(s)>=0,[(A>>j)&1 for j in range(D)],np.eye(D)[a],b/2,np.eye(4)[k],base]
    return feature.astype(np.float32),base.astype(np.float32)

class Model(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(26,32),nn.Tanh(),nn.Linear(32,5))
    def forward(self,x):return self.net(x)

def raw_prediction(model,mode,x,base):
    xx=x.clone()
    if mode=='blind':xx[:,8:12]=0
    pred=model(xx)
    return pred+base if mode=='repair' else pred

def infer(model,mode,s,A,b,k,a):
    x,base=features(s,A,b,k,a)
    with torch.no_grad():v=raw_prediction(model,mode,torch.tensor(x)[None],torch.tensor(base)[None])[0].numpy()
    v=np.clip(v,0,1);v[0]=min(v[0],np.log(2))
    # All learned controls get identical immediate feasibility constraints.
    for j in range(D):
        if s[j]>=0 or not A&(1<<j):v[j+1]=0
    v[a+1]=1
    return v

def evaluate(model,mode,A,c):
    @functools.lru_cache(None)
    def rollout(s,b):
        if not b:return entropy(s)
        candidates=[(entropy(s),-1)]
        for action in available(s,A):
            values=[]
            for k in range(4):
                if mode=='exact_feasible':v=np.array(successor(s,A,b,k,action))
                elif mode=='naive_source':v=np.array(successor(s,ALL,b,k,action))
                else:v=infer(model,mode,s,A,b,k,action)
                values.append(float(v[0]+v[1:]@c))
            candidates.append((min(values),action))
        action=min(candidates)[1]
        if action==-1:return entropy(s)
        return c[action]+sum(p*rollout(t,b-1) for t,p in branches(s,action))
    return rollout((-1,)*D,2)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((ROOT/'artifacts/manifests/acquisition_repair_probe_freeze.json').read_text())
    for p,h in frozen.items():assert sha(ROOT/p)==h
    rows=[]
    for s,A,b,k in itertools.product(STATES,TRAIN_A,(1,2),range(4)):
        for a in available(s,A):
            x,base=features(s,A,b,k,a);rows.append((x,base,successor(s,A,b,k,a)))
    x,base,target=[torch.tensor(np.array(v),dtype=torch.float32) for v in zip(*rows)]
    np.savez_compressed(OUT/'training_inputs.npz',features=x.numpy(),source_successors=base.numpy(),targets=target.numpy())
    results=[];max_replay=0.
    for seed,mode in itertools.product(SEEDS,MODES):
        assert time.monotonic()-start<120,'Finite CPU limit reached'
        torch.manual_seed(seed);model=Model();opt=torch.optim.Adam(model.parameters(),lr=.002);rng=np.random.default_rng(seed);trace=[]
        for step in range(600):
            ix=rng.integers(0,len(x),256);pred=raw_prediction(model,mode,x[ix],base[ix]);loss=(pred-target[ix]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
            if (step+1)%100==0:trace.append(dict(step=step+1,mse=float(loss.detach())))
        path=OUT/f'{mode}_{seed}.npz';np.savez_compressed(path,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        reload=Model()
        with np.load(path) as z:reload.load_state_dict({k:torch.tensor(z[k]) for k in z.files})
        with torch.no_grad():max_replay=max(max_replay,float((model(x[:64])-reload(x[:64])).abs().max()))
        values=[evaluate(model,mode,A,c) for A in TEST_A for c in TEST_COSTS]
        control=[evaluate(model,mode,ALL,c) for c in POLICIES]
        results.append(dict(mode=mode,seed=seed,mean=float(np.mean(values)),risks=values,source_control=control,trace=trace,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(path)))
    references={mode:[evaluate(None,mode,A,c) for A in TEST_A for c in TEST_COSTS] for mode in ('naive_source','exact_feasible')}
    references['optimal']=[optimal((-1,)*D,A,2,c)[0] for A in TEST_A for c in TEST_COSTS]
    means={m:float(np.mean([r['mean'] for r in results if r['mode']==m])) for m in MODES}
    means.update({k:float(np.mean(v)) for k,v in references.items()})
    gate=all(means['repair']<=.98*means[m] for m in ('direct','blind','naive_source'))
    assert max_replay==0 and all(v>=-1e-12 for v in references['optimal'])
    save(OUT/'report.json',dict(results=results,reference_risks=references,means=means,mechanism_gate=gate,
        training_examples=len(x),checkpoints=9,max_checkpoint_replay_error=max_replay,seconds=time.monotonic()-start,cloud_calls=0,
        scope='Privileged synthetic population teacher; held-out availability combinations; no real-data or novelty evidence'))
    save(OUT/'manifest.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(dict(means=means,gate=gate,examples=len(x),seconds=time.monotonic()-start)))

if __name__=='__main__':main()
