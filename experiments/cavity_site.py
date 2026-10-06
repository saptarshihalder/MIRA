"""Finite source-trained site replacement architecture diagnostic."""
import argparse, hashlib, itertools, json, os, threading, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/local/cavity_site_v1'
OUT=ROOT/'artifacts/reports/cavity_site_v1'
FREEZE=ROOT/'artifacts/manifests/cavity_site_freeze.json'
SEEDS=(819001,819002,819003)
MODES=('cavity','aggregate','static','mlp')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}

def context(x,y,m):
    n=m.sum(0).clip(1);mean=(x*m).sum(0)/n
    xc=(x-mean)*m;scale=np.sqrt((xc*xc).sum(0)/n).clip(.1)
    xs=xc/scale;ym=y.mean();ys=y.std().clip(.1)
    cross=(xs*(y-ym)[:,None]).sum(0)/n
    pairs=(xs.T@xs)/(m.T@m).clip(1)
    c=np.concatenate([mean,scale,np.full(5,ym),np.full(5,ys),cross,m.mean(0),pairs.ravel()])
    return c.astype('float32'),mean,scale

def generate(n,seed):
    rng=np.random.default_rng(seed);out={k:[] for k in ('context','x','y','mask','sx','sy','sm')}
    for _ in range(n):
        a=rng.choice([-1.,1.],5)*rng.uniform(.6,1.8,5);b=rng.normal(0,.4,5)
        f=rng.uniform(.5,2.5,5);d=rng.normal(0,.7,5);noise=rng.uniform(.15,.8,5)
        y=rng.normal(size=96);u=rng.normal(size=96)
        x=y[:,None]*a+b+.4*np.sin(y[:,None]*f)+u[:,None]*d+rng.normal(size=(96,5))*noise
        sm=(rng.random((48,5))>.2).astype(float);c,mean,scale=context(x[:48],y[:48],sm)
        mask=np.ones((48,5));drop=rng.integers(0,6,48)
        for i,j in enumerate(drop):
            if j<5:mask[i,j]=0
        for k,v in dict(context=c,x=(x[48:]-mean)/scale,y=y[48:],mask=mask,sx=x[:48],sy=y[:48],sm=sm).items():out[k].append(v)
    return {k:np.asarray(v,dtype='float32') for k,v in out.items()}

def prepare():
    DATA.mkdir(parents=True,exist_ok=False)
    for name,n,seed in [('source',512,819101),('development',64,819102)]:
        np.savez_compressed(DATA/f'{name}.npz',**generate(n,seed))
    paths=[ROOT/'experiments/cavity_site.py',ROOT/'docs/CAVITY_SITE_PROTOCOL.md']+sorted(DATA.iterdir())
    save(FREEZE,{p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    print('Prepared; commit freeze before fitting.')

class Model(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.net=nn.Sequential(nn.Linear(65 if mode=='mlp' else 64,64 if mode=='mlp' else 32),nn.Tanh(),
            nn.Linear(64 if mode=='mlp' else 32,64 if mode=='mlp' else 32),nn.Tanh(),nn.Linear(64 if mode=='mlp' else 32,2))
    def forward(self,c,x,m):
        if self.mode=='mlp':
            h=self.net(torch.cat((c,x*m,m),-1));return h[:,0],h[:,1].clamp(-5,3)
        batch=len(x);tau=torch.zeros_like(x);eta=torch.zeros_like(x)
        for _ in range(3):
            precision=1+tau.sum(1,keepdim=True);natural=eta.sum(1,keepdim=True)
            cp=precision.expand(-1,5);cn=natural.expand(-1,5)
            if self.mode=='cavity':cp=cp-tau;cn=cn-eta
            if self.mode=='static':cp=torch.ones_like(x);cn=torch.zeros_like(x)
            inp=torch.cat((c[:,None].expand(-1,5,-1),(x*m)[:,:,None],torch.eye(5).expand(batch,-1,-1),
                           (cn/cp)[:,:,None],cp.log()[:,:,None],m.mean(1)[:,None,None].expand(-1,5,1)),-1)
            h=self.net(inp);tau=nn.functional.softplus(h[:,:,1])*m;eta=tau*h[:,:,0]
        precision=1+tau.sum(1);return eta.sum(1)/precision,-precision.log()

def nll(mean,lv,y):return .5*(np.log(2*np.pi)+lv+(y-mean)**2*np.exp(-lv))

def predict(model,c,x,m):
    mus=[];lvs=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            mu,lv=model(*(torch.tensor(v[start:start+1024]) for v in (c,x,m)))
            mus.append(mu.numpy());lvs.append(lv.numpy())
    return np.concatenate(mus),np.concatenate(lvs)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);lim=threadpool_limits(limits=1)
    for f,h in json.loads(FREEZE.read_text(encoding='utf-8')).items():assert sha(ROOT/f)==h,f
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    timer=threading.Timer(600,lambda:os._exit(124));timer.daemon=True;timer.start()
    source=load(DATA/'source.npz');training=[]
    c=torch.tensor(source['context']);x=torch.tensor(source['x']);y=torch.tensor(source['y']);m=torch.tensor(source['mask'])
    for seed in SEEDS:
        for mode in MODES:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Model(mode)
            opt=torch.optim.AdamW(model.parameters(),lr=.0018,weight_decay=.0001);trace=[]
            for step in range(3000):
                t=rng.integers(0,512,128);q=rng.integers(0,48,128)
                mu,lv=model(c[t],x[t,q],m[t,q]);loss=.5*(lv+(y[t,q]-mu).square()*(-lv).exp()).mean()
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                if (step+1)%100==0:trace.append(float(loss.detach()))
            torch.save(model.state_dict(),OUT/f'{mode}_{seed}.pt')
            training.append(dict(mode=mode,seed=seed,updates=3000,parameters=sum(p.numel() for p in model.parameters()),trace=trace))
            save(OUT/'training.json',training)
    save(OUT/'fitted_before_scoring.json',{p.name:sha(p) for p in OUT.iterdir()})
    dev=load(DATA/'development.npz');masks=np.ones((10,5),dtype='float32')
    for i,pair in enumerate(itertools.combinations(range(5),2)):masks[i,list(pair)]=0
    c=np.repeat(dev['context'],480,axis=0);x=np.repeat(dev['x'][:,None],10,axis=1).reshape(-1,5)
    y=np.repeat(dev['y'][:,None],10,axis=1).reshape(-1);m=np.tile(np.repeat(masks,48,axis=0),(64,1))
    summary={}
    for seed in SEEDS:
        for mode in MODES:
            model=Model(mode);model.load_state_dict(torch.load(OUT/f'{mode}_{seed}.pt',weights_only=True));model.eval()
            mu,lv=predict(model,c,x,m);loss=nll(mu,lv,y);mse=(mu-y)**2;coverage=np.abs(mu-y)<=1.96*np.exp(lv/2)
            np.savez_compressed(OUT/f'pred_{mode}_{seed}.npz',mu=mu,lv=lv,nll=loss,mse=mse,coverage=coverage,cell_nll=loss.reshape(64,10,48).mean(2),cell_mse=mse.reshape(64,10,48).mean(2),cell_coverage=coverage.reshape(64,10,48).mean(2))
            summary[f'{mode}_{seed}']=dict(nll=float(loss.mean()),mse=float(mse.mean()),coverage=float(coverage.mean()))
    rmu=[];rlv=[]
    for t in range(64):
        _,mean,scale=context(dev['sx'][t],dev['sy'][t],dev['sm'][t])
        sx=(dev['sx'][t]-mean)/scale*dev['sm'][t]
        for mask in masks:
            take=mask.astype(bool);design=np.column_stack((np.ones(48),sx[:,take]));penalty=np.eye(4);penalty[0,0]=0
            inv=np.linalg.inv(design.T@design+penalty);w=inv@design.T@dev['sy'][t]
            leverage=np.sum((design@inv)*design,axis=1)
            loo=(dev['sy'][t]-design@w)/(1-leverage).clip(.05)
            variance=max(float((loo**2).mean()),.01)
            rmu.extend(np.column_stack((np.ones(48),dev['x'][t][:,take]))@w);rlv.extend([np.log(variance)]*48)
    mu=np.array(rmu);lv=np.array(rlv);loss=nll(mu,lv,y);mse=(mu-y)**2;coverage=np.abs(mu-y)<=1.96*np.exp(lv/2)
    np.savez_compressed(OUT/'pred_ridge.npz',mu=mu,lv=lv,nll=loss,mse=mse,coverage=coverage,cell_nll=loss.reshape(64,10,48).mean(2),cell_mse=mse.reshape(64,10,48).mean(2),cell_coverage=coverage.reshape(64,10,48).mean(2))
    summary['ridge']=dict(nll=float(loss.mean()),mse=float(mse.mean()),coverage=float(coverage.mean()))
    gates={}
    for seed in SEEDS:
        main=summary[f'cavity_{seed}'];controls=[f'{v}_{seed}' for v in ('aggregate','static','mlp')]+['ridge']
        gains={v:summary[v]['nll']-main['nll'] for v in controls}
        gates[str(seed)]=dict(gains=gains,passed=all(g>=.01 for g in gains.values()) and all(main['mse']<=1.01*summary[v]['mse'] for v in controls))
    report=dict(summary=summary,gates=gates,passed=all(g['passed'] for g in gates.values()),seconds=time.monotonic()-start,
        cloud_calls=0,source_tasks=512,development_tasks=64,total_updates=36000,novelty_established=False)
    save(OUT/'report.json',report);save(OUT/'manifest.json',{p.name:sha(p) for p in OUT.iterdir() if p.name!='manifest.json'})
    timer.cancel();lim.restore_original_limits();print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else main()
