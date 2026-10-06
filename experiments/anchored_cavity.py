"""Support-moment initialization and explicit availability repair; fresh panel."""
import argparse,shutil
import numpy as np
import torch
from torch import nn
import cavity_site as p

DATA=p.ROOT/'artifacts/local/anchored_cavity_v1'
OUT=p.ROOT/'artifacts/reports/anchored_cavity_v1'
FREEZE=p.ROOT/'artifacts/manifests/anchored_cavity_freeze.json'

class Model(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;size=64 if mode=='mlp' else 32
        self.net=nn.Sequential(nn.Linear(75 if mode=='mlp' else 69,size),nn.Tanh(),nn.Linear(size,size),nn.Tanh(),nn.Linear(size,2))
        if mode!='mlp':
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,c,x,m):
        ym=c[:,10];yv=c[:,15].square().clamp_min(.01);cross=c[:,20:25]
        cross=torch.maximum(torch.minimum(cross,.95*yv.sqrt()[:,None]),-.95*yv.sqrt()[:,None])
        variance=(yv[:,None]-cross.square()).clamp_min(.01)
        tau0=cross.square()/(variance*yv[:,None])+.001
        eta0=ym[:,None]*tau0+cross*x/variance
        tau0=tau0*m;eta0=eta0*m
        if self.mode=='mlp':
            h=self.net(torch.cat((c,x*m,m,tau0,eta0),-1));return h[:,0],h[:,1].clamp(-5,3)
        tau,eta=tau0,eta0
        for _ in range(3):
            cp=(1/yv[:,None]+tau.sum(1,keepdim=True)).expand(-1,5)
            cn=(ym[:,None]/yv[:,None]+eta.sum(1,keepdim=True)).expand(-1,5)
            if self.mode=='cavity':cp=cp-tau;cn=cn-eta
            if self.mode=='static':cp=(1/yv[:,None]).expand(-1,5);cn=(ym/yv)[:,None].expand(-1,5)
            inp=torch.cat((c[:,None].expand(-1,5,-1),(x*m)[:,:,None],torch.eye(5).expand(len(x),-1,-1),
                (cn/cp)[:,:,None],cp.log()[:,:,None],m.mean(1)[:,None,None].expand(-1,5,1),m[:,None].expand(-1,5,-1)),-1)
            h=self.net(inp);scale=torch.exp(4*torch.tanh(h[:,:,1]/4))
            tau=tau0*scale;eta=eta0*scale+tau*h[:,:,0]
        precision=1/yv+tau.sum(1)
        return (ym/yv+eta.sum(1))/precision,-precision.log()

def prepare():
    DATA.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(p.DATA/'source.npz',DATA/'source.npz')
    np.savez_compressed(DATA/'development.npz',**p.generate(64,819202))
    files=[p.ROOT/'experiments/cavity_site.py',p.ROOT/'experiments/anchored_cavity.py',p.ROOT/'docs/ANCHORED_CAVITY_PROTOCOL.md']+sorted(DATA.iterdir())
    p.save(FREEZE,{f.relative_to(p.ROOT).as_posix():p.sha(f) for f in files})
    print('Repair prepared; prior development not opened; commit before fit.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:
        p.DATA=DATA;p.OUT=OUT;p.FREEZE=FREEZE;p.Model=Model;p.main()
