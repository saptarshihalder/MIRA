"""Query-conditioned trained operator with disjoint support verification."""
import importlib.util
from pathlib import Path
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('frozen_spectral',Path(__file__).resolve().parents[1]/'spectral_v1/model.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class CrossfitOperator(base.SpectralFilter):
    def __init__(self,kind='query',steps=6,ridge=.05):
        super().__init__('global' if kind=='scalar' else 'learned',steps,ridge)
        self.variant=kind
        self.query_head=nn.Sequential(nn.Linear(5,16),nn.Tanh(),nn.Linear(16,1)).double()
        nn.init.zeros_(self.query_head[-1].weight)
        nn.init.constant_(self.query_head[-1].bias,1.4)
        self.query_head.requires_grad_(kind!='no_query')

    def prepare(self,inputs):
        qx,qm,qz,sx,sm,sy,sz,tx,tm,ty,tz=inputs
        if tx.shape[1]<4:raise ValueError('Need two nonempty fitting/verification halves')
        parts=[]
        for parity in (0,1):
            fit=torch.arange(parity,tx.shape[1],2,device=tx.device)
            val=torch.arange(1-parity,tx.shape[1],2,device=tx.device)
            joined=[torch.cat((tx[:,val],qx),1),torch.cat((tm[:,val],qm),1),torch.cat((tz[:,val],qz),1),sx,sm,sy,sz,tx[:,fit],tm[:,fit],ty[:,fit],tz[:,fit]]
            p=super().prepare(joined)
            p['verification_y']=ty[:,val].double()
            p['leverage']=((p['q']@p['v']).square()/p['eigen'][:,None]).sum(-1)/len(fit)
            parts.append(p)
        return parts

    def candidate(self,p,diagnostics=False):
        value=super().solve(p,diagnostics)
        raw,objective=value if diagnostics else (value,None)
        if self.variant=='no_query':probability=raw.sigmoid()
        else:
            correction=raw-p['qz']
            shift=(p['y']-p['z'].sigmoid()).mean(1)-(p['sy']-p['sz'].sigmoid()).mean(1)
            context=torch.stack((p['qz'].clamp(-6,6)/3,correction.tanh(),correction.abs().clamp_max(6)/3,p['leverage'].log1p().clamp_max(6)/3,shift[:,None].expand_as(raw)), -1)
            alpha=self.query_head(context).squeeze(-1).sigmoid()
            probability=(1-alpha)*p['qz'].sigmoid()+alpha*raw.sigmoid()
        return probability,objective

    def predict(self,parts,diagnostics=False):
        candidates=[];hard=[];soft=[];accepts=[];objectives=[];gains=[];ses=[]
        for p in parts:
            probability,objective=self.candidate(p,diagnostics)
            n=p['verification_y'].shape[1]
            verified=probability[:,:n].clamp(1e-8,1-1e-8)
            y=p['verification_y'];z=p['qz'][:,:n]
            losses=torch.nn.functional.softplus(z)-y*z
            improvement=losses+y*verified.log()+(1-y)*torch.log1p(-verified)
            mean=improvement.mean(1);se=(improvement.var(1,unbiased=True)/n+1e-16).sqrt()
            margin=mean-1.645*se
            accepted=margin>0
            soft_weight=(margin/(se+.005)).sigmoid()
            frozen=p['qz'][:,n:].sigmoid();candidate=probability[:,n:]
            candidates.append(candidate)
            hard.append(torch.where(accepted[:,None],candidate,frozen))
            soft.append(soft_weight[:,None]*candidate+(1-soft_weight[:,None])*frozen)
            accepts.append(accepted);gains.append(mean);ses.append(se)
            if objective is not None:objectives.append(objective)
        return dict(hard=torch.stack(hard).mean(0),soft=torch.stack(soft).mean(0),unguarded=torch.stack(candidates).mean(0),accepted=torch.stack(accepts),gains=torch.stack(gains),ses=torch.stack(ses),objectives=objectives)

    def forward(self,*inputs):
        result=self.predict(self.prepare(inputs))
        logits=torch.logit(result['hard'].clamp(1e-8,1-1e-8))
        return torch.where(result['accepted'].any(0)[:,None],logits,inputs[2].double())
