"""Learn spectral adaptation under a convex support-objective descent bound."""
import torch
from torch import nn


def features(x, mask):
    x = torch.where(mask, torch.zeros_like(x), x).double()
    m = mask.double()
    return torch.cat((torch.ones_like(x[..., :1]), x, m, (x[..., :, None] * m[..., None, :]).flatten(-2)), -1)


class SpectralFilter(nn.Module):
    def __init__(self, kind='learned', steps=6, ridge=.05):
        super().__init__()
        self.kind, self.steps, self.ridge = kind, steps, ridge
        self.network = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 1)).double()
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -1.)
        self.scalar = nn.Parameter(torch.tensor(-1., dtype=torch.float64), requires_grad=kind=='global')
        self.network.requires_grad_(kind=='learned')

    def prepare(self, inputs):
        qx,qm,qz,sx,sm,sy,sz,tx,tm,ty,tz=inputs
        a=features(tx,tm)
        scale=a.square().mean(1,keepdim=True).sqrt().clamp_min(.1)
        a=a/scale
        s=features(sx,sm)/scale
        q=features(qx,qm)/scale
        eye=torch.eye(a.shape[-1],device=a.device,dtype=a.dtype)
        h=.25*(a.transpose(1,2)@a)/a.shape[1]+self.ridge*eye
        eigen,v=torch.linalg.eigh(h)
        return dict(a=a,s=s,q=q,y=ty.double(),z=tz.double(),sy=sy.double(),sz=sz.double(),qz=qz.double(),eigen=eigen,v=v)

    def solve(self,p,diagnostics=False):
        a,s,v=p['a'],p['s'],p['v']
        w=torch.zeros_like(p['eigen'])
        objectives=[]
        def objective(w):
            logits=p['z']+(a*w[:,None]).sum(-1)
            return (torch.nn.functional.softplus(logits)-p['y']*logits).mean(1)+.5*self.ridge*w.square().sum(1)
        if diagnostics: objectives.append(objective(w))
        for step in range(self.steps):
            residual=(p['z']+(a*w[:,None]).sum(-1)).sigmoid()-p['y']
            g=(a*residual[...,None]).mean(1)+self.ridge*w
            projected=(v.transpose(1,2)@g[...,None]).squeeze(-1)
            source_residual=(p['sz']+(s*w[:,None]).sum(-1)).sigmoid()-p['sy']
            source_g=(s*source_residual[...,None]).mean(1)+self.ridge*w
            source_projected=(v.transpose(1,2)@source_g[...,None]).squeeze(-1)
            row_scores=(a@v)*residual[...,None]
            se=(row_scores.var(1,unbiased=False)/a.shape[1]+1e-8).sqrt()
            if self.kind=='learned':
                context=torch.stack((p['eigen'].log(),(projected.abs()/se).clamp_max(10),
                    (source_projected.abs()/se).clamp_max(10),
                    (source_projected*projected/(se.square()+1e-8)).clamp(-10,10),
                    se.log().clamp(-12,2),torch.full_like(projected,step/self.steps)), -1)
                gate=self.network(context).squeeze(-1).sigmoid()
            elif self.kind=='global': gate=self.scalar.sigmoid()
            else: gate=torch.ones_like(projected)
            w=w-(v@(gate*projected/p['eigen'])[...,None]).squeeze(-1)
            if diagnostics: objectives.append(objective(w))
        logits=p['qz']+(p['q']*w[:,None]).sum(-1)
        return (logits,torch.stack(objectives)) if diagnostics else logits

    def forward(self,*inputs):
        return self.solve(self.prepare(inputs))
