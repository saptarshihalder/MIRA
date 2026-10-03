"""Calibration anchor plus learned correction outside its fitting-support span."""
import torch
from model import CrossfitOperator
from support_head import SupportHead
from recover_evaluation import stable_platt

class AnchoredQuery(CrossfitOperator):
    def prepare(self, inputs):
        parts=super().prepare(inputs)
        for p in parts:
            p['original_qz']=p['qz'].clone()
            # All nuisance fits use only this branch's fitting labels.
            z,y=p['z'],p['y'];calibrated=[];sz=[];qz=[]
            a=[];s=[];q=[]
            for i in range(len(z)):
                _,certificate=stable_platt(dict(target_logit=z[i].numpy(),target_y=y[i].numpy(),query_logit=z[i].numpy()),.1)
                w=torch.tensor(certificate['weights'],dtype=torch.float64)
                calibrated.append((1+w[0])*z[i]+w[1]);sz.append((1+w[0])*p['sz'][i]+w[1]);qz.append((1+w[0])*p['qz'][i]+w[1])
                design=torch.stack((torch.ones_like(z[i]),z[i]),-1)
                projection=torch.linalg.pinv(design)@p['a'][i]
                a.append(p['a'][i]-design@projection)
                s.append(p['s'][i]-torch.stack((torch.ones_like(p['sz'][i]),p['sz'][i]),-1)@projection)
                q.append(p['q'][i]-torch.stack((torch.ones_like(p['qz'][i]),p['qz'][i]),-1)@projection)
            p.update(z=torch.stack(calibrated),sz=torch.stack(sz),qz=torch.stack(qz),a=torch.stack(a),s=torch.stack(s),q=torch.stack(q))
            n=p['a'].shape[1];h=.25*(p['a'].transpose(1,2)@p['a'])/n+self.ridge*torch.eye(p['a'].shape[-1],dtype=torch.float64)
            p['eigen'],p['v']=torch.linalg.eigh(h)
            p['leverage']=((p['q']@p['v']).square()/p['eigen'][:,None]).sum(-1)/n
        return parts

    def predict(self,parts,diagnostics=False):
        hard=[];soft=[];raw=[];accepted=[];anchor_accept=[]
        for p in parts:
            candidate,_=self.candidate(p,diagnostics);n=p['verification_y'].shape[1];y=p['verification_y']
            original=p['original_qz'].sigmoid();anchor=p['qz'].sigmoid()
            def loss(prob):
                v=prob[:,:n].clamp(1e-8,1-1e-8)
                return -y*v.log()-(1-y)*torch.log1p(-v)
            def decision(gain):
                se=(gain.var(1,unbiased=True)/n+1e-16).sqrt();margin=gain.mean(1)-1.645*se
                return margin>0,(margin/(se+.005)).sigmoid()
            anchor_ok,anchor_weight=decision(loss(original)-loss(anchor))
            relative_ok,relative_weight=decision(loss(anchor)-loss(candidate))
            original_ok,original_weight=decision(loss(original)-loss(candidate))
            ok=relative_ok&original_ok
            base=torch.where(anchor_ok[:,None],anchor[:,n:],original[:,n:])
            soft_base=anchor_weight[:,None]*anchor[:,n:]+(1-anchor_weight[:,None])*original[:,n:]
            weight=(relative_weight*original_weight)[:,None]
            hard.append(torch.where(ok[:,None],candidate[:,n:],base))
            soft.append(weight*candidate[:,n:]+(1-weight)*soft_base)
            raw.append(candidate[:,n:]);accepted.append(ok);anchor_accept.append(anchor_ok)
        return dict(hard=torch.stack(hard).mean(0),soft=torch.stack(soft).mean(0),unguarded=torch.stack(raw).mean(0),accepted=torch.stack(accepted),anchor_accepted=torch.stack(anchor_accept))

    def forward(self,*inputs):
        return torch.logit(self.predict(self.prepare(inputs))['hard'].clamp(1e-8,1-1e-8))

class AnchoredSupport(SupportHead,AnchoredQuery):
    pass

class AnchorOnly(AnchoredQuery):
    def candidate(self,p,diagnostics=False):return p['qz'].sigmoid(),None
