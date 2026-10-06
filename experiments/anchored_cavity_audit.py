"""Independent per-sensor anchored update, reusing only the score-audit harness."""
import torch
import anchored_cavity as model_spec
import cavity_site_audit as audit

def forward(w,mode,c,x,m):
    def net(v):
        v=torch.tanh(v@w['net.0.weight'].T+w['net.0.bias'])
        v=torch.tanh(v@w['net.2.weight'].T+w['net.2.bias'])
        return v@w['net.4.weight'].T+w['net.4.bias']
    mean=c[:,10];var=c[:,15]**2;var=var.clamp_min(.01)
    covariance=torch.minimum(torch.maximum(c[:,20:25],-.95*torch.sqrt(var)[:,None]),.95*torch.sqrt(var)[:,None])
    residual=(var[:,None]-covariance**2).clamp_min(.01)
    base_precision=(covariance**2/(residual*var[:,None])+.001)*m
    base_natural=(mean[:,None]*(covariance**2/(residual*var[:,None])+.001)+covariance*x/residual)*m
    if mode=='mlp':
        h=net(torch.cat([c,x*m,m,base_precision,base_natural],1));return h[:,0],h[:,1].clamp(-5,3)
    precision=base_precision.clone();natural=base_natural.clone()
    for _ in range(3):
        revised=[]
        for j in range(5):
            members=[k for k in range(5) if k!=j] if mode=='cavity' else list(range(5))
            cp=1/var+precision[:,members].sum(1);cn=mean/var+natural[:,members].sum(1)
            if mode=='static':cp=1/var;cn=mean/var
            ident=torch.zeros((len(x),5));ident[:,j]=1
            v=torch.cat([c,(x[:,j]*m[:,j])[:,None],ident,(cn/cp)[:,None],torch.log(cp)[:,None],m.mean(1)[:,None],m],1)
            h=net(v);scale=torch.exp(4*torch.tanh(h[:,1]/4));tau=base_precision[:,j]*scale
            revised.append((tau,base_natural[:,j]*scale+tau*h[:,0]))
        precision=torch.stack([v[0] for v in revised],1);natural=torch.stack([v[1] for v in revised],1)
    total=1/var+precision.sum(1);return (mean/var+natural.sum(1))/total,-torch.log(total)

if __name__=='__main__':
    audit.p.DATA=model_spec.DATA;audit.p.OUT=model_spec.OUT;audit.p.FREEZE=model_spec.FREEZE
    audit.forward=forward;audit.main()
