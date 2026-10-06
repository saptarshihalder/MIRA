"""Replay trained weights with a separate site-update implementation."""
import json,time
import numpy as np
import torch
import cavity_site as p

def forward(w,mode,c,x,m):
    def net(v):
        v=torch.tanh(v@w['net.0.weight'].T+w['net.0.bias'])
        v=torch.tanh(v@w['net.2.weight'].T+w['net.2.bias'])
        return v@w['net.4.weight'].T+w['net.4.bias']
    if mode=='mlp':
        o=net(torch.cat([c,x*m,m],dim=1));return o[:,0],o[:,1].clamp(-5,3)
    precision=torch.zeros_like(x);natural=torch.zeros_like(x)
    for _ in range(3):
        updates=[]
        for j in range(5):
            keep=[k for k in range(5) if k!=j] if mode=='cavity' else list(range(5))
            cp=1+precision[:,keep].sum(1);cn=natural[:,keep].sum(1)
            if mode=='static':cp=torch.ones_like(cp);cn=torch.zeros_like(cn)
            ident=torch.zeros((len(x),5));ident[:,j]=1
            inp=torch.cat([c,(x[:,j]*m[:,j])[:,None],ident,(cn/cp)[:,None],torch.log(cp)[:,None],m.mean(1)[:,None]],1)
            o=net(inp);tau=torch.logaddexp(torch.zeros_like(o[:,1]),o[:,1])*m[:,j]
            updates.append((tau,tau*o[:,0]))
        precision=torch.stack([u[0] for u in updates],1);natural=torch.stack([u[1] for u in updates],1)
    total=1+precision.sum(1);return natural.sum(1)/total,-torch.log(total)

def main():
    start=time.monotonic();torch.set_num_threads(1)
    for manifest,base in [(p.FREEZE,p.ROOT),(p.OUT/'manifest.json',p.OUT),(p.OUT/'fitted_before_scoring.json',p.OUT)]:
        for f,h in json.loads(manifest.read_text(encoding='utf-8')).items():assert p.sha(base/f)==h,f
    d=p.load(p.DATA/'development.npz');r=json.loads((p.OUT/'report.json').read_text(encoding='utf-8'))
    masks=np.ones((10,5),dtype=np.float32)
    pairs=[(j,k) for j in range(5) for k in range(j+1,5)]
    for i,pair in enumerate(pairs):masks[i,list(pair)]=0
    c=np.repeat(d['context'],480,0);x=np.repeat(d['x'][:,None],10,1).reshape(-1,5)
    y=np.repeat(d['y'][:,None],10,1).reshape(-1);m=np.tile(np.repeat(masks,48,0),(64,1));error=0.;cells=0
    for split in ('source','development'):
        data=p.load(p.DATA/f'{split}.npz')
        for sx,sy,sm,context in zip(data['sx'],data['sy'],data['sm'],data['context']):
            rebuilt=p.context(sx,sy,sm)[0];assert np.max(abs(rebuilt-context))<2e-5
    for key,summary in r['summary'].items():
        saved=p.load(p.OUT/f'pred_{key}.npz');mus=[];lvs=[]
        if key!='ridge':
            mode=key.rsplit('_',1)[0];w=torch.load(p.OUT/f'{key}.pt',weights_only=True)
            with torch.no_grad():
                for start_at in range(0,len(x),512):
                    mu,lv=forward(w,mode,*(torch.tensor(v[start_at:start_at+512]) for v in (c,x,m)))
                    mus.extend(mu.numpy());lvs.extend(lv.numpy())
        else:
            for t in range(64):
                sx,sy,sm=d['sx'][t],d['sy'][t],d['sm'][t];n=sm.sum(0).clip(1)
                mean=(sx*sm).sum(0)/n;scale=np.sqrt((((sx-mean)*sm)**2).sum(0)/n).clip(.1)
                z=(sx-mean)/scale*sm
                for mask in masks:
                    keep=mask.astype(bool);a=np.column_stack([np.ones(48),z[:,keep]])
                    penalty=np.diag([0.,1.,1.,1.]);normal=a.T@a+penalty
                    beta=np.linalg.solve(normal,a.T@sy);hat=np.sum(a*np.linalg.solve(normal,a.T).T,1)
                    var=max(float(np.mean(((sy-a@beta)/(1-hat).clip(.05))**2)),.01)
                    mus.extend(np.column_stack([np.ones(48),d['x'][t][:,keep]])@beta);lvs.extend([np.log(var)]*48)
        mu=np.array(mus);lv=np.array(lvs);err=max(float(abs(mu-saved['mu']).max()),float(abs(lv-saved['lv']).max()))
        assert err<2e-5,(key,err);error=max(error,err)
        loss=.5*(np.log(2*np.pi)+saved['lv']+(y-saved['mu'])**2/np.exp(saved['lv']))
        mse=(y-saved['mu'])**2;coverage=abs(y-saved['mu'])<=1.96*np.exp(saved['lv']/2)
        for name,arr in [('nll',loss),('mse',mse),('coverage',coverage)]:
            assert np.allclose(arr,saved[name],rtol=1e-6,atol=2e-5),(key,name)
            assert abs(float(arr.mean())-summary[name])<2e-5
            assert np.allclose(arr.reshape(64,10,48).mean(2),saved['cell_'+name],rtol=1e-6,atol=2e-5)
        cells+=640
    for seed in p.SEEDS:
        s=r['summary'];main=s[f'cavity_{seed}'];controls=[f'{v}_{seed}' for v in ('aggregate','static','mlp')]+['ridge']
        gains={v:s[v]['nll']-main['nll'] for v in controls};g=r['gates'][str(seed)]
        assert gains==g['gains'];assert g['passed']==(all(v>=.01 for v in gains.values()) and all(main['mse']<=1.01*s[v]['mse'] for v in controls))
    assert r['passed']==all(g['passed'] for g in r['gates'].values())
    result=dict(passed=True,model_prediction_rows=len(y)*13,task_mask_cells=cells,max_prediction_error=error,seconds=time.monotonic()-start)
    p.save(p.OUT/'audit.json',result);print(json.dumps(result))

if __name__=='__main__':main()
