"""Independent recursive policy replay of the source-only adequacy diagnostic."""
import functools
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import predictor_adequacy as e
import support_bellman_probe as p


def risk(pred, truth, avail, cost, oracle):
    @functools.lru_cache(None)
    def solve(state, h):
        i = p.INDEX[state]
        prob = float(np.clip(pred[i], 1e-6, 1-1e-6))
        matching = np.all((np.array(state) < 0) | (p.BITS == state), axis=1)
        target = float(truth[matching].mean()) if oracle else float(pred[i])
        best = (-target*np.log(prob)-(1-target)*np.log1p(-prob), 3)
        if h:
            for a in range(3):
                if avail[a] and state[a] < 0:
                    children=[]
                    for bit in (0, 1):
                        child=list(state); child[a]=bit
                        children.append(solve(tuple(child), h-1)[0])
                    best=min(best, (cost+sum(children)/2, a))
        return best
    vals=[]
    for x, target in zip(p.BITS, truth):
        state=[-1]*3; paid=0.
        for h in (2, 1):
            a=solve(tuple(state),h)[1]
            if a == 3: break
            assert avail[a] and state[a]<0
            state[a]=int(x[a]); paid+=cost
        prob=float(np.clip(pred[p.INDEX[tuple(state)]],1e-6,1-1e-6))
        vals.append(-target*np.log(prob)-(1-target)*np.log1p(-prob)+paid)
    return sum(vals)/8


def main():
    start=time.monotonic(); torch.set_num_threads(1)
    limiter=threadpool_limits(limits=1)
    for manifest, base in [(e.FREEZE,p.ROOT),(e.OUT/'manifest.json',e.OUT),
                           (e.OUT/'fitted_before_scoring.json',e.OUT)]:
        for path,h in json.loads(manifest.read_text()).items(): assert p.sha(base/path)==h,path
    validation=e.read(e.DATA/'validation.npz')
    metadata=json.loads((e.DATA/'tasks.json').read_text())['validation']
    for t, m in enumerate(metadata):
        x,y=validation['support_x'][t],validation['support_y'][t]
        stats=[]
        for state in p.STATES:
            take=np.ones(len(x),dtype=bool)
            for a in range(3):
                if state[a]>=0: take &= x[:,a]==state[a]
            stats.extend((take.sum()/64, (y[take].sum()+1)/(take.sum()+2)))
        assert np.array_equal(np.array(stats,dtype=np.float32),validation['stats'][t])
        other=[a for a in range(3) if a!=m['selector']]
        truth=np.array([m['noise']+(1-2*m['noise'])*(int(b[other[int(b[m['selector']])]]) ^ m['parity']) for b in p.BITS])
        assert np.array_equal(truth,validation['teacher_full'][t])
    report=json.loads((e.OUT/'report.json').read_text())
    maxerr=0.; cells=0
    for key, summary in report['summary'].items():
        saved=e.read(e.OUT/f'scores_{key}.npz')
        if key.startswith(('small_', 'large_')):
            model=p.Predictor(); model.load_state_dict(torch.load(e.OUT/f'{key}.pt',weights_only=True))
            pred=p.predictions(model.eval(),validation['stats'])
            if key.startswith('small_') and key.endswith('_200'):
                original=torch.load(p.OUT/f'predictor_{key.split("_")[1]}.pt',weights_only=True)
                assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
        elif key.startswith('ridge_'):
            r=e.read(e.OUT/f'{key}.npz')
            logits=np.column_stack((np.ones(96),(validation['stats']-r['mean'])/r['scale']))@r['coef']
            pred=1/(1+np.exp(-logits))
        elif key=='population_bayes': pred=validation['teacher_state']
        else:
            pred=[]
            for x,y in zip(validation['support_x'],validation['support_y']):
                compatible=np.all((x[:,None]<0)|(x[:,None]==p.BITS[None]),axis=2)
                table=np.full(8,.5)
                for _ in range(40):
                    w=compatible*np.where(y[:,None],table,1-table)
                    w=w/w.sum(1,keepdims=True)
                    table=(1+(w*y[:,None]).sum(0))/(2+w.sum(0))
                pred.append([table[np.all((s<0)|(p.BITS==s),axis=1)].mean() for s in p.STATES])
            pred=np.array(pred)
        assert np.allclose(pred,saved['predictions'],rtol=0,atol=1e-7),key
        nll=-validation['teacher_state']*np.log(np.clip(pred,1e-6,1-1e-6))-(1-validation['teacher_state'])*np.log1p(-np.clip(pred,1e-6,1-1e-6))
        assert abs(float(nll.mean())-summary['state_nll'])<1e-7
        replay=np.zeros_like(saved['scores'])
        for t in range(96):
            for ai,avail in enumerate(p.AVAIL):
                for ci,cost in enumerate(p.COSTS):
                    for oracle in (0,1):
                        replay[t,ai,ci,oracle]=risk(pred[t],validation['teacher_full'][t],avail,cost,oracle)
                        cells+=1
        err=float(np.max(np.abs(replay-saved['scores'])))
        assert err<1e-6,(key,err)
        maxerr=max(maxerr,err)
        assert abs(float(replay[...,0].mean())-summary['risk'])<1e-6
        assert abs(float(replay[...,1].mean())-summary['ceiling'])<1e-6
    for seed,g in report['gates'].items():
        s=report['summary']; r=s[f'large_{seed}_2000']['risk']
        gains={k:s[k]['risk']-r for k in ('counts','ridge_large')}
        assert gains==g['gains']
        assert g['utility_passed']==all(v>=.01 for v in gains.values())
        assert g['optimization_gain']==s[f'large_{seed}_200']['risk']-r
        assert g['optimization_passed']==(g['optimization_gain']>=.005)
    assert report['utility_passed']==all(g['utility_passed'] for g in report['gates'].values())
    assert report['optimization_passed']==all(g['optimization_passed'] for g in report['gates'].values())
    result=dict(passed=True,cells=cells,model_tables=len(report['summary']),max_error=maxerr,
                source_boundary='No closed test panel opened',seconds=time.monotonic()-start)
    p.save(e.OUT/'audit.json',result)
    limiter.restore_original_limits(); print(json.dumps(result))


if __name__=='__main__': main()
