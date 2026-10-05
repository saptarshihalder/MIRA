"""Replay frozen models and independently enumerate synthetic policy outcomes."""
import json, time, functools
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import support_bellman_probe as p

def independent_dp(pred, truth, avail, cost):
    action = np.full((3, 27), 3, dtype=int)
    @functools.lru_cache(None)
    def value(state, horizon):
        i = p.INDEX[state]
        q = float(truth[np.all((np.array(state) < 0) | (p.BITS == np.array(state)), axis=1)].mean())
        prob = float(np.clip(pred[i], 1e-6, 1-1e-6))
        candidates = [(float(-q*np.log(prob)-(1-q)*np.log1p(-prob)), 3)]
        if horizon:
            for a in range(3):
                if not avail[a] or state[a] >= 0: continue
                children=[]
                for bit in (0,1):
                    s=list(state);s[a]=bit;children.append(value(tuple(s),horizon-1))
                candidates.append((cost+sum(children)/2,a))
        risk,a=min(candidates,key=lambda x:(x[0],x[1]));action[horizon,i]=a
        return risk
    value((-1,-1,-1),2)
    return action, value((-1,-1,-1),2)

def replay(pred, action, truth, avail, cost):
    probs=[];paid=[];paths=[]
    for x in p.BITS:
        s=[-1]*3;path=[-1,-1];charge=0.
        for step in range(2):
            a=int(action[2-step,p.INDEX[tuple(s)]])
            if a==3:break
            assert avail[a] and s[a]<0
            s[a]=int(x[a]);path[step]=a;charge+=cost
        probs.append(pred[p.INDEX[tuple(s)]]);paid.append(charge);paths.append(path)
    probs=np.array(probs);loss=-truth*np.log(np.clip(probs,1e-6,1-1e-6))-(1-truth)*np.log(np.clip(1-probs,1e-6,1-1e-6))
    return probs,np.array(paid),np.array(paths),loss

def main():
    start=time.monotonic();torch.set_num_threads(1);lim=threadpool_limits(limits=1)
    for manifest,base in ((p.FREEZE,p.ROOT),(p.OUT/'manifest.json',p.OUT),(p.OUT/'fitted_before_scoring.json',p.OUT)):
        for f,h in json.loads(manifest.read_text()).items():assert p.sha(base/f)==h,f
    with np.load(p.DATA/'test.npz') as z:test={k:z[k] for k in z.files}
    with np.load(p.DATA/'source.npz') as z:source={k:z[k] for k in z.files}
    for data in (source,test):
        for x,y,stats in zip(data['support_x'],data['support_y'],data['stats']):
            assert np.array_equal(p.support_statistics(x,y),stats)
    report=json.loads((p.OUT/'report.json').read_text());maxerr=0.;cells=0;tie_orders=0;ceilings={};source_checks={}
    for seed in p.SEEDS:
        predictor=p.Predictor();predictor.load_state_dict(torch.load(p.OUT/f'predictor_{seed}.pt',weights_only=True));predictor.eval()
        agents={}
        for mode in p.METHODS[:3]:
            model=p.Agent(predictor);model.load_state_dict(torch.load(p.OUT/f'{mode}_{seed}.pt',weights_only=True));model.eval()
            assert all(torch.equal(v,model.predictor.state_dict()[k]) for k,v in predictor.state_dict().items());agents[mode]=model
        assert len({sum(x.numel() for x in a.parameters()) for a in agents.values()})==1
        pred=p.predictions(predictor,test['stats']);counts=[p.count_belief(x,y) for x,y in zip(test['support_x'],test['support_y'])]
        with np.load(p.OUT/f'paths_{seed}.npz') as z:rows={k:z[k] for k in z.files}
        recomputed={m:[] for m in p.METHODS}
        for j in range(len(rows['task'])):
            t=int(rows['task'][j]);ai=int(rows['availability'][j]);ci=int(rows['cost'][j]);mode=str(rows['method'][j])
            avail,cost=p.AVAIL[ai],p.COSTS[ci];pr=pred[t]
            if mode in agents:act=p.learned_actions(agents[mode],test['stats'][t],avail,cost,mode=='blind_q')
            elif mode=='shared_stop':act=np.full((3,27),3)
            else:
                if mode=='support_counts_dp':pr=counts[t]
                if mode=='population_bayes_dp':pr=test['teacher_state'][t]
                # Conditional averages of these full-state probabilities yield the
                # stored partial-state probabilities for the coherent count model.
                if mode.startswith('population'):
                    act,_=independent_dp(pr,test['teacher_full'][t],avail,cost)
                else:
                    # Discriminative predictor may be incoherent: use its own entropy
                    # at each partial state, preserving the declared plug-in baseline.
                    @functools.lru_cache(None)
                    def val(s,h):
                        v=float(p.cross_entropy(pr[s],pr[s]));best=(v,3)
                        if h:
                            for a in range(3):
                                if avail[a] and p.STATES[s,a]<0:
                                    candidate=(cost+sum(val(int(k),h-1)[0] for k in p.CHILD[s,a])/2,a)
                                    best=min(best,candidate)
                        return best
                    act=np.array([[val(s,h)[1] for s in range(27)] for h in range(3)])
            probability,paid,path,loss=replay(pr,act,test['teacher_full'][t],avail,cost)
            if not np.array_equal(path,rows['path'][j]):
                assert mode.endswith('_dp') and np.array_equal(np.sort(path,axis=1),np.sort(rows['path'][j],axis=1)), (seed,j,mode)
                tie_orders+=1
            for a,b in ((probability,rows['probability'][j]),(paid,rows['paid'][j]),(loss,rows['expected_nll'][j])):maxerr=max(maxerr,float(np.max(np.abs(a-b))))
            recomputed[mode].append(float((loss+paid).mean()));cells+=1
        means={m:float(np.mean(v)) for m,v in recomputed.items()}
        assert all(abs(means[m]-report['means'][str(seed)][m])<1e-6 for m in means)
        ceiling=means['support_counts_dp']-means['population_shared_dp']
        ceilings[str(seed)]={'best_possible_gain_over_counts_with_frozen_predictor':ceiling,'required_gain':.01,'gate_feasible_for_any_policy':ceiling>=.01}
        # Post hoc source-only ceiling diagnostic; no new fit or task generation.
        source_pred=p.predictions(predictor,source['stats']);oracle=[];count_score=[]
        for t in range(len(source_pred)):
            cp=p.count_belief(source['support_x'][t],source['support_y'][t])
            for avail in p.AVAIL:
                for cost in p.COSTS:
                    _,risk=independent_dp(source_pred[t],source['teacher_full'][t],avail,cost);oracle.append(risk)
                    ca=np.nanargmin(p.solve(p.cross_entropy(cp,cp)[None],avail,cost),axis=-1)[:,0]
                    _,pay,_,loss=replay(cp,ca,source['teacher_full'][t],avail,cost);count_score.append(float((pay+loss).mean()))
        source_checks[str(seed)]={'oracle_shared_risk':float(np.mean(oracle)),'count_dp_risk':float(np.mean(count_score)),'best_possible_gain':float(np.mean(count_score)-np.mean(oracle))}
        gate=all(means[m]-means['support_q']>=.01 for m in ('myopic_q','blind_q','support_counts_dp')) and means['support_q']<=means['shared_predictor_dp']+.01
        assert gate==report['gates'][str(seed)]['passed']
    assert maxerr<1e-6 and cells==13824
    result=dict(passed=True,checkpoints=12,policy_cells=cells,equivalent_acquisition_order_ties=tie_orders,audit_repair="Allow equal acquired sets in different DP tie orders and 1e-6 float32/float64 replay tolerance; require predictions and costs to agree; no scientific changes",max_replay_error=maxerr,shared_predictors_exact=True,gate_passed=report['passed'],posthoc_test_predictor_ceiling=ceilings,posthoc_source_predictor_ceiling=source_checks,seconds=time.monotonic()-start,auditor_sha256=p.sha(p.ROOT/'experiments/support_bellman_audit.py'),scope='Independent path/outcome enumeration and dynamic-program recursion; learned Q forward uses frozen model code; no retraining')
    p.save(p.OUT/'audit.json',result);p.save(p.OUT/'manifest.json',{f.name:p.sha(f) for f in p.OUT.iterdir() if f.is_file() and f.name!='manifest.json'})
    print(json.dumps(result));lim.restore_original_limits()

if __name__=='__main__':main()
