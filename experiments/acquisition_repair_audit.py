"""Validate trained probe artifacts and independently enumerate deployed risks."""
import json,itertools
import numpy as np
import torch
import acquisition_repair_probe as p

def main():
    torch.set_num_threads(1);report=json.loads((p.OUT/'report.json').read_text());assert len(report['results'])==9
    for name,digest in json.loads((p.ROOT/'artifacts/manifests/acquisition_repair_probe_freeze.json').read_text()).items():assert p.sha(p.ROOT/name)==digest
    assert abs(sum(w for _,_,w in p.JOINT)-1)<1e-12
    with np.load(p.OUT/'training_inputs.npz') as a:
        x=a['features'];target=a['targets'];assert len(x)==3456
        assert np.all(target[:,1:]>=0) and np.all(target[:,1:]<=1+1e-6)
        assert np.all(target[:,1:].sum(1)<=2*x[:,16]+1e-6)
        assert np.all(np.abs(target[:,1:]*(1-x[:,8:12]))<1e-6)
    maximum=0.;source={};costs=list(itertools.product(p.TEST_A,p.TEST_COSTS))
    for row in report['results']:
        model=p.Model();path=p.OUT/f'{row["mode"]}_{row["seed"]}.npz';assert p.sha(path)==row['checkpoint_sha256']
        with np.load(path) as a:model.load_state_dict({k:torch.tensor(a[k]) for k in a.files})
        values=[]
        # Enumerate complete source worlds. Policy sees only acquired entries.
        for A,c in costs:
            total=0.
            for full,y,weight in p.JOINT:
                state=(-1,)*4;cost=0.
                for budget in (2,1):
                    choices=[(p.entropy(state),-1)]
                    for action in p.available(state,A):
                        q=min(float(v[0]+v[1:]@c) for v in (p.infer(model,row['mode'],state,A,budget,k,action) for k in range(4)))
                        choices.append((q,action))
                    action=min(choices)[1]
                    if action<0:break
                    assert A&(1<<action) and state[action]<0
                    cost+=c[action];s=list(state);s[action]=full[action];state=tuple(s)
                py=sum(yy*w for _,yy,w in p.conditional(state))
                total+=weight*(cost-np.log(py if y else 1-py))
            values.append(total)
        maximum=max(maximum,float(np.max(np.abs(np.array(values)-row['risks']))))
        assert abs(np.mean(values)-row['mean'])<1e-10
        source.setdefault(row['mode'],[]).append(np.mean(row['source_control']))
    assert maximum<1e-10
    m=report['means'];gate=all(m['repair']<=.98*m[k] for k in ('direct','blind','naive_source'));assert gate==report['mechanism_gate']
    optimal=np.array(report['reference_risks']['optimal'])
    assert all(np.all(np.array(row['risks'])>=optimal-1e-10) for row in report['results'])
    assert np.allclose(report['reference_risks']['exact_feasible'],optimal,atol=1e-12,rtol=0)
    result=dict(passed=True,checkpoints=9,independently_enumerated_cells=216,max_risk_replay_error=maximum,
        training_feasibility_checked=True,source_control_means={k:float(np.mean(v)) for k,v in source.items()},mechanism_gate=gate,
        auditor_sha256=p.sha(p.ROOT/'experiments/acquisition_repair_audit.py'))
    p.save(p.OUT/'audit.json',result)
    p.save(p.OUT/'manifest.json',{x.name:p.sha(x) for x in p.OUT.iterdir() if x.is_file() and x.name!='manifest.json'})
    print(json.dumps(result))

if __name__=='__main__':main()
