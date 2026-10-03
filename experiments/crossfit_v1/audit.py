"""Reconstruct saved losses/data, verify split decisions and replay all checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
import run

ROOT=run.ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default=str(ROOT/'artifacts/runs/crossfit_v1_gpu/results'));args=parser.parse_args();folder=Path(args.folder)
    torch.set_num_threads(2);report=json.loads((folder/'report.json').read_text());plan=report['plan']
    assert report['device']=='cuda' and report['gpu']
    assert report['protocol_sha256']==run.v1.checksum(ROOT/'configs/crossfit_v1.json')
    assert report['model_sha256']==run.v1.checksum(Path(__file__).with_name('model.py'))
    assert report['runner_sha256']==run.v1.checksum(Path(__file__).with_name('run.py'))
    models={};nll_error=brier_error=cpu_error=baseline_error=0.;arrays=replays=0;baseline_decisions=0;max_increase=-float('inf')
    for name in run.NAMES:
        model=run.make_model(name,plan,'cpu')
        with np.load(folder/(name+'_weights.npz'),allow_pickle=False) as state:
            assert all(np.isfinite(state[k]).all() for k in state.files)
            model.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files})
        trace=report['models'][name]
        assert trace['updates']==500
        assert (trace['max_gradient'] is not None and trace['max_gradient']>0) or (trace.get('nonzero_gradient_assertion_passed') and trace.get('changed_state_arrays',0)>0)
        models[name]=model.eval()
    expected={(r['seed'],r['width'],r['regime']):r for r in report['results']};assert len(expected)==160
    decision_checks=0;id_checks=0
    for seed in plan['development_seeds']:
        for width in plan['development_widths']:
            for episode in run.v1.world(seed,width,plan):
                regime=episode['regime'];row=expected[(seed,width,regime)]
                with np.load(folder/'predictions'/f'{seed}_{width}_{regime}.npz',allow_pickle=False) as saved:
                    for key in run.v1.INPUT_KEYS:
                        if key.endswith('_logit'):assert np.allclose(saved[key],episode[key],atol=1e-6,rtol=0)
                        else:assert np.array_equal(saved[key],episode[key])
                    assert np.array_equal(saved['labels'],episode['query_y'])
                    identities=[]
                    for key in ('source_fit_ids','source_ids','target_ids','query_ids'):
                        assert np.array_equal(saved[key],episode[key]);identities.append(set(map(tuple,saved[key].tolist())));id_checks+=1
                    assert all(not identities[i]&identities[j] for i in range(4) for j in range(i+1,4))
                    for name,metrics in row['metrics'].items():
                        p=saved[name];assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                        clipped=p.astype(float).clip(1e-7,1-1e-7);y=saved['labels'].astype(float)
                        actual=dict(nll=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()),brier=float(((p-y)**2).mean()));arrays+=1
                        nll_error=max(nll_error,abs(actual['nll']-metrics['nll']));brier_error=max(brier_error,abs(actual['brier']-metrics['brier']))
                    inputs=[torch.tensor(saved[k][None],dtype=torch.bool if k.endswith('_mask') else torch.float32) for k in run.v1.INPUT_KEYS]
                    for name,model in models.items():
                        with torch.no_grad():output=model.predict(model.prepare(inputs),True)
                        for field,suffix in (('hard',''),('unguarded','_unguarded')):
                            cpu_error=max(cpu_error,float(np.abs(output[field][0].numpy()-saved[name+suffix]).max()));replays+=1
                        assert output['accepted'][:,0].tolist()==row['decisions'][name]['accepted'];decision_checks+=2
                        assert np.allclose(output['gains'][:,0].numpy(),row['decisions'][name]['gains'],atol=1e-10,rtol=0)
                        max_increase=max(max_increase,max(float((o[1:]-o[:-1]).max()) for o in output['objectives']))
                    assert row['control_audits']['support_logistic'].get('converged',True)
                    def design(x,m):
                        return np.column_stack((x,m.astype(np.float32),(x[:,:,None]*m[:,None,:]).reshape(len(x),-1)))
                    guarded=[];baseline_accept=[]
                    for parity in (0,1):
                        fit=np.arange(parity,len(saved['target_y']),2);val=np.arange(1-parity,len(saved['target_y']),2)
                        estimator=LogisticRegression(C=plan['support_logistic_c'],max_iter=1000,tol=1e-7,solver='lbfgs')
                        estimator.fit(design(saved['target_x'][fit],saved['target_mask'][fit]),saved['target_y'][fit]);assert estimator.n_iter_[0]<1000
                        vx=np.concatenate((saved['target_x'][val],saved['query_x']));vm=np.concatenate((saved['target_mask'][val],saved['query_mask']))
                        probability=estimator.predict_proba(design(vx,vm))[:,1];n=len(val);label=saved['target_y'][val];z=saved['target_logit'][val]
                        pv=probability[:n].clip(1e-8,1-1e-8)
                        difference=np.logaddexp(0,z)-label*z+label*np.log(pv)+(1-label)*np.log1p(-pv)
                        keep=bool(difference.mean()>1.645*difference.std(ddof=1)/np.sqrt(n));baseline_accept.append(keep)
                        guarded.append(probability[n:] if keep else saved['frozen'])
                    assert baseline_accept==row['decisions']['guarded_logistic']['accepted'];baseline_decisions+=2
                    baseline_error=max(baseline_error,float(np.abs(np.mean(guarded,axis=0)-saved['guarded_logistic']).max()))
    contrasts={};gate=True;neural=True
    for regime in run.v1.REGIMES:
        contrasts[regime]={}
        for control in ('frozen','guarded_logistic','no_query','scalar','support_logistic'):
            effect=np.array([np.mean([r['metrics'][control]['nll']-r['metrics']['query']['nll'] for r in report['results'] if r['seed']==seed and r['regime']==regime]) for seed in plan['development_seeds']])
            radius=t.ppf(.975,len(effect)-1)*effect.std(ddof=1)/len(effect)**.5
            mean=float(effect.mean());low=float(mean-radius);high=float(mean+radius)
            contrasts[regime][control]=dict(gain=mean,lower95=low,upper95=high,n=20)
            if control=='frozen' and regime in ('no_shift','ignorable_shift'):gate &= -low<=.001
            if control=='frozen' and regime=='sign_flip':gate &= mean>=.003 and low>0
            if regime=='sign_flip' and control in ('guarded_logistic','no_query','scalar'):neural &= mean>=.003 and low>0
    accepted={regime:{name:sum(sum(r['decisions'][name]['accepted']) for r in report['results'] if r['regime']==regime) for name in (*run.NAMES,'guarded_logistic')} for regime in run.v1.REGIMES}
    audit=dict(passed=nll_error<1e-10 and brier_error<1e-10 and cpu_error<1e-5 and baseline_error<1e-8 and max_increase<=1e-8,files=160,probability_arrays=arrays,cpu_prediction_arrays=replays,split_decisions=decision_checks,baseline_split_decisions=baseline_decisions,max_baseline_replay_error=baseline_error,id_checks=id_checks,max_nll_error=nll_error,max_brier_error=brier_error,max_cpu_error=cpu_error,max_support_objective_increase=max_increase,robustness_development_gate=bool(gate),neural_specific_signflip_gate=bool(neural),contrasts=contrasts,accepted_folds=accepted,scope='Exploratory unadjusted95% seed-level t intervals, widths averaged; no confirmation or universal risk guarantee')
    out=ROOT/'artifacts/reports/crossfit_v1_gpu';out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit));assert audit['passed']

if __name__=='__main__':main()
