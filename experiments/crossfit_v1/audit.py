"""Reconstruct saved losses/data, verify split decisions and replay all checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
import run

ROOT=run.ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',default=str(ROOT/'artifacts/runs/crossfit_v1_gpu/results'));args=parser.parse_args();folder=Path(args.folder)
    torch.set_num_threads(2);report=json.loads((folder/'report.json').read_text());plan=report['plan']
    assert report['device']=='cuda' and report['gpu']
    assert report['protocol_sha256']==run.v1.checksum(ROOT/'configs/crossfit_v1.json')
    assert report['model_sha256']==run.v1.checksum(Path(__file__).with_name('model.py'))
    assert report['runner_sha256']==run.v1.checksum(Path(__file__).with_name('run.py'))
    models={};nll_error=brier_error=cpu_error=0.;arrays=replays=0;max_increase=-float('inf')
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
                        actual=run.v1.metrics(p,saved['labels']);arrays+=1
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
    audit=dict(passed=nll_error<1e-10 and brier_error<1e-10 and cpu_error<1e-5 and max_increase<=1e-8,files=160,probability_arrays=arrays,cpu_prediction_arrays=replays,split_decisions=decision_checks,id_checks=id_checks,max_nll_error=nll_error,max_brier_error=brier_error,max_cpu_error=cpu_error,max_support_objective_increase=max_increase,robustness_development_gate=bool(gate),neural_specific_signflip_gate=bool(neural),contrasts=contrasts,accepted_folds=accepted,scope='Exploratory unadjusted95% seed-level t intervals, widths averaged; no confirmation or universal risk guarantee')
    out=ROOT/'artifacts/reports/crossfit_v1_gpu';out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit));assert audit['passed']

if __name__=='__main__':main()
