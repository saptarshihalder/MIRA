"""Recompute every loss and checkpoint prediction, plus seed-level t intervals."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import t
import torch
import run
from model import SpectralFilter

ROOT=run.ROOT
def main():
    torch.set_num_threads(2);folder=ROOT/'artifacts/runs/spectral_v1_gpu/results';report=json.loads((folder/'report.json').read_text());plan=report['plan']
    assert report['gpu'] and report['device']=='cuda'
    assert report['protocol_sha256']==run.v1.checksum(ROOT/'configs/spectral_v1.json')
    assert report['model_sha256']==run.v1.checksum(Path(__file__).with_name('model.py'))
    models={}
    for name in ('learned','global','full'):
        model=SpectralFilter(name)
        with np.load(folder/(name+'_weights.npz'),allow_pickle=False) as state:model.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files})
        models[name]=model.eval()
    nll_error=cpu_error=largest_increase=0.;files=arrays=replays=0
    for row in report['rows']:
        seed,width,regime=(row[k] for k in ('seed','width','regime'))
        episode=next(e for e in run.v1.world(seed,width,plan) if e['regime']==regime)
        with np.load(folder/'predictions'/f'{seed}_{width}_{regime}.npz') as saved:
            for key in run.v1.INPUT_KEYS:
                assert np.allclose(saved[key],episode[key],atol=1e-6,rtol=0),key
            assert np.array_equal(saved['labels'],episode['query_y'])
            for name in row['metrics']:
                nll_error=max(nll_error,abs(run.v1.metrics(saved[name],saved['labels'])['nll']-row['metrics'][name]['nll']));arrays+=1
            inputs=run.v1.tensor_input([episode],'cpu')
            for name,model in models.items():
                with torch.no_grad():logits,obj=model.solve(model.prepare(inputs),True)
                largest_increase=max(largest_increase,float((obj[1:]-obj[:-1]).max()))
                cpu_error=max(cpu_error,float(np.abs(logits.sigmoid()[0].numpy()-saved[name]).max()));replays+=1
        files+=1
    intervals={};gate=True
    for regime in run.v1.REGIMES:
        intervals[regime]={}
        for control in ('frozen','target_platt','support_logistic','global','full'):
            effects=np.array([np.mean([r['metrics'][control]['nll']-r['metrics']['learned']['nll'] for r in report['rows'] if r['seed']==seed and r['regime']==regime]) for seed in plan['development_seeds']])
            radius=t.ppf(.975,len(effects)-1)*effects.std(ddof=1)/np.sqrt(len(effects))
            mean=float(effects.mean());low=float(mean-radius);high=float(mean+radius)
            intervals[regime][control]=dict(gain=mean,lower95=low,upper95=high,n_seeds=len(effects))
            if regime in ('sign_flip','nonlinear_shift') and control in ('full','global','support_logistic'):gate &= mean>=.003 and low>0
            if regime in ('no_shift','ignorable_shift') and control=='frozen':gate &= -low<=.001
    result=dict(passed=nll_error<1e-10 and cpu_error<1e-5 and largest_increase<=1e-8,files=files,probability_arrays=arrays,cpu_replays=replays,max_nll_error=nll_error,max_cpu_error=cpu_error,max_support_objective_increase=largest_increase,development_gate_pass=bool(gate),contrasts=intervals,uncertainty='unadjusted exploratory95% t intervals over8 latent seeds; widths averaged within seed; no confirmation')
    target=ROOT/'artifacts/reports/spectral_v1_gpu';target.mkdir(parents=True,exist_ok=True)
    (target/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert result['passed']
if __name__=='__main__':main()
