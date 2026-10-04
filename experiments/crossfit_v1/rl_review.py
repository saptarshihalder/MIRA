"""Portable saved-artifact review; optional deterministic single-fit reproduction."""
import argparse,json,csv
import numpy as np
import torch
import rl_run as r

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--retrain-one',action='store_true');cli.add_argument('--output');args=cli.parse_args()
    torch.set_num_threads(1);checked=0
    for folder in (r.OUT,r.ROOT/'artifacts/reports/annual_rl_baselines'):
        for name,digest in json.loads((folder/'manifest.json').read_text()).items():
            assert r.sha(folder/name)==digest,name;checked+=1
        assert json.loads((folder/'audit.json').read_text())['passed']
    unavailable_raw=[]
    for name,digest in json.loads(r.FREEZE.read_text()).items():
        p=r.ROOT/name
        if not p.exists() and name.startswith('artifacts/runs/'):unavailable_raw.append(name);continue
        assert r.sha(p)==digest,name;checked+=1
    # Exact expected score-function gradient equals full-information gradient.
    torch.manual_seed(123);logits=torch.randn(8,7,requires_grad=True);reward=torch.randn(8,7);p=logits.softmax(-1)
    exact=-(p*reward).sum(1).mean();score=-(p.detach()*p.log()*reward).sum(1).mean()
    g1=torch.autograd.grad(exact,logits,retain_graph=True)[0];g2=torch.autograd.grad(score,logits,retain_graph=True)[0]
    baseline=torch.randn(8,1);centered=-(p*(reward-baseline)).sum(1).mean();g3=torch.autograd.grad(centered,logits)[0]
    gradient_error=max(float((g1-g2).abs().max()),float((g1-g3).abs().max()));assert gradient_error<1e-6
    report=json.loads((r.OUT/'report.json').read_text());summary=json.loads((r.OUT/'summary.json').read_text());rows=[]
    for task,s in summary['tasks'].items():
        rr=[x for x in report['rows'] if x['task']==task and x['mode']=='reinforce'];zero=[]
        for x in rr:
            with np.load(r.OUT/f'episode{x["episode"]}_{x["seed"]}_reinforce_predictions.npz') as a:zero.append(float(a['policy'][:,3].mean()))
        seedmeans=[np.mean([x['nll'] for x in rr if x['seed']==seed]) for seed in r.SEEDS]
        best=min(('frozen','platt','additive','conditional_tree'),key=lambda k:s['means'][k])
        rows.append(dict(task=task,rl_nll=s['means']['reinforce'],best_baseline=best,best_nll=s['means'][best],
            anchor_gain=float(np.mean([x['anchor_nll']-x['nll'] for x in rr])),seed_min=float(min(seedmeans)),seed_max=float(max(seedmeans)),mean_zero_action_probability=float(np.mean(zero))))
    reproduction=None
    if args.retrain_one:
        spec=json.loads((r.INPUT/'index.json').read_text())[0]
        with np.load(r.ROOT/spec['input']) as a:model,trace=r.train(a['features'][:384],a['anchor'][:384],a['y'][:384],r.SEEDS[0],'reinforce')
        with np.load(r.OUT/'episode0_991001_reinforce_weights.npz') as w:
            error=max(float(np.max(np.abs(value.detach().numpy()-w[name]))) for name,value in model.state_dict().items())
        assert error==0,error;reproduction=dict(checkpoint='episode0_991001_reinforce',max_weight_error=error,seconds=trace['seconds'],query_labels_supplied=False)
    result=dict(artifact_review_passed=True,files_hashed=checked,raw_reconstruction_inputs_unavailable=unavailable_raw,
        gradient_identity_error=gradient_error,single_fit_reproduction=reproduction,tasks=rows,
        scope='Closed finite RL pilot complete as a negative result; no convergence, architectural novelty or venue-readiness claim')
    if args.output:
        out=r.Path(args.output);out.mkdir(parents=True,exist_ok=False);r.save(out/'review.json',result)
        with (out/'results.csv').open('w',newline='',encoding='utf8') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result))

if __name__=='__main__':main()
