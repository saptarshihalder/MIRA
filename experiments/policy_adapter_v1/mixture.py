"""New frozen developmental hypothesis: learn uncertainty over fitted corrections."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
import pilot as p

ROOT = p.ROOT


def experts(sf, sy, qf):
    """Vectorized ridge offset fits: all15 single-parity corrections, no query labels."""
    x = sf[:, 7:22].astype(float)
    w = np.zeros((15, 2))
    for iteration in range(20):
        probability = expit(sf[:, -1, None]+x*w[:, 0]+w[:, 1])
        residual, curvature = probability-sy[:, None], probability*(1-probability)
        gradient = np.column_stack(((x*residual).mean(0)+.01*w[:, 0], residual.mean(0)))
        h00, h01, h11 = (curvature*x*x).mean(0)+.01, (curvature*x).mean(0), curvature.mean(0)+1e-8
        determinant = h00*h11-h01*h01
        step = np.column_stack(((h11*gradient[:, 0]-h01*gradient[:, 1])/determinant,
                               (h00*gradient[:, 1]-h01*gradient[:, 0])/determinant))
        w -= np.clip(step, -2., 2.)
    probability = expit(qf[:, -1, None]+qf[:, 7:22]*w[:, 0]+w[:, 1])
    return np.column_stack((expit(qf[:, -1]), probability)), w


def prepare(e, plan):
    sf, sy, qf = e['support_f'], e['support_y'], e['query_f']
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=plan['cv_seed'])
    oof = np.empty((len(sy), 16))
    for a, b in cv.split(sf, sy):
        oof[b], _ = experts(sf[a], sy[a], sf[b])
    query, w = experts(sf, sy, qf)
    losses = -sy[:, None]*np.log(np.clip(oof, 1e-7, 1-1e-7))-(1-sy[:, None])*np.log1p(-np.clip(oof, 1e-7, 1-1e-7))
    residual = sy-expit(sf[:, -1])
    target_moment = np.mean((sf[:, 7:22]-sf[:, 7:22].mean(0))*residual[:, None], 0)
    source_f, source_y = e['source_f'], e['source_y']
    source_moment = np.mean((source_f[:, 7:22]-source_f[:, 7:22].mean(0))*(source_y-expit(source_f[:, -1]))[:, None], 0)
    moment = np.r_[0., target_moment]
    source = np.r_[0., source_moment]
    meanloss = losses.mean(0)
    null = np.r_[1., np.zeros(15)]
    context = np.column_stack((null, meanloss[0]-meanloss, losses.std(0),
        moment, np.abs(moment), source, np.abs(source), source*moment,
        np.r_[0., w[:, 0]], np.r_[0., w[:, 1]],
        np.full(16, sy.mean()), np.full(16, expit(sf[:, -1]).mean()))).astype(np.float32)
    chosen = 1+int(np.argmax(np.abs(target_moment)))
    return dict(context=context, probability=query.astype(np.float32),
        labels=e['query_y'].astype(np.float32), oracle=e['oracle'], seed=e['seed'], regime=e['regime'],
        cv_choice=int(np.argmin(meanloss)), moment_choice=chosen,
        moment_cv_gain=float(meanloss[0]-meanloss[chosen]))


class Mixture(nn.Module):
    def __init__(self, linear=False):
        super().__init__()
        self.score = nn.Linear(12, 1) if linear else nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
        last = self.score if linear else self.score[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, context, probability):
        score = self.score(context).squeeze(-1)+context[..., 0]*np.log(15.)
        weights = score.softmax(-1)
        return (probability*weights[:, None]).sum(-1)


def predict(model, context, probability):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(context)[None], torch.as_tensor(probability)[None])[0].numpy().astype(float)


def generate(plan, seeds, split):
    return [prepare(e, plan) for seed in range(*seeds) for e in p.world(seed, split, plan)]


def train(plan, out):
    started = time.monotonic()
    torch.set_num_threads(2)
    tasks = generate(plan, plan['train_seeds'], 'train')
    validation = generate(plan, plan['validation_seeds'], 'validation')
    data = [torch.from_numpy(np.stack([e[key] for e in tasks])) for key in ('context','probability','labels')]
    traces = {}
    for name in ('neural_mixture', 'linear_mixture'):
        torch.manual_seed(plan['training_seed'])
        model = Mixture(linear=name=='linear_mixture')
        optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
        rng = np.random.default_rng(plan['training_seed'])
        history = []
        for epoch in range(plan['epochs']):
            losses = []
            order = rng.permutation(len(tasks))
            for start in range(0,len(order),plan['batch_size']):
                index = order[start:start+plan['batch_size']]
                probability = model(data[0][index],data[1][index]).clamp(1e-7,1-1e-7)
                loss = nn.functional.binary_cross_entropy(probability,data[2][index])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
        np.savez_compressed(out/(name+'.npz'),**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        traces[name] = dict(parameters=sum(a.numel() for a in model.parameters()), train_nll=history,
            sha256=p.checksum(out/(name+'.npz')),
            validation_nll=float(np.mean([p.bce(predict(model,e['context'],e['probability']),e['labels']) for e in validation])))
    guards = {str(threshold):float(np.mean([p.bce(e['probability'][:,e['moment_choice']] if e['moment_cv_gain']>threshold else e['probability'][:,0],e['labels']) for e in validation])) for threshold in plan['guard_grid']}
    threshold = min(guards,key=guards.get)
    (out/'training.json').write_text(json.dumps(dict(models=traces, guard_threshold=float(threshold),guard_validation_nll=guards,
        source_sha256=p.checksum(__file__),base_source_sha256=p.checksum(p.__file__),protocol_sha256=p.checksum(ROOT/'configs/policy_mixture_v1.json'),
        seconds=time.monotonic()-started,device='cpu',train_episodes=len(tasks)),indent=2))
    print(json.dumps(dict(models={k:v['validation_nll'] for k,v in traces.items()},guard=float(threshold))),flush=True)


def evaluate(plan,out):
    torch.set_num_threads(2)
    meta=json.loads((out/'training.json').read_text())
    assert meta['source_sha256']==p.checksum(__file__) and meta['base_source_sha256']==p.checksum(p.__file__)
    assert meta['protocol_sha256']==p.checksum(ROOT/'configs/policy_mixture_v1.json')
    models={}
    for name in meta['models']:
        model=Mixture(linear=name=='linear_mixture')
        assert p.checksum(out/(name+'.npz'))==meta['models'][name]['sha256']
        with np.load(out/(name+'.npz')) as state:
            model.load_state_dict({k:torch.from_numpy(state[k].copy()) for k in state.files})
        models[name]=model
    results=[]
    (out/'predictions').mkdir()
    for seed in range(*plan['development_seeds']):
        for e in p.world(seed,'development',plan):
            task=prepare(e,plan)
            predictions=p.simple_predictions(e,plan)
            predictions.update({name:predict(model,task['context'],task['probability']) for name,model in models.items()})
            predictions['cv_select']=task['probability'][:,task['cv_choice']]
            predictions['calibrated_guard']=task['probability'][:,task['moment_choice']] if task['moment_cv_gain']>meta['guard_threshold'] else task['probability'][:,0]
            np.savez_compressed(out/'predictions'/f"{seed}_{e['regime']}.npz",**predictions,context=task['context'],experts=task['probability'],
                source_f=e['source_f'],source_y=e['source_y'],support_f=e['support_f'],support_y=e['support_y'],query_f=e['query_f'],query_y=e['query_y'],oracle=e['oracle'])
            results.append(dict(seed=seed,regime=e['regime'],expected={n:p.bce(pr,e['oracle']) for n,pr in predictions.items()},empirical={n:p.bce(pr,e['query_y']) for n,pr in predictions.items()}))
    names=list(results[0]['expected'])
    means={regime:{name:float(np.mean([r['expected'][name] for r in results if r['regime']==regime])) for name in names} for regime in p.REGIMES}
    gains={name:p.interval([np.mean([r['expected'][name]-r['expected']['neural_mixture'] for r in results if r['seed']==seed]) for seed in range(*plan['development_seeds'])]) for name in names if name!='neural_mixture'}
    harm=means['null']['neural_mixture']-means['null']['frozen']
    primary=('linear_mixture','moment','calibrated_guard','cv_select','cv_parity','cv_pooled')
    passed=all(gains[n]['low']>0 for n in primary) and gains['moment']['mean']>=.01 and harm<=.01
    report=dict(gate_pass=passed,gains=gains,null_harm=harm,expected_nll=means,results=results,
        scope='New synthetic development; same single-parity family, new independent worlds; finite-context uncertainty/shrinkage hypothesis only')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','expected_nll')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('train','evaluate'))
    args=parser.parse_args()
    plan=json.loads((ROOT/'configs/policy_mixture_v1.json').read_text())
    out=ROOT/'artifacts/reports/policy_mixture_v1'
    if args.phase=='train':
        if out.exists(): raise ValueError('Preserve prior output')
        out.mkdir(parents=True)
    elif (out/'predictions').exists(): raise ValueError('Preserve prior predictions')
    (train if args.phase=='train' else evaluate)(plan,out)
