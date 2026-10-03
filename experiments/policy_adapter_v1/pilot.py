"""Development-only predictive-loss meta-training. Never import frozen compiler code."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
CODES = np.array([a for a in itertools.product((0, 1), repeat=4) if any(a)], dtype=np.int64)
MASKS = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.int64)
SIGNS = 1. - 2. * ((MASKS @ CODES.T) % 2)
REGIMES = ('unchanged', 'reversal', 'rate', 'block', 'null')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mechanism(rng):
    return dict(active=int(rng.integers(15)), signal=float(rng.choice([-2., 0., 2.])),
                rate=float(rng.choice([-.8, 0., .8])), block=float(rng.choice([0., .5])),
                value=float(rng.choice([-.3, 0., .3])))


def policy_probs(u, y, policy):
    logits = (policy['rate'] * MASKS.sum(1)[None] +
              .5 * policy['signal'] * (2*y-1)[:, None] * SIGNS[:, policy['active']][None] +
              policy['block'] * ((2*MASKS[:, 0]-1)*(2*MASKS[:, 1]-1))[None] +
              policy['value'] * u[:, None] * (MASKS[:, 0]-.5)[None])
    return np.exp(logits-logsumexp(logits, axis=1, keepdims=True))


def latent(rng, count, beta, bias):
    u = rng.normal(size=count)
    x = .25*u[:, None] + rng.normal(size=(count, 4))
    y = (rng.random(count) < expit(beta*u+bias)).astype(np.int64)
    return u, x, y


def observe(rows, policy, rng, beta, bias):
    u, x, y = rows
    distribution = policy_probs(u, y, policy)
    index = (rng.random(len(y))[:, None] > np.cumsum(distribution, axis=1)).sum(1)
    mask = MASKS[np.minimum(index, 15)]
    observed = x.copy()
    observed[mask.astype(bool)] = np.nan
    p0 = policy_probs(u, np.zeros(len(y)), policy)[np.arange(len(y)), index]
    p1 = policy_probs(u, np.ones(len(y)), policy)[np.arange(len(y)), index]
    oracle = expit(beta*u+bias+np.log(p1/p0))
    return dict(u=u, x=observed, mask=mask, y=y.copy(), oracle=oracle)


def features(observation, frozen):
    # Only observed inputs and frozen coefficients: no labels, oracle or policy.
    u, mask = observation['u'], observation['mask']
    values = np.where(mask.astype(bool), 0., observation['x'])
    parity = 1.-2.*((mask @ CODES.T) % 2)
    offset = frozen[0]*u+frozen[1]
    return np.column_stack((u, np.tanh(u), np.clip(u*u-1., -3., 5.), values,
                            parity, offset)).astype(np.float32)


def world(seed, split, plan):
    rng = np.random.default_rng([seed, plan['namespace']])
    beta, bias = float(rng.uniform(.4, 1.4)), float(rng.choice([-1.1, 0., 1.1]))
    source_policy = mechanism(rng)
    source_rows = latent(rng, plan['source_labels'], beta, bias)
    support_rows = latent(rng, plan['target_labels'], beta, bias)
    query_rows = latent(rng, plan['queries'], beta, bias)
    source = observe(source_rows, source_policy, rng, beta, bias)
    base = LogisticRegression(C=10., max_iter=200).fit(source['u'][:, None], source['y'])
    frozen = np.array([base.coef_[0, 0], base.intercept_[0]])
    policies = []
    if split in ('train', 'validation'):
        for cell in range(2):
            target_policy = mechanism(rng)
            policies.append((str(cell), target_policy))
    else:
        for regime in REGIMES:
            target_policy = source_policy.copy()
            if regime == 'reversal':
                target_policy['signal'] = -source_policy['signal']
            elif regime == 'rate':
                target_policy['rate'] = float(rng.choice([-1.4, 1.4]))
            elif regime == 'block':
                target_policy['block'] = 1.2
                target_policy['value'] = float(rng.choice([-.8, .8]))
            elif regime == 'null':
                target_policy['signal'] = 0.
            policies.append((regime, target_policy))
    episodes = []
    for cell, policy in policies:
        support = observe(support_rows, policy, rng, beta, bias)
        query = observe(query_rows, policy, rng, beta, bias)
        # Equal episode counts/update budgets for no-shift pretraining.
        same_support = observe(support_rows, source_policy, rng, beta, bias)
        same_query = observe(query_rows, source_policy, rng, beta, bias)
        episodes.append(dict(seed=seed, regime=cell, source=source, support=support, query=query,
            source_f=features(source, frozen), support_f=features(support, frozen),
            query_f=features(query, frozen), source_y=source['y'], support_y=support['y'],
            query_y=query['y'], oracle=query['oracle'], frozen=frozen,
            same_support_f=features(same_support, frozen), same_query_f=features(same_query, frozen),
            source_policy=source_policy, target_policy=policy, beta=beta, bias=bias))
    return episodes


class ContextAdapter(nn.Module):
    def __init__(self, width=23):
        super().__init__()
        self.row = nn.Sequential(nn.Linear(width+2, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU())
        context_width = 2*width+18
        self.hyper = nn.Sequential(nn.Linear(2*context_width, 64), nn.SiLU(), nn.Linear(64, width+1))
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

    def encode(self, f, y):
        probability = f[..., -1].sigmoid()
        residual = y-probability
        centered = f-f.mean(1, keepdim=True)
        learned = self.row(torch.cat((f, y[..., None], residual[..., None]), -1)).mean(1)
        return torch.cat((f.mean(1), (centered*residual[..., None]).mean(1), learned,
                          y.mean(1, keepdim=True), residual.mean(1, keepdim=True)), -1)

    def forward(self, source_f, source_y, support_f, support_y, query_f):
        context = torch.cat((self.encode(source_f, source_y), self.encode(support_f, support_y)), -1)
        weights = self.hyper(context)
        return query_f[..., -1]+(query_f*weights[:, None, :-1]).sum(-1)+weights[:, -1, None]


class OrdinaryAdapter(nn.Module):
    def __init__(self, width=23):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 100), nn.SiLU(), nn.Linear(100, 100), nn.SiLU(), nn.Linear(100, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source_f, source_y, support_f, support_y, query_f):
        return query_f[..., -1]+self.net(query_f).squeeze(-1)


def tensor_batch(episodes, same=False):
    keys = ('source_f', 'source_y', 'support_f', 'support_y', 'query_f', 'query_y')
    return tuple(torch.from_numpy(np.stack([e[('same_'+key) if same and key in ('support_f', 'query_f') else key]
                         for e in episodes])).float() for key in keys)


def predict(model, source_f, source_y, support_f, support_y, query_f):
    model.eval()
    arrays = (source_f, source_y, support_f, support_y, query_f)
    with torch.no_grad():
        p = model(*(torch.as_tensor(a, dtype=torch.float32)[None] for a in arrays)).sigmoid()[0]
    return p.numpy().astype(np.float64)


def bce(p, y):
    p = np.clip(p, 1e-7, 1.-1e-7)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def train(plan, output):
    started = time.monotonic()
    torch.set_num_threads(2)
    training = [e for seed in range(*plan['train_seeds']) for e in world(seed, 'train', plan)]
    validation = [e for seed in range(*plan['validation_seeds']) for e in world(seed, 'validation', plan)]
    batches, no_shift_batches = tensor_batch(training), tensor_batch(training, same=True)
    val = tensor_batch(validation)
    traces = {}
    for name in ('context', 'ordinary', 'no_shift'):
        torch.manual_seed(plan['training_seed'])
        model = OrdinaryAdapter() if name == 'ordinary' else ContextAdapter()
        optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
        data = no_shift_batches if name == 'no_shift' else batches
        order_rng = np.random.default_rng(plan['training_seed'])
        history = []
        for epoch in range(plan['epochs']):
            model.train()
            losses = []
            order = order_rng.permutation(len(training))
            for start in range(0, len(order), plan['batch_size']):
                index = order[start:start+plan['batch_size']]
                inputs = tuple(a[index] for a in data)
                loss = nn.functional.binary_cross_entropy_with_logits(model(*inputs[:5]), inputs[5])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val_loss = float(nn.functional.binary_cross_entropy_with_logits(model(*val[:5]), val[5]))
            history.append(dict(epoch=epoch+1, train_nll=float(np.mean(losses)), validation_nll=val_loss))
        # Fixed final epoch; validation is diagnostic, never checkpoint selection.
        state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(output/(name+'_weights.npz'), **state)
        traces[name] = dict(parameters=sum(p.numel() for p in model.parameters()), epochs=history,
                            sha256=checksum(output/(name+'_weights.npz')))
        print(json.dumps(dict(model=name, validation_nll=val_loss)), flush=True)
    metadata = dict(protocol_sha256=checksum(ROOT/'configs/policy_adapter_v1.json'),
        source_sha256=checksum(__file__), seconds=time.monotonic()-started,
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        device='cpu', train_episodes=len(training), validation_episodes=len(validation), models=traces)
    (output/'training.json').write_text(json.dumps(metadata, indent=2))


def offset_fit(x, y, offset, penalty):
    def objective(w):
        z = offset+x@w
        loss = np.mean(np.logaddexp(0., z)-y*z)+.5*penalty*np.sum(w[:-1]**2)
        grad = x.T@(expit(z)-y)/len(y)
        grad[:-1] += penalty*w[:-1]
        return loss, grad
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter':150})
    if not result.success:
        raise RuntimeError('Baseline optimizer did not converge: '+str(result.message))
    return result.x


def simple_predictions(e, plan):
    sf, sy, qf = e['support_f'].astype(float), e['support_y'], e['query_f'].astype(float)
    source_f, source_y = e['source_f'].astype(float), e['source_y']
    predictions = {'frozen':expit(qf[:, -1])}
    intercept = offset_fit(np.ones((len(sy), 1)), sy, sf[:, -1], 0.)
    predictions['intercept'] = expit(qf[:, -1]+intercept[0])
    for name, columns, pooled in (('cv_mask', list(range(7))+[7, 8, 10, 14], False),
                                  ('cv_parity', list(range(22)), False),
                                  ('cv_pooled', list(range(22)), True)):
        def design(f, domain):
            x = np.column_stack((f[:, columns], np.ones(len(f))))
            return np.column_stack((x, domain*x)) if pooled else x
        sx, qx = design(sf, 1.), design(qf, 1.)
        source_x = design(source_f, 0.)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=plan['cv_seed'])
        candidates = []
        for penalty in plan['ridge_grid']:
            loss = []
            for a, b in cv.split(sf, sy):
                x, y, offset = sx[a], sy[a], sf[a, -1]
                if pooled:
                    x, y, offset = np.vstack((source_x, x)), np.r_[source_y, y], np.r_[source_f[:, -1], offset]
                w = offset_fit(x, y, offset, penalty)
                loss.append(bce(expit(sf[b, -1]+sx[b]@w), sy[b]))
            candidates.append(float(np.mean(loss)))
        penalty = plan['ridge_grid'][int(np.argmin(candidates))]
        x, y, offset = sx, sy, sf[:, -1]
        if pooled:
            x, y, offset = np.vstack((source_x, x)), np.r_[source_y, y], np.r_[source_f[:, -1], offset]
        w = offset_fit(x, y, offset, penalty)
        predictions[name] = expit(qf[:, -1]+qx@w)
    # Calibrated strongest centered parity moment; selection/scaling use support only.
    residual = sy-expit(sf[:, -1]+intercept[0])
    moment = np.mean((sf[:, 7:22]-sf[:, 7:22].mean(0))*residual[:, None], 0)
    column = 7+int(np.argmax(np.abs(moment)))
    sx = np.column_stack((sf[:, column], np.ones(len(sf))))
    w = offset_fit(sx, sy, sf[:, -1], 0.01)
    predictions['moment'] = expit(qf[:, -1]+np.column_stack((qf[:, column], np.ones(len(qf))))@w)
    return predictions


def interval(values):
    values = np.asarray(values)
    mean = float(values.mean())
    radius = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean, low=mean-radius, high=mean+radius, independent_tasks=len(values))


def evaluate(plan, output):
    torch.set_num_threads(2)
    training = json.loads((output/'training.json').read_text())
    if training['protocol_sha256'] != checksum(ROOT/'configs/policy_adapter_v1.json') or training['source_sha256'] != checksum(__file__):
        raise ValueError('Source/protocol changed after training')
    models = {}
    for name in ('context', 'ordinary', 'no_shift'):
        model = OrdinaryAdapter() if name == 'ordinary' else ContextAdapter()
        with np.load(output/(name+'_weights.npz')) as weights:
            model.load_state_dict({k:torch.from_numpy(weights[k].copy()) for k in weights.files})
        if checksum(output/(name+'_weights.npz')) != training['models'][name]['sha256']:
            raise ValueError('Checkpoint changed')
        models[name] = model
    results = []
    saved = output/'predictions'
    saved.mkdir()
    for seed in range(*plan['development_seeds']):
        for e in world(seed, 'development', plan):
            predictions = simple_predictions(e, plan)
            for name, model in models.items():
                predictions[name] = predict(model, e['source_f'], e['source_y'], e['support_f'], e['support_y'], e['query_f'])
            np.savez_compressed(saved/(str(seed)+'_'+e['regime']+'.npz'),
                **predictions, source_f=e['source_f'], source_y=e['source_y'], support_f=e['support_f'],
                support_y=e['support_y'], query_f=e['query_f'], query_y=e['query_y'], oracle=e['oracle'],
                source_ids=np.arange(plan['source_labels']),
                support_ids=np.arange(plan['source_labels'],plan['source_labels']+plan['target_labels']),
                query_ids=np.arange(plan['source_labels']+plan['target_labels'],plan['source_labels']+plan['target_labels']+plan['queries']))
            results.append(dict(seed=seed, regime=e['regime'], source_policy=e['source_policy'],
                target_policy=e['target_policy'], beta=e['beta'], bias=e['bias'],
                expected={name:bce(p, e['oracle']) for name,p in predictions.items()},
                empirical={name:bce(p, e['query_y']) for name,p in predictions.items()}))
    names = list(results[0]['expected'])
    means = {regime:{name:float(np.mean([e['expected'][name] for e in results if e['regime']==regime]))
                     for name in names} for regime in REGIMES}
    gains = {}
    for comparator in names:
        if comparator == 'context':
            continue
        gains[comparator] = interval([np.mean([e['expected'][comparator]-e['expected']['context']
                  for e in results if e['seed']==seed]) for seed in range(*plan['development_seeds'])])
    null_harm = means['null']['context']-means['null']['frozen']
    unchanged_harm = means['unchanged']['context']-means['unchanged']['frozen']
    primary = ('ordinary', 'no_shift', 'intercept', 'cv_mask', 'cv_parity', 'cv_pooled', 'moment')
    passed = gains['ordinary']['mean'] >= .01 and all(gains[n]['low']>0 for n in primary) and max(null_harm, unchanged_harm)<=.01
    report = dict(gate_pass=passed, expected_nll=means, gains=gains, null_harm=null_harm,
                  unchanged_harm=unchanged_harm, episodes=len(results), results=results,
                  scope='CPU synthetic development; linear frozen proxy, not TFM, clinical, confirmation or venue-readiness evidence')
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('results', 'expected_nll')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'evaluate'))
    parser.add_argument('--out', default='artifacts/reports/policy_adapter_v1')
    args = parser.parse_args()
    plan = json.loads((ROOT/'configs/policy_adapter_v1.json').read_text())
    output = ROOT/args.out
    if args.phase == 'train':
        if output.exists():
            raise ValueError('Preserve previous output')
        output.mkdir(parents=True)
    elif (output/'report.json').exists() or (output/'predictions').exists():
        raise ValueError('Preserve previous evaluation output')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        (train if args.phase == 'train' else evaluate)(plan, output)
