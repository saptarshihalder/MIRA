"""Frozen source-only duration x coverage diagnostic; no closed test access."""
import argparse
import json
import os
import threading
import time
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import support_bellman_probe as p

DATA = p.ROOT / 'artifacts/local/predictor_adequacy_v1'
OUT = p.ROOT / 'artifacts/reports/predictor_adequacy_v1'
FREEZE = p.ROOT / 'artifacts/manifests/predictor_adequacy_freeze.json'
STEPS = (200, 2000)
CONFIG = dict(extra_seed=817101, validation_seed=817102, extra_tasks=2880,
              validation_tasks=96, steps=STEPS, seeds=p.SEEDS, wall_seconds=600,
              cloud_calls=0, ridge=1., optimization_margin=.005, utility_margin=.01)


def read(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def prepare():
    DATA.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name, n, seed in [('extra', 2880, 817101), ('validation', 96, 817102)]:
        arrays, metadata[name] = p.make_tasks(n, seed)
        np.savez_compressed(DATA / (name + '.npz'), **arrays)
    p.save(DATA / 'tasks.json', metadata)
    p.save(DATA / 'config.json', CONFIG)
    files = [p.ROOT / 'experiments/predictor_adequacy.py', p.ROOT / 'experiments/support_bellman_probe.py',
             p.ROOT / 'docs/PREDICTOR_ADEQUACY_PROTOCOL.md', p.DATA / 'source.npz']
    files += sorted(DATA.iterdir())
    p.save(FREEZE, {f.relative_to(p.ROOT).as_posix(): p.sha(f) for f in files})
    print('Prepared and hashed source-only diagnostic; commit before fitting.')


def fit(source, seed, tag):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = p.Predictor()
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    stats = torch.tensor(source['stats'])
    target = torch.tensor(source['teacher_state'], dtype=torch.float32)
    sx = torch.tensor(p.STATE_X)
    trace = []
    for step in range(1, 2001):
        task = rng.integers(0, len(stats), 512)
        state = rng.integers(0, 27, len(task))
        loss = nn.functional.binary_cross_entropy_with_logits(model(stats[task], sx[state]), target[task, state])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            trace.append(dict(step=step, loss=float(loss.detach())))
        if step in STEPS:
            torch.save(model.state_dict(), OUT / f'{tag}_{seed}_{step}.pt')
            if tag == 'small' and step == 200:
                original = torch.load(p.OUT / f'predictor_{seed}.pt', weights_only=True)
                assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    return dict(source=tag, seed=seed, trace=trace, updates=2000,
                parameters=sum(v.numel() for v in model.parameters()))


def ridge_fit(source, tag):
    x = source['stats'].astype(float)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    target = np.clip(source['teacher_state'], 1e-6, 1 - 1e-6)
    target = np.log(target / (1 - target))
    penalty = np.eye(55); penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ target)
    np.savez_compressed(OUT / f'ridge_{tag}.npz', mean=mean, scale=scale, coef=coef)


def policies(pred, truth):
    values = np.empty((len(pred), 4, 3, 2))
    for ai, avail in enumerate(p.AVAIL):
        for ci, cost in enumerate(p.COSTS):
            for oracle in (0, 1):
                terminal = p.cross_entropy(truth['teacher_state'] if oracle else pred, pred)
                actions = np.nanargmin(p.solve(terminal, avail, cost), axis=-1)
                for t in range(len(pred)):
                    pr, paid, _ = p.rollout(pred[t], actions[:, t], avail, cost)
                    values[t, ai, ci, oracle] = (p.cross_entropy(truth['teacher_full'][t], pr) + paid).mean()
    return values


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    limiter = threadpool_limits(limits=1)
    for path, h in json.loads(FREEZE.read_text()).items():
        assert p.sha(p.ROOT / path) == h, path
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    timer = threading.Timer(600, lambda: os._exit(124)); timer.daemon = True; timer.start()
    small, extra = read(p.DATA / 'source.npz'), read(DATA / 'extra.npz')
    large = {k: np.concatenate((small[k], extra[k])) for k in small}
    training = []
    for tag, source in [('small', small), ('large', large)]:
        ridge_fit(source, tag)
        for seed in p.SEEDS:
            training.append(fit(source, seed, tag))
            p.save(OUT / 'training.json', training)
    p.save(OUT / 'fitted_before_scoring.json', {f.name: p.sha(f) for f in OUT.iterdir()})
    validation = read(DATA / 'validation.npz')
    preds = {}
    for tag in ('small', 'large'):
        ridge = read(OUT / f'ridge_{tag}.npz')
        logits = np.column_stack((np.ones(96), (validation['stats'] - ridge['mean']) / ridge['scale'])) @ ridge['coef']
        preds[f'ridge_{tag}'] = 1 / (1 + np.exp(-logits))
        for seed in p.SEEDS:
            for step in STEPS:
                key = f'{tag}_{seed}_{step}'
                model = p.Predictor()
                model.load_state_dict(torch.load(OUT / (key + '.pt'), weights_only=True))
                preds[key] = p.predictions(model.eval(), validation['stats'])
    preds['counts'] = np.array([p.count_belief(x, y) for x, y in zip(validation['support_x'], validation['support_y'])])
    preds['population_bayes'] = validation['teacher_state']
    summary = {}
    for key, pred in preds.items():
        scores = policies(pred, validation)
        nll = p.cross_entropy(validation['teacher_state'], pred)
        summary[key] = dict(risk=float(scores[..., 0].mean()), ceiling=float(scores[..., 1].mean()),
                            state_nll=float(nll.mean()))
        np.savez_compressed(OUT / f'scores_{key}.npz', predictions=pred, scores=scores, nll=nll)
    gates = {}
    for seed in p.SEEDS:
        risk = summary[f'large_{seed}_2000']['risk']
        gains = {k: summary[k]['risk'] - risk for k in ('counts', 'ridge_large')}
        opt_gain = summary[f'large_{seed}_200']['risk'] - risk
        gates[str(seed)] = dict(gains=gains, optimization_gain=opt_gain,
                               optimization_passed=opt_gain >= .005,
                               utility_passed=all(v >= .01 for v in gains.values()))
    report = dict(summary=summary, gates=gates, config=CONFIG,
                  optimization_passed=all(v['optimization_passed'] for v in gates.values()),
                  utility_passed=all(v['utility_passed'] for v in gates.values()),
                  seconds=time.monotonic()-start, cloud_calls=0, total_updates=12000,
                  original_small_checkpoint_weights_exact=True, novelty_claim=False)
    p.save(OUT / 'report.json', report)
    p.save(OUT / 'manifest.json', {f.name: p.sha(f) for f in OUT.iterdir() if f.name != 'manifest.json'})
    timer.cancel(); limiter.restore_original_limits()
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
