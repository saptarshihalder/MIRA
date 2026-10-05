"""Bounded privileged-source mechanism diagnostic; prepare, freeze, then fit once."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/local/support_bellman_probe_v1'
OUT = ROOT / 'artifacts/reports/support_bellman_probe_v1'
FREEZE = ROOT / 'artifacts/manifests/support_bellman_probe_freeze.json'
SEEDS = (814001, 814002, 814003)
COSTS = (.01, .03, .05)
AVAIL = np.array(((1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)), dtype=np.int64)
CONFIG = dict(source_tasks=192, test_tasks=48, support_size=64, missing_probability=.2,
              source_seed=814101, test_seed=814102, seeds=SEEDS, costs=COSTS,
              availability=AVAIL.tolist(), predictor_steps=200, q_steps=150,
              batch_size=512, learning_rate=.003, wall_seconds=180, em_steps=40,
              superiority=.01, planner_noninferiority=.01, horizon=2)
STATES = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)
BITS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)
INDEX = {tuple(s): i for i, s in enumerate(STATES)}
MATCH = np.all((STATES[:, None] < 0) | (STATES[:, None] == BITS[None]), axis=2)
STATE_X = np.concatenate((np.maximum(STATES, 0), STATES >= 0), axis=1).astype(np.float32)
CHILD = {}
for i, s in enumerate(STATES):
    for a in np.where(s < 0)[0]:
        children = []
        for v in (0, 1):
            t = s.copy(); t[a] = v; children.append(INDEX[tuple(t)])
        CHILD[i, a] = np.array(children)
METHODS = ('support_q', 'myopic_q', 'blind_q', 'support_counts_dp',
           'shared_predictor_dp', 'shared_stop', 'population_shared_dp', 'population_bayes_dp')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def conditional(p_full):
    return (p_full @ MATCH.T) / MATCH.sum(1)


def cross_entropy(q, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -q * np.log(p) - (1 - q) * np.log1p(-p)


def support_statistics(x, y):
    matches = np.all((STATES[:, None] < 0) | (STATES[:, None] == x[None]), axis=2)
    count = matches.sum(1)
    positive = matches @ y
    return np.stack((count / CONFIG['support_size'], (positive + 1) / (count + 2)), axis=1).ravel().astype(np.float32)


def make_tasks(number, seed):
    rng = np.random.default_rng(seed)
    families = [(i % 3, (i // 3) % 2) for i in range(number)]
    rng.shuffle(families)
    stats, values, labels, q_full, metadata = [], [], [], [], []
    for selector, parity in families:
        noise = float(rng.uniform(.03, .2))
        other = [j for j in range(3) if j != selector]
        clean = BITS[np.arange(8), np.where(BITS[:, selector] == 0, other[0], other[1])] ^ parity
        py = noise + (1 - 2 * noise) * clean
        ix = rng.integers(0, 8, CONFIG['support_size'])
        y = (rng.random(len(ix)) < py[ix]).astype(np.int64)
        x = BITS[ix].copy(); x[rng.random(x.shape) < .2] = -1
        stats.append(support_statistics(x, y)); values.append(x); labels.append(y); q_full.append(py)
        metadata.append(dict(selector=selector, parity=parity, noise=noise))
    return dict(stats=np.array(stats), support_x=np.array(values), support_y=np.array(labels),
                teacher_full=np.array(q_full), teacher_state=conditional(np.array(q_full))), metadata


def prepare():
    DATA.mkdir(parents=True, exist_ok=False)
    tasks = {}
    for split in ('source', 'test'):
        arrays, metadata = make_tasks(CONFIG[f'{split}_tasks'], CONFIG[f'{split}_seed'])
        np.savez_compressed(DATA / f'{split}.npz', **arrays)
        tasks[split] = metadata
    save(DATA / 'tasks.json', tasks)
    save(DATA / 'config.json', CONFIG)
    save(DATA / 'manifest.json', {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                                  for p in DATA.iterdir() if p.is_file()})
    print('Prepared fixed source/test tasks; root must freeze code, protocol and these files before fitting.')


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(54, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(38, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, support, state):
        return self.decoder(torch.cat((self.encoder(support), state), -1)).squeeze(-1)


class Agent(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = Predictor(); self.predictor.load_state_dict(predictor.state_dict())
        self.predictor.requires_grad_(False)
        self.encoder = nn.Sequential(nn.Linear(54, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(43, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, support, state, avail, cost, horizon):
        return self.decoder(torch.cat((self.encoder(support), state, avail, cost, horizon), -1))


def predictions(model, stats):
    n = len(stats)
    with torch.no_grad():
        p = model(torch.tensor(np.repeat(stats, 27, axis=0)), torch.tensor(np.tile(STATE_X, (n, 1))))
    return p.sigmoid().numpy().reshape(n, 27)


def solve(terminal, avail, cost):
    """Uniform independent feature transitions; output [h, task, state, action]."""
    n = len(terminal)
    q = np.full((3, n, 27, 4), np.nan)
    q[0, :, :, 3] = terminal
    previous = terminal
    for h in (1, 2):
        q[h, :, :, 3] = terminal
        for (s, a), child in CHILD.items():
            if avail[a]: q[h, :, s, a] = cost + previous[:, child].mean(1)
        previous = np.nanmin(q[h], axis=-1)
    return q


def fit_predictor(source, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Predictor(); opt = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    stats, target = torch.tensor(source['stats']), torch.tensor(source['teacher_state'], dtype=torch.float32)
    sx = torch.tensor(STATE_X); trace = []
    for step in range(CONFIG['predictor_steps']):
        task = rng.integers(0, len(stats), CONFIG['batch_size']); state = rng.integers(0, 27, len(task))
        loss = nn.functional.binary_cross_entropy_with_logits(model(stats[task], sx[state]), target[task, state])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 50 == 0: trace.append(float(loss.detach()))
    return model.eval(), trace


def source_targets(source, pred):
    terminal = cross_entropy(source['teacher_state'], pred)
    targets = np.zeros((len(pred), 2, 4, 3, 27, 4), dtype=np.float32)
    valid = np.zeros_like(targets, dtype=bool)
    for ai, avail in enumerate(AVAIL):
        for ci, cost in enumerate(COSTS):
            q = solve(terminal, avail, cost)[1:].transpose(1, 0, 2, 3)
            allowed_state = np.all((STATES < 0) | (avail[None] == 1), axis=1)
            valid[:, :, ai, ci] = np.isfinite(q) & allowed_state[None, None, :, None]
            targets[:, :, ai, ci] = np.nan_to_num(q)
    return torch.tensor(targets), torch.tensor(valid)


def fit_agent(predictor, source, targets, valid, mode, seed):
    torch.manual_seed(seed + 1); rng = np.random.default_rng(seed + 1)
    model = Agent(predictor)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=CONFIG['learning_rate'])
    stats = torch.tensor(source['stats']); sx = torch.tensor(STATE_X); av = torch.tensor(AVAIL, dtype=torch.float32)
    costs = torch.tensor(COSTS, dtype=torch.float32); trace = []
    for step in range(CONFIG['q_steps']):
        n = CONFIG['batch_size']
        t = rng.integers(0, len(stats), n); h = rng.integers(0, 2, n)
        a = rng.integers(0, 4, n); c = rng.integers(0, 3, n); s = rng.integers(0, 27, n)
        actual_h = np.zeros(n, dtype=int) if mode == 'myopic_q' else h
        mask = valid[t, actual_h, a, c, s]
        support = torch.zeros_like(stats[t]) if mode == 'blind_q' else stats[t]
        estimate = model(support, sx[s], av[a], costs[c, None], torch.tensor((h + 1)[:, None] / 2, dtype=torch.float32))
        loss = ((estimate - targets[t, actual_h, a, c, s]).square() * mask).sum() / mask.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 50 == 0: trace.append(float(loss.detach()))
    return model.eval(), trace


def learned_actions(model, stats, avail, cost, blind=False):
    support = np.zeros((54,), dtype=np.float32) if blind else stats
    with torch.no_grad():
        q = model(torch.tensor(np.tile(support, (54, 1))), torch.tensor(np.tile(STATE_X, (2, 1))),
                  torch.tensor(np.tile(avail, (54, 1)), dtype=torch.float32),
                  torch.full((54, 1), cost), torch.tensor(np.repeat((.5, 1.), 27)[:, None], dtype=torch.float32))
    q = q.numpy().reshape(2, 27, 4)
    legal = (STATES < 0) & avail[None].astype(bool)
    q[:, :, :3] = np.where(legal[None], q[:, :, :3], np.inf)
    action = np.full((3, 27), 3, dtype=np.int64); action[1:] = q.argmin(-1)
    return action


def count_belief(x, y):
    """Uniform-feature Bernoulli table; EM integrates missing support coordinates."""
    compatible = np.all((x[:, None] < 0) | (x[:, None] == BITS[None]), axis=2)
    p = np.full(8, .5)
    for _ in range(CONFIG['em_steps']):
        weight = compatible * np.where(y[:, None] == 1, p[None], 1 - p[None])
        weight /= weight.sum(1, keepdims=True)
        p = (1 + (weight * y[:, None]).sum(0)) / (2 + weight.sum(0))
    return conditional(p[None])[0]


def rollout(pred, action, avail, cost):
    ids = np.zeros(8, dtype=int); paths = np.full((8, 2), -1, dtype=np.int64)
    paid = np.zeros(8); done = np.zeros(8, dtype=bool)
    for h in (2, 1):
        for i in range(8):
            if done[i]: continue
            a = int(action[h, ids[i]])
            if a == 3: done[i] = True; continue
            assert avail[a] and STATES[ids[i], a] == -1
            ids[i] = CHILD[ids[i], a][BITS[i, a]]
            paths[i, 2 - h] = a; paid[i] += cost
    return pred[ids], paid, paths


def evaluate(test, fitted):
    records, details, timing = [], {}, []
    for seed, predictor, agents in fitted:
        shared = predictions(predictor, test['stats'])
        arrays = {k: [] for k in ('seed', 'task', 'availability', 'cost', 'method', 'probability', 'paid', 'path', 'expected_nll')}
        for t, stats in enumerate(test['stats']):
            counts = count_belief(test['support_x'][t], test['support_y'][t])
            for ai, avail in enumerate(AVAIL):
                for ci, cost in enumerate(COSTS):
                    for method in METHODS:
                        start = time.perf_counter_ns(); pred = shared[t]
                        if method in agents:
                            action = learned_actions(agents[method], stats, avail, cost, method == 'blind_q')
                            network_states, backup_states = 54, 0
                        elif method == 'shared_stop':
                            action = np.full((3, 27), 3); network_states, backup_states = 0, 0
                        else:
                            if method == 'support_counts_dp': pred = counts
                            if method == 'population_bayes_dp': pred = test['teacher_state'][t]
                            q = test['teacher_state'][t] if method.startswith('population') else pred
                            action = np.nanargmin(solve(cross_entropy(q, pred)[None], avail, cost), axis=-1)[:, 0]
                            network_states, backup_states = 0, 54
                        latency = (time.perf_counter_ns() - start) / 1e6
                        probability, paid, path = rollout(pred, action, avail, cost)
                        loss = cross_entropy(test['teacher_full'][t], probability)
                        row = dict(seed=seed, task=t, availability=ai, cost=ci, method=method,
                                   risk=float((loss + paid).mean()), nll=float(loss.mean()), paid=float(paid.mean()))
                        records.append(row)
                        timing.append(dict(seed=seed, task=t, availability=ai, cost=ci, method=method,
                                           table_ms=latency, q_network_states=network_states, backup_states=backup_states))
                        for key in ('seed', 'task', 'availability', 'cost', 'method'): arrays[key].append(row[key])
                        for key, val in (('probability', probability), ('paid', paid), ('path', path), ('expected_nll', loss)):
                            arrays[key].append(val)
        np.savez_compressed(OUT / f'paths_{seed}.npz', **{k: np.array(v) for k, v in arrays.items()}, bits=BITS)
        details[str(seed)] = {m: float(np.mean([r['risk'] for r in records if r['seed'] == seed and r['method'] == m])) for m in METHODS}
    gates = {}
    for seed, means in details.items():
        gains = {m: means[m] - means['support_q'] for m in ('myopic_q', 'blind_q', 'support_counts_dp')}
        noninferior = means['support_q'] <= means['shared_predictor_dp'] + CONFIG['planner_noninferiority']
        gates[seed] = dict(gains=gains, direct_planner_noninferior=noninferior,
                           passed=all(g >= CONFIG['superiority'] for g in gains.values()) and noninferior)
    return dict(records=records, timing=timing, means=details, gates=gates,
                passed=all(g['passed'] for g in gates.values()))


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    limiter = threadpool_limits(limits=1)
    frozen = json.loads(FREEZE.read_text())
    required = ['experiments/support_bellman_probe.py', 'docs/SUPPORT_BELLMAN_PROTOCOL.md']
    required += [str(p.relative_to(ROOT)).replace('\\', '/') for p in DATA.iterdir() if p.is_file()]
    for path in required: assert path in frozen and sha(ROOT / path) == frozen[path], path
    assert json.loads((DATA / 'config.json').read_text()) == json.loads(json.dumps(CONFIG))
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    watchdog = threading.Timer(CONFIG['wall_seconds'], lambda: os._exit(124)); watchdog.daemon = True; watchdog.start()
    with np.load(DATA / 'source.npz') as z: source = {k: z[k] for k in z.files}
    fitted, training = [], []
    for seed in SEEDS:
        predictor, trace = fit_predictor(source, seed)
        torch.save(predictor.state_dict(), OUT / f'predictor_{seed}.pt')
        training.append(dict(seed=seed, method='predictor', trace=trace, updates=CONFIG['predictor_steps']))
        pred = predictions(predictor, source['stats']); targets, valid = source_targets(source, pred)
        agents = {}
        for mode in METHODS[:3]:
            model, trace = fit_agent(predictor, source, targets, valid, mode, seed)
            torch.save(model.state_dict(), OUT / f'{mode}_{seed}.pt')
            agents[mode] = model
            training.append(dict(seed=seed, method=mode, trace=trace, updates=CONFIG['q_steps'],
                                 parameters=sum(p.numel() for p in model.parameters()),
                                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        assert len({sum(p.numel() for p in m.parameters()) for m in agents.values()}) == 1
        fitted.append((seed, predictor, agents)); save(OUT / 'training.json', training)
    save(OUT / 'fitted_before_scoring.json', {p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    # The optimizer never opens held-out task arrays or receives their population parameters.
    with np.load(DATA / 'test.npz') as z: test = {k: z[k] for k in z.files}
    report = evaluate(test, fitted)
    report.update(seconds=time.monotonic() - start, total_updates=sum(t['updates'] for t in training),
                  config=CONFIG, cloud_calls=0, novelty_claim=False, privileged_source_teacher=True,
                  transfer_scope='New tasks/support draws within the same six Boolean families',
                  latency_scope='Full policy-table construction; excludes common prediction table and support-count fit')
    save(OUT / 'report.json', report)
    save(OUT / 'manifest.json', {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'})
    watchdog.cancel(); limiter.restore_original_limits()
    print(json.dumps({k: report[k] for k in ('means', 'gates', 'passed', 'seconds', 'total_updates')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
