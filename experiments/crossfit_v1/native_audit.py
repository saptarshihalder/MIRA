"""Independent replay and accounting audit of the fixed native family pilot.

Run after native_train.py has finished. This never fits a learned checkpoint or
changes the pretraining freeze; the auditor has a separately recorded hash.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from model import CrossfitOperator
from support_head import SupportHead


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'artifacts/reports/native_meta_inputs'
OUT = ROOT / 'artifacts/reports/native_meta_v1'
SEEDS = (760001, 760002, 760003)
LEARNERS = ('query', 'task_scalar', 'support_head')
TASKS = ('brfss_diabete4', 'brfss_asthma3', 'road_ksi')
INPUT_KEYS = ('query_x', 'query_mask', 'query_logit', 'source_x',
              'source_mask', 'source_y', 'source_logit', 'target_x',
              'target_mask', 'target_y', 'target_logit')
PREDICTIONS = ('frozen', 'target_platt', 'support_logistic', *LEARNERS,
               'guarded_logistic', 'native_linear_converged',
               'global_platt', 'global_simplex')
HISTORICAL = ('native_linear_converged', 'global_platt', 'global_simplex')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def family(task):
    require(task.startswith(('brfss_', 'road_')), 'Unknown dataset family: ' + task)
    return 'brfss' if task.startswith('brfss_') else 'road'


def metric(probability, labels):
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    require(probability.shape == labels.shape, 'Prediction/label shape mismatch')
    require(np.isfinite(probability).all(), 'Nonfinite prediction')
    require(((probability >= 0) & (probability <= 1)).all(), 'Prediction outside [0,1]')
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return dict(nll=float(-(labels * np.log(clipped) +
                           (1 - labels) * np.log1p(-clipped)).mean()),
                brier=float(((probability - labels) ** 2).mean()))


def equalize(module, arguments):
    context = arguments[0]
    return (context.mean(1, keepdim=True).expand_as(context),)


def instantiate(name, seed):
    torch.manual_seed(seed)
    model = SupportHead('query') if name == 'support_head' else CrossfitOperator('query')
    if name == 'task_scalar':
        model.network.register_forward_pre_hook(equalize)
    require(sum(p.numel() for p in model.parameters() if p.requires_grad) == 370,
            'Capacity mismatch for ' + name)
    return model.eval()


def inputs(episode):
    return [torch.as_tensor(episode[key][None],
                            dtype=torch.bool if key.endswith('_mask') else torch.float32)
            for key in INPUT_KEYS]


def design(episode, prefix):
    x, mask = episode[prefix + '_x'], episode[prefix + '_mask']
    interaction = (x[:, :, None] * mask[:, None, :]).reshape(len(x), -1)
    return np.column_stack((x, mask.astype(np.float32), interaction))


def logistic(episode, regularization):
    labels = episode['target_y']
    if len(np.unique(labels)) < 2:
        return np.full(len(episode['query_x']), (labels.sum() + 1) / (len(labels) + 2))
    model = LogisticRegression(C=regularization, max_iter=1000,
                               tol=1e-7, solver='lbfgs')
    model.fit(design(episode, 'target'), labels)
    require(model.n_iter_[0] < 1000, 'Independent logistic replay did not converge')
    return model.predict_proba(design(episode, 'query'))[:, 1]


def platt(episode, ridge):
    z, y = episode['target_logit'].astype(float), episode['target_y'].astype(float)
    a = np.column_stack((z, np.ones(len(z))))
    weight = np.zeros(2)
    def objective(w):
        score = z + a @ w
        return np.mean(np.logaddexp(0, score) - y * score) + ridge * (w @ w) / 2
    for _ in range(100):
        probability = expit(z + a @ weight)
        gradient = a.T @ (probability - y) / len(y) + ridge * weight
        if np.abs(gradient).max() < 1e-10:
            break
        hessian = a.T @ (a * (probability * (1 - probability))[:, None]) / len(y)
        direction = np.linalg.solve(hessian + ridge * np.eye(2), gradient)
        old, step = objective(weight), 1.
        for _ in range(30):
            candidate = weight - step * direction
            if objective(candidate) <= old - 1e-4 * step * (gradient @ direction) + 1e-14:
                break
            step *= .5
        weight = candidate
    gradient = a.T @ (expit(z + a @ weight) - y) / len(y) + ridge * weight
    require(np.abs(gradient).max() < 1e-8, 'Independent Platt replay stationarity failure')
    return expit((1 + weight[0]) * episode['query_logit'] + weight[1])


def guarded_logistic(episode, regularization):
    frozen, candidates, decisions = expit(episode['query_logit']).astype(float), [], []
    for parity in (0, 1):
        fitting = np.arange(parity, len(episode['target_y']), 2)
        verification = np.arange(1 - parity, len(episode['target_y']), 2)
        inner = dict(episode)
        for suffix in ('x', 'mask', 'y', 'logit'):
            inner['target_' + suffix] = episode['target_' + suffix][fitting]
        for suffix in ('x', 'mask', 'logit'):
            inner['query_' + suffix] = np.concatenate((episode['target_' + suffix][verification],
                                                       episode['query_' + suffix]))
        probability = logistic(inner, regularization)
        n = len(verification)
        y, z = episode['target_y'][verification], episode['target_logit'][verification]
        checked = np.clip(probability[:n], 1e-8, 1 - 1e-8)
        gain = np.logaddexp(0, z) - y * z + y * np.log(checked) + (1 - y) * np.log1p(-checked)
        accepted = bool(gain.mean() > 1.645 * gain.std(ddof=1) / np.sqrt(n))
        candidates.append(probability[n:] if accepted else frozen)
        decisions.append(accepted)
    return np.mean(candidates, axis=0), decisions


def compare(left, right, label, tolerance=1e-12):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape, 'Replay shape mismatch: ' + label)
    require(np.isfinite(left).all() and np.isfinite(right).all(), 'Nonfinite replay: ' + label)
    error = float(np.abs(left - right).max()) if left.size else 0.
    require(error <= tolerance, 'Replay mismatch: ' + label + ' error=' + str(error))
    return error


def audit():
    started = time.monotonic()
    torch.set_num_threads(2)
    report = json.loads((OUT / 'report.json').read_text())
    training = json.loads((OUT / 'training.json').read_text())
    freeze_path = ROOT / 'artifacts/manifests/native_meta_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    require(report['freeze_sha256'] == sha(freeze_path), 'Freeze identity changed')
    for filename, digest in freeze.items():
        require(sha(ROOT / filename) == digest, 'Frozen file changed: ' + filename)
    for filename in ('experiments/crossfit_v1/native_train.py',
                     'docs/NATIVE_META_PROTOCOL.md',
                     'artifacts/reports/crossfit_v1_gpu/report.json',
                     'artifacts/reports/native_meta_inputs/manifest.json',
                     'artifacts/reports/native_meta_inputs/raw_replay_audit.json'):
        require(filename in freeze, 'Essential pretraining input absent from freeze: ' + filename)
    prepared_manifest = json.loads((INPUT / 'manifest.json').read_text())
    for filename, digest in prepared_manifest.items():
        require(sha(INPUT / filename) == digest, 'Prepared input changed: ' + filename)
    raw_audit = json.loads((INPUT / 'raw_replay_audit.json').read_text())
    require(raw_audit['passed'] and raw_audit['raw_replayed_episode_files'] == 122,
            'Pretraining raw replay audit not complete')
    config = json.loads((ROOT / 'configs/large_native_v3.json').read_text())
    prepared_report = json.loads((INPUT / 'report.json').read_text())
    require({t['task'] for t in prepared_report['tasks']} == set(TASKS), 'Prepared tasks mismatch')
    for task in prepared_report['tasks']:
        name = task['task']
        spec = next(s for s in config['tasks'] if s['name'] == name)
        raw = ROOT / 'artifacts/runs/large_native_data' / spec['npz']
        old = ROOT / f'artifacts/runs/large_native_a100_v3/results/{name}_seed171001'
        require(sha(raw) == spec['npz_sha256'] == task['raw_sha256'], 'Raw dataset hash mismatch')
        require(sha(old / 'boundaries_and_meta.npz') == task['boundary_sha256'], 'Boundary hash mismatch')
        require(sha(old / 'source_backbone.json') == task['backbone_sha256'], 'Backbone hash mismatch')

    training_files = sorted((INPUT / 'training').glob('*.npz'))
    development_files = sorted((INPUT / 'development').glob('*.npz'))
    require(len(training_files) == 96 and len(development_files) == 26, 'Episode inventory mismatch')
    require(all(str(p.relative_to(INPUT)).replace('\\', '/') in prepared_manifest
                for p in training_files + development_files), 'Unmanifested episode input')
    expected_training = {source: [str(p.relative_to(ROOT)).replace('\\', '/')
                                 for p in training_files if family(p.stem) == source]
                         for source in ('brfss', 'road')}
    require(len(expected_training['brfss']) == 64 and len(expected_training['road']) == 32,
            'Training family counts mismatch')
    expected_cells = set()
    for path in development_files:
        task, group = path.stem.split('_development_')
        require(task in TASKS, 'Unknown development task')
        source = 'road' if family(task) == 'brfss' else 'brfss'
        expected_cells.update((source, seed, task, group) for seed in SEEDS)
    require(len(expected_cells) == 78, 'Development cell grid incomplete')
    expected_models = {f'{source}_{seed}_{name}' for source in ('brfss', 'road')
                       for seed in SEEDS for name in LEARNERS}
    require(set(training) == expected_models, 'Training checkpoint grid mismatch')
    require({p.name for p in OUT.glob('*_weights.npz')} ==
            {key + '_weights.npz' for key in expected_models}, 'Checkpoint file inventory mismatch')
    models, checkpoint_hashes = {}, {}
    for key in sorted(expected_models):
        trace = training[key]
        source, seed, name = trace['source_family'], trace['initialization'], trace['variant']
        require(key == f'{source}_{seed}_{name}', 'Training metadata identity mismatch')
        require(trace['updates'] == 500 and trace['parameters'] == 370,
                'Training updates/capacity mismatch')
        require(trace['training_input_files'] == expected_training[source] and
                trace['training_episodes'] == len(expected_training[source]), 'Training inputs mismatch')
        require(all(family(Path(p).stem) == source for p in trace['training_input_files']),
                'Target-family meta-training leakage')
        require(np.isfinite(trace['max_gradient']) and trace['max_gradient'] > 0, 'Missing gradient evidence')
        require([v['update'] for v in trace['trace']] == list(range(1, 452, 50)) + [500],
                'Training trace updates mismatch')
        require(all(np.isfinite(v['loss']) for v in trace['trace']), 'Nonfinite training loss')
        path = OUT / (key + '_weights.npz')
        require(sha(path) == trace['checkpoint_sha256'], 'Checkpoint hash mismatch: ' + key)
        model = instantiate(name, seed)
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state = read(path)
        require(set(state) == set(original), 'Checkpoint tensor inventory mismatch: ' + key)
        require(all(np.isfinite(v).all() for v in state.values()), 'Nonfinite checkpoint tensor')
        model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
        require(any(not torch.equal(original[k], v) for k, v in model.state_dict().items()),
                'Checkpoint unchanged from initialization')
        models[(source, seed, name)] = model
        checkpoint_hashes[key] = sha(path)

    plan = json.loads((ROOT / 'artifacts/reports/crossfit_v1_gpu/report.json').read_text())['plan']
    rows, seen, expected_prediction_files = [], set(), set()
    max_metric_error = max_model_error = max_control_error = 0.
    adaptation = {name: [] for name in (*LEARNERS, 'guarded_logistic')}
    for row in report['rows']:
        cell = (row['source_family'], row['initialization'], row['task'], row['group'])
        require(cell in expected_cells and cell not in seen, 'Unexpected/duplicate development cell')
        seen.add(cell)
        source, seed, task, group = cell
        require(row['target_family'] == family(task) != source, 'Family holdout metadata mismatch')
        input_path = INPUT / 'development' / f'{task}_development_{group}.npz'
        require(row['input_file'] == str(input_path.relative_to(ROOT)).replace('\\', '/') and
                row['input_sha256'] == sha(input_path), 'Development input identity mismatch')
        expected_file = f'predictions/{source}_{seed}_{task}_development_{group}.npz'
        require(row['prediction_file'] == expected_file, 'Prediction file identity mismatch')
        expected_prediction_files.add(expected_file)
        e, saved = read(input_path), read(OUT / expected_file)
        require(np.array_equal(saved['labels'], e['query_y']), 'Saved labels changed')
        require(np.isin(e['query_y'], [0, 1]).all(), 'Nonbinary query labels')
        for key in (*INPUT_KEYS, 'source_ids', 'target_ids', 'query_ids'):
            require(np.array_equal(saved[key], e[key]), 'Saved episode input changed: ' + key)
        metrics = {name: metric(saved[name], saved['labels']) for name in PREDICTIONS}
        require(set(row['metrics']) == set(PREDICTIONS), 'Reported prediction inventory mismatch')
        for name, value in metrics.items():
            for key in ('nll', 'brier'):
                max_metric_error = max(max_metric_error, compare(value[key], row['metrics'][name][key],
                                                                expected_file + ':' + name + ':' + key))
        tensors = inputs(e)
        require(set(row['decisions']) == set(LEARNERS), 'Decision inventory mismatch')
        for name in LEARNERS:
            model = models[(source, seed, name)]
            with torch.no_grad():
                replay = model.predict(model.prepare(tensors))
            max_model_error = max(max_model_error, compare(replay['hard'][0].numpy(), saved[name],
                                                          expected_file + ':' + name))
            decisions = replay['accepted'][:, 0].tolist()
            require(decisions == row['decisions'][name], 'Adaptation decision replay mismatch')
            adaptation[name].append(float(np.mean(decisions)))
        controls = dict(frozen=expit(e['query_logit']).astype(float),
                        target_platt=platt(e, plan['platt_ridge']),
                        support_logistic=logistic(e, plan['support_logistic_c']))
        controls['guarded_logistic'], decisions = guarded_logistic(e, plan['support_logistic_c'])
        adaptation['guarded_logistic'].append(float(np.mean(decisions)))
        for name, prediction in controls.items():
            max_control_error = max(max_control_error, compare(prediction, saved[name],
                                                              expected_file + ':' + name, 1e-10))
        old_path = ROOT / f'artifacts/runs/large_native_a100_v3/results/{task}_seed171001/predictions/development_{group}.npz'
        old = read(old_path)
        require(np.array_equal(old['labels'], e['query_y']) and
                np.array_equal(old['query_ids'], e['query_ids']), 'Historical panel identity mismatch')
        for name in HISTORICAL:
            compare(old[name], saved[name], expected_file + ':' + name, 0.)
        rows.append(dict(row, metrics=metrics))
    require(seen == expected_cells and len(rows) == 78, 'Missing development cells')
    require({str(p.relative_to(OUT)).replace('\\', '/') for p in (OUT / 'predictions').glob('*.npz')} ==
            expected_prediction_files, 'Prediction artifact inventory mismatch')
    require(json.loads((OUT / 'partial_rows.json').read_text()) == report['rows'], 'Partial/final rows mismatch')

    means, brier_means, individual, gains, harms = {}, {}, {}, {}, {}
    for task in TASKS:
        subset = [row for row in rows if row['task'] == task]
        require(len(subset) == {'brfss_diabete4': 30, 'brfss_asthma3': 30, 'road_ksi': 18}[task],
                'Per-task development cell count mismatch')
        means[task] = {name: float(np.mean([row['metrics'][name]['nll'] for row in subset]))
                       for name in PREDICTIONS}
        brier_means[task] = {name: float(np.mean([row['metrics'][name]['brier'] for row in subset]))
                            for name in PREDICTIONS}
        individual[task] = {str(seed): float(np.mean([row['metrics']['query']['nll']
                            for row in subset if row['initialization'] == seed])) for seed in SEEDS}
        gains[task] = {name: value - means[task]['query'] for name, value in means[task].items()}
        group_gains = [float(np.mean([row['metrics']['frozen']['nll'] - row['metrics']['query']['nll']
                        for row in subset if row['group'] == group])) for group in sorted({r['group'] for r in subset})]
        harms[task] = dict(groups=len(group_gains), harmed_groups=sum(g < 0 for g in group_gains),
                           worst_group_gain=float(min(group_gains)), mean_group_gain=float(np.mean(group_gains)))
        for name in PREDICTIONS:
            compare(means[task][name], report['means'][task][name], task + ':' + name + ':mean')
            compare(gains[task][name], report['gains'][task][name], task + ':' + name + ':gain')
        for seed in SEEDS:
            compare(individual[task][str(seed)], report['initialization_nll'][task][str(seed)],
                    task + ':' + str(seed) + ':initialization')
    gate = all(value['query'] < value[name] for value in means.values()
               for name in ('frozen', 'task_scalar', 'support_head', 'guarded_logistic', 'target_platt'))
    gate = gate and all(gains[task][name] >= .001 for task in TASKS
                        for name in ('frozen', 'task_scalar', 'support_head'))
    require(gate == report['development_gate'], 'Development gate replay mismatch')
    require(report['cloud_calls'] == 0 and report['device'] == 'cpu', 'Compute accounting mismatch')
    return dict(passed=True,scope='Independent audit of used native family-transfer development; no confirmation claim',
                cells=78,checkpoints=18,initializations=list(SEEDS),parameters_per_model=370,updates_per_model=500,
                family_separation_verified=True,freeze_verified_files=len(freeze),prepared_input_hashes=len(prepared_manifest),
                raw_replay_audit_sha256=sha(INPUT / 'raw_replay_audit.json'),
                max_saved_metric_error=max_metric_error,max_checkpoint_prediction_error=max_model_error,
                max_control_prediction_error=max_control_error,decisions_replayed=True,
                task_means=means,task_brier=brier_means,initialization_nll=individual,gains=gains,
                realized_group_harm=harms,adaptation_half_coverage={k:float(np.mean(v)) for k,v in adaptation.items()},
                development_gate=bool(gate),checkpoint_sha256=checkpoint_hashes,freeze_sha256=sha(freeze_path),
                auditor_source_sha256=sha(Path(__file__)),auditor_frozen_before_training=False,
                seconds=time.monotonic()-started)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUT,
                        help='Completed result folder, absolute or relative to repository root')
    args = parser.parse_args()
    OUT = args.folder if args.folder.is_absolute() else ROOT / args.folder
    OUT = OUT.resolve()
    require((OUT / 'report.json').exists(), 'Run after native_train.py completes')
    try:
        result = audit()
    except Exception as error:
        result = dict(passed=False,error_type=type(error).__name__,error=str(error),
                      auditor_source_sha256=sha(Path(__file__)),auditor_frozen_before_training=False)
        (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
        raise
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest = dict(auditor_source_sha256=sha(Path(__file__)),
                    freeze_sha256=result['freeze_sha256'],
                    files={str(p.relative_to(OUT)).replace('\\', '/'): sha(p)
                           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('passed','cells','checkpoints','max_saved_metric_error',
                                           'max_checkpoint_prediction_error','max_control_prediction_error',
                                           'development_gate','seconds')}))


if __name__ == '__main__':
    main()
