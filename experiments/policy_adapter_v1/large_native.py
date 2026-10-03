"""Bounded, grouped native CUDA development; all outcomes and controls retained."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch import nn

import mixture as m
import native as n
import pilot as p

ROOT = p.ROOT
DEFAULTS = dict(backbone_labels=50000, min_backbone_labels=16384,
                source_labels=128, train_episodes=32, target_labels=128,
                queries=512, min_group_rows=640, max_group_queries=2048,
                epochs=200, batch_size=8, learning_rate=.001, cv_seed=171011,
                support_intercept_ridge=.01, support_platt_ridge=.1,
                global_platt_ridge=.001, lbfgs_max_iter=2000,
                lbfgs_tolerance=1e-7)


def ordered(ids, data, seed):
    """Order by public row identity and seed; labels/features never enter."""
    return np.asarray(sorted(ids, key=lambda i: hashlib.sha256(
        f"mira-large-native:{seed}:{int(data['year'][i])}:{int(data['row_id'][i])}".encode()
    ).digest()), dtype=np.int64)


def load_data(path, columns):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in ('x', 'y', 'groups', 'year', 'row_id')}
    count = len(data['y'])
    if data['x'].shape != (count, len(columns)) or any(data[key].shape != (count,) for key in ('y', 'groups', 'year', 'row_id')):
        raise ValueError('Native array dimensions do not match the column manifest')
    if not np.isin(data['y'], [0, 1]).all() or not np.isin(data['year'], [2024, 2025]).all():
        raise ValueError('Require binary outcomes and declared 2024/2025 years')
    for key in ('groups', 'row_id'):
        if not np.isfinite(data[key]).all() or not np.equal(data[key], np.floor(data[key])).all():
            raise ValueError('Groups and row identities must be finite integers')
        data[key] = data[key].astype(np.int64)
    identity = np.column_stack((data['year'], data['row_id']))
    if len(np.unique(identity, axis=0)) != count:
        raise ValueError('Duplicate (year,row_id) identity')
    if np.isinf(data['x']).any():
        raise ValueError('Native features contain infinity')
    data['x'] = data['x'].astype(np.float32)
    data['y'] = data['y'].astype(np.int64)
    return data


def partitions(data, plan, seed):
    """Reserve disjoint within-group meta episodes before source fitting."""
    groups, years = data['groups'], data['year']
    source = np.flatnonzero((years == 2024) & (groups % 5 < 3))
    width = plan['target_labels'] + plan['queries']
    queues = {int(group): ordered(source[groups[source] == group], data, seed)
              for group in sorted(np.unique(groups[source]))}
    offsets = {group: 0 for group in queues}
    training = []
    while len(training) < plan['train_episodes']:
        progress = False
        for group, ids in queues.items():
            if len(training) == plan['train_episodes']:
                break
            start = offsets[group]
            if start + width <= len(ids):
                training.append((group, ids[start:start + width]))
                offsets[group] += width
                progress = True
        if not progress:
            raise ValueError('Insufficient disjoint within-source-group meta episodes')
    meta_ids = np.concatenate([ids for _, ids in training])
    remaining = ordered(np.setdiff1d(source, meta_ids), data, seed)
    fit_count = min(plan['backbone_labels'], len(remaining) - plan['source_labels'])
    if fit_count < plan['min_backbone_labels']:
        raise ValueError('Insufficient source labels after reserving meta/context rows')
    fit = remaining[:fit_count]
    context = remaining[fit_count:fit_count + plan['source_labels']]
    panels = {}
    omitted = {}
    for split, residue in (('validation', 3), ('development', 4)):
        pool = np.flatnonzero((years == 2025) & (groups % 5 == residue))
        episodes, excluded = [], []
        for group in sorted(np.unique(groups[pool])):
            ids = ordered(pool[groups[pool] == group], data, seed)
            needed = max(plan['min_group_rows'], plan['target_labels'] + plan['queries'])
            if len(ids) < needed:
                excluded.append(dict(group=int(group), rows=len(ids), reason='prespecified minimum group rows'))
                continue
            ids = ids[:plan['target_labels'] + plan['max_group_queries']]
            episodes.append((int(group), ids))
        if not episodes:
            raise ValueError('No eligible held-out groups in ' + split)
        panels[split] = episodes
        omitted[split] = excluded
    all_parts = [fit, context, meta_ids] + [ids for episodes in panels.values() for _, ids in episodes]
    if len(np.unique(np.concatenate(all_parts))) != sum(map(len, all_parts)):
        raise ValueError('Row boundary overlap')
    return dict(fit=fit, context=context, training=training, **panels, omitted=omitted,
                source_available=len(source), source_unused=len(remaining) - fit_count - len(context))


def calibration(z, labels, kind='platt', ridge=.001):
    """Ridge offset calibration; intercept is finite even for one-class support."""
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels, dtype=float)
    design = np.ones((len(z), 1)) if kind == 'intercept' else np.column_stack((z, np.ones(len(z))))
    def objective(theta):
        score = z + design @ theta
        loss = np.mean(np.logaddexp(0., score) - labels * score) + .5 * ridge * (theta @ theta)
        gradient = design.T @ (expit(score) - labels) / len(labels) + ridge * theta
        return loss, gradient
    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=2000, gtol=1e-9, ftol=1e-13))
    coefficient = np.array([1., result.x[0]]) if kind == 'intercept' else np.array([1. + result.x[0], result.x[1]])
    audit = dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit),
                 gradient_max=float(np.max(np.abs(result.jac))), objective=float(result.fun), ridge=float(ridge))
    if not result.success:
        raise RuntimeError('Calibration convergence failed: ' + str(result.message))
    return coefficient, audit


def simplex_stack(probability, labels):
    """Convex global expert stack with analytic gradient and simplex audit."""
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=float)
    width = probability.shape[1]
    def objective(weight):
        prediction = np.clip(probability @ weight, 1e-12, 1 - 1e-12)
        loss = -np.mean(labels * np.log(prediction) + (1 - labels) * np.log1p(-prediction))
        gradient = probability.T @ ((prediction - labels) / (prediction * (1 - prediction))) / len(labels)
        return loss, gradient
    result = minimize(objective, np.full(width, 1 / width), jac=True, method='SLSQP',
                      bounds=[(0., 1.)] * width,
                      constraints=[dict(type='eq', fun=lambda weight: weight.sum() - 1,
                                        jac=lambda weight: np.ones_like(weight))],
                      options=dict(maxiter=2000, ftol=1e-11))
    weights = result.x
    gradient = objective(weights)[1]
    active = weights > 1e-7
    multiplier = float(np.mean(gradient[active]))
    kkt = max(float(np.max(np.abs(gradient[active] - multiplier))),
              float(max(0., np.max(multiplier - gradient[~active]))) if (~active).any() else 0.)
    audit = dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit),
                 objective=float(result.fun), simplex_error=float(abs(weights.sum() - 1)),
                 minimum_weight=float(weights.min()), kkt_residual=kkt)
    if not result.success or audit['simplex_error'] > 1e-7 or weights.min() < -1e-8:
        raise RuntimeError('Simplex stack convergence failed: ' + str(audit))
    return weights, audit


def tensors(tasks, device, dtype=torch.float32):
    return [torch.as_tensor(np.stack([task[key] for task in tasks]), device=device, dtype=dtype)
            for key in ('context', 'probability', 'labels')]


def fit_mixture(tasks, plan, seed, linear=False, converged=False, device='cuda'):
    torch.manual_seed(seed)
    dtype = torch.float64 if converged else torch.float32
    model = m.Mixture(linear=linear).to(device=device, dtype=dtype)
    context, probability, labels = tensors(tasks, device, dtype)
    trace = []
    updates = 0
    def loss_function(index=None):
        c, pr, y = (context, probability, labels) if index is None else (context[index], probability[index], labels[index])
        return nn.functional.binary_cross_entropy(model(c, pr).clamp(1e-7, 1 - 1e-7), y)
    if converged:
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=plan['lbfgs_max_iter'],
                                      tolerance_grad=plan['lbfgs_tolerance'], tolerance_change=1e-12,
                                      history_size=50, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = loss_function()
            loss.backward()
            trace.append(float(loss.detach().cpu()))
            return loss
        optimizer.step(closure)
        closure()
        gradient_max = max(float(parameter.grad.detach().abs().max().cpu()) for parameter in model.parameters())
        updates = int(optimizer.state[next(model.parameters())].get('n_iter', 0))
        optimization = dict(algorithm='full-batch float64 L-BFGS strong Wolfe',
                            gradient_max=gradient_max, tolerance=plan['lbfgs_tolerance'],
                            converged=gradient_max <= plan['lbfgs_tolerance'],
                            success=gradient_max <= plan['lbfgs_tolerance'],
                            function_evaluations=len(trace), starts=1,
                            iterations_limit=plan['lbfgs_max_iter'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
        rng = np.random.default_rng(seed)
        for _ in range(plan['epochs']):
            order = rng.permutation(len(tasks))
            losses = []
            for start in range(0, len(tasks), plan['batch_size']):
                index = torch.as_tensor(order[start:start + plan['batch_size']], device=device)
                optimizer.zero_grad()
                loss = loss_function(index)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                updates += 1
            trace.append(float(np.mean(losses)))
        optimization = dict(algorithm='fixed Adam updates', converged=None)
    return model, dict(parameters=sum(parameter.numel() for parameter in model.parameters()),
                       parameter_device=str(next(model.parameters()).device), updates=updates,
                       train_nll=trace, optimization=optimization,
                       linear_constant_terms='bias and context columns10/11 cancel across experts' if linear else None)


def prediction(model, task):
    parameter = next(model.parameters())
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(task['context'], device=parameter.device, dtype=parameter.dtype)[None],
                     torch.as_tensor(task['probability'], device=parameter.device, dtype=parameter.dtype)[None])[0].cpu().numpy().astype(float)


def prepared_episode(data, ids, group, source_f, source_y, backbone, selected, mean, scale, plan):
    support_count = plan['target_labels']
    feature = n.correction_features(data['x'][ids], selected, backbone.predict_proba(data['x'][ids])[:, 1], mean, scale)
    episode = dict(source_f=source_f, source_y=source_y, support_f=feature[:support_count],
                   support_y=data['y'][ids[:support_count]], query_f=feature[support_count:],
                   query_y=data['y'][ids[support_count:]], oracle=None, seed=int(group), regime='native')
    task = m.prepare(episode, plan)
    task.update(group=int(group), support_ids=ids[:support_count], query_ids=ids[support_count:],
                support_f=episode['support_f'], support_y=episode['support_y'], query_f=episode['query_f'])
    return task


def identity(data, ids):
    return np.column_stack((data['year'][ids], data['row_id'][ids])).astype(np.int64)


def execute_task(spec, plan, seed, data_dir, out):
    from xgboost import XGBClassifier
    import xgboost
    started = time.monotonic()
    path = data_dir / spec['npz']
    digest = p.checksum(path)
    expected = spec.get('npz_sha256')
    if expected is not None and expected != digest:
        raise ValueError('Native NPZ source hash mismatch')
    data = load_data(path, spec['columns'])
    parts = partitions(data, plan, seed)
    out.mkdir(parents=True)
    fit, context = parts['fit'], parts['context']
    backbone = XGBClassifier(device='cuda', tree_method='hist', n_estimators=200, max_depth=5,
                             learning_rate=.05, reg_lambda=1., n_jobs=4, random_state=seed)
    backbone.fit(data['x'][fit], data['y'][fit])
    backbone_config = json.loads(backbone.get_booster().save_config())
    source_device = backbone_config['learner']['generic_param']['device']
    if not source_device.startswith('cuda'):
        raise RuntimeError('Source XGBoost must train on CUDA')
    backbone.save_model(out / 'source_backbone.json')
    missing = np.isnan(data['x'][fit]).mean(0)
    selected = sorted(np.flatnonzero((missing > 0) & (missing < 1)),
                      key=lambda index: (-missing[index], spec['columns'][index]))[:4]
    if len(selected) != 4:
        raise ValueError('Need four nonconstant naturally incomplete channels')
    mean = np.nanmean(data['x'][fit][:, selected], axis=0)
    scale = np.maximum(np.nanstd(data['x'][fit][:, selected], axis=0), 1e-8)
    source_f = n.correction_features(data['x'][context], selected,
                                     backbone.predict_proba(data['x'][context])[:, 1], mean, scale)
    tasks = {split: [prepared_episode(data, ids, group, source_f, data['y'][context],
                                     backbone, selected, mean, scale, plan) for group, ids in parts[split]]
             for split in ('training', 'validation', 'development')}
    models, traces = {}, {}
    for name, linear, converged in (('native_scratch', False, False), ('native_linear', True, False),
                                   ('native_linear_converged', True, True)):
        models[name], traces[name] = fit_mixture(tasks['training'], plan, seed, linear, converged)
        weights = out / (name + '_weights.npz')
        np.savez_compressed(weights, **{key: value.detach().cpu().numpy() for key, value in models[name].state_dict().items()})
        traces[name]['checkpoint_sha256'] = p.checksum(weights)
        traces[name]['validation_nll'] = float(np.mean([p.bce(prediction(models[name], task), task['labels']) for task in tasks['validation']]))
    train_probability = np.concatenate([task['probability'] for task in tasks['training']]).astype(float)
    train_labels = np.concatenate([task['labels'] for task in tasks['training']])
    global_platt, platt_audit = calibration(logit(np.clip(train_probability[:, 0], 1e-7, 1 - 1e-7)),
                                          train_labels, ridge=plan['global_platt_ridge'])
    stack_weights, stack_audit = simplex_stack(train_probability, train_labels)
    np.savez_compressed(out / 'calibration_weights.npz', global_platt=global_platt, global_simplex=stack_weights)
    results = []
    (out / 'predictions').mkdir()
    for split in ('validation', 'development'):
        for task in tasks[split]:
            base = task['probability'][:, 0].astype(float)
            support_z = task['support_f'][:, -1].astype(float)
            query_z = task['query_f'][:, -1].astype(float)
            intercept, intercept_audit = calibration(support_z, task['support_y'], 'intercept', plan['support_intercept_ridge'])
            support_platt, support_audit = calibration(support_z, task['support_y'], ridge=plan['support_platt_ridge'])
            predictions = {name: prediction(model, task) for name, model in models.items()}
            predictions.update(frozen=base, cv_select=task['probability'][:, task['cv_choice']],
                               moment=task['probability'][:, task['moment_choice']],
                               global_platt=expit(global_platt[0] * query_z + global_platt[1]),
                               global_simplex=task['probability'].astype(float) @ stack_weights,
                               support_intercept=expit(query_z + intercept[1]),
                               support_platt=expit(support_platt[0] * query_z + support_platt[1]))
            np.savez_compressed(out / 'predictions' / f"{split}_group{task['group']}.npz", **predictions,
                                labels=task['labels'], experts=task['probability'], context=task['context'],
                                support_f=task['support_f'], support_y=task['support_y'], query_f=task['query_f'],
                                support_ids=identity(data, task['support_ids']), query_ids=identity(data, task['query_ids']),
                                support_intercept_weights=intercept, support_platt_weights=support_platt)
            results.append(dict(split=split, group=task['group'], support=len(task['support_ids']), queries=len(base),
                                nll={name: p.bce(value, task['labels']) for name, value in predictions.items()},
                                calibration=dict(intercept=intercept_audit, platt=support_audit)))
    boundaries = dict(source_fit_ids=identity(data, fit), source_context_ids=identity(data, context),
                      source_f=source_f, source_y=data['y'][context],
                      meta_support_ids=np.stack([identity(data, task['support_ids']) for task in tasks['training']]),
                      meta_query_ids=np.stack([identity(data, task['query_ids']) for task in tasks['training']]),
                      meta_context=np.stack([task['context'] for task in tasks['training']]),
                      meta_experts=np.stack([task['probability'] for task in tasks['training']]),
                      meta_labels=np.stack([task['labels'] for task in tasks['training']]))
    np.savez_compressed(out / 'boundaries_and_meta.npz', **boundaries)
    torch.cuda.synchronize()
    report = dict(task=spec['name'], seed=seed, primary_model='native_scratch', results=results,
                  npz_sha256=digest, source_backbone_sha256=p.checksum(out / 'source_backbone.json'),
                  boundary_sha256=p.checksum(out / 'boundaries_and_meta.npz'),
                  calibration_sha256=p.checksum(out / 'calibration_weights.npz'),
                  selected_columns=[spec['columns'][index] for index in selected], selected_missing_rates=missing[selected].tolist(),
                  eligible_rows=len(data['y']), source_available=parts['source_available'], source_unused=parts['source_unused'],
                  source_fit_labels=len(fit), source_context_labels=len(context), native_meta_episodes=len(tasks['training']),
                  omitted_groups=parts['omitted'], training_models=traces,
                  global_calibration=dict(platt=platt_audit, simplex=stack_audit, meta_query_labels=len(train_labels)),
                  partition='2024 groups%5<3 source;2025 groups%5==3 validation;2025 groups%5==4 development',
                  boundary_disjoint=True, source_backbone_device=source_device, gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, xgboost=xgboost.__version__, python=platform.python_version(),
                  seconds=time.monotonic() - started,
                  scope='Grouped retrospective native CUDA development; three seeds share rows and groups, not independent dataset replication; no confirmation or gate claim')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(task=spec['name'], seed=seed, gpu=report['gpu'], seconds=report['seconds'],
                          source_fit_labels=len(fit), development_groups=sum(row['split'] == 'development' for row in results))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/large_native_v1.json'))
    parser.add_argument('--data', default=str(ROOT / 'artifacts/runs/large_native_data'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--task')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required; no CPU substitute')
    torch.set_num_threads(4)
    config = Path(args.config)
    plan = {**DEFAULTS, **json.loads(config.read_text())}
    seeds = plan['training_seeds']
    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError('Seed was not frozen in protocol')
        seeds = [args.seed]
    specs = plan['tasks']
    if args.task is not None:
        specs = [spec for spec in specs if spec['name'] == args.task]
        if not specs:
            raise ValueError('Task was not frozen in protocol')
    out = Path(args.out)
    if out.exists():
        raise ValueError('Preserve prior outputs')
    out.mkdir(parents=True)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            reports = [execute_task(spec, plan, seed, Path(args.data), out / f"{spec['name']}_seed{seed}")
                       for spec in specs for seed in seeds]
            (out / 'manifest.json').write_text(json.dumps(dict(protocol_sha256=p.checksum(config),
                source_sha256=p.checksum(__file__), reused_sources={str(Path(file).name): p.checksum(file) for file in (m.__file__, n.__file__, p.__file__)},
                reports=[dict(task=report['task'], seed=report['seed'], seconds=report['seconds']) for report in reports]), indent=2))
        finally:
            (out / 'warnings.json').write_text(json.dumps([str(warning.message) for warning in recorded], indent=2))


if __name__ == '__main__':
    main()
