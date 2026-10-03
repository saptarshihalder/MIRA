"""Finite v2 meta-regularization development using the frozen v1 data law."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / 'experiments/bridge_v1/run.py'
_spec = importlib.util.spec_from_file_location('bridge_v1_frozen_runner', V1_PATH)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)
DEFAULTS = {**v1.DEFAULTS, 'updates': 600, 'newton_steps': 4, 'fixed_penalty': .1}
MAIN = 'contextual_regularizer'
COMPARATORS = ('frozen', 'target_platt', 'support_logistic', 'fixed_regularizer',
               'global_regularizer', 'type_regularizer', 'target_only_regularizer',
               'global_grid_control')
SMOKE_SEEDS = {940001, 940002, 950001, 950002, 960001, 960002}


def validate_plan(plan):
    v1.validate_plan(plan)
    if plan['updates'] > 600 or plan['updates'] < 1:
        raise ValueError('V2 has a fixed600update maximum')
    seeds = plan['train_seeds'] + plan['validation_seeds'] + plan['development_seeds']
    if set(seeds) & SMOKE_SEEDS:
        raise ValueError('V2 smoke namespace cannot enter scientific evaluation')
    if plan['newton_steps'] != 4 or plan['fixed_penalty'] != .1:
        raise ValueError('Preserve frozen four-step solver and .1fixed control')
    if plan['batch_size'] > len(plan['train_seeds']) * len(v1.REGIMES):
        raise ValueError('Meta batch exceeds the number of training episodes per width')
    expected = plan.get('frozen_v1_source_sha256')
    if expected is not None and expected != v1.checksum(V1_PATH):
        raise ValueError('Frozen v1 generator/scoring source changed')


def make_models(plan):
    from model import (SupportRegularizer, GlobalRegularizer, FeatureTypeRegularizer,
                       FixedRegularizer, TargetOnlyRegularizer)
    common = dict(steps=plan['newton_steps'], solver_dtype=torch.float64)
    constructors = {
        MAIN: lambda: SupportRegularizer(width=plan['width'], **common),
        'global_regularizer': lambda: GlobalRegularizer(**common),
        'type_regularizer': lambda: FeatureTypeRegularizer(width=plan['width'], **common),
        'target_only_regularizer': lambda: TargetOnlyRegularizer(width=plan['width'], **common),
        'fixed_regularizer': lambda: FixedRegularizer(penalty=plan['fixed_penalty'], **common),
    }
    models = {}
    for name, constructor in constructors.items():
        torch.manual_seed(plan['training_seed'])
        models[name] = constructor()
    return models


def hash_episodes(episodes):
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(f"{episode['seed']}:{episode['width']}:{episode['regime']}".encode())
        for key in v1.INPUT_KEYS + ('query_y',):
            value = np.ascontiguousarray(episode[key])
            digest.update(f'{key}:{value.dtype}:{value.shape}'.encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def select_global_grid(cache, plan, device):
    """Search global penalties using only labeled training meta-query episodes."""
    from model import FixedRegularizer
    started = time.monotonic()
    objectives = []
    batches = 0
    episode_fits = 0
    query_labels = sum(int(labels.numel()) for _, labels in cache.values())
    for penalty in np.logspace(-2, 0, 11):
        candidate = FixedRegularizer(penalty=float(penalty), steps=plan['newton_steps'],
                                     solver_dtype=torch.float64).to(device)
        candidate.eval()
        loss_sum = 0.
        queries = 0
        with torch.no_grad():
            for width in plan['train_widths']:
                inputs, labels = cache[width]
                for start in range(0, len(labels), plan['batch_size']):
                    stop = min(start + plan['batch_size'], len(labels))
                    logits = candidate(*(value[start:stop] for value in inputs))
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, labels[start:stop].to(logits.dtype), reduction='sum')
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite global penalty grid objective')
                    loss_sum += float(loss.cpu())
                    queries += int(labels[start:stop].numel())
                    batches += 1
                    episode_fits += stop - start
        objectives.append(dict(penalty=float(penalty), actual_penalty=float(candidate.penalty.cpu()),
                               train_meta_query_nll=loss_sum / queries))
    # Deterministic lowest-index tie resolution; no held-out data are consulted.
    selected = min(range(len(objectives)), key=lambda index: objectives[index]['train_meta_query_nll'])
    winner = FixedRegularizer(penalty=objectives[selected]['penalty'], steps=plan['newton_steps'],
                              solver_dtype=torch.float64).to(device)
    audit = dict(objectives=objectives, selected_index=selected,
                 selected_penalty=objectives[selected]['penalty'],
                 actual_selected_penalty=float(winner.penalty.cpu()), selection_split='train_meta_query_only',
                 training_meta_query_labels=query_labels, candidate_count=11,
                 search_cost=dict(forward_batches=batches, episode_fits=episode_fits,
                                  newton_batch_steps=batches * plan['newton_steps'],
                                  backward_updates=0, seconds=time.monotonic() - started))
    return winner, audit


def train_models(episodes, plan, device, out):
    cache = {}
    for width in plan['train_widths']:
        subset = [episode for episode in episodes if episode['width'] == width]
        cache[width] = (v1.tensor_input(subset, device), torch.as_tensor(
            np.stack([episode['query_y'] for episode in subset]), device=device, dtype=torch.float64))
    models = make_models(plan)
    traces = {}
    for name, model in models.items():
        started = time.monotonic()
        model.to(device)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        history = []
        updates = 0
        if parameters:
            optimizer = torch.optim.Adam(parameters, lr=plan['learning_rate'])
            rng = np.random.default_rng(plan['training_seed'])
            model.train()
            for update in range(plan['updates']):
                width = plan['train_widths'][int(rng.integers(len(plan['train_widths'])))]
                inputs, labels = cache[width]
                index = torch.as_tensor(rng.choice(len(labels), size=plan['batch_size'], replace=False), device=device)
                optimizer.zero_grad()
                corrected_logit = model(*(value[index] for value in inputs))
                loss = nn.functional.binary_cross_entropy_with_logits(corrected_logit, labels[index].to(corrected_logit.dtype))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite meta-query loss in ' + name)
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise RuntimeError('Nonfinite regularizer gradient in ' + name)
                optimizer.step()
                updates += 1
                if update % 10 == 0 or update + 1 == plan['updates']:
                    history.append(dict(update=update + 1, nll=float(loss.detach().cpu())))
        if device == 'cuda':
            torch.cuda.synchronize()
        checkpoint = out / (name + '_weights.npz')
        np.savez_compressed(checkpoint, **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
        traces[name] = dict(parameters=sum(parameter.numel() for parameter in parameters), updates=updates,
            parameter_device=str(parameters[0].device) if parameters else None, train_nll=history,
            solver_device=str(parameters[0].device) if parameters else str(model.penalty.device),
            checkpoint_sha256=v1.checksum(checkpoint), seconds=time.monotonic() - started,
            conditioning='Five coordinate-type penalties;1857storednetworkparameters, support statistics excluded' if name == 'type_regularizer' else
                         'Target-support statistics only' if name == 'target_only_regularizer' else
                         'Source and target support statistics' if name == MAIN else
                         'One global scalar' if name == 'global_regularizer' else 'Fixed .1scalar',
            optimization='Fixed final Adam update; identical meta-query labels/batches for learned controls' if parameters else
                         'Prespecified .1penalty support solver; zero trained meta parameters')
    grid_model, grid_audit = select_global_grid(cache, plan, device)
    grid_checkpoint = out / 'global_grid_control_weights.npz'
    np.savez_compressed(grid_checkpoint, **{key: value.detach().cpu().numpy() for key, value in grid_model.state_dict().items()})
    models['global_grid_control'] = grid_model
    traces['global_grid_control'] = dict(parameters=0, updates=0, parameter_device=None,
        solver_device=str(grid_model.penalty.device), checkpoint_sha256=v1.checksum(grid_checkpoint),
        optimization='11fixed global penalties selected by all training meta-query labels; no validation/development selection',
        conditioning='One selected global scalar; fixed four-step support solve', selection=grid_audit)
    return models, traces


def summarize(results, plan, smoke=False):
    seeds = sorted({row['seed'] for row in results})
    gains = {}
    for name in COMPARATORS:
        values = [np.mean([row['metrics'][name]['nll'] - row['metrics'][MAIN]['nll']
                           for row in results if row['seed'] == seed and row['regime'] in v1.SHIFTED]) for seed in seeds]
        gains[name] = v1.interval(values, plan)
    harms = {}
    for regime in ('no_shift', 'ignorable_shift'):
        values = [np.mean([row['metrics'][MAIN]['nll'] - row['metrics']['frozen']['nll']
                           for row in results if row['seed'] == seed and row['regime'] == regime]) for seed in seeds]
        harms[regime] = v1.interval(values, plan)
    passed = all(gain['mean'] >= plan['useful_effect_nll'] and gain['low'] > 0 for gain in gains.values())
    passed = passed and all(harm['high'] <= plan['max_null_harm_nll'] for harm in harms.values())
    passed = passed and all(row['control_audits']['support_logistic'].get('converged', True) for row in results)
    return dict(gate_pass=None if smoke else bool(passed), primary_model=MAIN,
                shifted_nll_gains=gains, null_harms=harms,
                gate_rule=dict(comparators=list(COMPARATORS), useful_effect_nll=plan['useful_effect_nll'],
                               max_null_harm_upper95_nll=plan['max_null_harm_nll']),
                interval_scope='Paired bootstrap of generator seed; average widths/shifted regimes inside each seed;12 development seeds are uncertainty units',
                scope='Separate engineering smoke; excluded from scientific development' if smoke else
                      'One finite new synthetic meta-regularization development; prior negative v1/native results retained; no confirmed novelty or native efficacy')


def diagnostic_summary(model, episode, device):
    """Record actual fixed-step support solve quality, without a convergence claim."""
    if not hasattr(model, 'diagnostics'):
        return dict(available=False, convergence_claim=False)
    model.eval()
    with torch.no_grad():
        diagnostic = model.diagnostics(*v1.tensor_input([episode], device))
    result = {}
    for key, value in diagnostic.items():
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            result[key] = float(array.item()) if array.size == 1 else dict(
                minimum=float(array.min()), maximum=float(array.max()), mean=float(array.mean()))
        else:
            result[key] = value
    result['convergence_claim'] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/bridge_v2.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = Path(args.config)
    plan = {**DEFAULTS, **json.loads(config.read_text())}
    validate_plan(plan)
    if args.smoke:
        plan.update(source_fit_labels=256, source_labels=16, target_labels=16, queries=32,
                    training_queries=16, train_widths=[6], development_widths=[6],
                    train_seeds=[940001, 940002], validation_seeds=[950001, 950002],
                    development_seeds=[960001, 960002], updates=2, batch_size=2)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA requested but unavailable')
    torch.set_num_threads(4)
    out = Path(args.out)
    if out.exists():
        raise ValueError('Preserve earlier outputs; use a fresh v2directory')
    out.mkdir(parents=True)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            training = [episode for seed in plan['train_seeds'] for width in plan['train_widths']
                        for episode in v1.world(seed, width, plan, training=True)]
            training_hash = hash_episodes(training)
            models, traces = train_models(training, plan, device, out)
            del training
            (out / 'predictions').mkdir()
            panels, data_hashes, metadata = {}, {}, []
            for split in ('validation', 'development'):
                seeds = plan[split + '_seeds']
                episodes = [episode for seed in seeds for width in plan['development_widths']
                            for episode in v1.world(seed, width, plan)]
                data_hashes[split] = hash_episodes(episodes)
                rows = []
                for episode in episodes:
                    predictions, audits = v1.controls(episode, plan)
                    predictions.update({name: v1.model_probability(model, episode, device) for name, model in models.items()})
                    solver_audits = {name: diagnostic_summary(model, episode, device) for name, model in models.items()}
                    name = f"{split}_seed{episode['seed']}_width{episode['width']}_{episode['regime']}"
                    np.savez_compressed(out / 'predictions' / (name + '.npz'), **predictions,
                        labels=episode['query_y'], **{key: episode[key] for key in v1.INPUT_KEYS},
                        **{key: episode[key] for key in ('source_fit_ids', 'source_ids', 'target_ids', 'query_ids',
                                                        'source_frozen_coefficient', 'source_frozen_intercept')})
                    rows.append(dict(split=split, seed=episode['seed'], width=episode['width'], regime=episode['regime'],
                        control_audits=audits, solver_audits=solver_audits,
                        metrics={name: v1.metrics(value, episode['query_y']) for name, value in predictions.items()}))
                    metadata.append(dict(split=split, seed=episode['seed'], width=episode['width'],
                                         regime=episode['regime'], **episode['generator_metadata']))
                panels[split] = rows
                del episodes
            report = summarize(panels['development'], plan, args.smoke)
            failed_controls = [dict(split=row['split'], seed=row['seed'], width=row['width'], regime=row['regime'])
                for rows in panels.values() for row in rows if not row['control_audits']['support_logistic'].get('converged', True)]
            if failed_controls and not args.smoke:
                report['gate_pass'] = False
            report.update(results=panels['development'], validation_results=panels['validation'],
                unconverged_support_controls=failed_controls, training_models=traces, device=device,
                gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None, updates=plan['updates'],
                training_data_sha256=training_hash, validation_data_sha256=data_hashes['validation'],
                development_data_sha256=data_hashes['development'], protocol_sha256=v1.checksum(config),
                source_sha256=v1.checksum(__file__), model_source_sha256=v1.checksum(Path(__file__).with_name('model.py')),
                frozen_v1_source_sha256=v1.checksum(V1_PATH), effective_plan=plan,
                fixed_solver='Four differentiable Newton steps, float64, same basis/offset for every regularizer; not a convergence guarantee',
                torch=torch.__version__, python=platform.python_version(), seconds=time.monotonic() - started)
            (out / 'generator_metadata.json').write_text(json.dumps(metadata, indent=2))
            (out / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({key: report[key] for key in ('gate_pass', 'shifted_nll_gains', 'null_harms', 'device', 'gpu', 'seconds')}), flush=True)
        finally:
            (out / 'warnings.json').write_text(json.dumps([str(item.message) for item in recorded], indent=2))


if __name__ == '__main__':
    main()
