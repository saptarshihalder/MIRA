"""Engineering-only recursive JEPA pilot; diagnostic scores cannot pass a gate."""
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

ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / 'experiments/bridge_v1/run.py'
_spec = importlib.util.spec_from_file_location('recursive_frozen_bridge_v1', V1_PATH)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)
NAMES = ('full_jepa', 'supervised_only', 'no_recursion')
SMOKE_SEEDS = {480001, 480002, 490001, 490002}
REQUIRED = ('training_seed', 'width', 'steps', 'updates', 'batch_size', 'learning_rate',
            'teacher_decay', 'jepa_weight', 'variance_weight', 'covariance_weight',
            'deep_supervision_weight', 'train_seeds', 'development_seeds')


def validate_plan(plan):
    if any(key not in plan for key in REQUIRED):
        raise ValueError('Require every explicit training/loss/EMA protocol field')
    v1.validate_plan(plan)
    if plan['updates'] != 64 or plan['steps'] != 3:
        raise ValueError('Engineering pilot fixes64updates and3recursive steps')
    if set(plan['train_seeds'] + plan['development_seeds']) & SMOKE_SEEDS:
        raise ValueError('Keep smoke seeds outside engineering pilot worlds')
    if plan.get('validation_used', False):
        raise ValueError('No validation-based model or checkpoint selection')
    if not 0 <= plan['teacher_decay'] < 1:
        raise ValueError('Teacher decay must be in[0,1)')


def model_spec(name, plan):
    if name not in NAMES:
        raise ValueError('Unknown engineering control')
    return dict(width=plan['width'], steps=1 if name == 'no_recursion' else plan['steps'],
                use_jepa=name != 'supervised_only')


def make_model(name, plan):
    from model import RecursiveJEPA
    torch.manual_seed(plan['training_seed'])
    return RecursiveJEPA(**model_spec(name, plan))


def restore_model(name, plan, checkpoint, device='cpu'):
    model = make_model(name, plan).to(device)
    with np.load(checkpoint, allow_pickle=False) as state:
        model.load_state_dict({key: torch.from_numpy(state[key].copy()).to(device) for key in state.files})
    model.eval()
    return model


def hash_episodes(episodes):
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(f"{episode['seed']}:{episode['width']}:{episode['regime']}".encode())
        for key in v1.INPUT_KEYS + ('query_y',):
            value = np.ascontiguousarray(episode[key])
            digest.update(f'{key}:{value.dtype}:{value.shape}'.encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def collapse_stats(details):
    result = {}
    for side, latent in details['step_latents'][-1].items():
        value = latent.detach().float().reshape(-1, latent.shape[-1])
        standard_deviation = value.std(0, unbiased=False)
        centered = value - value.mean(0)
        covariance = centered.T @ centered / max(1, len(value) - 1)
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        result[side] = dict(mean_std=float(standard_deviation.mean().cpu()),
            minimum_std=float(standard_deviation.min().cpu()),
            fraction_dimensions_std_below_01=float((standard_deviation < .01).float().mean().cpu()),
            off_diagonal_covariance_mean_square=float(off_diagonal.square().mean().cpu()),
            all_finite=bool(torch.isfinite(value).all()))
    return result


def loss_weights(name, plan):
    auxiliary = name != 'supervised_only'
    return dict(jepa_weight=plan['jepa_weight'] if auxiliary else 0.,
                variance_weight=plan['variance_weight'] if auxiliary else 0.,
                covariance_weight=plan['covariance_weight'] if auxiliary else 0.,
                deep_weight=plan['deep_supervision_weight'])


def train_models(episodes, plan, device, out):
    cache = {}
    for width in plan['train_widths']:
        subset = [episode for episode in episodes if episode['width'] == width]
        cache[width] = (v1.tensor_input(subset, device), torch.as_tensor(
            np.stack([episode['query_y'] for episode in subset]), device=device, dtype=torch.float32))
    models, traces = {}, {}
    for name in NAMES:
        started = time.monotonic()
        model = make_model(name, plan).to(device)
        trainable = {key: value for key, value in model.named_parameters() if value.requires_grad}
        optimizer = torch.optim.Adam(trainable.values(), lr=plan['learning_rate'])
        rng = np.random.default_rng(plan['training_seed'])
        gradient_max = {key: 0. for key in trainable}
        history, collapse = [], {}
        teacher_changes = 0
        model.train()
        for update in range(plan['updates']):
            width = plan['train_widths'][int(rng.integers(len(plan['train_widths'])))]
            inputs, labels = cache[width]
            index = torch.as_tensor(rng.choice(len(labels), size=plan['batch_size'], replace=False), device=device)
            batch = [value[index] for value in inputs]
            optimizer.zero_grad()
            loss, breakdown = model.training_loss(batch, labels[index], **loss_weights(name, plan))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite engineering loss in ' + name)
            loss.backward()
            for key, parameter in trainable.items():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError('Nonfinite gradient in ' + name + ':' + key)
                    gradient_max[key] = max(gradient_max[key], float(parameter.grad.abs().max().detach().cpu()))
            optimizer.step()
            teacher_before = {key: value.detach().clone() for key, value in model.state_dict().items() if key.startswith('teacher.')}
            model.update_teacher(decay=plan['teacher_decay'])
            teacher_changes += int(any(not torch.equal(teacher_before[key], model.state_dict()[key]) for key in teacher_before))
            if update == 0 or update + 1 == plan['updates']:
                with torch.no_grad():
                    collapse[str(update + 1)] = collapse_stats(model.forward_details(*batch))
            if update % 8 == 0 or update + 1 == plan['updates']:
                history.append(dict(update=update + 1, **breakdown))
        if device == 'cuda':
            torch.cuda.synchronize()
        checkpoint = out / (name + '_weights.npz')
        np.savez_compressed(checkpoint, **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
        traces[name] = dict(spec=model_spec(name, plan), parameters=sum(value.numel() for value in model.parameters()),
            trainable_parameters=sum(value.numel() for value in trainable.values()),
            frozen_parameters=sum(value.numel() for value in model.parameters() if not value.requires_grad),
            parameter_device=str(next(iter(trainable.values())).device), updates=plan['updates'],
            loss_weights=loss_weights(name, plan), training_trace=history, collapse_stats=collapse,
            gradient_paths={key: dict(max_abs_gradient=value, nonzero_observed=value > 0) for key, value in gradient_max.items()},
            teacher_ema=dict(decay=plan['teacher_decay'], update_calls=plan['updates'],
                             observed_teacher_updates=int(model.teacher_updates.detach().cpu()),
                             iterations_changing_teacher_parameters=teacher_changes,
                             teacher_gradients_absent=all(value.grad is None for key, value in model.named_parameters() if key.startswith('teacher.'))),
            checkpoint_sha256=v1.checksum(checkpoint), seconds=time.monotonic() - started)
        models[name] = model
    return models, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/recursive_jepa_v1.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = Path(args.config)
    raw = json.loads(config.read_text())
    if any(key not in raw for key in REQUIRED):
        raise ValueError('Missing explicit canonical prototype protocol fields')
    plan = {**v1.DEFAULTS, **raw}
    validate_plan(plan)
    if args.smoke:
        plan.update(source_fit_labels=256, source_labels=16, target_labels=16, queries=32,
            training_queries=16, train_widths=[6], development_widths=[6], train_seeds=[480001, 480002],
            development_seeds=[490001, 490002], updates=2, batch_size=2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA requires a real GPU; CPU substitution forbidden')
    torch.set_num_threads(2)
    out = Path(args.out)
    if out.exists():
        raise ValueError('Preserve prior engineering outputs')
    out.mkdir(parents=True)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            training = [episode for seed in plan['train_seeds'] for width in plan['train_widths']
                        for episode in v1.world(seed, width, plan, training=True)]
            training_hash = hash_episodes(training)
            models, traces = train_models(training, plan, args.device, out)
            del training
            episodes = [episode for seed in plan['development_seeds'] for width in plan['development_widths']
                        for episode in v1.world(seed, width, plan)]
            evaluation_hash = hash_episodes(episodes)
            (out / 'predictions').mkdir()
            results, metadata = [], []
            for episode in episodes:
                predictions, audits = v1.controls(episode, plan)
                predictions.update({name: v1.model_probability(model, episode, args.device) for name, model in models.items()})
                latent_diagnostics = {}
                for name, model in models.items():
                    with torch.no_grad():
                        latent_diagnostics[name] = collapse_stats(model.forward_details(*v1.tensor_input([episode], args.device)))
                np.savez_compressed(out / 'predictions' / f"seed{episode['seed']}_width{episode['width']}_{episode['regime']}.npz",
                    **predictions, labels=episode['query_y'], **{key: episode[key] for key in v1.INPUT_KEYS},
                    **{key: episode[key] for key in ('source_fit_ids', 'source_ids', 'target_ids', 'query_ids',
                                                    'source_frozen_coefficient', 'source_frozen_intercept')})
                results.append(dict(seed=episode['seed'], width=episode['width'], regime=episode['regime'],
                    control_audits=audits, collapse_stats=latent_diagnostics,
                    metrics={name: v1.metrics(value, episode['query_y']) for name, value in predictions.items()}))
                metadata.append(dict(seed=episode['seed'], width=episode['width'], regime=episode['regime'], **episode['generator_metadata']))
            means = {regime: {name: {metric: float(np.mean([row['metrics'][name][metric] for row in results if row['regime'] == regime]))
                        for metric in ('nll', 'brier')} for name in results[0]['metrics']} for regime in v1.REGIMES}
            report = dict(engineering_only=True, gate_pass=None, smoke=bool(args.smoke),
                scope='Bounded recursive JEPA engineering pilot; all scores diagnostic; no efficacy gate, validation selection, confirmation or native claim',
                results=results, regime_means=means, training_models=traces, effective_plan=plan,
                validation_used=False, unused_validation_seeds=plan['validation_seeds'],
                device=args.device, gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                updates=plan['updates'], training_data_sha256=training_hash, development_data_sha256=evaluation_hash,
                protocol_sha256=v1.checksum(config), source_sha256=v1.checksum(__file__),
                model_source_sha256=v1.checksum(Path(__file__).with_name('model.py')), frozen_v1_source_sha256=v1.checksum(V1_PATH),
                torch=torch.__version__, python=platform.python_version(), seconds=time.monotonic() - started)
            (out / 'generator_metadata.json').write_text(json.dumps(metadata, indent=2))
            (out / 'training_trace.json').write_text(json.dumps({name: trace['training_trace'] for name, trace in traces.items()}, indent=2))
            (out / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(dict(engineering_only=True, gate_pass=None, device=args.device, gpu=report['gpu'],
                                  updates=plan['updates'], seconds=report['seconds'])), flush=True)
        finally:
            (out / 'warnings.json').write_text(json.dumps([str(item.message) for item in recorded], indent=2))


if __name__ == '__main__':
    main()
