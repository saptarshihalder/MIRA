"""Frozen synthetic Bridge development; generator metadata never enters learners."""
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
from sklearn.linear_model import LogisticRegression
import sklearn
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
REGIMES = ('sign_flip', 'nonlinear_shift', 'no_shift', 'ignorable_shift')
SHIFTED = ('sign_flip', 'nonlinear_shift')
DEFAULTS = dict(source_fit_labels=2048, source_labels=128, target_labels=128,
                queries=256, training_queries=64, train_widths=[6, 10],
                development_widths=[6, 10, 14], updates=800, batch_size=8,
                learning_rate=.001, width=32, heads=4,
                platt_ridge=.1, support_logistic_c=.1, bootstrap_samples=2000,
                bootstrap_seed=191301, useful_effect_nll=.003, max_null_harm_nll=.001)
INPUT_KEYS = ('query_x', 'query_mask', 'query_logit', 'source_x', 'source_mask',
              'source_y', 'source_logit', 'target_x', 'target_mask', 'target_y', 'target_logit')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_for(seed, width, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(width), int(stream)]))


def identities(seed, width, stream, count, law=0):
    """Semantic identities expose independent generator streams and row positions."""
    return np.column_stack((np.full(count, seed), np.full(count, width), np.full(count, law),
                            np.full(count, stream), np.arange(count))).astype(np.int64)


def covariates(seed, width, stream, count, beta, bias):
    rng = rng_for(seed, width, stream)
    x = rng.normal(size=(count, width))
    probability = expit(x @ beta + .45 * np.tanh(x[:, 0] * x[:, 1]) + bias)
    y = (rng.random(count) < probability).astype(np.float32)
    return x, y


def observe(x, y, policy, seed, width, stream):
    """Evaluator's retrospective label-dependent acquisition mechanism."""
    if policy['regime'] == 'ignorable_shift':
        acquisition = policy.get('base_offset', .9) + policy.get('anchor_effect', .6) * np.tanh(x[:, 0, None]) + policy['base'][None]
    else:
        anchor = np.tanh(x[:, 0])
        if policy['regime'] == 'nonlinear_shift':
            anchor = np.tanh(x[:, 0] * x[:, 1]) + .35 * np.sin(x[:, 0])
        acquisition = policy['base'][None] + (2 * y[:, None] - 1) * anchor[:, None] * policy['effect'][None]
    probability = expit(acquisition)
    mask = rng_for(seed, width, stream).random(x.shape) < probability
    mask[:, :2] = False  # two observed anchors; no hidden value enters a learner
    observed = x.copy()
    observed[mask] = 0.
    return observed.astype(np.float32), mask


def frozen_design(x, mask):
    return np.column_stack((x, mask.astype(np.float32)))


def support_design(x, mask):
    """Fixed generic value-mask interaction basis; no mechanism coefficients."""
    interaction = (x[:, :, None] * mask[:, None, :]).reshape(len(x), -1)
    return np.column_stack((x, mask.astype(np.float32), interaction))


def world(seed, width, plan, training=False):
    if width < 3:
        raise ValueError('Need two observed anchors and at least one incomplete feature')
    parameters = rng_for(seed, width, 1)
    beta = parameters.normal(size=width)
    beta[:2] += [.6, -.4]
    beta *= 1.3 / np.linalg.norm(beta)
    bias = float(parameters.uniform(-.25, .25))
    base = -.85 + .2 * parameters.normal(size=width)
    effect = parameters.choice([-1., 1.], size=width) * parameters.uniform(1.1, 1.8, size=width)
    source_policy = dict(regime='no_shift', base=base, effect=effect)
    fit_raw, fit_y = covariates(seed, width, 10, plan['source_fit_labels'], beta, bias)
    fit_x, fit_mask = observe(fit_raw, fit_y, source_policy, seed, width, 20)
    frozen = LogisticRegression(C=1., max_iter=500, tol=1e-7, solver='lbfgs')
    frozen.fit(frozen_design(fit_x, fit_mask), fit_y)
    if frozen.n_iter_[0] >= 500:
        raise RuntimeError('Source frozen logistic control did not converge')
    source_raw, source_y = covariates(seed, width, 11, plan['source_labels'], beta, bias)
    source_x, source_mask = observe(source_raw, source_y, source_policy, seed, width, 21)
    source_logit = frozen.decision_function(frozen_design(source_x, source_mask)).astype(np.float32)
    target_raw, target_y = covariates(seed, width, 12, plan['target_labels'], beta, bias)
    query_count = plan['training_queries'] if training else plan['queries']
    query_raw, query_y = covariates(seed, width, 13, query_count, beta, bias)
    episodes = []
    for index, regime in enumerate(plan.get('regimes', REGIMES)):
        if regime not in REGIMES:
            raise ValueError('Unknown prespecified regime')
        target_policy = dict(regime=regime, base=base.copy(), effect=effect.copy())
        if regime == 'sign_flip':
            target_policy['effect'] *= -1
        law = int(regime == 'ignorable_shift')
        episode_beta = beta
        episode_frozen = frozen
        episode_source_x, episode_source_mask = source_x, source_mask
        episode_source_y, episode_source_logit = source_y, source_logit
        episode_target_y, episode_query_y = target_y, query_y
        episode_source_policy = source_policy
        if law:
            # Both acquisition laws are MAR and outcome depends on observed anchors.
            episode_beta = beta.copy()
            episode_beta[2:] = 0.
            episode_source_policy = dict(regime='ignorable_shift', base=base.copy(), effect=np.zeros(width),
                                         base_offset=0., anchor_effect=.25)
            target_policy.update(effect=np.zeros(width), base_offset=.9, anchor_effect=.6)
            null_fit_raw, null_fit_y = covariates(seed, width, 10, plan['source_fit_labels'], episode_beta, bias)
            null_fit_x, null_fit_mask = observe(null_fit_raw, null_fit_y, episode_source_policy, seed, width, 120)
            episode_frozen = LogisticRegression(C=1., max_iter=500, tol=1e-7, solver='lbfgs')
            episode_frozen.fit(frozen_design(null_fit_x, null_fit_mask), null_fit_y)
            if episode_frozen.n_iter_[0] >= 500:
                raise RuntimeError('MAR source frozen logistic control did not converge')
            null_source_raw, episode_source_y = covariates(seed, width, 11, plan['source_labels'], episode_beta, bias)
            episode_source_x, episode_source_mask = observe(null_source_raw, episode_source_y, episode_source_policy, seed, width, 121)
            episode_source_logit = episode_frozen.decision_function(frozen_design(episode_source_x, episode_source_mask)).astype(np.float32)
            _, episode_target_y = covariates(seed, width, 12, plan['target_labels'], episode_beta, bias)
            _, episode_query_y = covariates(seed, width, 13, query_count, episode_beta, bias)
        target_x, target_mask = observe(target_raw, episode_target_y, target_policy, seed, width, 30 + index)
        query_x, query_mask = observe(query_raw, episode_query_y, target_policy, seed, width, 40 + index)
        episodes.append(dict(seed=int(seed), width=int(width), regime=regime,
            source_x=episode_source_x, source_mask=episode_source_mask, source_y=episode_source_y, source_logit=episode_source_logit,
            target_x=target_x, target_mask=target_mask, target_y=episode_target_y,
            target_logit=episode_frozen.decision_function(frozen_design(target_x, target_mask)).astype(np.float32),
            query_x=query_x, query_mask=query_mask,
            query_logit=episode_frozen.decision_function(frozen_design(query_x, query_mask)).astype(np.float32),
            query_y=episode_query_y, source_fit_ids=identities(seed, width, 10, len(fit_y), law),
            source_ids=identities(seed, width, 11, len(source_y), law), target_ids=identities(seed, width, 12, len(target_y), law),
            query_ids=identities(seed, width, 13, len(query_y), law),
            generator_metadata=dict(beta=episode_beta.tolist(), bias=bias, source_base=base.tolist(),
                source_effect=episode_source_policy['effect'].tolist(), target_effect=target_policy['effect'].tolist(),
                source_policy=episode_source_policy['regime'], target_policy=regime,
                mar_offsets=[episode_source_policy.get('base_offset'), target_policy.get('base_offset')],
                mar_anchor_effects=[episode_source_policy.get('anchor_effect'), target_policy.get('anchor_effect')],
                outcome_law=law, observed_anchors=[0, 1], source_fit_iterations=int(episode_frozen.n_iter_[0]),
                source_fit_converged=bool(episode_frozen.n_iter_[0] < 500)),
            source_frozen_coefficient=episode_frozen.coef_[0].copy(), source_frozen_intercept=episode_frozen.intercept_.copy()))
    return episodes


def validate_plan(plan):
    train, dev = list(plan['train_seeds']), list(plan['development_seeds'])
    validation = list(plan['validation_seeds'])
    lists = (train, validation, dev)
    if any(not seeds or len(set(seeds)) != len(seeds) for seeds in lists):
        raise ValueError('Require distinct explicit train/validation/development seed lists')
    if len(set(train + validation + dev)) != len(train + validation + dev):
        raise ValueError('Train/validation/development world seed overlap')
    if len(dev) * len(plan['development_widths']) > 48:
        raise ValueError('Development exceeds48world cap')
    if not 1 <= plan['updates'] <= 800 or min(plan['source_labels'], plan['target_labels'], plan['queries']) < 1:
        raise ValueError('Require bounded updates and positive label/query budgets')
    if tuple(plan.get('regimes', REGIMES)) != REGIMES:
        raise ValueError('Preserve all four regime controls in their frozen order')
    if set(train + validation + dev) & {910001, 910002, 920001, 920002, 930001, 930002}:
        raise ValueError('Smoke namespace cannot enter scientific worlds')


def make_models(plan):
    from model import Bridge, GenericContextual, LinearResidual
    constructors = dict(bridge=lambda: Bridge(width=plan['width'], heads=plan['heads']),
        generic_contextual=lambda: GenericContextual(width=plan['width'], heads=plan['heads']),
        linear_residual=LinearResidual,
        target_only=lambda: Bridge(width=plan['width'], heads=plan['heads'], max_potential=4., ablation='target_only'),
        no_query=lambda: Bridge(width=plan['width'], heads=plan['heads'], ablation='no_query'))
    models = {}
    for name, constructor in constructors.items():
        torch.manual_seed(plan['training_seed'])
        models[name] = constructor()
    return models


def tensor_input(episodes, device):
    return [torch.as_tensor(np.stack([episode[key] for episode in episodes]), device=device,
                            dtype=torch.bool if key.endswith('_mask') else torch.float32)
            for key in INPUT_KEYS]


def model_probability(model, episode, device):
    """Only the declared observed query and labeled support arrays are supplied."""
    model.eval()
    with torch.no_grad():
        result = model(*tensor_input([episode], device)).sigmoid()
    return result[0].detach().cpu().numpy().astype(float)


def train_models(episodes, plan, device, out):
    cache = {}
    for width in plan['train_widths']:
        subset = [episode for episode in episodes if episode['width'] == width]
        cache[width] = (tensor_input(subset, device), torch.as_tensor(
            np.stack([episode['query_y'] for episode in subset]), device=device, dtype=torch.float32))
    torch.manual_seed(plan['training_seed'])
    models = make_models(plan)
    traces = {}
    for name, model in models.items():
        start = time.monotonic()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=plan['learning_rate'])
        rng = np.random.default_rng(plan['training_seed'])
        history = []
        model.train()
        for update in range(plan['updates']):
            width = plan['train_widths'][int(rng.integers(len(plan['train_widths'])))]
            inputs, labels = cache[width]
            selected = torch.as_tensor(rng.choice(len(labels), size=plan['batch_size'], replace=False), device=device)
            optimizer.zero_grad()
            corrected_logit = model(*(value[selected] for value in inputs))
            loss = nn.functional.binary_cross_entropy_with_logits(corrected_logit, labels[selected])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite predictive training loss in ' + name)
            loss.backward()
            optimizer.step()
            if update % 10 == 0 or update + 1 == plan['updates']:
                history.append(dict(update=update + 1, nll=float(loss.detach().cpu())))
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        checkpoint = out / (name + '_weights.npz')
        np.savez_compressed(checkpoint, **{key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
        traces[name] = dict(parameters=sum(value.numel() for value in model.parameters()), updates=plan['updates'],
                            parameter_device=str(next(model.parameters()).device), train_nll=history,
                            checkpoint_sha256=checksum(checkpoint), seconds=time.monotonic() - start)
    return models, traces


def target_platt(episode, ridge):
    z, labels = episode['target_logit'].astype(float), episode['target_y'].astype(float)
    design = np.column_stack((z, np.ones(len(z))))
    def objective(theta):
        score = z + design @ theta
        loss = np.mean(np.logaddexp(0., score) - labels * score) + .5 * ridge * (theta @ theta)
        gradient = design.T @ (expit(score) - labels) / len(z) + ridge * theta
        return loss, gradient
    fit = minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=1000, gtol=1e-9, ftol=1e-13))
    if not fit.success:
        raise RuntimeError('Target Platt control did not converge: ' + str(fit.message))
    return expit((1 + fit.x[0]) * episode['query_logit'] + fit.x[1]), dict(
        success=bool(fit.success), gradient_max=float(np.abs(fit.jac).max()), iterations=int(fit.nit),
        weights=fit.x.tolist(), ridge=float(ridge))


def controls(episode, plan):
    predictions = dict(frozen=expit(episode['query_logit']).astype(float))
    predictions['target_platt'], platt_audit = target_platt(episode, plan['platt_ridge'])
    labels = episode['target_y']
    if len(np.unique(labels)) < 2:
        predictions['support_logistic'] = np.full(len(episode['query_x']), (labels.sum() + 1) / (len(labels) + 2))
        logistic_audit = dict(single_class_fallback='Laplace-smoothed support prevalence')
    else:
        model = LogisticRegression(C=plan['support_logistic_c'], max_iter=1000, tol=1e-7, solver='lbfgs')
        model.fit(support_design(episode['target_x'], episode['target_mask']), labels)
        predictions['support_logistic'] = model.predict_proba(support_design(episode['query_x'], episode['query_mask']))[:, 1]
        logistic_audit = dict(converged=bool(model.n_iter_[0] < 1000), iterations=int(model.n_iter_[0]),
                              c=float(plan['support_logistic_c']), parameters=int(model.coef_.size + 1))
    return predictions, dict(target_platt=platt_audit, support_logistic=logistic_audit)


def metrics(probability, labels):
    probability = np.asarray(probability, dtype=float)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return dict(nll=float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
                brier=float(np.mean((probability - labels) ** 2)))


def interval(values, plan):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(plan['bootstrap_seed'])
    sampled = values[rng.integers(len(values), size=(plan['bootstrap_samples'], len(values)))].mean(1)
    low, high = np.quantile(sampled, [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high), independent_world_seeds=len(values),
                bootstrap_samples=plan['bootstrap_samples'])


def summarize(results, plan, smoke=False):
    seeds = sorted(set(row['seed'] for row in results))
    names = list(results[0]['metrics'])
    gains = {}
    for name in names:
        if name == 'bridge':
            continue
        values = [np.mean([row['metrics'][name]['nll'] - row['metrics']['bridge']['nll'] for row in results
                           if row['seed'] == seed and row['regime'] in SHIFTED]) for seed in seeds]
        gains[name] = interval(values, plan)
    harms = {}
    for regime in ('no_shift', 'ignorable_shift'):
        values = [np.mean([row['metrics']['bridge']['nll'] - row['metrics']['frozen']['nll'] for row in results
                           if row['seed'] == seed and row['regime'] == regime]) for seed in seeds]
        harms[regime] = interval(values, plan)
    comparators = ('frozen', 'target_platt', 'support_logistic', 'linear_residual', 'generic_contextual')
    passed = all(gains[name]['mean'] >= plan['useful_effect_nll'] and gains[name]['low'] > 0 for name in comparators)
    passed = passed and max(value['high'] for value in harms.values()) <= plan['max_null_harm_nll']
    passed = passed and all(row['control_audits']['support_logistic'].get('converged', True) for row in results)
    return dict(gate_pass=None if smoke else bool(passed), shifted_nll_gains=gains, null_harms=harms,
                interval_scope='Paired bootstrap resamples generator seed; average widths/regimes within seed before resampling; exploratory development intervals',
                gate_rule=dict(minimum_shifted_gain=plan['useful_effect_nll'], lower95_above_zero_for_every_control=True,
                               comparators=list(comparators), maximum_null_upper95_harm=plan['max_null_harm_nll'],
                               require_support_logistic_convergence=True),
                scope='Engineering smoke, excluded from scientific development' if smoke else
                      'Controlled synthetic retrospective label-dependent acquisition development; no native efficacy or confirmed architecture novelty')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/bridge_v1.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = Path(args.config)
    plan = {**DEFAULTS, **json.loads(config.read_text())}
    validate_plan(plan)
    if args.smoke:
        plan.update(source_fit_labels=256, source_labels=16, target_labels=16, queries=32, training_queries=16,
                    train_widths=[6], development_widths=[6], train_seeds=[910001, 910002],
                    validation_seeds=[920001, 920002], development_seeds=[930001, 930002], updates=2, batch_size=2)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA requested but unavailable')
    torch.set_num_threads(4)
    out = Path(args.out)
    if out.exists():
        raise ValueError('Preserve prior Bridge output; use a fresh directory')
    out.mkdir(parents=True)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        try:
            training = [episode for seed in plan['train_seeds'] for width in plan['train_widths']
                        for episode in world(seed, width, plan, training=True)]
            data_digest = hashlib.sha256()
            for episode in training:
                for key in INPUT_KEYS + ('query_y',):
                    data_digest.update(np.ascontiguousarray(episode[key]).tobytes())
            models, traces = train_models(training, plan, device, out)
            del training
            (out / 'predictions').mkdir()
            results, validation_results, metadata = [], [], []
            evaluation_digest = hashlib.sha256()
            validation_digest = hashlib.sha256()
            for split, split_seeds in (('validation', plan['validation_seeds']), ('development', plan['development_seeds'])):
                for seed in split_seeds:
                    for width in plan['development_widths']:
                        for episode in world(seed, width, plan):
                            predictions, audits = controls(episode, plan)
                            predictions.update({name: model_probability(model, episode, device) for name, model in models.items()})
                            np.savez_compressed(out / 'predictions' / f"{split}_seed{seed}_width{width}_{episode['regime']}.npz",
                                **predictions, labels=episode['query_y'], **{key: episode[key] for key in INPUT_KEYS},
                                **{key: episode[key] for key in ('source_fit_ids', 'source_ids', 'target_ids', 'query_ids',
                                                                'source_frozen_coefficient', 'source_frozen_intercept')})
                            for key in INPUT_KEYS + ('query_y',):
                                (validation_digest if split == 'validation' else evaluation_digest).update(np.ascontiguousarray(episode[key]).tobytes())
                            destination = validation_results if split == 'validation' else results
                            destination.append(dict(split=split, seed=seed, width=width, regime=episode['regime'], control_audits=audits,
                                                    metrics={name: metrics(probability, episode['query_y']) for name, probability in predictions.items()}))
                            metadata.append(dict(split=split, seed=seed, width=width, regime=episode['regime'], **episode['generator_metadata']))
            report = summarize(results, plan, args.smoke)
            failed_controls = [dict(split=row['split'], seed=row['seed'], width=row['width'], regime=row['regime'])
                               for row in validation_results + results
                               if not row['control_audits']['support_logistic'].get('converged', True)]
            report['unconverged_support_controls'] = failed_controls
            if failed_controls and not args.smoke:
                report['gate_pass'] = False
            report.update(results=results, validation_results=validation_results, training_models=traces, device=device,
                gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                updates=plan['updates'], train_world_seeds=len(plan['train_seeds']), development_world_seeds=len(plan['development_seeds']),
                development_worlds=len(plan['development_seeds']) * len(plan['development_widths']),
                validation_world_seeds=len(plan['validation_seeds']), validation_data_sha256=validation_digest.hexdigest(),
                training_data_sha256=data_digest.hexdigest(), development_data_sha256=evaluation_digest.hexdigest(),
                protocol_sha256=checksum(config), source_sha256=checksum(__file__),
                model_source_sha256=checksum(Path(__file__).with_name('model.py')), effective_plan=plan,
                correction_ranges=dict(bridge=[-4., 4.], generic_contextual=[-4., 4.], target_only=[-4., 4.], no_query=[-4., 4.]),
                torch=torch.__version__, sklearn=sklearn.__version__, python=platform.python_version(),
                seconds=time.monotonic() - started)
            (out / 'generator_metadata.json').write_text(json.dumps(metadata, indent=2))
            (out / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({key: report[key] for key in ('gate_pass', 'shifted_nll_gains', 'null_harms', 'device', 'gpu', 'seconds')}), flush=True)
        finally:
            (out / 'warnings.json').write_text(json.dumps([str(warning.message) for warning in recorded], indent=2))


if __name__ == '__main__':
    main()
