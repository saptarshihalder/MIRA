"""Independent saved-result audit; never trains or calls a provider."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values, plan):
    values = np.asarray(values, dtype=float)
    random = np.random.default_rng(plan['bootstrap_seed'])
    draws = random.integers(len(values), size=(plan['bootstrap_samples'], len(values)))
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=(1, 2), default=1)
    parser.add_argument('--out', required=True, help='Existing results directory')
    parser.add_argument('--audit-output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    directory = Path(args.out)
    report = json.loads((directory / 'report.json').read_text())
    plan = report['effective_plan']
    code = ROOT / f'experiments/bridge_v{args.version}'
    config = ROOT / f'configs/bridge_v{args.version}.json'
    checks, errors = {}, []

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            errors.append(name)

    check('exact_protocol_hash', report['protocol_sha256'] == sha(config))
    check('exact_runner_hash', report['source_sha256'] == sha(code / 'run.py'))
    check('exact_model_hash', report['model_source_sha256'] == sha(code / 'model.py'))
    if args.version == 2:
        check('exact_frozen_generator_hash', report['frozen_v1_source_sha256'] == sha(ROOT / 'experiments/bridge_v1/run.py'))
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('audited_bridge_runner', code / 'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    generator = runner if args.version == 1 else runner.v1
    generator.validate_plan(plan)
    torch.set_num_threads(4)
    models = runner.make_models(plan)
    if args.version == 2:
        from model import FixedRegularizer
        selection = report['training_models']['global_grid_control']['selection']
        models['global_grid_control'] = FixedRegularizer(penalty=selection['selected_penalty'], steps=4, solver_dtype=torch.float64)
    for name, model in models.items():
        checkpoint = directory / (name + '_weights.npz')
        check(name + '_checkpoint_hash', sha(checkpoint) == report['training_models'][name]['checkpoint_sha256'])
        with np.load(checkpoint, allow_pickle=False) as weights:
            model.load_state_dict({key: torch.as_tensor(weights[key].copy()) for key in weights.files}, strict=True)
        model.eval()
    rows = report['validation_results'] + report['results']
    expected = {(row['split'], row['seed'], row['width'], row['regime']): row for row in rows}
    metadata = {(row['split'], row['seed'], row['width'], row['regime']): row
                for row in json.loads((directory / 'generator_metadata.json').read_text())}
    check('unique_report_rows', len(expected) == len(rows))
    files = list((directory / 'predictions').glob('*.npz'))
    check('complete_file_count', len(files) == len(rows))
    max_metric_error = max_source_logit_error = max_cpu_error = max_regenerated_numeric_error = 0.
    max_regenerated_coefficient_error = max_metadata_float_error = 0.
    probabilities = files_checked = cpu_files = cpu_predictions = 0
    audited_rows = []
    split_hashes, generator_hashes = {}, {}
    chosen_cpu = set()
    numerical_matches = boundary_matches = all_probability_bounds = True
    for split in ('validation', 'development'):
        digest = hashlib.sha256()
        regenerated_digest = hashlib.sha256()
        for seed in plan[split + '_seeds']:
            for width in plan['development_widths']:
                episodes = generator.world(seed, width, plan)
                for episode in episodes:
                    identity = (split, seed, width, episode['regime'])
                    row = expected[identity]
                    path = directory / 'predictions' / f'{split}_seed{seed}_width{width}_{episode["regime"]}.npz'
                    with np.load(path, allow_pickle=False) as saved:
                        keys = generator.INPUT_KEYS + ('query_y',)
                        if args.version == 2:
                            digest.update(f'{seed}:{width}:{episode["regime"]}'.encode())
                            regenerated_digest.update(f'{seed}:{width}:{episode["regime"]}'.encode())
                        for key in keys:
                            actual = saved['labels' if key == 'query_y' else key]
                            wanted = episode[key]
                            if actual.dtype.kind in 'bfiu':
                                difference = float(np.max(np.abs(actual.astype(float) - wanted.astype(float))))
                                max_regenerated_numeric_error = max(max_regenerated_numeric_error, difference)
                                # Fitted logits tolerate platform BLAS differences; values/masks/labels are exact.
                                numerical_matches &= difference < 1e-6 if key.endswith('_logit') else np.array_equal(actual, wanted)
                            value = np.ascontiguousarray(actual)
                            if args.version == 2:
                                digest.update(f'{key}:{value.dtype}:{value.shape}'.encode())
                            digest.update(value.tobytes())
                            regenerated = np.ascontiguousarray(wanted)
                            if args.version == 2:
                                regenerated_digest.update(f'{key}:{regenerated.dtype}:{regenerated.shape}'.encode())
                            regenerated_digest.update(regenerated.tobytes())
                        for key in ('source_frozen_coefficient', 'source_frozen_intercept'):
                            max_regenerated_coefficient_error = max(max_regenerated_coefficient_error, float(np.abs(saved[key] - episode[key]).max()))
                        id_sets = []
                        for key in ('source_fit_ids', 'source_ids', 'target_ids', 'query_ids'):
                            actual = saved[key]
                            boundary_matches &= np.array_equal(actual, episode[key])
                            id_sets.append(set(map(tuple, actual.tolist())))
                            boundary_matches &= len(id_sets[-1]) == len(actual)
                        boundary_matches &= all(not id_sets[a] & id_sets[b] for a in range(4) for b in range(a + 1, 4))
                        for prefix in ('source', 'target', 'query'):
                            design = np.column_stack((saved[prefix + '_x'], saved[prefix + '_mask'].astype(np.float32)))
                            reconstructed = (design @ saved['source_frozen_coefficient'] + saved['source_frozen_intercept']).astype(np.float32)
                            max_source_logit_error = max(max_source_logit_error, float(np.abs(reconstructed - saved[prefix + '_logit']).max()))
                        new_metrics = {}
                        for name in row['metrics']:
                            probability = saved[name].astype(float)
                            labels = saved['labels'].astype(float)
                            all_probability_bounds &= bool(np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all())
                            clipped = np.clip(probability, 1e-7, 1 - 1e-7)
                            values = dict(nll=float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
                                          brier=float(np.mean((probability - labels) ** 2)))
                            max_metric_error = max(max_metric_error, *(abs(values[key] - row['metrics'][name][key]) for key in values))
                            new_metrics[name] = values
                            probabilities += 1
                        cpu_key = (split, width, episode['regime'])
                        if cpu_key not in chosen_cpu:
                            chosen_cpu.add(cpu_key)
                            for name, model in models.items():
                                inputs = [torch.as_tensor(saved[key].copy()[None], dtype=torch.bool if key.endswith('_mask') else torch.float32)
                                          for key in generator.INPUT_KEYS]
                                with torch.no_grad():
                                    probability = model(*inputs).sigmoid()[0].numpy().astype(float)
                                max_cpu_error = max(max_cpu_error, float(np.abs(probability - saved[name]).max()))
                                cpu_predictions += 1
                            cpu_files += 1
                        for key, value in episode['generator_metadata'].items():
                            actual = metadata[identity][key]
                            if isinstance(value, float) or isinstance(value, list) and value and all(isinstance(item, float) for item in value):
                                difference = float(np.abs(np.asarray(actual) - np.asarray(value)).max())
                                max_metadata_float_error = max(max_metadata_float_error, difference)
                                matches = difference < 1e-12
                            else:
                                matches = actual == value
                            check(f'metadata_{split}_{seed}_{width}_{episode["regime"]}_{key}', matches)
                    audited_rows.append(dict(split=split, seed=seed, width=width, regime=episode['regime'], metrics=new_metrics))
                    files_checked += 1
        split_hashes[split] = digest.hexdigest()
        generator_hashes[split] = regenerated_digest.hexdigest()
        check(split + '_data_hash', split_hashes[split] == report[split + '_data_sha256'])
    check('generator_values_masks_labels_exact_fitted_logits_tolerance_1e-6', numerical_matches)
    check('regenerated_source_coefficients_tolerance_1e-7', max_regenerated_coefficient_error < 1e-7)
    check('all_ids_match_and_disjoint', boundary_matches)
    check('all_probability_bounds', all_probability_bounds)
    check('all_saved_metric_agreement', max_metric_error < 1e-10)
    check('source_prediction_logit_agreement', max_source_logit_error < 1e-6)
    check('cpu_checkpoint_prediction_agreement', max_cpu_error < 2e-5)
    primary = 'bridge' if args.version == 1 else 'contextual_regularizer'
    dev = [row for row in audited_rows if row['split'] == 'development']
    seeds = sorted({row['seed'] for row in dev})
    comparators = report['gate_rule']['comparators']
    gains = {name: interval([np.mean([row['metrics'][name]['nll'] - row['metrics'][primary]['nll']
                                      for row in dev if row['seed'] == seed and row['regime'] in ('sign_flip', 'nonlinear_shift')])
                             for seed in seeds], plan) for name in comparators}
    harms = {regime: interval([np.mean([row['metrics'][primary]['nll'] - row['metrics']['frozen']['nll']
                                       for row in dev if row['seed'] == seed and row['regime'] == regime])
                              for seed in seeds], plan) for regime in ('no_shift', 'ignorable_shift')}
    independent_gate = all(value['mean'] >= plan['useful_effect_nll'] and value['low'] > 0 for value in gains.values())
    independent_gate &= all(value['high'] <= plan['max_null_harm_nll'] for value in harms.values())
    independent_gate &= not bool(report['unconverged_support_controls'])
    check('independent_gate_agreement', bool(independent_gate) == report['gate_pass'])
    for family, values in (('shifted_nll_gains', gains), ('null_harms', harms)):
        check(family + '_interval_agreement', all(abs(value[key] - report[family][name][key]) < 1e-10
                                                for name, value in values.items() for key in value))
    audit = dict(passed=not errors, errors=errors, checks_passed=sum(checks.values()), checks_total=len(checks),
                 version=args.version, protocol_sha256=report['protocol_sha256'], auditor_sha256=sha(__file__),
                 files_checked=files_checked, probability_arrays_checked=probabilities,
                 cpu_checkpoint_files=cpu_files, cpu_checkpoint_predictions=cpu_predictions,
                 cpu_subset='First seed in each split, width and regime; every saved checkpoint',
                 max_metric_error=max_metric_error, max_source_logit_error=max_source_logit_error,
                 max_cpu_probability_error=max_cpu_error, max_regenerated_numeric_error=max_regenerated_numeric_error,
                 max_regenerated_coefficient_error=max_regenerated_coefficient_error,
                 max_metadata_float_error=max_metadata_float_error,
                 saved_data_hashes=split_hashes, regenerated_data_hashes=generator_hashes,
                 regeneration_note='Source-fit floating logits may vary by platform; masks, observed values and all labels require exact equality. Metadata floats tolerance1e-12; fitted coefficients1e-7; fitted logits1e-6.',
                 independent_gate_pass=bool(independent_gate),
                 independent_shifted_gains=gains, independent_null_harms=harms,
                 scope='Saved artifacts, NumPy scoring, frozen generator regeneration and CPU checkpoint replay; no retraining/provider calls',
                 seconds=time.monotonic() - started)
    target = Path(args.audit_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: audit[key] for key in ('passed', 'errors', 'files_checked', 'probability_arrays_checked', 'cpu_checkpoint_predictions', 'max_metric_error', 'max_cpu_probability_error', 'independent_gate_pass', 'seconds')}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
