"""Independent engineering-result audit; no training or cloud calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit-output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result_dir = Path(args.out)
    code = Path(__file__).parent
    report = json.loads((result_dir / 'report.json').read_text())
    plan = report['effective_plan']
    errors = []
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            errors.append(name)

    check('exact_runner_hash', sha(code / 'run.py') == report['source_sha256'])
    check('exact_model_hash', sha(code / 'model.py') == report['model_source_sha256'])
    check('exact_protocol_hash', sha(ROOT / 'configs/recursive_jepa_v1.json') == report['protocol_sha256'])
    check('exact_frozen_generator_hash', sha(ROOT / 'experiments/bridge_v1/run.py') == report['frozen_v1_source_sha256'])
    check('engineering_only_no_efficacy_gate', report.get('gate_pass') is None)
    check('actual_cuda_training', report['device'] == 'cuda' and bool(report.get('gpu')))
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('audited_recursive_runner', code / 'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    generator = runner.v1
    torch.set_num_threads(4)
    models = {name: runner.make_model(name, plan) for name in runner.NAMES}
    teacher_arrays = {}
    for name, model in models.items():
        checkpoint = result_dir / (name + '_weights.npz')
        trace = report['training_models'][name]
        check(name + '_checkpoint_hash', sha(checkpoint) == trace['checkpoint_sha256'])
        check(name + '_actual_update_count', trace['updates'] == plan['updates'] == 64)
        with np.load(checkpoint, allow_pickle=False) as state:
            check(name + '_finite_checkpoint', all(np.isfinite(state[key]).all() for key in state.files))
            teacher_keys = [key for key in state.files if 'teacher' in key or 'ema' in key]
            teacher_arrays[name] = len(teacher_keys)
            model.load_state_dict({key: torch.as_tensor(state[key].copy()) for key in state.files}, strict=True)
            check(name + '_saved_ema_update_count', int(state['teacher_updates']) == plan['updates'])
            gaps = [float(np.abs(state[key] - state[key.replace('teacher.', 'encoder.', 1)]).max())
                    for key in state.files if key.startswith('teacher.')]
            check(name + '_teacher_differs_from_final_online_encoder', bool(gaps) and max(gaps) > 1e-8)
        check(name + '_teacher_parameters_frozen', all(not value.requires_grad for value in model.teacher.parameters()))
        model.eval()
    rows = report['results']
    expected = {(row['seed'], row['width'], row['regime']): row for row in rows}
    check('unique_report_rows', len(expected) == len(rows))
    paths = list((result_dir / 'predictions').glob('*.npz'))
    check('complete_prediction_files', len(paths) == len(rows))
    probabilities = cpu_predictions = 0
    max_nll_error = max_brier_error = max_cpu_error = max_logit_error = max_generator_error = 0.
    all_bounds = boundaries = visible_arrays_match = True
    hidden_values_zero = frozen_intervention_exact = sentinel_invariance = True
    seen_ids = set()
    for seed in plan['development_seeds']:
        for width in plan['development_widths']:
            for episode in generator.world(seed, width, plan):
                row = expected[(seed, width, episode['regime'])]
                path = result_dir / 'predictions' / f'seed{seed}_width{width}_{episode["regime"]}.npz'
                with np.load(path, allow_pickle=False) as saved:
                    for key in generator.INPUT_KEYS + ('query_y',):
                        actual = saved['labels' if key == 'query_y' else key]
                        desired = episode[key]
                        difference = float(np.abs(actual.astype(float) - desired.astype(float)).max())
                        max_generator_error = max(max_generator_error, difference)
                        visible_arrays_match &= difference < 1e-6 if key.endswith('_logit') else np.array_equal(actual, desired)
                    id_sets = []
                    for key in ('source_fit_ids', 'source_ids', 'target_ids', 'query_ids'):
                        boundaries &= np.array_equal(saved[key], episode[key])
                        identities = set(map(tuple, saved[key].tolist()))
                        boundaries &= len(identities) == len(saved[key])
                        id_sets.append(identities)
                    boundaries &= all(not id_sets[a] & id_sets[b] for a in range(4) for b in range(a + 1, 4))
                    seen_ids |= id_sets[-1]
                    for prefix in ('source', 'target', 'query'):
                        x, mask = saved[prefix + '_x'], saved[prefix + '_mask']
                        hidden_values_zero &= bool((x[mask] == 0).all())
                        design = np.column_stack((x, mask.astype(np.float32)))
                        logits = (design @ saved['source_frozen_coefficient'] + saved['source_frozen_intercept']).astype(np.float32)
                        max_logit_error = max(max_logit_error, float(np.abs(logits - saved[prefix + '_logit']).max()))
                    labels = saved['labels'].astype(float)
                    for name, values in row['metrics'].items():
                        probability = saved[name].astype(float)
                        all_bounds &= bool(np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all())
                        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
                        nll = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
                        brier = np.mean((probability - labels) ** 2)
                        max_nll_error = max(max_nll_error, abs(float(nll) - values['nll']))
                        max_brier_error = max(max_brier_error, abs(float(brier) - values['brier']))
                        probabilities += 1
                    inputs = [torch.as_tensor(saved[key].copy()[None], dtype=torch.bool if key.endswith('_mask') else torch.float32)
                              for key in generator.INPUT_KEYS]
                    for name, model in models.items():
                        with torch.no_grad():
                            probability = model(*inputs).sigmoid()[0].numpy().astype(float)
                            frozen_intervention_exact &= torch.equal(model(*inputs, force_frozen=True), inputs[2])
                            corrupted = [value.clone() for value in inputs]
                            for value_index, mask_index in ((0, 1), (3, 4), (7, 8)):
                                corrupted[value_index][corrupted[mask_index]] = float('nan')
                            sentinel = model(*corrupted).sigmoid()[0].numpy().astype(float)
                            sentinel_invariance &= np.array_equal(probability, sentinel)
                        max_cpu_error = max(max_cpu_error, float(np.abs(probability - saved[name]).max()))
                        cpu_predictions += 1
    check('regenerated_values_masks_labels_exact_fitted_logits_tolerance1e-6', visible_arrays_match)
    check('source_support_query_ids_match_disjoint', boundaries)
    check('initially_hidden_values_are_zero', hidden_values_zero)
    check('force_frozen_returns_exact_frozen_logits', frozen_intervention_exact)
    check('initially_hidden_sentinel_invariance', sentinel_invariance)
    check('all_probability_bounds', all_bounds)
    check('all_nll_agree', max_nll_error < 1e-10)
    check('all_brier_agree', max_brier_error < 1e-10)
    check('all_source_logits_reconstructed', max_logit_error < 1e-6)
    check('all_cpu_checkpoint_predictions_agree', max_cpu_error <= 1e-5)
    audit = dict(passed=not errors, errors=errors, checks=checks,
                 auditor_sha256=sha(__file__), protocol_sha256=report['protocol_sha256'],
                 files_checked=len(paths), probability_arrays_checked=probabilities,
                 cpu_checkpoint_predictions=cpu_predictions, saved_teacher_state_arrays=teacher_arrays,
                 max_nll_error=max_nll_error, max_brier_error=max_brier_error,
                 max_cpu_probability_error=max_cpu_error, max_source_logit_error=max_logit_error,
                 max_generator_numeric_error=max_generator_error,
                 scope='Engineering-only64-update GPU pilot: all saved scores/bounds/row identities/source logits/state hashes and CPU checkpoint replay. No efficacy, novelty, universal safety or collapse-resistance conclusion.',
                 seconds=time.monotonic() - started)
    target = Path(args.audit_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: audit[key] for key in ('passed', 'errors', 'files_checked', 'probability_arrays_checked', 'cpu_checkpoint_predictions', 'max_cpu_probability_error', 'seconds')}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
