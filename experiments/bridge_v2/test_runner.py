"""V2 fixed-data, matched-control, query-label and strict-gate checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
import torch
from torch import nn

import run as runner
import model as models_module


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.plan = {**runner.DEFAULTS, 'train_seeds': [900111, 900112],
            'validation_seeds': [900121], 'development_seeds': [900131, 900132],
            'training_seed': 900141, 'source_fit_labels': 256, 'source_labels': 32,
            'target_labels': 32, 'queries': 32, 'training_queries': 16, 'batch_size': 2}
        cls.episode = runner.v1.world(900151, 6, cls.plan)[0]

    def test_frozen_v1_data_law_and_declared_hash(self):
        regenerated = runner.v1.world(900151, 6, self.plan)[0]
        self.assertEqual(runner.hash_episodes([self.episode]), runner.hash_episodes([regenerated]))
        changed = copy.deepcopy(self.episode)
        changed['query_y'] = 1 - changed['query_y']
        self.assertNotEqual(runner.hash_episodes([self.episode]), runner.hash_episodes([changed]))
        runner.validate_plan({**self.plan, 'frozen_v1_source_sha256': runner.v1.checksum(runner.V1_PATH)})
        with self.assertRaisesRegex(ValueError, 'source changed'):
            runner.validate_plan({**self.plan, 'frozen_v1_source_sha256': '0' * 64})

    def test_splits_update_bound_and_solver_revision_rejected(self):
        runner.validate_plan(self.plan)
        for changed, message in (({'updates': 601}, '600update'),
                                 ({'validation_seeds': self.plan['train_seeds']}, 'overlap'),
                                 ({'train_seeds': [940001]}, 'smoke'),
                                 ({'newton_steps': 5}, 'four-step')):
            with self.assertRaisesRegex(ValueError, message):
                runner.validate_plan({**self.plan, **changed})

    def test_initial_models_are_matched_fixed_penalty_solver(self):
        models = runner.make_models(self.plan)
        prediction = {name: runner.v1.model_probability(model, self.episode, 'cpu') for name, model in models.items()}
        for probability in prediction.values():
            np.testing.assert_allclose(probability, prediction['fixed_regularizer'], atol=1e-6)
            self.assertTrue(np.isfinite(probability).all())
        counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()}
        self.assertEqual(counts['contextual_regularizer'], counts['type_regularizer'])
        self.assertEqual(counts['contextual_regularizer'], counts['target_only_regularizer'])
        self.assertEqual(counts['global_regularizer'], 1)
        self.assertEqual(counts['fixed_regularizer'], 0)

    def test_query_labels_and_parameters_never_enter_solver_prediction(self):
        model = runner.make_models(self.plan)[runner.MAIN]
        first = runner.v1.model_probability(model, self.episode, 'cpu')
        changed = copy.deepcopy(self.episode)
        changed['query_y'] = np.full(1, np.nan)
        changed['generator_metadata'] = dict(privileged='neverread')
        np.testing.assert_array_equal(first, runner.v1.model_probability(model, changed, 'cpu'))
        del changed['query_y']
        del changed['generator_metadata']
        np.testing.assert_array_equal(first, runner.v1.model_probability(model, changed, 'cpu'))

    def test_support_solver_diagnostics_are_retained_without_convergence_claim(self):
        model = runner.make_models(self.plan)['fixed_regularizer']
        audit = runner.diagnostic_summary(model, self.episode, 'cpu')
        self.assertEqual(audit['newton_steps'], 4)
        self.assertEqual(audit['solver_dtype'], 'torch.float64')
        self.assertTrue(audit['all_finite'])
        self.assertIn('max_abs_gradient', audit)
        self.assertFalse(audit['convergence_claim'])

    def test_prediction_api_uses_sigmoid_of_logits(self):
        class ConstantLogit(nn.Module):
            def forward(self, *inputs):
                return torch.full_like(inputs[2], -2.)
        np.testing.assert_allclose(runner.v1.model_probability(ConstantLogit(), self.episode, 'cpu'), expit(-2.), atol=1e-7)

    def test_global_grid_uses_training_meta_labels_and_ignores_heldout_plan(self):
        class GridModel(nn.Module):
            def __init__(self, penalty, steps, solver_dtype):
                super().__init__()
                self.register_buffer('penalty', torch.tensor(penalty))
            def forward(self, *inputs):
                return self.penalty.expand_as(inputs[2])
        inputs = runner.v1.tensor_input([self.episode], 'cpu')
        labels = torch.zeros_like(inputs[2])
        plan = {**self.plan, 'train_widths': [6]}
        with patch.object(models_module, 'FixedRegularizer', GridModel):
            first, first_audit = runner.select_global_grid({6: (inputs, labels)}, plan, 'cpu')
            heldout_change = {**plan, 'development_seeds': [999991], 'validation_seeds': [999992]}
            _, changed_audit = runner.select_global_grid({6: (inputs, labels)}, heldout_change, 'cpu')
            second, second_audit = runner.select_global_grid({6: (inputs, 1 - labels)}, plan, 'cpu')
        self.assertAlmostEqual(float(first.penalty), .01, places=6)
        self.assertAlmostEqual(float(second.penalty), 1., places=6)
        self.assertEqual(first_audit['objectives'], changed_audit['objectives'])
        self.assertEqual(first_audit['selection_split'], 'train_meta_query_only')
        self.assertEqual(first_audit['candidate_count'], 11)
        self.assertEqual(first_audit['training_meta_query_labels'], len(self.episode['query_y']))
        self.assertEqual(first_audit['search_cost']['forward_batches'], 11)
        self.assertEqual(first_audit['search_cost']['backward_updates'], 0)
        self.assertNotEqual(first_audit['selected_index'], second_audit['selected_index'])

    def test_selected_grid_penalty_survives_numpy_checkpoint_restore(self):
        selected = models_module.FixedRegularizer(penalty=float(np.logspace(-2, 0, 11)[8]))
        expected = runner.v1.model_probability(selected, self.episode, 'cpu')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'grid.npz'
            np.savez_compressed(path, **{key: value.numpy() for key, value in selected.state_dict().items()})
            restored = models_module.FixedRegularizer(penalty=.1)
            with np.load(path) as state:
                restored.load_state_dict({key: torch.from_numpy(state[key].copy()) for key in state.files})
        self.assertNotEqual(float(restored.penalty), .1)
        self.assertEqual(float(restored.penalty), float(selected.penalty))
        np.testing.assert_array_equal(expected, runner.v1.model_probability(restored, self.episode, 'cpu'))

    def test_gate_uses_all_controls_and_null_upper_bound(self):
        rows = []
        names = (runner.MAIN,) + runner.COMPARATORS
        for seed, harm in ((1, -.004), (2, .004)):
            for regime in runner.v1.REGIMES:
                score = {name: .25 for name in names}
                score[runner.MAIN] = .24 if regime in runner.v1.SHIFTED else .25 + harm
                rows.append(dict(seed=seed, width=6, regime=regime,
                    metrics={name: dict(nll=value, brier=.1) for name, value in score.items()},
                    control_audits=dict(support_logistic=dict(converged=True))))
        report = runner.summarize(rows, self.plan)
        self.assertEqual(set(report['shifted_nll_gains']), set(runner.COMPARATORS))
        self.assertGreater(report['null_harms']['no_shift']['high'], .001)
        self.assertFalse(report['gate_pass'])
        self.assertIsNone(runner.summarize(rows, self.plan, smoke=True)['gate_pass'])


if __name__ == '__main__':
    unittest.main()
