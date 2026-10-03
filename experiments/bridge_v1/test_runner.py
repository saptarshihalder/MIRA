"""Scientific boundaries and logit API checks for the synthetic runner."""
import copy
import unittest

import numpy as np
from scipy.special import expit
import torch
from torch import nn

import run as runner


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.plan = {**runner.DEFAULTS, 'train_seeds': [900011, 900012],
                    'validation_seeds': [900021], 'development_seeds': [900031, 900032],
                    'training_seed': 900041, 'source_fit_labels': 256,
                    'source_labels': 32, 'target_labels': 32, 'queries': 32,
                    'training_queries': 16, 'batch_size': 2}
        cls.episodes = runner.world(900051, 6, cls.plan)

    def test_source_fit_support_and_queries_are_disjoint(self):
        for episode in self.episodes:
            ids = [episode[key] for key in ('source_fit_ids', 'source_ids', 'target_ids', 'query_ids')]
            combined = np.concatenate(ids)
            self.assertEqual(len(np.unique(combined, axis=0)), sum(map(len, ids)))
        for regime in ('sign_flip', 'nonlinear_shift'):
            episode = next(episode for episode in self.episodes if episode['regime'] == regime)
            no_shift = next(episode for episode in self.episodes if episode['regime'] == 'no_shift')
            for key in ('source_x', 'source_mask', 'source_y', 'source_logit', 'source_frozen_coefficient'):
                np.testing.assert_array_equal(episode[key], no_shift[key])

    def test_split_seed_overlap_and_world_cap_rejected(self):
        runner.validate_plan(self.plan)
        overlap = {**self.plan, 'validation_seeds': self.plan['train_seeds']}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            runner.validate_plan(overlap)
        too_many = {**self.plan, 'development_seeds': list(range(901001, 901018))}
        with self.assertRaisesRegex(ValueError, '48world'):
            runner.validate_plan(too_many)
        train_ids = runner.identities(900011, 6, 10, 16)
        dev_ids = runner.identities(900031, 6, 10, 16)
        self.assertEqual(len(np.unique(np.concatenate((train_ids, dev_ids)), axis=0)), 32)

    def test_observed_anchors_and_true_missing_encoding(self):
        for episode in self.episodes:
            for prefix in ('source', 'target', 'query'):
                mask = episode[prefix + '_mask']
                x = episode[prefix + '_x']
                self.assertEqual(mask.dtype, np.bool_)
                self.assertFalse(mask[:, :2].any())
                self.assertTrue(np.all(x[mask] == 0))

    def test_mar_null_does_not_use_labels_and_has_anchor_only_outcome(self):
        episode = next(episode for episode in self.episodes if episode['regime'] == 'ignorable_shift')
        meta = episode['generator_metadata']
        np.testing.assert_array_equal(meta['beta'][2:], np.zeros(4))
        self.assertEqual(meta['source_policy'], 'ignorable_shift')
        self.assertEqual(meta['target_policy'], 'ignorable_shift')
        policy = dict(regime='ignorable_shift', base=np.zeros(6), effect=np.zeros(6),
                      base_offset=.9, anchor_effect=.6)
        x = np.arange(60).reshape(10, 6) / 10
        first = runner.observe(x, np.zeros(10), policy, 900099, 6, 1)
        second = runner.observe(x, np.ones(10), policy, 900099, 6, 1)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_query_labels_and_generator_metadata_do_not_enter_prediction(self):
        class SupportOnlyModel(nn.Module):
            def forward(self, query_x, query_mask, query_logit, source_x, source_mask,
                        source_y, source_logit, target_x, target_mask, target_y, target_logit):
                return query_logit + .1 * (target_y.mean(1) - source_y.mean(1))[:, None]
        first = self.episodes[0]
        changed = copy.deepcopy(first)
        changed['query_y'] = np.full(1, np.nan)
        changed['generator_metadata'] = dict(privileged='must never be read')
        np.testing.assert_array_equal(runner.model_probability(SupportOnlyModel(), first, 'cpu'),
                                      runner.model_probability(SupportOnlyModel(), changed, 'cpu'))
        del changed['query_y']
        del changed['generator_metadata']
        self.assertEqual(len(runner.model_probability(SupportOnlyModel(), changed, 'cpu')), len(first['query_x']))
        first_control, _ = runner.controls(first, self.plan)
        changed_control, _ = runner.controls(changed, self.plan)
        for name in first_control:
            np.testing.assert_array_equal(first_control[name], changed_control[name])

    def test_forward_logits_are_sigmoided_and_bce_logits_is_stable(self):
        class LogitModel(nn.Module):
            def forward(self, *inputs):
                return torch.full_like(inputs[2], 2.)
        probability = runner.model_probability(LogitModel(), self.episodes[0], 'cpu')
        np.testing.assert_allclose(probability, expit(2.), atol=1e-7)
        logit = torch.tensor([-1000., 1000.], requires_grad=True)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor([1., 0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logit.grad).all())

    def test_matched_correction_ranges_and_model_budget(self):
        models = runner.make_models(self.plan)
        self.assertEqual(models['target_only'].max_potential, 4.)
        self.assertEqual(models['bridge'].max_potential, 2.)
        self.assertEqual(models['generic_contextual'].max_potential, 2.)
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(abs(count(models['bridge']) - count(models['generic_contextual'])), 100)

    def test_gate_requires_null_upper_interval_and_each_comparator(self):
        names = ('bridge', 'generic_contextual', 'linear_residual', 'target_only', 'no_query',
                 'frozen', 'target_platt', 'support_logistic')
        results = []
        for seed, null_harm in ((1, -.004), (2, .004)):
            for regime in runner.REGIMES:
                nll = {name: .25 for name in names}
                nll['bridge'] = .24 if regime in runner.SHIFTED else .25 + null_harm
                results.append(dict(seed=seed, width=6, regime=regime,
                    metrics={name: dict(nll=value, brier=.1) for name, value in nll.items()},
                    control_audits=dict(support_logistic=dict(converged=True))))
        report = runner.summarize(results, self.plan)
        self.assertEqual(report['null_harms']['no_shift']['mean'], 0.)
        self.assertGreater(report['null_harms']['no_shift']['high'], .001)
        self.assertFalse(report['gate_pass'])


if __name__ == '__main__':
    unittest.main()
