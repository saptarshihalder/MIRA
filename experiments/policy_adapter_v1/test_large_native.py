"""Native partition, held-out label boundary, and numerical-control checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
import torch

import large_native as large


class NativeBoundaryTests(unittest.TestCase):
    def data(self):
        groups = np.repeat(np.arange(10), 100)
        years = np.tile(np.r_[np.full(50, 2024), np.full(50, 2025)], 10)
        rows = np.arange(len(groups))
        return dict(x=np.zeros((len(rows), 4), dtype=np.float32), y=rows % 2,
                    groups=groups, year=years, row_id=rows)

    def plan(self):
        return {**large.DEFAULTS, 'backbone_labels': 50, 'min_backbone_labels': 20,
                'source_labels': 8, 'train_episodes': 4, 'target_labels': 5,
                'queries': 7, 'min_group_rows': 12, 'max_group_queries': 20}

    def test_group_year_rows_and_label_independent_assignment(self):
        data = self.data()
        parts = large.partitions(data, self.plan(), 171001)
        for key in ('fit', 'context'):
            ids = parts[key]
            self.assertTrue(np.all(data['year'][ids] == 2024))
            self.assertTrue(np.all(data['groups'][ids] % 5 < 3))
        all_parts = [parts['fit'], parts['context']]
        for group, ids in parts['training']:
            self.assertEqual(len(ids), 12)
            self.assertTrue(np.all(data['groups'][ids] == group))
            self.assertTrue(np.all(data['year'][ids] == 2024))
            all_parts.append(ids)
        for split, residue in (('validation', 3), ('development', 4)):
            for group, ids in parts[split]:
                self.assertEqual(group % 5, residue)
                self.assertTrue(np.all(data['year'][ids] == 2025))
                self.assertTrue(np.all(data['groups'][ids] == group))
                self.assertEqual(len(ids), 25)
                all_parts.append(ids)
        self.assertEqual(len(np.unique(np.concatenate(all_parts))), sum(map(len, all_parts)))
        intervention = {**data, 'y': 1 - data['y'], 'x': data['x'] + 999}
        changed = large.partitions(intervention, self.plan(), 171001)
        for key in ('fit', 'context'):
            np.testing.assert_array_equal(parts[key], changed[key])
        for split in ('training', 'validation', 'development'):
            for (_, first), (_, second) in zip(parts[split], changed[split]):
                np.testing.assert_array_equal(first, second)

    def test_row_identity_order_survives_file_reordering(self):
        data = self.data()
        ids = np.arange(len(data['y']))
        order = large.ordered(ids, data, 171001)
        reverse = {key: value[::-1] for key, value in data.items()}
        reverse_order = large.ordered(ids, reverse, 171001)
        np.testing.assert_array_equal(large.identity(data, order), large.identity(reverse, reverse_order))

    def test_query_label_intervention_cannot_change_context_or_experts(self):
        rng = np.random.default_rng(12)
        count = 24
        x = rng.normal(size=(count, 4)).astype(np.float32)
        x[rng.random(x.shape) < .3] = np.nan
        data = dict(x=x, y=np.arange(count) % 2)
        class Backbone:
            def predict_proba(self, values):
                probability = expit(np.nan_to_num(values[:, 0], nan=0.) * .2)
                return np.column_stack((1 - probability, probability))
        backbone = Backbone()
        source_f = large.n.correction_features(x[:8], [0, 1, 2, 3],
                                               backbone.predict_proba(x[:8])[:, 1], np.zeros(4), np.ones(4))
        plan = {**large.DEFAULTS, 'target_labels': 8}
        first = large.prepared_episode(data, np.arange(8, 24), 1, source_f, data['y'][:8],
                                       backbone, [0, 1, 2, 3], np.zeros(4), np.ones(4), plan)
        modified = {**data, 'y': data['y'].copy()}
        modified['y'][16:] = 1 - modified['y'][16:]
        second = large.prepared_episode(modified, np.arange(8, 24), 1, source_f, data['y'][:8],
                                        backbone, [0, 1, 2, 3], np.zeros(4), np.ones(4), plan)
        np.testing.assert_array_equal(first['context'], second['context'])
        np.testing.assert_array_equal(first['probability'], second['probability'])
        self.assertFalse(np.array_equal(first['labels'], second['labels']))

    def test_duplicate_native_identity_rejected(self):
        data = self.data()
        data['row_id'][1] = data['row_id'][0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'native.npz'
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                large.load_data(path, ['a', 'b', 'c', 'd'])


class NumericalControlTests(unittest.TestCase):
    def test_lbfgs_reports_actual_gradient_and_fits_known_mixture(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(74)
        tasks = []
        for _ in range(4):
            context = rng.normal(size=(16, 12)).astype(np.float32)
            context[:, 0] = 0.
            context[0, 0] = 1.
            context[:, 10:] = .5
            probability = rng.uniform(.1, .9, size=(64, 16)).astype(np.float32)
            score = .5 * context[:, 1] + context[:, 0] * np.log(15.)
            weight = np.exp(score - score.max())
            weight /= weight.sum()
            tasks.append(dict(context=context, probability=probability,
                              labels=(probability @ weight).astype(np.float32)))
        plan = {**large.DEFAULTS, 'lbfgs_tolerance': 1e-7}
        model, trace = large.fit_mixture(tasks, plan, 171001, linear=True, converged=True, device='cpu')
        audit = trace['optimization']
        self.assertLess(audit['gradient_max'], 1e-7)
        self.assertTrue(audit['success'])
        self.assertEqual(trace['parameter_device'], 'cpu')
        for task in tasks:
            np.testing.assert_allclose(large.prediction(model, task), task['labels'], atol=1e-4)

    def test_simplex_recovers_better_expert_and_is_convex_feasible(self):
        y = np.tile([0., 1.], 50)
        probability = np.column_stack((np.full(len(y), .5), .1 + .8 * y, .9 - .8 * y))
        weight, audit = large.simplex_stack(probability, y)
        self.assertTrue(audit['success'])
        self.assertLess(audit['simplex_error'], 1e-9)
        self.assertGreater(weight[1], .999)
        self.assertLess(audit['kkt_residual'], 1e-6)
        self.assertLessEqual(large.p.bce(probability @ weight, y), large.p.bce(probability[:, 1], y) + 1e-8)

    def test_calibration_recovers_known_probabilities_and_finite_single_class(self):
        z = np.linspace(-3, 3, 101)
        truth = expit(1.4 * z + .7)
        coefficient, audit = large.calibration(z, truth, ridge=1e-8)
        self.assertTrue(audit['success'])
        np.testing.assert_allclose(coefficient, [1.4, .7], atol=1e-5)
        for labels in (np.zeros(16), np.ones(16)):
            coefficient, audit = large.calibration(np.zeros(16), labels, 'intercept', .01)
            self.assertTrue(audit['success'])
            self.assertTrue(np.isfinite(coefficient).all())
            self.assertTrue(0 < expit(coefficient[1]) < 1)


if __name__ == '__main__':
    unittest.main()
