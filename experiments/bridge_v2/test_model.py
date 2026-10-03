"""Check solver equivalence, input boundaries, symmetry and outer gradients."""
import inspect
import unittest

import numpy as np
import torch

from model import (SupportRegularizer, GlobalRegularizer, FeatureTypeRegularizer, TargetOnlyRegularizer,
                   FixedRegularizer, basis)


def fixture(columns=4):
    torch.manual_seed(122)
    def rows(n):
        return (torch.randn(2, n, columns), torch.rand(2, n, columns) < .3,
                torch.randint(2, (2, n)).float(), torch.randn(2, n))
    qx, qm, _, qlogit = rows(5)
    return qx, qm, qlogit, *rows(23), *rows(31)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(844)
        self.model = SupportRegularizer().eval()
        self.args = fixture()

    def test_fixed_solver_matches_independent_numpy_four_updates(self):
        model = FixedRegularizer()
        coefficients, penalty = model.fit_support(*self.args[3:])
        z = basis(self.args[7], self.args[8], self.args[10]).numpy()
        for b in range(2):
            beta = np.zeros(z.shape[-1])
            offset = self.args[10][b].double().numpy()
            labels = self.args[9][b].double().numpy()
            ridge = penalty[b].double().numpy()
            for _ in range(4):
                probability = 1 / (1 + np.exp(-(offset + z[b] @ beta)))
                gradient = z[b].T @ (probability - labels) / len(labels) + ridge * beta
                hessian = z[b].T @ (z[b] * (probability * (1 - probability))[:, None]) / len(labels)
                beta -= np.linalg.solve(hessian + np.diag(ridge), gradient)
            np.testing.assert_allclose(coefficients[b].detach().numpy(), beta, atol=2e-12, rtol=2e-11)

    def test_joint_field_permutation_variable_width_and_support_order(self):
        # Activate learned descriptor dependence while preserving the same architecture.
        with torch.no_grad():
            self.model.network[-1].weight.normal_(0, .1)
        output = self.model(*self.args)
        permutation = torch.tensor([2, 0, 3, 1])
        shuffled = tuple(a[..., permutation] if a.ndim == 3 else a for a in self.args)
        torch.testing.assert_close(output, self.model(*shuffled), atol=3e-7, rtol=1e-6)
        reversed_supports = tuple(a.flip(1) if i >= 3 else a for i, a in enumerate(self.args))
        torch.testing.assert_close(output, self.model(*reversed_supports), atol=3e-7, rtol=1e-6)
        for columns in (1, 6, 10, 14):
            self.assertTrue(torch.isfinite(self.model(*fixture(columns))).all())

    def test_query_order_and_chunking(self):
        output = self.model(*self.args)
        order = torch.tensor([3, 4, 0, 2, 1])
        shuffled = tuple(a[:, order] if i < 3 else a for i, a in enumerate(self.args))
        torch.testing.assert_close(output[:, order], self.model(*shuffled), atol=0, rtol=0)
        chunked = [self.model(*(a[:, start:stop] if i < 3 else a for i, a in enumerate(self.args)))
                   for start, stop in ((0, 2), (2, 5))]
        torch.testing.assert_close(output, torch.cat(chunked, 1), atol=0, rtol=0)

    def test_initial_controls_share_uniform_penalty_and_solver(self):
        models = (self.model, FeatureTypeRegularizer(), TargetOnlyRegularizer(), GlobalRegularizer(), FixedRegularizer())
        output = models[-1](*self.args)
        for model in models:
            torch.testing.assert_close(model(*self.args), output, atol=3e-7, rtol=1e-6)
            audit = model.diagnostics(*self.args)
            self.assertTrue(audit['all_finite'])
            self.assertEqual(audit['newton_steps'], 4)
            self.assertGreaterEqual(audit['penalty_min'], .01)
            self.assertLessEqual(audit['penalty_max'], 1.)
        counts = [sum(p.numel() for p in model.parameters()) for model in models]
        self.assertEqual(counts, [1857, 1857, 1857, 1, 0])

    def test_target_labels_and_learned_source_conditioning(self):
        changed = list(self.args)
        changed[9] = 1 - changed[9]
        self.assertGreater((self.model(*changed) - self.model(*self.args)).abs().max().item(), 1e-4)
        with torch.no_grad():
            self.model.network[-1].weight.normal_(0, .1)
        changed = list(self.args)
        changed[5] = 1 - changed[5]
        self.assertGreater((self.model(*changed) - self.model(*self.args)).abs().max().item(), 1e-6)
        type_control = FeatureTypeRegularizer()
        torch.testing.assert_close(type_control(*changed), type_control(*self.args), atol=0, rtol=0)
        target_control = TargetOnlyRegularizer()
        target_control.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(target_control(*changed), target_control(*self.args), atol=0, rtol=0)
        # Common calibration survives: identical supports do not force zero correction.
        q = self.args[:3]
        source = self.args[3:7]
        self.assertGreater((self.model(*q, *source, *source) - q[2]).abs().max().item(), 1e-4)

    def test_outer_nll_gradients_through_solver(self):
        labels = torch.tensor([[0., 1., 1., 0., 1.], [1., 0., 0., 1., 0.]])
        output = self.model(*self.args)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        loss.backward()
        gradient = self.model.network[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 1e-6)
        # Independent finite difference checks differentiation through all Newton solves.
        original = self.model.network[-1].bias.detach().clone()
        epsilon = .002
        losses = []
        with torch.no_grad():
            for sign in (1, -1):
                self.model.network[-1].bias.copy_(original + sign * epsilon)
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(self.model(*self.args), labels).item())
            self.model.network[-1].bias.copy_(original)
        numeric = (losses[0] - losses[1]) / (2 * epsilon)
        self.assertAlmostEqual(numeric, self.model.network[-1].bias.grad.item(), delta=4e-5)

    def test_masks_boundaries_and_query_labels_absent(self):
        changed = list(self.args)
        for values, masks in ((0, 1), (3, 4), (7, 8)):
            changed[values] = changed[values].clone()
            changed[values][changed[masks]] = float('nan')
        torch.testing.assert_close(self.model(*changed), self.model(*self.args), atol=0, rtol=0)
        for cls in (SupportRegularizer, GlobalRegularizer, FeatureTypeRegularizer, TargetOnlyRegularizer, FixedRegularizer):
            self.assertNotIn('query_y', inspect.signature(cls.forward).parameters)
        changed[9] = changed[9].clone()
        changed[9][0, 0] = 2
        with self.assertRaises(ValueError):
            self.model(*changed)
        with self.assertRaises(ValueError):
            SupportRegularizer(steps=5)


if __name__ == '__main__':
    unittest.main()
