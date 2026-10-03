"""Boundary, recurrence, teacher and objective checks; no efficacy experiment."""
import inspect
import unittest

import torch
from model import RecursiveJEPA


def fixture(columns=6):
    torch.manual_seed(618)
    def rows(n):
        return (torch.randn(2, n, columns), torch.rand(2, n, columns) < .3,
                torch.randint(2, (2, n)).float(), torch.randn(2, n))
    qx, qm, _, qlogit = rows(5)
    return (qx, qm, qlogit, *rows(11), *rows(13))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(211)
        self.model = RecursiveJEPA()
        self.inputs = fixture()
        self.labels = torch.tensor([[0., 1., 0., 1., 1.], [1., 0., 1., 0., 0.]])

    def test_teacher_no_grad_ema_explicit_and_eval_invariant(self):
        teacher_before = {k: v.clone() for k, v in self.model.teacher.state_dict().items()}
        loss, breakdown = self.model.training_loss(self.inputs, self.labels)
        loss.backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.teacher.parameters()))
        self.assertEqual(int(self.model.teacher_updates), 0)
        self.assertTrue(all(torch.equal(v, teacher_before[k]) for k, v in self.model.teacher.state_dict().items()))
        self.assertGreater(breakdown['variance'], 0)
        self.assertGreater(breakdown['jepa_mse'], 0)
        with torch.no_grad():
            for p in self.model.encoder.parameters():
                p.add_(.01)
        student = dict(self.model.encoder.state_dict())
        self.model.update_teacher(.8)
        self.assertEqual(int(self.model.teacher_updates), 1)
        for key, value in self.model.teacher.state_dict().items():
            torch.testing.assert_close(value, .8 * teacher_before[key] + .2 * student[key], atol=1e-7, rtol=1e-5)
        self.model.train()
        self.assertFalse(self.model.teacher.training)
        self.model.eval()
        snapshot = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            a, b = self.model(*self.inputs), self.model(*self.inputs)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertTrue(all(torch.equal(v, snapshot[k]) for k, v in self.model.state_dict().items()))

    def test_outer_gradients_reach_every_step_and_support_labels(self):
        details = self.model.forward_details(*self.inputs)
        for state in details['step_latents']:
            for latent in state.values():
                latent.retain_grad()
        for coefficient in details['step_coefficients']:
            coefficient.retain_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(details['logits'], self.labels).backward()
        for state in details['step_latents']:
            for latent in state.values():
                self.assertIsNotNone(latent.grad)
                self.assertTrue(torch.isfinite(latent.grad).all())
                self.assertGreater(latent.grad.abs().sum().item(), 0)
        for coefficient in details['step_coefficients']:
            self.assertTrue(torch.isfinite(coefficient.grad).all())
            self.assertGreater(coefficient.grad.abs().sum().item(), 0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.model.refiner.parameters()), 0)
        changed = list(self.inputs)
        changed[5] = 1 - changed[5]
        self.assertGreater((self.model(*changed) - self.model(*self.inputs)).abs().max().item(), 1e-6)
        changed = list(self.inputs)
        changed[9] = 1 - changed[9]
        self.assertGreater((self.model(*changed) - self.model(*self.inputs)).abs().max().item(), 1e-4)

    def test_hidden_values_never_enter_prediction_or_corruption_teacher(self):
        changed = list(self.inputs)
        for x, mask in ((0, 1), (3, 4), (7, 8)):
            changed[x] = changed[x].clone()
            changed[x][changed[mask]] = float('nan')
        torch.testing.assert_close(self.model(*changed), self.model(*self.inputs), atol=0, rtol=0)
        torch.manual_seed(711)
        loss_a, info_a = self.model.training_loss(self.inputs, self.labels)
        torch.manual_seed(711)
        loss_b, info_b = self.model.training_loss(changed, self.labels)
        torch.testing.assert_close(loss_a, loss_b, atol=0, rtol=0)
        self.assertEqual(info_a, info_b)
        weak_x, weak_mask, rich_x, rich_mask = self.model.corruption_views(changed[0], changed[1])
        self.assertTrue((weak_mask | rich_mask).equal(weak_mask))
        self.assertTrue((rich_mask | self.inputs[1]).equal(rich_mask))
        self.assertTrue(torch.isfinite(weak_x).all() and torch.isfinite(rich_x).all())
        self.assertTrue((rich_x[self.inputs[1]] == 0).all())
        self.assertNotIn('query_y', inspect.signature(RecursiveJEPA.forward).parameters)

    def test_column_row_permutations_and_widths(self):
        output = self.model(*self.inputs)
        order = torch.tensor([4, 0, 5, 2, 1, 3])
        shuffled = tuple(x[..., order] if x.ndim == 3 else x for x in self.inputs)
        torch.testing.assert_close(output, self.model(*shuffled), atol=4e-7, rtol=1e-6)
        reverse = tuple(x.flip(1) if i >= 3 else x for i, x in enumerate(self.inputs))
        torch.testing.assert_close(output, self.model(*reverse), atol=4e-7, rtol=1e-6)
        order = torch.tensor([4, 1, 3, 0, 2])
        shuffled = tuple(x[:, order] if i < 3 else x for i, x in enumerate(self.inputs))
        torch.testing.assert_close(output[:, order], self.model(*shuffled), atol=0, rtol=0)
        chunks = [self.model(*(x[:, start:stop] if i < 3 else x for i, x in enumerate(self.inputs)))
                  for start, stop in ((0, 2), (2, 5))]
        torch.testing.assert_close(output, torch.cat(chunks, 1), atol=0, rtol=0)
        for width in (1, 10, 14):
            self.assertTrue(torch.isfinite(self.model(*fixture(width))).all())

    def test_bounds_frozen_action_and_objective_breakdown(self):
        details = self.model.forward_details(*self.inputs)
        self.assertEqual(len(details['step_logits']), 3)
        self.assertLessEqual((details['logits'] - self.inputs[2]).abs().max().item(), 2 + 1e-6)
        self.assertLessEqual(details['coefficients'].abs().max().item(), self.model.steps * self.model.head_step + 1e-6)
        for state in details['step_latents']:
            self.assertTrue(all(latent.abs().max() <= 1 for latent in state.values()))
        self.assertTrue(torch.equal(self.model(*self.inputs, force_frozen=True), self.inputs[2]))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(self.model(*self.inputs)).all())
        loss, info = self.model.training_loss(self.inputs, self.labels)
        expected = info['supervised_bce'] + .2 * info['deep_bce'] + .1 * info['jepa_mse'] + .01 * info['variance'] + .001 * info['covariance']
        self.assertAlmostEqual(loss.item(), expected, delta=2e-7)
        supervised = RecursiveJEPA(use_jepa=False)
        _, info = supervised.training_loss(self.inputs, self.labels)
        self.assertEqual([info[k] for k in ('jepa_mse', 'variance', 'covariance')], [0, 0, 0])
        one_step = RecursiveJEPA(steps=1)
        _, info = one_step.training_loss(self.inputs, self.labels)
        self.assertEqual(info['deep_bce'], 0)
        self.assertEqual(sum(p.numel() for p in one_step.parameters()), sum(p.numel() for p in self.model.parameters()))


if __name__ == '__main__':
    unittest.main()
