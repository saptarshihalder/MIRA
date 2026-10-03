"""Scientific invariants, input boundaries, and actual trainability on CPU."""
import inspect
import unittest

import torch
from model import Bridge, GenericContextual, LinearResidual


def fixture(columns=6):
    torch.manual_seed(71)
    def rows(n):
        return (torch.randn(2, n, columns), torch.rand(2, n, columns) < .3,
                torch.randint(2, (2, n)).float(), torch.randn(2, n))
    qx, qm, _, qlogit = rows(5)
    return (qx, qm, qlogit, *rows(7), *rows(9))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(811)
        self.model = Bridge().eval()
        self.args = fixture()

    def test_identity_swap_composition_and_bounds(self):
        q = self.args[:3]
        source, target = self.args[3:7], self.args[7:]
        identity = self.model(*q, *source, *source)
        self.assertTrue(torch.equal(identity, q[2]))
        residual = self.model(*self.args) - q[2]
        swapped = self.model(*q, *target, *source) - q[2]
        torch.testing.assert_close(residual, -swapped, atol=3e-7, rtol=1e-5)
        third = tuple(a.flip(1) if i != 2 else 1 - a.flip(1) for i, a in enumerate(target))
        bc = self.model(*q, *target, *third) - q[2]
        ac = self.model(*q, *source, *third) - q[2]
        torch.testing.assert_close(residual + bc, ac, atol=3e-7, rtol=1e-5)
        self.assertLessEqual(residual.abs().max().item(), 4.0 + 1e-6)

    def test_joint_field_permutation_and_variable_width(self):
        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        permuted = tuple(a[..., permutation] if a.ndim == 3 else a for a in self.args)
        torch.testing.assert_close(self.model(*self.args), self.model(*permuted), atol=5e-7, rtol=1e-5)
        for columns in (1, 10, 14):
            output = self.model(*fixture(columns))
            self.assertEqual(output.shape, (2, 5))
            self.assertTrue(torch.isfinite(output).all())

    def test_query_order_batching_and_support_order(self):
        output = self.model(*self.args)
        query_order = torch.tensor([4, 1, 3, 0, 2])
        shuffled = tuple(a[:, query_order] if i < 3 else a for i, a in enumerate(self.args))
        torch.testing.assert_close(output[:, query_order], self.model(*shuffled), atol=4e-7, rtol=1e-5)
        parts = [self.model(*(a[:, start:stop] if i < 3 else a for i, a in enumerate(self.args)))
                 for start, stop in ((0, 2), (2, 5))]
        torch.testing.assert_close(output, torch.cat(parts, 1), atol=4e-7, rtol=1e-5)
        reverse = tuple(a.flip(1) if i >= 3 else a for i, a in enumerate(self.args))
        torch.testing.assert_close(output, self.model(*reverse), atol=5e-7, rtol=1e-5)
        batched = torch.cat([self.model(*(a[b:b + 1] for a in self.args)) for b in range(2)], 0)
        torch.testing.assert_close(output, batched, atol=5e-7, rtol=1e-5)

    def test_masked_values_ignored_and_query_labels_absent(self):
        changed = list(self.args)
        for value, mask in ((0, 1), (3, 4), (7, 8)):
            changed[value] = changed[value].clone()
            changed[value][changed[mask]] = float('nan')
        torch.testing.assert_close(self.model(*self.args), self.model(*changed), atol=0, rtol=0)
        for cls in (Bridge, GenericContextual, LinearResidual):
            self.assertNotIn('query_y', inspect.signature(cls.forward).parameters)
        changed[0][0, 0, 0] = float('inf')
        changed[1] = changed[1].clone()
        changed[1][0, 0, 0] = False
        with self.assertRaises(ValueError):
            self.model(*changed)

    def test_learns_finite_query_specific_support_response(self):
        output = self.model(*self.args)
        residual = output - self.args[2]
        self.assertGreater(residual.var(1).mean().item(), 1e-9)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.002)
        labels = torch.tensor([[0., 1., 1., 0., 1.], [1., 0., 0., 1., 0.]])
        before = torch.nn.functional.binary_cross_entropy_with_logits(output, labels).item()
        for _ in range(8):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.model(*self.args), labels)
            loss.backward()
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            optimizer.step()
        after = torch.nn.functional.binary_cross_entropy_with_logits(self.model(*self.args), labels).item()
        self.assertLess(after, before)
        # A gradient to a support label establishes that the labels actually enter h.
        args = list(self.args)
        args[9] = args[9].clone().requires_grad_(True)
        self.model(*args).sum().backward()
        self.assertGreater(args[9].grad.abs().sum().item(), 0)

    def test_controls_and_ablations(self):
        for model in (GenericContextual(), LinearResidual(),
                      *(Bridge(ablation=a) for a in ('target_only', 'no_mask', 'no_query'))):
            output = model(*self.args)
            self.assertEqual(output.shape, self.args[2].shape)
            self.assertTrue(torch.isfinite(output).all())
            output.sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        model = Bridge(ablation='no_query')
        residual = model(*self.args) - self.args[2]
        torch.testing.assert_close(residual, residual[:, :1].expand_as(residual), atol=3e-7, rtol=1e-5)
        counts = [sum(p.numel() for p in cls().parameters()) for cls in (Bridge, GenericContextual)]
        self.assertLess(max(counts), 100000)
        self.assertLess(abs(counts[0] - counts[1]) / counts[0], .01)


if __name__ == '__main__':
    unittest.main()
