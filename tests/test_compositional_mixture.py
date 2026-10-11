"""Structural/oracle fixtures only: no training or publication evidence."""
import copy
import math

import pytest
import torch

from experiments.compositional_mixture.model import CompositionalMixture, Query


def fixture(p=7):
    rng = torch.Generator().manual_seed(20261117)
    rand = lambda *shape: torch.randn(*shape, generator=rng, dtype=torch.float64)
    b, rows, n = 2, 9, 4
    anchor = dict(A=rand(b, p), D=rand(b, p, 2), psi=rand(b, p).exp(),
                  mx=rand(b, p), my=rand(b), vy=rand(b).exp())
    return dict(sx=rand(b, rows, p), sy=rand(b, rows), sm=(rand(b, rows, p) > -.5).double(),
                anchor=anchor, tid=torch.tensor([0, 1, 1, 0]), qx=rand(n, p),
                qm=(rand(n, p) > -.6).double(), qy=rand(n))


def model():
    with torch.random.fork_rng():
        torch.manual_seed(41)
        return CompositionalMixture().double()


def compare(left, right):
    for name in ('log_weights', 'means', 'variances', 'log_evidence'):
        torch.testing.assert_close(getattr(left, name), getattr(right, name), atol=3e-10, rtol=3e-10)


def test_exact_density_and_posterior_against_dense_joint_gaussian():
    batch = fixture()
    query = model().query(batch)
    pred = query.decode(query.summarize())
    for i in range(4):
        keep = batch['qm'][i].bool()
        for c in range(3):
            w = query.loadings[i, c, keep]
            covariance = w @ w.T + torch.diag(query.noise[i, c, keep])
            residual = query.qx[i, keep] - query.locations[i, c, keep]
            mvn = torch.distributions.MultivariateNormal(query.locations[i, c, keep], covariance_matrix=covariance)
            torch.testing.assert_close(pred.log_evidence[i, c], mvn.log_prob(query.qx[i, keep]), atol=1e-10, rtol=1e-10)
            cross = query.vy[i].sqrt() * w[:, 0]
            mean = query.my[i] + cross @ torch.linalg.solve(covariance, residual)
            variance = query.vy[i] - cross @ torch.linalg.solve(covariance, cross)
            torch.testing.assert_close(pred.means[i, c], mean, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(pred.variances[i, c], variance, atol=1e-10, rtol=1e-10)
    target = batch['qy']
    reference_nll = -torch.logsumexp(pred.log_weights + torch.distributions.Normal(
        pred.means, pred.variances.sqrt()).log_prob(target[:, None]), -1)
    torch.testing.assert_close(pred.nll(target), reference_nll)


def test_disjoint_merge_associativity_deletion_and_duplicate_guards():
    query = model().query(fixture())
    a, b, c = [query.summarize(indices) for indices in ([0, 2], [1, 3, 4], [5, 6])]
    compare(query.decode(a.merge(b).merge(c)), query.decode(a.merge(b.merge(c))))
    compare(query.decode(a.merge(b).merge(c)), query.decode(query.summarize()))
    compare(query.decode(a.merge(b).merge(c).remove(b)), query.decode(query.summarize([0, 2, 5, 6])))
    compare(query.decode(query.summarize([]).merge(a)), query.decode(a))
    with pytest.raises(ValueError):
        a.merge(a)
    with pytest.raises(ValueError):
        a.remove(b)
    with pytest.raises(ValueError):
        a.merge(model().query(fixture()).summarize([1]))
    with pytest.raises(ValueError):
        query.summarize([1, 1])


def restrict(batch, order):
    result = copy.deepcopy(batch)
    for key in ('sx', 'sm', 'qx', 'qm'):
        result[key] = batch[key][..., order]
    for key in ('A', 'D', 'psi', 'mx'):
        result['anchor'][key] = batch['anchor'][key][:, order]
    return result


def test_subset_recompilation_permutation_and_support_order_consistency():
    net, batch = model(), fixture()
    original = net.query(batch)
    order = [6, 1, 4]
    subset = net.query(restrict(batch, order))
    compare(original.decode(original.summarize(order)), subset.decode(subset.summarize()))
    shuffled = restrict(batch, [6, 5, 4, 3, 2, 1, 0])
    compare(net.distribution(batch), net.distribution(shuffled))
    for key in ('sx', 'sy', 'sm'):
        shuffled[key] = shuffled[key].flip(1)
    compare(net.distribution(batch), net.distribution(shuffled))


def test_factor_rotation_masks_empty_queries_and_query_label_boundary():
    net, batch = model(), fixture()
    expected = net.distribution(batch)
    batch['anchor']['D'] = batch['anchor']['D'] @ torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
    batch['sx'][~batch['sm'].bool()] = float('nan')
    batch['qx'][~batch['qm'].bool()] = float('nan')
    batch['qy'].fill_(float('nan'))
    compare(expected, net.distribution(batch))
    batch['qm'].zero_()
    empty = net.distribution(batch)
    mean, lv = empty.moments()
    torch.testing.assert_close(mean, batch['anchor']['my'][batch['tid']])
    torch.testing.assert_close(lv.exp(), batch['anchor']['vy'][batch['tid']])
    torch.testing.assert_close(empty.log_evidence, torch.zeros_like(empty.log_evidence))


def test_gradients_reach_support_encoder_new_factors_and_mixture_weights():
    net, batch = model(), fixture()
    net.distribution(batch).nll(batch['qy']).mean().backward()
    for parameter in net.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert parameter.grad.norm() > 1e-12
    factor_grad = net.head[-1].weight.grad.reshape(3, 6, -1)[:, 4:]
    assert factor_grad.norm() > 1e-12
    assert sum(p.numel() for p in net.parameters()) == 1317


def test_exact_mixture_supports_nonlinear_mean_without_query_set_attention():
    # Fixed constructed example, not a trained performance result.
    x = torch.tensor([0., 1., 2.], dtype=torch.float64)
    loadings = torch.tensor([[[[1.]], [[-1.]]]], dtype=torch.float64).expand(3, -1, -1, -1)
    locations = torch.tensor([[[0.], [2.]]], dtype=torch.float64).expand(3, -1, -1)
    query = Query(loadings, locations, torch.full((3, 2, 1), .25, dtype=torch.float64),
                  torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64),
                  torch.zeros(2, dtype=torch.float64), x[:, None], torch.ones(3, 1, dtype=torch.float64))
    pred = query.decode(query.summarize())
    mean, _ = pred.moments()
    assert abs(mean[0] - 2 * mean[1] + mean[2]) > .1
    assert not torch.allclose(pred.log_weights[0], pred.log_weights[1])


def test_invalid_masks_and_nonfinite_observed_values_fail_explicitly():
    net, batch = model(), fixture()
    batch['qm'][0, 0] = .5
    with pytest.raises(ValueError):
        net.distribution(batch)
    batch['qm'][0, 0] = 1
    batch['qx'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        net.distribution(batch)
    for p in (1, 16):
        mean, lv = net(fixture(p))
        assert bool(torch.isfinite(mean).all() and torch.isfinite(lv).all())
