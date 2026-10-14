"""Tempered product of experts (protocol v5): closed forms and the Hölder bound."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'lifted_cavity'))
from site_pool import gauss_nll, lin_pool_nll, log_pool, log_pool_slack  # noqa: E402


def test_log_pool_matches_normalized_numerical_product():
    rng = np.random.default_rng(0)
    y = np.linspace(-30, 30, 200001)
    for _ in range(20):
        ma, mb = rng.normal(0, 2, 2); va, vb = np.exp(rng.normal(0, 1, 2)); b = rng.uniform(.05, .95)
        logp = -(1 - b) * gauss_nll(ma, va, y) - b * gauss_nll(mb, vb, y)
        z = np.trapz(np.exp(logp), y)
        mu, var = log_pool(ma, va, mb, vb, b)
        assert abs(np.trapz(y * np.exp(logp), y) / z - mu) < 1e-6
        assert abs(-np.log(z) - log_pool_slack(ma, va, mb, vb, b)) < 1e-6


def test_pools_never_worse_than_average_nll_for_every_outcome():
    rng = np.random.default_rng(1)
    ma, mb, y = rng.normal(0, 3, (3, 5000)); va, vb = np.exp(rng.normal(0, 1.5, (2, 5000)))
    avg = .5 * gauss_nll(ma, va, y) + .5 * gauss_nll(mb, vb, y)
    mu, var = log_pool(ma, va, mb, vb)
    assert (gauss_nll(mu, var, y) <= avg + 1e-10).all()
    assert (lin_pool_nll(ma, va, mb, vb, y) <= avg + 1e-10).all()
    assert (log_pool_slack(ma, va, mb, vb) >= -1e-12).all()


def test_identical_experts_pool_to_themselves():
    mu, var = log_pool(1.3, .7, 1.3, .7)
    assert np.isclose(mu, 1.3) and np.isclose(var, .7) and abs(log_pool_slack(1.3, .7, 1.3, .7)) < 1e-12


def _times_site(mu, var, s_mu, s_var):
    """Multiply a Gaussian predictive by a Gaussian site N(s_mu; y, s_var) and renormalize (a Bayesian update)."""
    prec = 1 / var + 1 / s_var
    return (mu / var + s_mu / s_var) / prec, 1 / prec


def test_naturality_pool_commutes_with_site_updates_iff_weights_sum_to_one():
    """Universal property U2 (Genest 1984, external Bayesianity): pool o update = update o pool."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        ma, mb, sm = rng.normal(0, 2, 3); va, vb, sv = np.exp(rng.normal(0, 1, 3)); b = rng.uniform(.05, .95)
        up_then_pool = log_pool(*_times_site(ma, va, sm, sv), *_times_site(mb, vb, sm, sv), b)
        pool_then_up = _times_site(*log_pool(ma, va, mb, vb, b), sm, sv)
        assert np.allclose(up_then_pool, pool_then_up)
    # weights 0.7 + 0.7 (not summing to one) break the square: the site would be counted 1.4 times
    pa, pb = _times_site(0., 1., 2., .5), _times_site(1., 2., 2., .5)
    prec = .7 / pa[1] + .7 / pb[1]
    unnorm = ((.7 * pa[0] / pa[1] + .7 * pb[0] / pb[1]) / prec, 1 / prec)
    prec0 = .7 / 1. + .7 / 2.
    other = _times_site((.7 * 0. / 1. + .7 * 1. / 2.) / prec0, 1 / prec0, 2., .5)
    assert not np.allclose(unnorm, other)


def test_free_monoid_static_sites_are_a_mask_homomorphism():
    """Universal property U1: with static sites the posterior natural parameters are prior + sum over observed
    sensors, so F(A u B) = F(A) + F(B) - F(empty) for disjoint masks, and every mask follows from P site evaluations."""
    import torch
    import models
    torch.manual_seed(0)
    N, P, K = 6, 5, 1
    W = torch.randn(N, P, 1 + K, dtype=torch.float64); o = torch.randn(N, P, dtype=torch.float64)
    psi = torch.rand(N, P, dtype=torch.float64) + .1
    pp = torch.eye(1 + K, dtype=torch.float64).expand(N, -1, -1); pe = torch.zeros(N, 1 + K, dtype=torch.float64)
    net = models.LiftedCavity(K=K)
    F = lambda m: net.posterior(W, o, psi, m, pp, pe)[2:]
    A = torch.tensor([1., 1, 0, 0, 0], dtype=torch.float64).expand(N, P); B = torch.tensor([0., 0, 1, 0, 1], dtype=torch.float64).expand(N, P)
    (pA, nA), (pB, nB), (pAB, nAB), (p0, n0) = F(A), F(B), F(A + B), F(torch.zeros(N, P, dtype=torch.float64))
    assert torch.allclose(pAB, pA + pB - p0) and torch.allclose(nAB, nA + nB - n0)
