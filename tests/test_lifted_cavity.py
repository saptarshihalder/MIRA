"""Checks for the lifted cavity network: oracle correctness, exactness of lifted sites, 1-D PoE sign failure,
initialisation identity, mask handling and sensor exchangeability."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'lifted_cavity'))
sys.path.insert(0, str(ROOT / 'experiments'))

torch = pytest.importorskip('torch')
import family as fam  # noqa: E402
import fa as fa_mod  # noqa: E402
import data  # noqa: E402
import models  # noqa: E402


def test_batched_context_matches_repository_context():
    import cavity_site as repo
    T = fam.make_tasks(np.random.default_rng(1), 8)
    for t in range(8):
        c, _, _ = repo.context(T['raw_sx'][t], T['sy'][t].astype(float), T['sm'][t].astype(float))
        assert np.allclose(c, T['c'][t], atol=2e-5)


def test_oracle_moments_match_importance_sampling():
    T = fam.make_tasks(np.random.default_rng(2), 2)
    S = np.array([0, 2, 4])
    _, pm, pv = fam.oracle(T['prm'], 0, S, T['raw_qx'][0], T['qy'][0].astype(float))
    a, b, f, d, s = (T['prm'][k][0, S] for k in ('a', 'b', 'f', 'd', 's'))
    prec = np.linalg.inv(np.outer(d, d) + np.diag(s ** 2))
    ys = np.random.default_rng(0).normal(size=300000)
    r = T['raw_qx'][0][0, S] - (ys[:, None] * a + b + .4 * np.sin(ys[:, None] * f))
    lw = -.5 * np.einsum('ni,ij,nj->n', r, prec, r); w = np.exp(lw - lw.max()); w /= w.sum()
    m = (w * ys).sum()
    assert abs(m - pm[0]) < 1e-2 and abs((w * (ys - m) ** 2).sum() / pv[0] - 1) < .08


def test_lifted_sites_are_exact_for_every_mask_in_a_linear_gaussian_factor_world():
    rng = np.random.default_rng(3)
    P, K = 6, 2
    A, D, psi = rng.normal(size=P), rng.normal(size=(P, K)), rng.uniform(.05, .5, P)
    vy, my, mx = 1.7, .3, rng.normal(size=P)
    cov_x = vy * np.outer(A, A) + D @ D.T + np.diag(psi)
    fa = dict(mx=mx, my=my, vy=vy, A=A, D=D, psi=psi)
    qx = rng.normal(size=(4, P))
    for k in range(1, P + 1):
        for S in [np.sort(rng.choice(P, k, replace=False)) for _ in range(5)]:
            mean, var = fa_mod.lifted_poe(fa, qx, S)
            beta = np.linalg.solve(cov_x[np.ix_(S, S)], vy * A[S])
            assert np.allclose(mean, my + (qx[:, S] - mx[S]) @ beta, atol=1e-9)
            assert np.allclose(var, vy - vy * A[S] @ beta, atol=1e-9)


def test_shared_noise_flips_bayes_weights_so_positive_1d_sites_cannot_be_mask_consistent():
    # x1 = y + u, x2 = y + .5u + small noise: Bayes weight on x1 is positive alone and negative in the pair.
    a, d, s = np.array([1., 1.]), np.array([1., .5]), np.array([.05, .05])
    cov = np.outer(a, a) + np.outer(d, d) + np.diag(s ** 2)
    single = a[0] / cov[0, 0]
    pair = np.linalg.solve(cov, a)
    assert single > 0 and pair[0] < 0
    # any product of positive-precision sites eta_j(x_j)/tau_j keeps the singleton sign of each coefficient


def _panel(n=6, P=5, seed=4):
    pool = data.build_pool(n, seed, P=P)
    return pool


def test_untrained_lifted_model_equals_closed_form_and_ignores_missing_values():
    pool = _panel()
    torch.manual_seed(0)
    for K in (0, 1, 2):
        net = models.LiftedCavity(K=K)
        _, batch = next(data.eval_batches(pool, fam.mask_bank(2), tasks_per_batch=3))
        with torch.no_grad():
            mu, lv = net(batch)
            mu0, lv0 = models.closed_form(batch['anchors'][K], batch['qx'], batch['qm'], batch['tid'])
        assert torch.allclose(mu, mu0, atol=1e-4) and torch.allclose(lv, lv0, atol=1e-4)
    # perturb a trained-like (random last layer) model: values of masked sensors must not matter
    net = models.LiftedCavity(K=1)
    torch.nn.init.normal_(net.site[-1].weight, std=.1)
    _, batch = next(data.eval_batches(pool, fam.mask_bank(2), tasks_per_batch=3))
    with torch.no_grad():
        mu, lv = net(batch)
        b2 = dict(batch); b2['qx'] = batch['qx'] + 10 * (1 - batch['qm']) * torch.randn_like(batch['qx'])
        mu2, lv2 = net(b2)
    assert torch.allclose(mu, mu2, atol=1e-4) and torch.allclose(lv, lv2, atol=1e-4)


def test_lifted_model_is_sensor_exchangeable_and_runs_on_more_sensors():
    pool = _panel(P=8, seed=5)
    net = models.LiftedCavity(K=1)
    torch.nn.init.normal_(net.site[-1].weight, std=.1)
    _, batch = next(data.eval_batches(pool, fam.mask_bank(3, 8), tasks_per_batch=2))
    perm = torch.randperm(8)
    pb = dict(batch)
    for key in ('qx', 'qm'):
        pb[key] = batch[key][:, perm]
    for key in ('sx', 'sm'):
        pb[key] = batch[key][:, :, perm]
    pb['anchors'] = {K: {k: (v[:, perm] if v.dim() > 1 and v.shape[1] == 8 else v) for k, v in a.items()} for K, a in batch['anchors'].items()}
    with torch.no_grad():
        mu, lv = net(batch); mu2, lv2 = net(pb)
    assert torch.allclose(mu, mu2, atol=1e-4) and torch.allclose(lv, lv2, atol=1e-4)
