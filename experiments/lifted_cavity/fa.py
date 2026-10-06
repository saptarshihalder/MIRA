"""Supervised factor-analysis Gaussian (lifted PoE initialization), support-only, closed form.

Joint model in standardized sensor units:  x = mu_x + A (y - mu_y) + D h + e,  h ~ N(0, I_K), e ~ N(0, diag(psi)).
Given an estimate of the joint moments of (x, y), A = Cov(x,y)/Var(y), and (D, psi) come from principal-factor
iterations on the residual covariance R = Cov(x) - A A^T Var(y).  The PoE over the lifted latent z = (y, h)
with one rank-1 Gaussian site per observed sensor is exactly the conditional Gaussian of y given x_S under this
structured covariance, for every missingness pattern S.
"""
import numpy as np


def principal_factors(R, K, iters=50, floor=1e-3):
    p = len(R)
    psi = np.clip(np.diag(R) * .5, floor, None)
    for _ in range(iters):
        vals, vecs = np.linalg.eigh(R - np.diag(psi))
        top = np.argsort(vals)[::-1][:K]
        D = vecs[:, top] * np.sqrt(np.clip(vals[top], 0, None))
        psi = np.clip(np.diag(R) - (D * D).sum(1), floor, None)
    return D, psi


def fit_fa(mu, sig, K=1):
    p = len(mu) - 1
    vy = sig[p, p]
    A = sig[:p, p] / vy
    R = sig[:p, :p] - np.outer(A, A) * vy
    R = (R + R.T) / 2
    D, psi = principal_factors(R, K)
    return dict(mx=mu[:p], my=mu[p], vy=vy, A=A, D=D, psi=psi)


def lifted_poe(fa, qx, S, n=None):
    """Posterior over z=(y,h) as a product of per-sensor rank-1 Gaussian sites (exact under the FA model)."""
    K = fa['D'].shape[1]
    P0 = np.diag(np.r_[1 / fa['vy'], np.ones(K)])
    h0 = np.r_[fa['my'] / fa['vy'], np.zeros(K)]
    W = np.column_stack((fa['A'], fa['D']))[S]                 # (|S|, 1+K)
    lam = (W.T / fa['psi'][S]) @ W
    prec = P0 + lam
    obs = qx[:, S] - fa['mx'][S] + fa['A'][S] * fa['my']     # x~ + A mu_y  (per query)
    eta = h0[None] + (obs / fa['psi'][S]) @ W
    cov = np.linalg.inv(prec)
    mean = eta @ cov.T
    var = cov[0, 0]
    if n is not None:
        var = var * n / max(n - len(S) - 1, 1)
    return mean[:, 0], np.full(len(qx), var)
