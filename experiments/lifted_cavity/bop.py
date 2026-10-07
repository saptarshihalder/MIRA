"""Bayes-optimal predictive under the true task prior (privileged ceiling), by batched HMC over task parameters.

The ceiling knows the generator family and its prior, but not the task's parameters: it sees exactly what every
method sees (48 labeled support rows with their missingness, and the query's observed sensors). Per task:
  theta = (a, b, f, d, s) for every sensor;  x_j = a_j y + b_j + nl * sin(f_j y) + d_j u + s_j eps_j,  u, eps ~ N(0,1).
The support likelihood marginalises u analytically (rank-1 plus diagonal covariance over the observed sensors).
Posterior samples come from HMC with per-chain step-size and diagonal-mass adaptation, started from MAP fits.
The sign of each a_j is fixed by the support correlation (|a_j| >= .6 makes it unambiguous in practice).
Predictive density of a query, exactly given the samples (the query's own x reweights the samples):
  p(y | x_S, D) = sum_m p(y, x_S | theta_m) / sum_m p(x_S | theta_m),
with p(x_S | theta_m) by trapezoid quadrature over y on a uniform grid.
"""
import math
import numpy as np
import torch

LOG2PI = math.log(2 * math.pi)
torch.set_default_dtype(torch.float64)


def unpack(z, sign, P):
    za, b, zf, d, zs = z.split(P, -1)
    a = sign * (.6 + 1.2 * torch.sigmoid(za))
    f = .5 + 2. * torch.sigmoid(zf)
    s = .15 + .65 * torch.sigmoid(zs)
    return a, b, f, d, s


def log_prior(z, P):
    za, b, zf, d, zs = z.split(P, -1)
    lj = lambda u: torch.nn.functional.logsigmoid(u) + torch.nn.functional.logsigmoid(-u)
    return (lj(za) + lj(zf) + lj(zs) - .5 * (b / .4) ** 2 - .5 * (d / .7) ** 2).sum(-1)


def loglik_rows(a, b, f, d, s, nl, x, y, m):
    """Log density of observed sensors per row. a..s (..., P); x, m (..., n, P); y (..., n)."""
    mu = a[..., None, :] * y[..., None] + b[..., None, :] + nl * torch.sin(f[..., None, :] * y[..., None])
    r = (x - mu) * m
    w = m / (s[..., None, :] ** 2)
    q1 = (w * r * r).sum(-1)
    g = (w * d[..., None, :] * r).sum(-1)
    c = 1 + (w * d[..., None, :] ** 2).sum(-1)
    logdet = (m * torch.log(s[..., None, :] ** 2)).sum(-1) + torch.log(c)
    return -.5 * (m.sum(-1) * LOG2PI + logdet + q1 - g * g / c)


def make_target(x, y, m, sign, nl, P):
    def logp(z):
        a, b, f, d, s = unpack(z, sign, P)
        return loglik_rows(a, b, f, d, s, nl, x, y, m).sum(-1) + log_prior(z, P)
    return logp


def grad_logp(logp, z):
    z = z.detach().requires_grad_(True)
    lp = logp(z)
    g, = torch.autograd.grad(lp.sum(), z)
    return lp.detach(), g.detach()


def sample(x, y, m, nl, chains=4, warm=(300, 200), draws=50, thin=4, L=12, seed=0):
    """x, m (T, n, P) raw support with missing entries zeroed and mask; y (T, n). Returns samples (T, chains*draws/thin, 5P)."""
    gen = torch.Generator().manual_seed(seed)
    T, n, P = x.shape
    xc = x - (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    sign = torch.sign(((xc * m) * (y - y.mean(1, keepdim=True))[..., None]).sum(1))
    sign[sign == 0] = 1
    X, Y, M, SG = (v[:, None].expand(T, chains, *v.shape[1:]) for v in (x, y, m, sign))
    logp = make_target(X, Y, M, SG, nl, P)
    # initial points: prior draws, then MAP by Adam
    z = torch.randn(T, chains, 5 * P, generator=gen)
    z[..., P:2 * P] *= .4; z[..., 3 * P:4 * P] *= .7
    z = z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=.05)
    for _ in range(400):
        loss = -logp(z).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    z = z.detach()
    eps = torch.full((T, chains, 1), .02)
    minv = torch.ones(T, chains, 5 * P)
    lp, g = grad_logp(logp, z)
    kept, acc_hist = [], []
    total = warm[0] + warm[1] + draws * thin
    buf = []
    for it in range(total):
        p0 = torch.randn(z.shape, generator=gen) / minv.sqrt()
        zn, p, gn = z.clone(), p0.clone(), g
        p = p + .5 * eps * gn
        for l in range(L):
            zn = zn + eps * minv * p
            lpn, gn = grad_logp(logp, zn)
            if l < L - 1:
                p = p + eps * gn
        p = p + .5 * eps * gn
        h0 = lp - .5 * (p0 * p0 * minv).sum(-1)
        h1 = lpn - .5 * (p * p * minv).sum(-1)
        logacc = (h1 - h0).nan_to_num(nan=-1e9).clamp(max=0)
        acc = torch.rand(logacc.shape, generator=gen).log() < logacc
        z = torch.where(acc[..., None], zn, z); lp = torch.where(acc, lpn, lp); g = torch.where(acc[..., None], gn, g)
        if it < warm[0] + warm[1]:
            eps = eps * torch.exp(.05 * (logacc.exp()[..., None] - .75))
            if warm[0] // 2 <= it < warm[0]:
                buf.append(z)
            if it == warm[0] - 1:
                minv = torch.stack(buf).var(0).clamp(1e-5, 10.); buf = []
        elif (it - warm[0] - warm[1]) % thin == 0:
            kept.append(z); acc_hist.append(acc.double().mean().item())
    S = torch.stack(kept, 2)                                         # (T, chains, draws, 5P)
    return S, sign, float(np.mean(acc_hist))


def rhat(S):
    """Split-free Gelman-Rubin R-hat per parameter; S (T, chains, draws, D)."""
    C, N = S.shape[1], S.shape[2]
    cm = S.mean(2); cv = S.var(2)
    W = cv.mean(1); B = N * cm.var(1)
    return torch.sqrt(((N - 1) / N * W + B / N) / W.clamp_min(1e-12))


def predictive_nll(S, sign, nl, qx, qy, masks, grid=np.linspace(-4.5, 4.5, 301)):
    """S (chains, draws, 5P) for one task; qx (Q, P) raw; qy (Q,); masks (K, P). Returns NLL (K, Q).

    The Gaussian quadratic form is expanded so that every sensor sum is a matrix product:
    sum_j w_j (x_j - mu_j(y))^2 = sum_j w_j x_j^2 - 2 sum_j w_j x_j mu_j(y) + sum_j w_j mu_j(y)^2.
    """
    P = qx.shape[1]
    th = S.reshape(-1, 5 * P)
    a, b, f, d, s = unpack(th, sign, P)                               # (Mc, P)
    G = torch.as_tensor(grid)
    mu = a[:, None, :] * G[None, :, None] + b[:, None, :] + nl * torch.sin(f[:, None, :] * G[None, :, None])   # (Mc, Gr, P)
    mut = a[:, None, :] * qy[None, :, None] + b[:, None, :] + nl * torch.sin(f[:, None, :] * qy[None, :, None])  # (Mc, Q, P)
    x = qx
    out = []
    for mk in masks:
        m = torch.as_tensor(mk, dtype=torch.float64)
        w = m / s ** 2                                                # (Mc, P)
        wd = w * d
        cc = 1 + (wd * d).sum(-1)                                     # (Mc,)
        logdet = (m * torch.log(s ** 2)).sum(-1) + torch.log(cc)     # (Mc,)
        A = w @ (x * x).T                                             # (Mc, Q)
        Bm = torch.bmm(x[None] * w[:, None, :], mu.transpose(1, 2))   # (Mc, Q, Gr)
        C = (w[:, None, :] * mu * mu).sum(-1)                         # (Mc, Gr)
        G1 = wd @ x.T                                                 # (Mc, Q)
        G2 = (wd[:, None, :] * mu).sum(-1)                            # (Mc, Gr)
        quad = A[:, :, None] - 2 * Bm + C[:, None, :] - (G1[:, :, None] - G2[:, None, :]) ** 2 / cc[:, None, None]
        lg = -.5 * (m.sum() * LOG2PI + logdet[:, None, None] + quad) - .5 * (LOG2PI + G[None, None, :] ** 2)
        mx = lg.amax(-1, keepdim=True)
        logpx = torch.log(torch.trapezoid(torch.exp(lg - mx), G, dim=-1)) + mx[..., 0]   # (Mc, Q)
        r = (x[None] - mut) * m                                      # (Mc, Q, P)
        qt = (w[:, None, :] * r * r).sum(-1) - (wd[:, None, :] * r).sum(-1) ** 2 / cc[:, None]
        lt = -.5 * (m.sum() * LOG2PI + logdet[:, None] + qt) - .5 * (LOG2PI + qy[None] ** 2)
        out.append(-(torch.logsumexp(lt, 0) - torch.logsumexp(logpx, 0)))
    return torch.stack(out)


def rhat_params(S, sign, P):
    """R-hat on the identifiable parameters (a, b, f, |d|, s); d's sign is a symmetric non-identifiability."""
    a, b, f, d, s = unpack(S, sign[:, None, None, :], P)
    return rhat(torch.cat((a, b, f, d.abs(), s), -1))
