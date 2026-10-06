"""Batched (torch) support-only anchors: EM joint Gaussian with missing sensors, then supervised factor analysis."""
import numpy as np
import torch

P = 5


@torch.no_grad()
def em_batched(sx, sy, sm, iters=8, ridge=1e-4):
    """sx (B,n,5) standardized with missing->0, sy (B,n), sm (B,n,5) in {0,1}. Returns mu (B,6), sig (B,6,6)."""
    sx, sy, sm = (torch.as_tensor(v, dtype=torch.float64) for v in (sx, sy, sm))
    B, n, P = sx.shape
    Z = torch.cat((sx, sy[..., None]), -1)
    O = torch.cat((sm, torch.ones(B, n, 1, dtype=sm.dtype)), -1)
    cnt = O.sum(1).clamp_min(1)
    mu = (Z * O).sum(1) / cnt
    var = (((Z - mu[:, None]) * O) ** 2).sum(1) / cnt + 1e-3
    sig = torch.diag_embed(var)
    eye = torch.eye(P + 1, dtype=torch.float64)
    Mo = torch.diag_embed(O)                                   # (B,n,6,6)
    for _ in range(iters):
        S = sig[:, None].expand(B, n, P + 1, P + 1)
        Sp = Mo @ S @ Mo + (eye - Mo)
        G = S @ Mo @ torch.linalg.inv(Sp) @ Mo                 # Sigma_{:,o} Sigma_oo^{-1}, zero on missing columns
        dev = ((Z - mu[:, None]) * O)[..., None]
        xh = mu[:, None] + (G @ dev)[..., 0]
        C = S - G @ S
        mu = xh.mean(1)
        d = xh - mu[:, None]
        sig = (d.transpose(1, 2) @ d + C.sum(1)) / n + ridge * eye
        sig = (sig + sig.transpose(1, 2)) / 2
    return mu, sig


@torch.no_grad()
def fa_batched(mu, sig, K=1, iters=40, floor=1e-3):
    """Supervised FA in standardized sensor units. Returns dict of (B,...) float64 tensors."""
    P = mu.shape[1] - 1
    vy = sig[:, P, P]
    A = sig[:, :P, P] / vy[:, None]
    R = sig[:, :P, :P] - A[:, :, None] * A[:, None, :] * vy[:, None, None]
    R = (R + R.transpose(1, 2)) / 2
    diagR = torch.diagonal(R, dim1=1, dim2=2)
    psi = (diagR * .5).clamp_min(floor)
    if K == 0:
        return dict(mx=mu[:, :P], my=mu[:, P], vy=vy, A=A, D=torch.zeros(len(mu), P, 0, dtype=mu.dtype), psi=diagR.clamp_min(floor))
    for _ in range(iters):
        vals, vecs = torch.linalg.eigh(R - torch.diag_embed(psi))
        vals, vecs = vals[:, -K:], vecs[:, :, -K:]
        D = vecs * vals.clamp_min(0).sqrt()[:, None, :]
        psi = (diagR - (D * D).sum(-1)).clamp_min(floor)
    return dict(mx=mu[:, :P], my=mu[:, P], vy=vy, A=A, D=D, psi=psi)
