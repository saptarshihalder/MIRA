"""Untrained successor prototype: gauge-free covariance transport readout.

Not part of frozen v4. Supplied tokens must come from a mask-respecting encoder.
Only the covariance Gram D D^T is used; no factor coordinates enter learning.
The analytic anchor is recovered at zero corrections. No utility guarantee.
"""
import math

import torch
from torch import nn


class CovarianceTransportHead(nn.Module):
    def __init__(self, feature_dim, max_log_scale=2.0):
        super().__init__()
        if feature_dim < 1 or max_log_scale <= 0:
            raise ValueError("Positive feature dimension and scale bound required")
        self.feature_dim = feature_dim
        self.max_log_scale = max_log_scale
        self.pair = nn.Parameter(torch.zeros(feature_dim, feature_dim))
        self.local = nn.Linear(feature_dim, 3)  # response offset, slope, covariance diagonal
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def forward(self, features, A, D, psi, mx, my, vy, qx, qm, return_covariance=False):
        """N queries, P sensors; factors may have any K, tokens have fixed d.

        Covariance transport is sensor-permutation equivariant conditional on
        equivariant tokens/anchors. Anchor fitting and an upstream encoder are
        outside this prototype's guarantees. Missing masks need not be random.
        This defines a conditional predictor, not an informative-mask model.
        """
        n, p, d = features.shape
        if d != self.feature_dim or p < 1 or any(x.shape != (n, p) for x in (A, psi, mx, qx, qm)):
            raise ValueError("Query/token/anchor shapes do not align")
        if D.ndim != 3 or D.shape[:2] != (n, p) or my.shape != (n,) or vy.shape != (n,):
            raise ValueError("Factor/prior shapes do not align")
        if not bool(((qm == 0) | (qm == 1)).all()) or not bool((psi > 0).all() and (vy > 0).all()):
            raise ValueError("Binary masks and positive anchor variances required")
        observed = qm.bool()
        tokens = torch.where(observed[..., None], features, torch.zeros_like(features))
        centered = torch.where(observed, qx - mx, torch.zeros_like(qx))
        if not all(bool(torch.isfinite(x).all()) for x in (tokens, centered, A, D, psi, mx, my, vy)):
            raise ValueError("Nonfinite observed inputs or anchors")
        offset, slope, diagonal = self.local(tokens).unbind(-1)
        pair = (self.pair + self.pair.T) / 2
        h = tokens @ pair @ tokens.transpose(-1, -2) / math.sqrt(d)
        h = h + torch.diag_embed(diagonal * qm)
        # Smooth Frobenius bound also bounds every eigenvalue; derivative at zero is one.
        norm2 = h.square().sum((-2, -1), keepdim=True)
        h = h * self.max_log_scale / torch.sqrt(self.max_log_scale ** 2 + norm2)
        anchor = D @ D.transpose(-1, -2) + torch.diag_embed(psi)
        eig, vectors = torch.linalg.eigh(anchor)
        if not bool((eig > 0).all()):
            raise ValueError("Anchor covariance lost positive definiteness")
        # The unique symmetric square root preserves sensor permutations, unlike Cholesky coordinates.
        root = (vectors * eig.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        covariance = root @ torch.matrix_exp(h) @ root
        covariance = (covariance + covariance.transpose(-1, -2)) / 2
        masked_covariance = covariance * (qm[..., :, None] * qm[..., None, :]) + torch.diag_embed(1 - qm)
        response = (A + slope) * qm
        residual = centered - offset * qm
        solved = torch.linalg.solve(masked_covariance, torch.stack((response, residual), dim=-1))
        variance = 1 / (1 / vy + (response * solved[..., 0]).sum(-1))
        mean = my + variance * (response * solved[..., 1]).sum(-1)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all() and (variance > 0).all()):
            raise ValueError("Nonfinite or nonpositive prediction")
        prediction = (mean, variance.log())
        return (*prediction, covariance) if return_covariance else prediction
