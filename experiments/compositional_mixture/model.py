"""Trainable, restriction-consistent mixture of shared-noise Gaussian models.

The universal property used here is the free commutative monoid extension of
sensor messages to additive sufficient statistics. That algebra and mixtures
of factor analyzers are established, not novelty claims. No fitted model yet.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


def finite(*values):
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Nonfinite observed values or parameters")


@dataclass(frozen=True)
class Evidence:
    precision: torch.Tensor
    natural: torch.Tensor
    constant: torch.Tensor
    sensors: frozenset
    context: object

    def merge(self, other):
        if self.context is not other.context or self.sensors & other.sensors:
            raise ValueError("Merge requires one query context and disjoint sensor IDs")
        return Evidence(self.precision + other.precision, self.natural + other.natural,
                        self.constant + other.constant, self.sensors | other.sensors, self.context)

    def remove(self, other):
        if self.context is not other.context or not other.sensors <= self.sensors:
            raise ValueError("Remove only a present subset from the same query context")
        return Evidence(self.precision - other.precision, self.natural - other.natural,
                        self.constant - other.constant, self.sensors - other.sensors, self.context)


@dataclass(frozen=True)
class Prediction:
    log_weights: torch.Tensor
    means: torch.Tensor
    variances: torch.Tensor
    log_evidence: torch.Tensor

    def moments(self):
        weights = self.log_weights.exp()
        mean = (weights * self.means).sum(-1)
        variance = (weights * (self.variances + (self.means - mean[:, None]).square())).sum(-1)
        return mean, variance.log()

    def nll(self, targets):
        # Evaluate the actual mixture, not its moment-matched Gaussian.
        log_density = -.5 * (math.log(2 * math.pi) + self.variances.log()
                              + (targets[:, None] - self.means).square() / self.variances)
        return -torch.logsumexp(self.log_weights + log_density, -1)


class Query:
    """One immutable-by-contract model/support/query snapshot; caches cannot cross fits.

    Evidence objects must come from this object's summarize method, unmodified.
    IDs track tensor columns, not physical sensor identity or duplicate readings.
    """
    def __init__(self, loadings, locations, noise, my, vy, logits, qx, qm):
        n, m, p, _ = loadings.shape
        if locations.shape != (n, m, p) or noise.shape != (n, m, p):
            raise ValueError("Component parameter shapes do not align")
        if qx.shape != (n, p) or qm.shape != (n, p) or my.shape != (n,) or vy.shape != (n,):
            raise ValueError("Query and prior shapes do not align")
        if logits.shape != (m,) or not bool(((qm == 0) | (qm == 1)).all()):
            raise ValueError("Mixture logits and binary mask required")
        if not bool((noise > 0).all() and (vy > 0).all()):
            raise ValueError("Positive variances required")
        qx = torch.where(qm.bool(), qx, torch.zeros_like(qx))
        finite(loadings, locations, noise, my, vy, logits, qx)
        self.loadings, self.locations, self.noise = loadings, locations, noise
        self.my, self.vy, self.logits, self.qx, self.qm = my, vy, logits, qx, qm

    def summarize(self, indices=None):
        p = self.qx.shape[1]
        indices = list(range(p)) if indices is None else list(indices)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= p for i in indices):
            raise ValueError("Unique in-range sensor indices required")
        w, noise = self.loadings[:, :, indices], self.noise[:, :, indices]
        observed = self.qm[:, None, indices].bool()
        residual = torch.where(observed, self.qx[:, None, indices] - self.locations[:, :, indices], 0.)
        inv_noise = observed.to(noise.dtype) / noise
        precision = torch.einsum('nmpi,nmp,nmpj->nmij', w, inv_noise, w)
        natural = torch.einsum('nmpi,nmp->nmi', w, residual * inv_noise)
        constant = (residual.square() * inv_noise
                    + observed * (noise.log() + math.log(2 * math.pi))).sum(-1)
        return Evidence(precision, natural, constant, frozenset(indices), self)

    def decode(self, evidence):
        if evidence.context is not self:
            raise ValueError("Evidence belongs to another query context")
        l = self.loadings.shape[-1]
        precision = evidence.precision + torch.eye(l, dtype=self.qx.dtype, device=self.qx.device)
        chol = torch.linalg.cholesky(precision)  # No jitter, retries or concealed failures.
        unit = torch.zeros_like(evidence.natural)
        unit[..., 0] = 1
        solved = torch.cholesky_solve(torch.stack((evidence.natural, unit), -1), chol)
        means = self.my[:, None] + self.vy.sqrt()[:, None] * solved[..., 0, 0]
        variances = self.vy[:, None] * solved[..., 0, 1]
        logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        log_evidence = -.5 * (evidence.constant + logdet
                              - (evidence.natural * solved[..., 0]).sum(-1))
        log_weights = torch.log_softmax(self.logits[None] + log_evidence, -1)
        finite(means, variances, log_weights, log_evidence)
        if not bool((variances > 0).all()):
            raise ValueError("Nonpositive posterior variance")
        return Prediction(log_weights, means, variances, log_evidence)


class CompositionalMixture(nn.Module):
    """Small support-only encoder; no sensor IDs, query labels or query-set pooling.

    For fixed support and fixed anchors, compiled component parameters restrict
    exactly to any sensor subset. Re-fitting anchors is outside that guarantee.
    """
    def __init__(self, components=3, extra_rank=2, width=16, hidden=24):
        super().__init__()
        if min(components, width, hidden) < 1 or extra_rank < 0:
            raise ValueError("Invalid model dimensions")
        self.components, self.extra_rank = components, extra_rank
        self.rows = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(width + 5, hidden), nn.Tanh(),
                                  nn.Linear(hidden, components * (4 + extra_rank)))
        # Small distinct components avoid dead gradients from zero new factors
        # and identical mixture components. Initialization is NOT exactly FA.
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.normal_(self.head[-1].bias, std=.01)
        self.logits = nn.Parameter(torch.zeros(components))

    def compile(self, sx, sy, sm, anchors):
        a, d, psi = (anchors[key] for key in ('A', 'D', 'psi'))
        mx, my, vy = (anchors[key] for key in ('mx', 'my', 'vy'))
        if sx.ndim != 3 or sx.shape != sm.shape or sy.shape != sx.shape[:2] or sx.shape[1] < 1:
            raise ValueError("Support shapes do not align")
        b, rows, p = sx.shape
        if any(t.shape != (b, p) for t in (a, psi, mx)) or d.ndim != 3 or d.shape[:2] != (b, p):
            raise ValueError("Sensor anchor shapes do not align")
        if my.shape != (b,) or vy.shape != (b,) or not bool(((sm == 0) | (sm == 1)).all()):
            raise ValueError("Prior shapes and binary support mask required")
        if not bool((psi > 0).all() and (vy > 0).all()):
            raise ValueError("Positive anchor variances required")
        centered = torch.where(sm.bool(), sx - mx[:, None], torch.zeros_like(sx))
        finite(centered, sy, a, d, psi, mx, my, vy)
        dy = sy - my[:, None]
        factor_var = d.square().sum(-1)
        scale = (a.square() * vy[:, None] + factor_var + psi).sqrt()
        features = torch.stack((dy[:, :, None].expand_as(sx) / vy.sqrt()[:, None, None],
                                centered / scale[:, None],
                                (centered - a[:, None] * dy[:, :, None]) / psi.sqrt()[:, None]), -1)
        features = torch.where(sm.bool()[..., None], features, torch.zeros_like(features))
        count = sm.sum(1)
        pooled = (self.rows(features) * sm[..., None]).sum(1) / count.clamp_min(1)[..., None]
        anchor_features = torch.stack((count / rows, a * vy.sqrt()[:, None] / scale,
                                       factor_var / (factor_var + psi), psi.log(),
                                       vy.log()[:, None].expand_as(psi)), -1)
        h = self.head(torch.cat((pooled, anchor_features), -1))
        h = h.reshape(b, p, self.components, 4 + self.extra_rank).permute(0, 2, 1, 3)
        offset, slope, log_noise, log_factor = h[..., :4].unbind(-1)
        corrected_a = a[:, None] + slope * psi.sqrt()[:, None] / vy.sqrt()[:, None, None]
        corrected_d = d[:, None] * (2 * torch.tanh(log_factor / 2)).exp()[..., None]
        extra = h[..., 4:] * psi.sqrt()[:, None, :, None]
        w = torch.cat(((corrected_a * vy.sqrt()[:, None, None])[..., None], corrected_d, extra), -1)
        locations = mx[:, None] + offset * psi.sqrt()[:, None]
        noise = psi[:, None] * (2 * torch.tanh(log_noise / 2)).exp()
        return w, locations, noise

    def query(self, batch):
        anchors, tid = batch['anchor'], batch['tid']
        w, locations, noise = self.compile(batch['sx'], batch['sy'], batch['sm'], anchors)
        return Query(w[tid], locations[tid], noise[tid], anchors['my'][tid], anchors['vy'][tid],
                     self.logits, batch['qx'], batch['qm'])

    def distribution(self, batch):
        query = self.query(batch)
        return query.decode(query.summarize())

    def forward(self, batch):
        return self.distribution(batch).moments()
