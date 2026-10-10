"""Universal pool transformer (UPT): one cell-token backbone, lifted sensor sites, a free target site and a gate.

The backbone and the lifted-site head are those of the LCT (lct.py). The target-column token of each query row also
feeds (i) a free Gaussian head (the transformer baseline's own output layer, pfn.CellPFN.out) and (ii) a gate. The
predictive is the linear pool, the pooling rule that commutes with marginalization (McConway 1981), i.e. with removing
sensors:
    p(y) = pi * N(y; lifted posterior)  +  (1 - pi) * N(y; free head),      pi = sigmoid(gate).
At initialization the site head is zero (the lifted posterior is the FA-Gaussian closed form) and the gate bias is
+4, so NLL_init <= NLL_FA - log(sigmoid(4)) = NLL_FA + 0.018 for every query and outcome.
forward() returns the moment-matched Gaussian (for squared error and the shared interfaces); nll() and predictive()
give the mixture, which is what training and scoring use.
"""
import math
import torch
from torch import nn
import lct
import models

LOG2PI = math.log(2 * math.pi)
GATE_BIAS = 4.


class UPT(lct.LCT):
    def __init__(self, d=64, layers=4, heads=4, ff=128, femb=16, K=1):
        super().__init__(d=d, layers=layers, heads=heads, ff=ff, femb=femb, K=K)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, GATE_BIAS)

    def _query_tokens(self, batch):
        """Query-row tokens of all P sensor columns and the target column, plus the support target scale."""
        bb = self.backbone
        sx, sy, sm = batch['sx'], batch['sy'], batch['sm']
        B, S, P = sx.shape
        nq = batch['tid'].numel() // B
        qx, qm = batch['qx'].reshape(B, nq, -1), batch['qm'].reshape(B, nq, -1)
        gen = None if self.training else torch.Generator().manual_seed(self.feature_seed)
        ym = sy.mean(1, keepdim=True); ys = sy.std(1, keepdim=True).clamp_min(.1)
        syn = (sy - ym) / ys
        X = torch.cat((sx * sm, qx * qm), 1); M = torch.cat((sm, qm), 1)
        tok = M[..., None] * bb.val(X[..., None]) + (1 - M[..., None]) * bb.missing
        fe = (torch.randn(B, 1, P, bb.femb_dim, generator=gen).to(sx.device) if gen is not None
              else torch.randn(B, 1, P, bb.femb_dim, device=sx.device))
        tok = tok + bb.femb(fe)
        tcol = torch.cat((bb.val(syn[..., None]), bb.query_target.expand(B, nq, -1)), 1) + bb.target_col
        z = torch.cat((tok, tcol[:, :, None]), 2)
        for blk in bb.blocks:
            z = blk(z, S)
        return z[:, S:], ym, ys

    def predictive(self, batch):
        """Mixture components per query: log pi, (mu_site, lv_site), log(1 - pi), (mu_free, lv_free)."""
        z, ym, ys = self._query_tokens(batch)
        B, nq, C, d = z.shape
        P = C - 1
        mu_s, lv_s = self._sites(batch, z[:, :, :P].reshape(B * nq, P, d))
        t = z[:, :, P]
        o = self.backbone.out(t)
        mu_f = (ym + ys * o[..., 0]).reshape(-1)
        lv_f = (5 * torch.tanh(o[..., 1] / 5) + 2 * ys.log()).reshape(-1)
        g = self.gate(t).reshape(-1)
        return nn.functional.logsigmoid(g), mu_s, lv_s, nn.functional.logsigmoid(-g), mu_f, lv_f

    def _sites(self, batch, h):
        """Lifted posterior from site corrections read off the query cell tokens h (N, P, d), exactly as lct.LCT."""
        anc, tid = batch['anchors'][self.K], batch['tid']
        qx, qm = batch['qx'], batch['qm']
        K = self.K
        out = self.head(h)
        A, D, psi0 = anc['A'][tid], anc['D'][tid], anc['psi'][tid]
        mx, my, vy = anc['mx'][tid], anc['my'][tid], anc['vy'][tid]
        xt = qx - mx
        mu0, _ = models.closed_form(anc, qx, qm, tid)
        yc = mu0.detach()[:, None]
        d_off, d_slope, d_D, s = out[..., 0], out[..., 1], out[..., 2:2 + K], out[..., 2 + K]
        slope = A + d_slope
        W = torch.cat((slope[..., None], D + d_D), -1)
        o = xt - (A * (yc - my[:, None]) + d_off) + slope * yc
        psi = psi0 * torch.exp(4 * torch.tanh(s / 4))
        N, P = qx.shape
        lam = (W[..., :, None] * W[..., None, :]) * (qm / psi)[..., None, None]
        prec = torch.diag_embed(torch.cat((1 / vy[:, None], torch.ones(N, K, dtype=vy.dtype, device=vy.device)), -1)) + lam.sum(1)
        nat = torch.cat(((my / vy)[:, None], torch.zeros(N, K, dtype=vy.dtype, device=vy.device)), -1) + (W * (qm * o / psi)[..., None]).sum(1)
        cov = torch.linalg.inv(prec)
        return (cov @ nat[..., None])[:, 0, 0], cov[:, 0, 0].log()

    def nll(self, batch, y=None):
        y = batch['qy'] if y is None else y
        la, ma, va, lb, mb, vb = self.predictive(batch)
        ca = la - .5 * (LOG2PI + va + (y - ma) ** 2 * torch.exp(-va))
        cb = lb - .5 * (LOG2PI + vb + (y - mb) ** 2 * torch.exp(-vb))
        return -torch.logaddexp(ca, cb)

    def forward(self, batch):
        la, ma, va, lb, mb, vb = self.predictive(batch)
        pa, pb = la.exp(), lb.exp()
        mu = pa * ma + pb * mb
        var = pa * (va.exp() + ma ** 2) + pb * (vb.exp() + mb ** 2) - mu ** 2
        return mu, var.clamp_min(1e-12).log()


def mixture_metrics(w, mu, var, y):
    """Per-query metrics of a Gaussian mixture (numpy; w, mu, var of shape (N, M)): NLL, squared error of the mean,
    90% central-interval coverage and the exact CRPS (Grimit et al. 2006)."""
    import numpy as np
    from scipy.stats import norm
    yy = y[:, None]
    nll = -np.log((w * norm.pdf(yy, mu, np.sqrt(var))).sum(1))
    m = (w * mu).sum(1)
    cdf = (w * norm.cdf(yy, mu, np.sqrt(var))).sum(1)
    A = lambda a, s2: 2 * np.sqrt(s2) * norm.pdf(a / np.sqrt(s2)) + a * (2 * norm.cdf(a / np.sqrt(s2)) - 1)
    crps = (w * A(yy - mu, var)).sum(1) - .5 * (w[:, :, None] * w[:, None, :] *
                                                A(mu[:, :, None] - mu[:, None, :], var[:, :, None] + var[:, None, :])).sum((1, 2))
    return dict(nll=nll, se=(y - m) ** 2, cov=((cdf >= .05) & (cdf <= .95)).astype(float), crps=crps)
