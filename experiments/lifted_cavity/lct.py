"""Lifted cavity transformer (LCT): the cell-token transformer of pfn.py used as the site network of a lifted PoE.

The transformer reads the support and query cells exactly as the baseline does. For every query row and every OBSERVED
sensor, its cell token is mapped (zero-initialized linear head) to a correction of that sensor's lifted site:
  offset, slope and nuisance loading of a local linearization of the sensor's response, plus a noise scale.
Sites start from the support-only FA anchor; the linearization point is the anchor's closed-form posterior mean.
The prediction is the y-marginal of prior x product of observed sites, so:
  * missing sensors contribute no site, for any mask and any width;
  * with the head at zero the model equals FA-Gaussian (as for the lifted cavity network).
"""
import torch
from torch import nn
import pfn
import models


class LCT(nn.Module):
    def __init__(self, d=64, layers=4, heads=4, ff=128, femb=16, K=1):
        super().__init__()
        self.K = K
        self.backbone = pfn.CellPFN(d=d, layers=layers, heads=heads, ff=ff, femb=femb)
        self.head = nn.Linear(d, 3 + K)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.feature_seed = 0

    def tokens(self, sx, sy, sm, qx, qm, gen=None):
        """Replicates CellPFN.core up to the final blocks and returns query-cell tokens (B, Q, P, d)."""
        bb = self.backbone
        B, S, P = sx.shape
        Q = qx.shape[1]
        ym = sy.mean(1, keepdim=True); ys = sy.std(1, keepdim=True).clamp_min(.1)
        syn = (sy - ym) / ys
        X = torch.cat((sx * sm, qx * qm), 1); M = torch.cat((sm, qm), 1)
        tok = M[..., None] * bb.val(X[..., None]) + (1 - M[..., None]) * bb.missing
        fe = (torch.randn(B, 1, P, bb.femb_dim, generator=gen).to(sx.device) if gen is not None
              else torch.randn(B, 1, P, bb.femb_dim, device=sx.device))
        tok = tok + bb.femb(fe)
        tcol = torch.cat((bb.val(syn[..., None]), bb.query_target.expand(B, Q, -1)), 1) + bb.target_col
        z = torch.cat((tok, tcol[:, :, None]), 2)
        for blk in bb.blocks:
            z = blk(z, S)
        return z[:, S:, :P]

    def forward(self, batch):
        anc, tid = batch['anchors'][self.K], batch['tid']
        B = batch['sx'].shape[0]
        nq = tid.numel() // B
        qx, qm = batch['qx'], batch['qm']
        gen = None if self.training else torch.Generator().manual_seed(self.feature_seed)          # CPU generator, moved to device
        h = self.tokens(batch['sx'], batch['sy'], batch['sm'], qx.reshape(B, nq, -1), qm.reshape(B, nq, -1), gen)
        out = self.head(h).reshape(B * nq, qx.shape[-1], -1)                 # (N, P, 3+K)
        K = self.K
        A, D, psi0 = anc['A'][tid], anc['D'][tid], anc['psi'][tid]
        mx, my, vy = anc['mx'][tid], anc['my'][tid], anc['vy'][tid]
        xt = qx - mx
        mu0, _ = models.closed_form(anc, qx, qm, tid)                         # anchor posterior mean: linearization point
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
