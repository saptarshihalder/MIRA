"""Lifted cavity network (learned EP over a target-plus-nuisance latent) and matched controls.

Notation: z = (y, h) with h in R^K a nuisance latent shared across sensors.  Every observed sensor j contributes
one Gaussian site over z in natural parameters (Lambda_j = w_j w_j^T / psi_j,  eta_j = w_j o_j / psi_j), i.e. a
rank-1 linearised likelihood x_j ~ N(w_j^T z + const, psi_j).  Missing sensors contribute no site, so any
missingness pattern is handled by omission.  Round 0 uses the closed-form support-FA sites (exact for a
linear-Gaussian K-factor world).  Each refinement round removes site j to form the cavity q_{-j}(z), and a shared
network maps (sensor value, the sensor's in-context support encoding, its anchor loadings, cavity statistics) to a
REPLACEMENT site: a local re-linearisation (offset, slope), a nuisance-loading correction and a noise scale.  The
output layer starts at zero, so the untrained model is exactly the closed-form FA-Gaussian conditional.
Sensors are exchangeable: no sensor identity enters, so the same weights apply to any number of sensors.
"""
import torch
from torch import nn


def gauss_nll(mu, logvar, y):
    return .5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar) + 1.8378770664093453)


class SupportEncoder(nn.Module):
    """DeepSets over one sensor's support scatter {(y_i, x_ij, r_ij)}: observed rows only."""

    def __init__(self, width=32, out=16):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.rho = nn.Sequential(nn.Linear(width + 1, out), nn.Tanh())

    def forward(self, sx, sy, sm, anc):
        # sx,sm (B,n,P); sy (B,n); anchor tensors (B,P)/(B,)
        ys = ((sy - anc['my'][:, None]) / anc['vy'].sqrt()[:, None])[..., None].expand_as(sx)
        r = (sx - anc['mx'][:, None] - anc['A'][:, None] * (sy - anc['my'][:, None])[..., None])
        feats = torch.stack((ys, sx - anc['mx'][:, None], r / anc['psi'].sqrt()[:, None]), -1)  # (B,n,P,3)
        h = self.phi(feats) * sm[..., None]
        cnt = sm.sum(1)                                                                             # (B,P)
        pooled = h.sum(1) / cnt.clamp_min(1)[..., None]
        return self.rho(torch.cat((pooled, (cnt / sm.shape[1])[..., None]), -1))                   # (B,P,out)


class LiftedCavity(nn.Module):
    def __init__(self, K=1, rounds=3, cavity=True, width=64, enc=16):
        super().__init__()
        self.K, self.rounds, self.cavity = K, rounds, cavity
        self.encoder = SupportEncoder(out=enc)
        nin = 1 + enc + 1 + K + 1 + (5 + 2 * K if cavity else 0) + 1
        self.site = nn.Sequential(nn.Linear(nin, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                  nn.Linear(width, 3 + K))
        nn.init.zeros_(self.site[-1].weight); nn.init.zeros_(self.site[-1].bias)

    def posterior(self, W, o, psi, m, prior_prec, prior_eta):
        # W (N,P,1+K), o (N,P), psi (N,P), m (N,P)
        lam = (W[..., :, None] * W[..., None, :]) * (m / psi)[..., None, None]     # (N,P,L,L)
        eta = W * (m * o / psi)[..., None]                                         # (N,P,L)
        prec = prior_prec + lam.sum(1)
        nat = prior_eta + eta.sum(1)
        return lam, eta, prec, nat

    def forward(self, batch):
        anc, qx, qm, tid = batch['anchors'][self.K], batch['qx'], batch['qm'], batch['tid']
        K = self.K
        e = self.encoder(batch['sx'], batch['sy'], batch['sm'], anc)[tid]                    # (N,P,enc)
        A, D, psi0 = anc['A'][tid], anc['D'][tid], anc['psi'][tid]
        mx, my, vy = anc['mx'][tid], anc['my'][tid], anc['vy'][tid]
        N, P = qx.shape
        xt = qx - mx                                                                          # centred sensor value
        prior_prec = torch.diag_embed(torch.cat((1 / vy[:, None], torch.ones(N, K, dtype=vy.dtype, device=vy.device)), -1))
        prior_eta = torch.cat(((my / vy)[:, None], torch.zeros(N, K, dtype=vy.dtype, device=vy.device)), -1)
        # round 0: closed-form FA sites; observation o_j = x~_j + A_j my  (so that w^T z reproduces A (y-my) + D h)
        W = torch.cat((A[..., None], D), -1)
        o = xt + A * my[:, None]
        psi = psi0
        static = torch.cat((xt[..., None], e, A[..., None], D, psi0.log()[..., None], qm.mean(1, keepdim=True)[:, :, None].expand(N, P, 1)), -1)
        for _ in range(self.rounds):
            lam, eta, prec, nat = self.posterior(W, o, psi, qm, prior_prec, prior_eta)
            if self.cavity:
                cprec = prec[:, None] - lam                                                   # (N,P,L,L)
                cnat = nat[:, None] - eta
                ccov = torch.linalg.inv(cprec)
                cmean = (ccov @ cnat[..., None])[..., 0]                                       # (N,P,L)
                ym = cmean[..., 0]
                yvar = ccov[..., 0, 0]
                hm = cmean[..., 1:]
                pred = A * (ym - my[:, None]) + (D * hm).sum(-1)
                pv = (W[..., None, :] @ ccov @ W[..., :, None])[..., 0, 0] + psi
                res = xt - pred
                cav = torch.cat((((ym - my[:, None]) / vy.sqrt()[:, None])[..., None], (yvar / vy[:, None]).log()[..., None],
                                 hm, ccov[..., 0, 1:], res[..., None], (res / pv.sqrt())[..., None], pv.log()[..., None]), -1)
                inp = torch.cat((static, cav), -1)
                yc = ym
            else:
                inp = static
                yc = my[:, None].expand(N, P)
            out = self.site(inp)
            d_off, d_slope, d_D, s = out[..., 0], out[..., 1], out[..., 2:2 + K], out[..., 2 + K]
            slope = A + d_slope
            Dn = D + d_D
            # local linearisation at the cavity mean: x~_j ~ g_j(yc) + slope (y - yc) + Dn h + e
            g = A * (yc - my[:, None]) + d_off
            W = torch.cat((slope[..., None], Dn), -1)
            o = xt - g + slope * yc
            psi = psi0 * torch.exp(4 * torch.tanh(s / 4))
        _, _, prec, nat = self.posterior(W, o, psi, qm, prior_prec, prior_eta)
        cov = torch.linalg.inv(prec)
        mean = (cov @ nat[..., None])[..., 0]
        return mean[:, 0], cov[:, 0, 0].log()


def closed_form(anc, qx, qm, tid):
    """FA-Gaussian conditional = lifted PoE with the round-0 (closed-form) sites. Returns mean, log-variance."""
    A, D, psi = anc['A'][tid], anc['D'][tid], anc['psi'][tid]
    mx, my, vy = anc['mx'][tid], anc['my'][tid], anc['vy'][tid]
    N, P = qx.shape; K = D.shape[-1]
    W = torch.cat((A[..., None], D), -1)
    o = qx - mx + A * my[:, None]
    lam = (W[..., :, None] * W[..., None, :]) * (qm / psi)[..., None, None]
    prec = torch.diag_embed(torch.cat((1 / vy[:, None], torch.ones(N, K, dtype=qx.dtype, device=qx.device)), -1)) + lam.sum(1)
    nat = torch.cat(((my / vy)[:, None], torch.zeros(N, K, dtype=qx.dtype, device=qx.device)), -1) + (W * (qm * o / psi)[..., None]).sum(1)
    cov = torch.linalg.inv(prec)
    return (cov @ nat[..., None])[:, 0, 0], cov[:, 0, 0].log()


class AnchorMLP(nn.Module):
    """Non-modular control: residual MLP on the closed-form FA-Gaussian prediction, with the same support encodings."""

    def __init__(self, K=1, width=64, enc=16, ctx=55):
        super().__init__()
        self.K = K
        self.encoder = SupportEncoder(out=enc)
        self.net = nn.Sequential(nn.Linear(5 + 5 + ctx + 2 + 5 * enc, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, batch):
        anc, tid = batch['anchors'][self.K], batch['tid']
        mu0, lv0 = closed_form(anc, batch['qx'], batch['qm'], tid)
        e = self.encoder(batch['sx'], batch['sy'], batch['sm'], anc)[tid] * batch['qm'][..., None]
        vy = anc['vy'][tid]; my = anc['my'][tid]
        inp = torch.cat((batch['qx'] * batch['qm'], batch['qm'], batch['c'][tid], ((mu0 - my) / vy.sqrt())[:, None],
                         (lv0 - vy.log())[:, None], e.flatten(1)), -1)
        h = self.net(inp)
        return mu0 + vy.sqrt() * h[:, 0], lv0 + h[:, 1]


class RepoWrapper(nn.Module):
    """Wrap the repository's anchored_cavity.Model (cavity / aggregate / static / mlp) with this batch interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        tid = batch['tid']
        return self.model(batch['c'][tid], batch['qx'] * batch['qm'], batch['qm'])
