"""Neural-process baselines (protocol v8), trained on the same prior and recipe as the transformer (pfn_train.py).

CNP   conditional neural process (Garnelo et al. 2018): MLP encoder of each labeled support row, mean-pooled, and an
      MLP decoder of (representation, query row) -> Gaussian.
TNP   transformer neural process, TNP-D (Nguyen & Grover 2022): one token per row; support tokens carry y, query tokens a
      target flag; queries attend only to support rows and support rows only to each other; Gaussian head.

A row is [x * m, m] padded to P_MAX sensors (missing and absent sensors are 0 with mask 0), so width is fixed at
P_MAX = 16 (the widest panel); y is standardized by the support mean and spread exactly as in pfn.CellPFN.
Interface identical to pfn.CellPFN: core(sx, sy, sm, qx, qm) -> mean, log-variance (B, Q); forward(batch) flattens.
"""
import torch
from torch import nn
import torch.nn.functional as F

P_MAX = 16


class _NP(nn.Module):
    def forward(self, batch):
        tid = batch['tid']; B = batch['sx'].shape[0]; nq = tid.numel() // B
        mu, lv = self.core(batch['sx'], batch['sy'], batch['sm'], batch['qx'].reshape(B, nq, -1), batch['qm'].reshape(B, nq, -1))
        return mu.reshape(-1), lv.reshape(-1)

    @staticmethod
    def rows(x, m):
        P = x.shape[-1]
        if P > P_MAX:
            raise ValueError(f'{P} sensors > P_MAX={P_MAX}')
        return torch.cat((F.pad(x * m, (0, P_MAX - P)), F.pad(m, (0, P_MAX - P))), -1)

    @staticmethod
    def scale(sy):
        ym = sy.mean(1, keepdim=True); ys = sy.std(1, keepdim=True).clamp_min(.1)
        return ym, ys, (sy - ym) / ys

    @staticmethod
    def head(o, ym, ys):
        return ym + ys * o[..., 0], 5 * torch.tanh(o[..., 1] / 5) + 2 * ys.log()


def _mlp(i, h, o, n):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class CNP(_NP):
    def __init__(self, h=192):
        super().__init__()
        self.enc = _mlp(2 * P_MAX + 1, h, h, 3)
        self.dec = _mlp(h + 2 * P_MAX, h, 2, 3)

    def core(self, sx, sy, sm, qx, qm):
        ym, ys, syn = self.scale(sy)
        r = self.enc(torch.cat((self.rows(sx, sm), syn[..., None]), -1)).mean(1)          # (B, h)
        q = self.rows(qx, qm)
        return self.head(self.dec(torch.cat((r[:, None].expand(-1, q.shape[1], -1), q), -1)), ym, ys)


class TNP(_NP):
    def __init__(self, d=64, layers=4, heads=4, ff=128):
        super().__init__()
        self.emb = _mlp(2 * P_MAX + 2, d, d, 2)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ff, dropout=0., batch_first=True, norm_first=True),
                                         layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), _mlp(d, d, 2, 2))

    def core(self, sx, sy, sm, qx, qm):
        ym, ys, syn = self.scale(sy)
        B, S, _ = sx.shape; Q = qx.shape[1]
        ctx = torch.cat((self.rows(sx, sm), syn[..., None], torch.zeros_like(syn[..., None])), -1)
        tgt = torch.cat((self.rows(qx, qm), torch.zeros(B, Q, 1, device=qx.device), torch.ones(B, Q, 1, device=qx.device)), -1)
        z = self.emb(torch.cat((ctx, tgt), 1))
        mask = torch.ones(S + Q, S + Q, dtype=torch.bool, device=sx.device)
        mask[:, :S] = False                                                                  # True = blocked: attend to support only
        h = self.enc(z, mask=mask)
        return self.head(self.out(h[:, S:]), ym, ys)
