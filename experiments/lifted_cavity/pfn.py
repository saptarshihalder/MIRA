"""TabPFN-v2-style cell-token transformer baseline, trained on the same task prior as the lifted cavity network.

Every (row, column) cell is a token. Columns are the P sensors plus the target. Each block alternates
  (1) attention across the columns of one row and
  (2) attention across rows within one column, where every row attends only to the labeled support rows,
followed by an MLP (pre-LayerNorm residual). Missing cells get a learned token. Sensors carry no identity: each
task draws random feature embeddings (as in TabPFN v2), so the model is width-agnostic and permutation-equivariant
in distribution. Queries never attend to each other, so predictions for different queries are independent.
"""
import math
import numpy as np
import torch
from torch import nn
import family as fam

LOG2PI = math.log(2 * math.pi)


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.fatt = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ratt = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, z, ns):
        B, R, C, d = z.shape
        x = z.reshape(B * R, C, d)
        h = self.ln1(x)
        x = x + self.fatt(h, h, h, need_weights=False)[0]
        x = x.reshape(B, R, C, d).transpose(1, 2).reshape(B * C, R, d)
        h = self.ln2(x)
        kv = h[:, :ns]
        x = x + self.ratt(h, kv, kv, need_weights=False)[0]
        z = x.reshape(B, C, R, d).transpose(1, 2)
        return z + self.mlp(self.ln3(z))


class CellPFN(nn.Module):
    def __init__(self, d=64, layers=4, heads=4, ff=128, femb=16):
        super().__init__()
        self.femb_dim = femb
        self.val = nn.Linear(1, d)
        self.missing = nn.Parameter(torch.randn(d) * .02)
        self.target_col = nn.Parameter(torch.randn(d) * .02)
        self.query_target = nn.Parameter(torch.randn(d) * .02)
        self.femb = nn.Linear(femb, d, bias=False)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.feature_seed = 0          # evaluation uses a fixed seed for the random feature embeddings

    def core(self, sx, sy, sm, qx, qm, gen=None):
        """sx,sm (B,S,P) standardized/masked; sy (B,S); qx,qm (B,Q,P). Returns mean, log-variance (B,Q)."""
        B, S, P = sx.shape
        Q = qx.shape[1]
        ym = sy.mean(1, keepdim=True)
        ys = sy.std(1, keepdim=True).clamp_min(.1)
        syn = (sy - ym) / ys
        X = torch.cat((sx * sm, qx * qm), 1)                      # (B,R,P)
        M = torch.cat((sm, qm), 1)
        tok = M[..., None] * self.val(X[..., None]) + (1 - M[..., None]) * self.missing
        fe = torch.randn(B, 1, P, self.femb_dim, generator=gen) if gen is not None else torch.randn(B, 1, P, self.femb_dim)
        tok = tok + self.femb(fe)
        tcol = torch.cat((self.val(syn[..., None]), self.query_target.expand(B, Q, -1)), 1) + self.target_col
        z = torch.cat((tok, tcol[:, :, None]), 2)                   # (B,R,P+1,d)
        for blk in self.blocks:
            z = blk(z, S)
        o = self.out(z[:, S:, P])
        mu = ym + ys * o[..., 0]
        lv = 5 * torch.tanh(o[..., 1] / 5) + 2 * ys.log()
        return mu, lv

    def forward(self, batch):
        """Adapter for the shared evaluation batches (queries ordered task-major with equal counts per task)."""
        tid = batch['tid']
        B = batch['sx'].shape[0]
        nq = tid.numel() // B
        qx = batch['qx'].reshape(B, nq, -1)
        qm = batch['qm'].reshape(B, nq, -1)
        gen = None
        if not self.training:
            gen = torch.Generator().manual_seed(self.feature_seed)
        mu, lv = self.core(batch['sx'], batch['sy'], batch['sm'], qx, qm, gen)
        return mu.reshape(-1), lv.reshape(-1)


def fresh_batch(rng, B=32, Q=16, nonlin=.4, P=5, all_masks=False):
    """Fresh tasks from the shared generator; source query masks (<=1 missing) unless all_masks."""
    T = fam.make_tasks(rng, B, nonlin=nonlin, P=P)
    q = rng.integers(0, T['qx'].shape[1], (B, Q))
    ar = np.arange(B)[:, None]
    qx, qy = T['qx'][ar, q], T['qy'][ar, q]
    if all_masks:
        k = rng.integers(0, P, (B, Q))                               # 0..P-1 missing, uniform
        qm = np.ones((B, Q, P), dtype='float32')
        for b in range(B):
            for i in range(Q):
                qm[b, i, rng.permutation(P)[:k[b, i]]] = 0
    else:
        qm = fam.source_query_masks(rng, B, Q, P)
    t = lambda a: torch.as_tensor(np.asarray(a, dtype='float32'))
    return t(T['sx']), t(T['sy']), t(T['sm']), t(qx), t(qm), t(qy)
