"""Opt-in successor backbone/readout; never loaded by frozen v4 runners.

Reuses the existing cell-token encoder, not the original factor-coordinate head.
No trained successor checkpoint or performance advantage is established.
"""
from pathlib import Path
import sys

import torch

source = str(Path(__file__).resolve().parents[1] / "lifted_cavity")
if source not in sys.path:
    sys.path.insert(0, source)
import lct
from experiments.lifted_transport.readout import CovarianceTransportHead


class CovarianceTransportPFN(lct.LCT):
    def __init__(self, d=64, layers=4, heads=4, ff=128, femb=16, K=1):
        super().__init__(d=d, layers=layers, heads=heads, ff=ff, femb=femb, K=K)
        self.head = CovarianceTransportHead(d)

    def forward(self, batch):
        anc, tid = batch["anchors"][self.K], batch["tid"]
        b = batch["sx"].shape[0]
        qx, qm = batch["qx"], batch["qm"]
        if tid.numel() % b:
            raise ValueError("Query/task counts do not align")
        q = tid.numel() // b
        # Sanitize before attention, since NaN * 0 remains NaN.
        sx = torch.where(batch["sm"].bool(), batch["sx"], torch.zeros_like(batch["sx"]))
        safe_qx = torch.where(qm.bool(), qx, torch.zeros_like(qx))
        generator = None if self.training else torch.Generator().manual_seed(self.feature_seed)
        features = self.tokens(sx, batch["sy"], batch["sm"], safe_qx.reshape(b, q, -1),
                               qm.reshape(b, q, -1), generator).reshape(*qx.shape, -1)
        return self.head(features, anc["A"][tid], anc["D"][tid], anc["psi"][tid],
                         anc["mx"][tid], anc["my"][tid], anc["vy"][tid], safe_qx, qm)
