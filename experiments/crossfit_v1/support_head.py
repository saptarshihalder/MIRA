"""Capacity-matched ablation: gate uses fitting-support context only."""
import torch
from model import CrossfitOperator, base


class SupportHead(CrossfitOperator):
    def candidate(self, p, diagnostics=False):
        # Solve once, predicting fitting rows as well as verification/query rows.
        joined = dict(p, q=torch.cat((p['a'], p['q']), 1),
                      qz=torch.cat((p['z'], p['qz']), 1))
        value = base.SpectralFilter.solve(self, joined, diagnostics)
        all_raw, objective = value if diagnostics else (value, None)
        n = p['a'].shape[1]
        fit_raw, raw = all_raw[:, :n], all_raw[:, n:]
        correction = fit_raw - p['z']
        leverage = ((p['a'] @ p['v']).square() / p['eigen'][:, None]).sum(-1) / n
        shift = (p['y'] - p['z'].sigmoid()).mean(1) - (p['sy'] - p['sz'].sigmoid()).mean(1)
        context = torch.stack((p['z'].clamp(-6, 6)/3, correction.tanh(),
            correction.abs().clamp_max(6)/3, leverage.log1p().clamp_max(6)/3,
            shift[:, None].expand_as(correction)), -1).mean(1, keepdim=True)
        alpha = self.query_head(context).squeeze(-1).sigmoid()
        return (1-alpha)*p['qz'].sigmoid()+alpha*raw.sigmoid(), objective
