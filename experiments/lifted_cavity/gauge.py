"""Version 2 inference contract for one-factor lifted models.

Moment-match the equal-weight predictions under the two equivalent factor signs.
This is a classical symmetry repair, not a new trained checkpoint or novelty claim.
The Gaussian returned below is NOT the exact two-component mixture density.
"""
import torch
from torch import nn

from models import LiftedCavity


def reverse_factor(batch):
    """Copy only dictionaries and the sign-changed tensor; never mutate an anchor."""
    other = dict(batch)
    other['anchors'] = dict(batch['anchors'])
    other['anchors'][1] = dict(batch['anchors'][1])
    other['anchors'][1]['D'] = -batch['anchors'][1]['D']
    return other


class SignAveragedCavity(nn.Module):
    """K=1 only. Two passes, shared weights, no evaluation-label-dependent choice.

    For predictions (m+, v+), (m-, v-), return
      m = (m+ + m-)/2,
      v = (v+ + v-)/2 + (m+ - m-)^2/4.
    Swapping D and -D swaps the components and leaves these moments unchanged.
    Arbitrary rotations for K>1 or a degenerate leading eigenspace are not solved.
    """
    def __init__(self, base):
        super().__init__()
        if not isinstance(base, LiftedCavity) or base.K != 1:
            raise ValueError('Sign averaging requires a one-factor LiftedCavity.')
        self.base = base

    def forward(self, batch):
        plus, lp = self.base(batch)
        minus, lm = self.base(reverse_factor(batch))
        mean = (plus + minus) * .5
        variance = (lp.exp() + lm.exp()) * .5 + (plus - minus).square() * .25
        return mean, variance.log()
