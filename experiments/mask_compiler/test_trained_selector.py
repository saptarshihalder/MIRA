import numpy as np
import pytest
import torch
from train_selector import Compiler, encode, MATRICES, transform
from mira.data import generate_episode, Observations, Support


def test_support_permutation_and_query_boundary():
    ep=generate_episode("pairwise",89101,.9,context=32,queries=8)
    obs=ep.support.observations
    rev=Support(Observations(obs.u[::-1],obs.values[::-1],obs.mask[::-1]),ep.support.labels[::-1])
    assert np.array_equal(encode(ep.support),encode(rev))
    with pytest.raises(TypeError):
        encode(ep.query)
    torch.manual_seed(9)
    model=Compiler()
    a, info=model.select(ep.support, confidence=1.)
    assert info['index']==0
    assert np.array_equal(transform(transform(obs.mask,a),a),obs.mask)
