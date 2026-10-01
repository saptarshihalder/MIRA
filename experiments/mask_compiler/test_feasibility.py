import itertools
import numpy as np
import pytest
from feasibility import basis, transform, library, fit_support
from mira.data import generate_episode


def test_exhaustive_inverses_and_nonbinary_rejection():
    masks = np.array(list(itertools.product((0, 1), repeat=4)))
    for a in library(4):
        assert np.array_equal(transform(transform(masks, a), a), masks)
    with pytest.raises(ValueError):
        basis([0, 0])
    with pytest.raises(ValueError):
        basis([1, 2])
    with pytest.raises(ValueError):
        basis([1, .5])
    with pytest.raises(ValueError):
        transform(masks, np.zeros((4, 4)))


def test_operational_inputs_and_cv_ignore_evaluator_targets():
    episode = generate_episode("pairwise", 73099, .9, context=32, queries=16)
    before = np.array(episode.query.observations.values, copy=True)
    p, selected = fit_support(episode.support, episode.query, library(4))
    # Perturb evaluator-only arrays while passing the same restricted learner objects.
    episode.targets.labels.setflags(write=True)
    episode.targets.labels[:] = 1-episode.targets.labels
    episode.targets.oracle.setflags(write=True)
    episode.targets.oracle[:] = .01
    p2, selected2 = fit_support(episode.support, episode.query, library(4))
    assert np.array_equal(p, p2) and selected == selected2
    assert np.array_equal(before, episode.query.observations.values, equal_nan=True)
    with pytest.raises(TypeError):
        fit_support(episode.support, episode.targets, library(4))
