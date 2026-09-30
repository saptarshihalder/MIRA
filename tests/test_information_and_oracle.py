from dataclasses import fields, replace
from itertools import product

import numpy as np
import pytest
from scipy.special import expit

from mira.data import (FAMILIES, EvaluationTargets, Mechanism, QueryInputs,
                       generate_episode, make_mechanism)
from mira.metrics import evaluate
from mira.models import cold_start_probability, predict_binary
from mira.representations import MODES, transform_pair


class SpyEstimator:
    classes_ = np.array([1, 0])  # Exercise class lookup instead of hard-coded [:,1].

    def fit(self, features, labels):
        self.context = features.copy()
        self.labels = labels.copy()
        return self

    def predict_proba(self, features):
        self.query = features.copy()
        positive = expit(features[:, 0] + self.labels.mean() - 0.5)
        return np.column_stack([positive, 1 - positive])

    def get_params(self, deep=False):
        return {"fake": True}


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_signal_oracle_is_exact_analytic_base(family):
    episode = generate_episode(family, 91, 0, context=16, queries=128)
    np.testing.assert_array_equal(episode.targets.oracle, episode.targets.analytic_base)
    metrics = evaluate(episode.targets.analytic_base, episode.targets)
    assert metrics["analytic_headroom"] == 0
    assert metrics["fraction_analytic_gain"] is None


@pytest.mark.parametrize("family", ["label_only", "pairwise", "value_dependent"])
def test_posterior_matches_enumerated_normalized_mask_likelihood(family):
    task = make_mechanism(family, 22, 0.9)
    mask = np.array(list(product((0, 1), repeat=task.dimensions)))
    for u_value in (-1.0, 1.0):
        u = np.full(len(mask), u_value)
        p0 = expit(task.beta * u_value)
        l1 = task.mask_likelihood(mask, np.ones(len(mask)), u)
        l0 = task.mask_likelihood(mask, np.zeros(len(mask)), u)
        assert np.sum(l0) == pytest.approx(1)
        assert np.sum(l1) == pytest.approx(1)
        np.testing.assert_allclose(task.posterior(mask, u), p0 * l1 / (p0 * l1 + (1 - p0) * l0), atol=1e-14)


def test_invalid_mechanism_rejects_non_normalizable_or_singular_settings():
    with pytest.raises(ValueError):
        Mechanism("bad", 1, (), 1, 0.8)
    with pytest.raises(ValueError):
        Mechanism("bad", 1, (0,), 1, 1.0)


@pytest.mark.parametrize("mode", MODES)
def test_query_targets_cannot_influence_transform_or_predictions(mode):
    episode = generate_episode("pairwise", 5, 0.8, context=64, queries=64)
    assert [field.name for field in fields(QueryInputs)] == ["observations"]
    assert not hasattr(episode.query, "labels") and not hasattr(episode.query, "oracle")
    original = transform_pair(episode.support, episode.query, mode, 99)
    altered_targets = EvaluationTargets(1 - episode.targets.labels, 1 - episode.targets.oracle,
                                        1 - episode.targets.analytic_base)
    # Alter evaluator state, then re-use precisely the same learner-accessible inputs.
    assert not np.array_equal(altered_targets.labels, episode.targets.labels)
    transformed = transform_pair(episode.support, episode.query, mode, 99)
    np.testing.assert_array_equal(original.context, transformed.context)
    np.testing.assert_array_equal(original.query, transformed.query)
    first, second = SpyEstimator(), SpyEstimator()
    np.testing.assert_array_equal(predict_binary(first, original), predict_binary(second, transformed))
    np.testing.assert_array_equal(first.labels, episode.support.labels)
    with pytest.raises(TypeError):
        transform_pair(episode.support, altered_targets, mode, 99)
    with pytest.raises(TypeError):
        QueryInputs(altered_targets)
    with pytest.raises(ValueError):
        episode.query.observations.values[0, 0] = 123


@pytest.mark.parametrize("family", FAMILIES)
def test_collision_control_pairs_masks_labels_and_oracle(family):
    gaussian = generate_episode(family, 123, 0.8, context=40, queries=80)
    collision = generate_episode(family, 123, 0.8, context=40, queries=80,
                                 value_distribution="zero_collision")
    for part in ("support", "query"):
        a, b = getattr(gaussian, part).observations, getattr(collision, part).observations
        np.testing.assert_array_equal(a.u, b.u)
        np.testing.assert_array_equal(a.mask, b.mask)
        assert np.all(b.values[np.isfinite(b.values)] == 0)
    np.testing.assert_array_equal(gaussian.support.labels, collision.support.labels)
    np.testing.assert_array_equal(gaussian.targets.labels, collision.targets.labels)
    np.testing.assert_array_equal(gaussian.targets.oracle, collision.targets.oracle)


def test_query_stream_independent_of_context_size_and_gamma_keeps_outcomes():
    a = generate_episode("sparse_pair", 99, 0.8, context=8, queries=30)
    b = generate_episode("sparse_pair", 99, 0.8, context=256, queries=30)
    np.testing.assert_array_equal(a.query.observations.values, b.query.observations.values)
    np.testing.assert_array_equal(a.targets.labels, b.targets.labels)
    c = generate_episode("sparse_pair", 99, 0, context=8, queries=30)
    np.testing.assert_array_equal(a.targets.labels, c.targets.labels)
    np.testing.assert_array_equal(a.query.observations.u, c.query.observations.u)


def test_cold_start_is_explicit_same_laplace_rule():
    np.testing.assert_array_equal(cold_start_probability(np.array([]), 2), [0.5, 0.5])
    np.testing.assert_allclose(cold_start_probability(np.ones(3), 2), [0.8, 0.8])
    assert cold_start_probability(np.array([0, 1]), 2) is None


def test_support_and_query_identity_sets_are_disjoint_and_overlap_is_rejected():
    episode = generate_episode("pairwise", 123, .8, context=12, queries=12)
    assert not np.intersect1d(episode.support_row_ids, episode.query_row_ids).size
    with pytest.raises(ValueError, match="disjoint"):
        replace(episode, query_row_ids=episode.support_row_ids)
