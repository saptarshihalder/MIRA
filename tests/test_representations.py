import numpy as np

from mira.data import Observations, QueryInputs, Support, generate_episode
from mira.representations import transform_pair


def test_imputation_uses_only_context_means_and_retains_empty_columns():
    context = Observations(np.array([-1, 1]), np.array([[2, np.nan], [4, np.nan]]),
                           np.array([[0, 1], [0, 1]]))
    query = QueryInputs(Observations(np.array([1, -1]), np.array([[np.nan, 50], [900, np.nan]]),
                                    np.array([[1, 0], [0, 1]])))
    support = Support(context, np.array([0, 1]))
    inputs = transform_pair(support, query, "imputed", 0)
    np.testing.assert_array_equal(inputs.context, [[-1, 2, 0], [1, 4, 0]])
    np.testing.assert_array_equal(inputs.query, [[1, 3, 50], [-1, 900, 0]])
    changed = QueryInputs(Observations(query.observations.u, np.array([[np.nan, -700], [0, np.nan]]),
                                      query.observations.mask))
    np.testing.assert_array_equal(inputs.context, transform_pair(support, changed, "imputed", 0).context)


def test_indicator_and_shuffled_controls_keep_native_inputs_and_marginals():
    episode = generate_episode("sparse_pair", 8, 0.8, context=128, queries=128)
    native = transform_pair(episode.support, episode.query, "native", 80)
    actual = transform_pair(episode.support, episode.query, "native_indicators", 80)
    shuffled = transform_pair(episode.support, episode.query, "native_shuffled", 80)
    repeat = transform_pair(episode.support, episode.query, "native_shuffled", 80)
    d = episode.mechanism.dimensions
    for key in ("context", "query"):
        a, b, c = getattr(native, key), getattr(actual, key), getattr(shuffled, key)
        np.testing.assert_array_equal(a, b[:, :d + 1])
        np.testing.assert_array_equal(a, c[:, :d + 1])
        np.testing.assert_array_equal(b[:, -d:].sum(axis=0), c[:, -d:].sum(axis=0))
        np.testing.assert_array_equal(c, getattr(repeat, key))
        assert not np.array_equal(b[:, -d:], c[:, -d:])
    wider = transform_pair(episode.support, episode.query, "native_width_control", 80, 30)
    assert wider.context.shape[1] == native.context.shape[1] + 30


def test_imputed_collision_loses_mask_information_until_indicators_added():
    episode = generate_episode("label_only", 3, 0.9, context=40, queries=80,
                               value_distribution="zero_collision")
    imputed = transform_pair(episode.support, episode.query, "imputed", 0)
    indicated = transform_pair(episode.support, episode.query, "imputed_indicators", 0)
    assert np.all(imputed.query[:, 1:] == 0)
    np.testing.assert_array_equal(indicated.query[:, :9], imputed.query)
    np.testing.assert_array_equal(indicated.query[:, -8:], episode.query.observations.mask)
