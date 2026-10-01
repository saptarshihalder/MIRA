"""Five checks for the extended law, unchanged defaults and value-only controls."""
from dataclasses import asdict
import hashlib
from itertools import product

import numpy as np
import pytest
from scipy.special import expit

from mira.data import FAMILIES, Mechanism, _sample, generate_episode, make_mechanism
from mira.runner import _configuration, parser


def test_bernoulli_law_normalization_bounded_signal_and_exact_posterior():
    for family in FAMILIES:
        for rate in (.1, .5, .9):
            task = make_mechanism(family, 77, .9, missing_rate=rate)
            mask = np.array(list(product((0, 1), repeat=task.dimensions)))
            for u_value in (-1., 1.):
                u = np.full(len(mask), u_value)
                likelihoods = [task.mask_likelihood(mask, np.full(len(mask), label), u) for label in (0, 1)]
                assert np.max(np.abs(task.signal(mask, u))) <= 1
                for likelihood in likelihoods:
                    assert likelihood.min() > 0
                    assert likelihood.sum() == pytest.approx(1, abs=1e-13)
                p0 = expit(task.beta * u_value)
                bayes = p0 * likelihoods[1] / (p0 * likelihoods[1] + (1 - p0) * likelihoods[0])
                np.testing.assert_allclose(task.posterior(mask, u), bayes, rtol=1e-13, atol=1e-14)
            zero = generate_episode(family, 77, 0, context=8, queries=16, missing_rate=rate)
            np.testing.assert_array_equal(zero.targets.oracle, zero.targets.analytic_base)


def test_final_bit_sampler_uses_conditional_probability_of_the_declared_law():
    # Every sign of U, label and first mask bit is tested just above and below
    # the final-bit likelihood threshold. This checks the sampler independently
    # against the normalized joint law rather than a noisy frequency estimate.
    task = Mechanism("test", 2, (0, 1), -1, .9, value_dependent=True, missing_rate=.1)
    cases = np.array(list(product((-1., 1.), (0, 1), (0, 1), (0, 1))))
    u, labels, rest, wanted = cases.T
    mask0 = np.column_stack([rest, np.zeros(len(cases))])
    mask1 = np.column_stack([rest, np.ones(len(cases))])
    l0 = task.mask_likelihood(mask0, labels, u)
    l1 = task.mask_likelihood(mask1, labels, u)
    threshold = l1 / (l0 + l1)

    class ControlledRNG:
        vector_calls = 0

        def choice(self, choices, size):
            return u.copy()

        def random(self, size):
            if isinstance(size, tuple):
                return np.column_stack([np.where(rest == 1, .01, .99), np.full(len(cases), .5)])
            self.vector_calls += 1
            if self.vector_calls == 1:
                return np.where(labels == 1, 0., .999)
            return threshold + np.where(wanted == 1, -1e-8, 1e-8)

        def normal(self, size):
            return np.zeros(size)

    observations, targets = _sample(task, len(cases), ControlledRNG(), np.random.default_rng(99))
    np.testing.assert_array_equal(observations.mask[:, 0], rest)
    np.testing.assert_array_equal(observations.mask[:, 1], wanted)
    np.testing.assert_array_equal(targets.labels, labels)


def test_original_default_draws_match_pre_extension_fixtures():
    # Captured before the extension; round only floats to tolerate platform
    # differences in elementary functions, not changes to RNG draws.
    fixtures = {
        ("label_only", "gaussian"): "7041a96bb225f658a7329838f2031d34ec6d4c8c82065dac02783d944065cf99",
        ("pairwise", "zero_collision"): "4c5c4a6a6dd08b0d645b8c4329eb89551938eea2bb1a3957262987e5c82a7af0",
        ("value_dependent", "gaussian"): "cedccbc2cdd75c77aa937feced0128e14e7cd31170be9f72558e34dce8028ba2",
        ("sparse_pair", "gaussian"): "a5b05edb5d295655cb2516fa6a804b117120eb42b6f7dc4e237500caf0d02219",
    }
    for (family, distribution), expected in fixtures.items():
        episode = generate_episode(family, 40000, .8, context=16, queries=32, value_distribution=distribution)
        digest = hashlib.sha256()
        arrays = [episode.support.observations.u, episode.support.observations.values,
                  episode.support.observations.mask, episode.support.labels,
                  episode.query.observations.u, episode.query.observations.values,
                  episode.query.observations.mask, episode.targets.labels,
                  episode.targets.oracle, episode.targets.analytic_base]
        for array in arrays:
            digest.update((np.round(array, 12) if array.dtype.kind == "f" else array).tobytes())
        assert digest.hexdigest() == expected


def test_parameter_boundaries_and_explicit_configuration():
    invalid = ({"missing_rate": 0}, {"missing_rate": 1}, {"missing_rate": np.nan},
               {"collision_probability": -.01}, {"collision_probability": 1.01},
               {"collision_probability": np.nan}, {"quantization_step": 0},
               {"quantization_step": np.inf}, {"quantization_step": np.nan})
    for parameters in invalid:
        with pytest.raises(ValueError):
            generate_episode("label_only", 3, .8, **parameters)
        option, value = next(iter(parameters.items()))
        with pytest.raises(ValueError):
            _configuration(parser().parse_args(["--" + option.replace("_", "-"), str(value)]))
    for collision_probability in (0, 1):
        episode = generate_episode("label_only", 3, .8, value_distribution="partial_collision",
                                   missing_rate=.1, collision_probability=collision_probability,
                                   quantization_step=2.5)
        metadata = asdict(episode.mechanism)
        assert metadata["missing_rate"] == .1 and metadata["quantization_step"] == 2.5
        assert metadata["collision_probability"] == collision_probability
    configuration = _configuration(parser().parse_args(["--value-distribution", "quantized", "--missing-rate", ".1",
                                                        "--quantization-step", "2.5", "--collision-probability", ".25"]))
    assert configuration["value_distribution"] == "quantized"
    assert configuration["missing_rate"] == .1 and configuration["quantization_step"] == 2.5
    assert configuration["collision_probability"] == .25


def test_value_controls_use_separate_rng_and_never_change_mechanism_targets():
    for family in FAMILIES:
        for rate in (.1, .5):
            baseline = generate_episode(family, 47, .8, context=32, queries=64, missing_rate=rate)
            for distribution in ("zero_collision", "partial_collision", "quantized"):
                altered = generate_episode(family, 47, .8, context=32, queries=64, missing_rate=rate,
                                           value_distribution=distribution, collision_probability=.25,
                                           quantization_step=2.5)
                for section in ("support", "query"):
                    first, second = getattr(baseline, section).observations, getattr(altered, section).observations
                    np.testing.assert_array_equal(first.u, second.u)
                    np.testing.assert_array_equal(first.mask, second.mask)
                    np.testing.assert_array_equal(np.isnan(first.values), np.isnan(second.values))
                    if distribution == "quantized":
                        observed = second.values[np.isfinite(second.values)] / 2.5
                        np.testing.assert_allclose(observed, np.round(observed))
                np.testing.assert_array_equal(baseline.support.labels, altered.support.labels)
                for field in ("labels", "oracle", "analytic_base"):
                    np.testing.assert_array_equal(getattr(baseline.targets, field), getattr(altered.targets, field))
                np.testing.assert_array_equal(baseline.support_row_ids, altered.support_row_ids)
                np.testing.assert_array_equal(baseline.query_row_ids, altered.query_row_ids)
            for probability, distribution in ((0, "gaussian"), (1, "zero_collision")):
                endpoint = generate_episode(family, 47, .8, context=32, queries=64, missing_rate=rate,
                                            value_distribution="partial_collision", collision_probability=probability)
                reference = generate_episode(family, 47, .8, context=32, queries=64, missing_rate=rate,
                                             value_distribution=distribution)
                np.testing.assert_array_equal(endpoint.support.observations.values, reference.support.observations.values)
                np.testing.assert_array_equal(endpoint.query.observations.values, reference.query.observations.values)
