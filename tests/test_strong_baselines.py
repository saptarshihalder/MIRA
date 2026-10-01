import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mira.data import generate_episode

SPEC = importlib.util.spec_from_file_location("strong_baselines", Path(__file__).resolve().parents[1] / "scripts/run_strong_baselines.py")
BASELINES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINES)


def test_saved_query_labels_oracles_and_mechanism_metadata_cannot_change_predictions(tmp_path):
    episode = generate_episode("pairwise", 91, .8, context=96, queries=32)
    support, query = episode.support.observations, episode.query.observations
    common = dict(uc=support.u, xc=support.values, mc=support.mask, yc=episode.support.labels,
                  uq=query.u, xq=query.values, mq=query.mask, support_row_ids=episode.support_row_ids,
                  query_row_ids=episode.query_row_ids)
    first, altered = tmp_path / "first.npz", tmp_path / "altered.npz"
    np.savez(first, **common, yq=episode.targets.labels, oracle=episode.targets.oracle, base=episode.targets.analytic_base)
    np.savez(altered, **common, yq=1 - episode.targets.labels, oracle=np.zeros(32), base=np.ones(32),
             active=np.array([99]), gamma=np.array([.999]))
    outputs = [BASELINES.predict_all(*BASELINES.load_inputs(path)) for path in (first, altered)]
    assert set(outputs[0]) == set(BASELINES.METHODS)
    for method in BASELINES.METHODS:
        np.testing.assert_array_equal(outputs[0][method][0], outputs[1][method][0])
        assert np.isfinite(outputs[0][method][0]).all()
        assert ((outputs[0][method][0] >= 0) & (outputs[0][method][0] <= 1)).all()
        for field in ("selected_c", "support_cv_losses"):
            assert outputs[0][method][1].get(field) == outputs[1][method][1].get(field)
    with pytest.raises(TypeError):
        BASELINES.predict_all(episode.support, episode.targets)


def test_pilot_reuse_matches_original_on_legal_inputs():
    episode = generate_episode("pairwise", 91, .8, context=64, queries=32)
    predictions = BASELINES.predict_all(episode.support, episode.query)
    context = {"u": episode.support.observations.u, "m": episode.support.observations.mask.astype(np.int64), "y": episode.support.labels}
    query = {"u": episode.query.observations.u, "m": episode.query.observations.mask.astype(np.int64),
             "p0": predictions["u_logistic_cv"][0]}
    np.testing.assert_array_equal(predictions["sparse_mixture_fitted_base"][0], BASELINES.load_pilot_mixture()(context, query))
