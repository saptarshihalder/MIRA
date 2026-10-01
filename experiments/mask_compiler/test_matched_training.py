"""Matched budget, portability, permutation, and evaluator boundary checks."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_training as matched
from mira.data import Observations, Support, EvaluationTargets


def test_identical_pretraining_tasks_and_parameter_budgets():
    moments, raw, lengths, teacher = matched.training_data(81000, 8)
    original_moments, original_teacher = matched.prototype_dataset(81000, 8)
    np.testing.assert_array_equal(moments, original_moments)
    np.testing.assert_array_equal(teacher, original_teacher)
    for x, n, moment in zip(raw, lengths, moments):
        mask = ((x[:n, 1:5] + 1) / 2).astype(np.uint8)
        parity = 1 - 2 * ((mask @ matched.CODES.T) % 2).astype(float)
        np.testing.assert_allclose(np.abs((parity * x[:n, 5, None]).mean(0)), moment[:15], atol=1e-7)
        assert not x[n:].any()
    assert sum(p.numel() for p in matched.Compiler().parameters()) == 6092
    assert sum(p.numel() for p in matched.DeepSetsSelector().parameters()) == 6092
    assert sum(p.numel() for p in matched.LinearSelector().parameters()) == 216


def test_npz_roundtrip_row_permutation_and_support_only_boundary(tmp_path):
    support, query, targets, _ = matched.development_episode(90000, 3, .9)
    order = np.random.default_rng(49).permutation(len(support.labels))
    obs = support.observations
    permuted = Support(Observations(obs.u[order], obs.values[order], obs.mask[order]), support.labels[order])
    for name, constructor in (("prototype", matched.Compiler), ("linear", matched.LinearSelector),
                              ("deepsets", matched.DeepSetsSelector)):
        torch.manual_seed(19)
        model = constructor().eval()
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, **{k: v.detach().numpy() for k, v in model.state_dict().items()})
        restored = matched.load_selector(name, path)
        matrix, info = model.select(support, confidence=0)
        matrix2, info2 = restored.select(support, confidence=0)
        np.testing.assert_array_equal(matrix, matrix2)
        assert info["max_probability"] == info2["max_probability"]
        pmatrix, pinfo = restored.select(permuted, confidence=0)
        np.testing.assert_array_equal(matrix, pmatrix)
        assert info["max_probability"] == pytest.approx(pinfo["max_probability"], abs=1e-6)
        np.testing.assert_array_equal(restored.select(support, confidence=1)[0], matched.MATRICES[0])
        for invalid in (query, targets):
            with pytest.raises(TypeError):
                restored.select(invalid)


def test_predictions_ignore_query_labels_and_oracle():
    support, query, targets, _ = matched.development_episode(90002, 2, .9)
    p, fit = matched.logistic_prediction(support, query, [matched.MATRICES[0]])
    changed = EvaluationTargets(1 - targets.labels, 1 - targets.oracle, 1 - targets.analytic_base)
    # Change evaluator state while preserving the observable query object.
    targets = changed
    p2, fit2 = matched.logistic_prediction(support, query, [matched.MATRICES[0]])
    np.testing.assert_array_equal(p, p2)
    assert fit == fit2
    with pytest.raises(TypeError):
        matched.logistic_prediction(support, targets, [matched.MATRICES[0]])
