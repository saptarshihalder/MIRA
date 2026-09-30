"""Explicit checkpoint requests; no model substitution or hidden oracle access."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .data import Array
from .representations import PreparedInputs

TABICL_V2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


class BinaryEstimator(Protocol):
    classes_: Array

    def fit(self, features: Array, labels: Array) -> Any: ...
    def predict_proba(self, features: Array) -> Array: ...
    def get_params(self, deep: bool = False) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ModelSettings:
    device: str = "cuda"
    ensembles: int = 4


def create_model(name: str, settings: ModelSettings, seed: int) -> BinaryEstimator:
    if name.startswith("tabpfn:"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        version = ModelVersion(name.split(":", 1)[1])
        return TabPFNClassifier.create_default_for_version(
            version, device=settings.device, n_estimators=settings.ensembles, random_state=seed)
    if name == "tabicl:v2":
        from tabicl import TabICLClassifier

        return TabICLClassifier(device=settings.device, n_estimators=settings.ensembles,
                                checkpoint_version=TABICL_V2_CHECKPOINT, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             reg_lambda=1.0, tree_method="hist", random_state=seed,
                             n_jobs=2, eval_metric="logloss")
    raise ValueError(f"Unsupported model: {name}; no silent fallback")


def predict_binary(model: BinaryEstimator, inputs: PreparedInputs) -> Array:
    """Only this restricted record crosses the learner boundary."""
    model.fit(inputs.context, inputs.labels)
    probabilities = np.asarray(model.predict_proba(inputs.query), dtype=float)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(inputs.query), len(classes)) or len(positive) != 1:
        raise ValueError("Backbone returned invalid binary probability shape/classes")
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("Backbone classes must be exactly {0,1}")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Backbone probabilities must be finite, normalized and in [0,1]")
    return probabilities[:, positive[0]]


def cold_start_probability(labels: Array, query_count: int) -> Array | None:
    """Declared Laplace fallback until both context classes exist."""
    if len(np.unique(labels)) < 2:
        return np.full(query_count, (float(np.sum(labels)) + 1) / (len(labels) + 2))
    return None
