"""Evaluation consumes evaluator-only targets after predictions are fixed."""
from __future__ import annotations

import numpy as np

from .data import Array, EvaluationTargets

PROBABILITY_EPSILON = 1e-6
HEADROOM_EPSILON = 1e-4


def expected_nll(probability: Array, true_probability: Array) -> float:
    p = np.clip(probability, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    return float(np.mean(-true_probability * np.log(p) - (1 - true_probability) * np.log1p(-p)))


def evaluate(probability: Array, targets: EvaluationTargets) -> dict[str, float | None]:
    p = np.asarray(probability, dtype=float)
    if p.shape != targets.labels.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid query probabilities")
    loss = expected_nll(p, targets.oracle)
    base_loss = expected_nll(targets.analytic_base, targets.oracle)
    oracle_loss = expected_nll(targets.oracle, targets.oracle)
    gap = base_loss - oracle_loss
    return {
        "expected_nll": loss,
        "empirical_nll": expected_nll(p, targets.labels),
        "expected_brier": float(np.mean(targets.oracle * (1 - p) ** 2 + (1 - targets.oracle) * p ** 2)),
        "empirical_brier": float(np.mean((p - targets.labels) ** 2)),
        "analytic_base_nll": base_loss,
        "oracle_nll": oracle_loss,
        "analytic_headroom": gap,
        "fraction_analytic_gain": (base_loss - loss) / gap if gap > HEADROOM_EPSILON else None,
    }
