"""Exact synthetic mechanisms adapted from the supplied headroom.py pilot.

P(M=m | Y=y,U=u) = 2^-d [1 + gamma (2y-1) f(m,u)],
f = sign * product(2 M_active-1) * U^value_dependent.
Observed Gaussian values are nuisance variables; U is always observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

Array = NDArray[np.generic]
FAMILIES: Mapping[str, tuple[int, int, bool]] = {
    "label_only": (8, 1, False),  # Pilot TFM's single signal among eight masks.
    "pairwise": (4, 2, False),
    "value_dependent": (8, 1, True),
    "sparse_pair": (16, 2, False),
}
VALUE_DISTRIBUTIONS = ("gaussian", "zero_collision")


def _readonly(value: Array, dtype: object) -> Array:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Observations:
    """Only observable features. A missing bit is exactly a NaN in values."""

    u: Array
    values: Array
    mask: Array

    def __post_init__(self) -> None:
        u = _readonly(self.u, np.float64)
        values = _readonly(self.values, np.float64)
        mask = _readonly(self.mask, np.uint8)
        if values.ndim != 2 or u.shape != (len(values),) or mask.shape != values.shape:
            raise ValueError("Observation shapes must be [n], [n,d], [n,d]")
        if not np.isin(self.mask, [0, 1]).all() or not np.isfinite(u).all():
            raise ValueError("Masks must be binary and U finite")
        if np.isinf(values).any() or not np.array_equal(np.isnan(values), mask.astype(bool)):
            raise ValueError("NaN pattern must equal the declared mask; infinities are invalid")
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)

    def __len__(self) -> int:
        return len(self.u)


@dataclass(frozen=True)
class Support:
    observations: Observations
    labels: Array

    def __post_init__(self) -> None:
        if not isinstance(self.observations, Observations):
            raise TypeError("Support requires observable inputs")
        if np.shape(self.labels) != (len(self.observations),) or not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Support requires one binary label per observation")
        object.__setattr__(self, "labels", _readonly(self.labels, np.int64))


@dataclass(frozen=True)
class QueryInputs:
    """The predictor receives this type, which has no labels or oracle field."""

    observations: Observations

    def __post_init__(self) -> None:
        if not isinstance(self.observations, Observations):
            raise TypeError("Query inputs require observations, never evaluation targets")


@dataclass(frozen=True)
class EvaluationTargets:
    """Evaluator-only outcomes and probabilities; never supplied to learners."""

    labels: Array
    oracle: Array
    analytic_base: Array

    def __post_init__(self) -> None:
        labels = _readonly(self.labels, np.int64)
        oracle = _readonly(self.oracle, np.float64)
        base = _readonly(self.analytic_base, np.float64)
        if labels.ndim != 1 or oracle.shape != labels.shape or base.shape != labels.shape:
            raise ValueError("Evaluation arrays must be aligned vectors")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Evaluation labels must be binary")
        for probability in (oracle, base):
            if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
                raise ValueError("Evaluation probabilities must lie in [0,1]")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "oracle", oracle)
        object.__setattr__(self, "analytic_base", base)


@dataclass(frozen=True)
class Mechanism:
    family: str
    dimensions: int
    active: tuple[int, ...]
    sign: int
    gamma: float
    beta: float = 0.8
    value_dependent: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.gamma < 1 or not np.isfinite(self.beta):
            raise ValueError("Require finite beta and 0 <= gamma < 1")
        if self.sign not in (-1, 1) or not self.active or len(set(self.active)) != len(self.active):
            raise ValueError("Require nonempty distinct active columns and sign +/-1")
        if min(self.active) < 0 or max(self.active) >= self.dimensions:
            raise ValueError("Active columns outside feature dimensions")

    def signal(self, mask: Array, u: Array) -> Array:
        value = self.sign * np.prod(2 * np.asarray(mask, dtype=float)[:, self.active] - 1, axis=1)
        return value * u if self.value_dependent else value

    def posterior(self, mask: Array, u: Array) -> Array:
        signal = self.gamma * self.signal(mask, u)
        return expit(self.beta * u + np.log1p(signal) - np.log1p(-signal))

    def mask_likelihood(self, mask: Array, labels: Array, u: Array) -> Array:
        return 2.0 ** (-self.dimensions) * (1 + self.gamma * (2 * labels - 1) * self.signal(mask, u))


@dataclass(frozen=True)
class Episode:
    """Evaluator container. Pass support and query separately to the predictor."""

    support: Support
    query: QueryInputs
    targets: EvaluationTargets
    mechanism: Mechanism
    seed: int
    support_row_ids: Array
    query_row_ids: Array

    def __post_init__(self) -> None:
        support_ids = _readonly(self.support_row_ids, str)
        query_ids = _readonly(self.query_row_ids, str)
        if support_ids.shape != (len(self.support.observations),) or query_ids.shape != (len(self.query.observations),):
            raise ValueError("Row identifiers must match their support/query observations")
        if (len(np.unique(support_ids)) != len(support_ids) or len(np.unique(query_ids)) != len(query_ids)
                or np.intersect1d(support_ids, query_ids).size):
            raise ValueError("Support/query row identifiers must be unique and disjoint")
        object.__setattr__(self, "support_row_ids", support_ids)
        object.__setattr__(self, "query_row_ids", query_ids)


def canonical_family(family: str) -> str:
    family = {"joint_pair": "pairwise"}.get(family, family)
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    return family


def make_mechanism(family: str, seed: int, gamma: float, beta: float = 0.8) -> Mechanism:
    family = canonical_family(family)
    dimensions, degree, value = FAMILIES[family]
    rng = np.random.default_rng(np.random.SeedSequence([seed, list(FAMILIES).index(family), 0]))
    active = tuple(sorted(int(index) for index in rng.choice(dimensions, degree, replace=False)))
    return Mechanism(family, dimensions, active, int(rng.choice([-1, 1])), gamma, beta, value)


def _sample(task: Mechanism, n: int, rng: np.random.Generator,
            value_distribution: str) -> tuple[Observations, EvaluationTargets]:
    u = rng.choice([-1.0, 1.0], size=n)
    base = expit(task.beta * u)
    labels = (rng.random(n) < base).astype(np.int64)
    mask = rng.integers(0, 2, size=(n, task.dimensions))
    rest = np.prod(2 * mask[:, task.active[:-1]] - 1, axis=1)
    if task.value_dependent:
        rest = rest * u
    missing_probability = (1 + task.gamma * task.sign * (2 * labels - 1) * rest) / 2
    mask[:, task.active[-1]] = rng.random(n) < missing_probability
    oracle = task.posterior(mask, u)
    values = rng.normal(size=mask.shape)
    if value_distribution == "zero_collision":
        # Deliberate collision with mean imputation. Consume the same random
        # draws so masks, outcomes, and the oracle are paired across variants.
        values[:] = 0
    values[mask == 1] = np.nan
    return Observations(u, values, mask), EvaluationTargets(labels, oracle, base)


def generate_episode(family: str, seed: int, gamma: float, context: int = 256,
                     queries: int = 1024, beta: float = 0.8,
                     value_distribution: str = "gaussian") -> Episode:
    if context < 0 or queries < 1 or seed < 0:
        raise ValueError("Require context >= 0, queries >= 1, seed >= 0")
    if value_distribution not in VALUE_DISTRIBUTIONS:
        raise ValueError(f"Unknown value distribution: {value_distribution}")
    task = make_mechanism(family, seed, gamma, beta)
    family_id = list(FAMILIES).index(task.family)
    # Disjoint random streams: changing context length cannot change query rows.
    context_rng = np.random.default_rng(np.random.SeedSequence([seed, family_id, 1]))
    query_rng = np.random.default_rng(np.random.SeedSequence([seed, family_id, 2]))
    observations, support_targets = _sample(task, context, context_rng, value_distribution)
    query, targets = _sample(task, queries, query_rng, value_distribution)
    support_ids = np.array([f"{task.family}:{seed}:support:{index}" for index in range(context)], dtype=str)
    query_ids = np.array([f"{task.family}:{seed}:query:{index}" for index in range(queries)], dtype=str)
    return Episode(Support(observations, support_targets.labels), QueryInputs(query), targets, task, seed,
                   support_ids, query_ids)
