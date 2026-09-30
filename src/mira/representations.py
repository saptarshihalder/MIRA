"""Context-only transforms. Neither query labels nor oracle values are accepted."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Array, QueryInputs, Support

MODES = ("native", "native_indicators", "native_shuffled", "imputed",
         "imputed_indicators", "native_width_control")


@dataclass(frozen=True)
class PreparedInputs:
    context: Array
    labels: Array
    query: Array
    mode: str


def _shuffled(mask: Array, rng: np.random.Generator, columns: int) -> Array:
    # Width sweeps repeat independently shuffled observed mask columns.
    return np.column_stack([rng.permutation(mask[:, index % mask.shape[1]]) for index in range(columns)])


def transform_pair(support: Support, query: QueryInputs, mode: str, seed: int,
                   width_control_columns: int | None = None) -> PreparedInputs:
    if not isinstance(support, Support) or not isinstance(query, QueryInputs):
        raise TypeError("Transforms accept Support and QueryInputs only")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    context, queries = support.observations, query.observations
    if context.values.shape[1] != queries.values.shape[1]:
        raise ValueError("Context and query feature widths differ")
    xc = np.column_stack([context.u, context.values])
    xq = np.column_stack([queries.u, queries.values])
    if mode.startswith("imputed"):
        observed = np.isfinite(xc)
        counts = observed.sum(axis=0)
        means = np.divide(np.nansum(xc, axis=0), counts, out=np.zeros(xc.shape[1]), where=counts > 0)
        xc = np.where(np.isnan(xc), means, xc)
        xq = np.where(np.isnan(xq), means, xq)
    if mode.endswith("indicators"):
        xc = np.column_stack([xc, context.mask])
        xq = np.column_stack([xq, queries.mask])
    elif mode in ("native_shuffled", "native_width_control"):
        columns = context.values.shape[1]
        if mode == "native_width_control" and width_control_columns is not None:
            if width_control_columns < 1:
                raise ValueError("Width control requires at least one added column")
            columns = width_control_columns
        context_rng = np.random.default_rng(np.random.SeedSequence([seed, 731, 1]))
        query_rng = np.random.default_rng(np.random.SeedSequence([seed, 731, 2]))
        xc = np.column_stack([xc, _shuffled(context.mask, context_rng, columns)])
        xq = np.column_stack([xq, _shuffled(queries.mask, query_rng, columns)])
    # Own arrays prevent backbones from modifying the immutable observation store.
    return PreparedInputs(xc.astype(np.float32), support.labels.copy(), xq.astype(np.float32), mode)
