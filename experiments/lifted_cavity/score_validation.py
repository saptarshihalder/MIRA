"""Validate score completeness against panel structure, without inspecting target values."""
from pathlib import Path
import numpy as np

GAUSSIAN_METRICS = ('nll', 'se', 'cov', 'crps')
TABPFN_METRICS = ('nll', 'se')


def expected_scores(panel, metrics=GAUSSIAN_METRICS, limit=None):
    if isinstance(panel, (str, Path)):
        import torch
        panel = torch.load(panel, weights_only=False, map_location='cpu')
    n = int(panel['pool']['n'])
    if n <= 0:
        raise ValueError('panel has no tasks')
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be positive')
        n = min(n, limit)
    if 'banks' in panel:
        expected = {f'k{k}_{m}': (n, len(bank)) for k, bank in panel['banks'].items() for m in metrics}
    else:
        expected = {f'e{e}_{m}': (n,) for e in panel['conds'] for m in metrics}
    if not expected or any(0 in shape for shape in expected.values()):
        raise ValueError('panel has no score conditions')
    return expected


def validate_arrays(arrays, expected):
    if set(arrays) != set(expected):
        raise ValueError(f'score keys differ: missing={sorted(set(expected) - set(arrays))}, '
                         f'unexpected={sorted(set(arrays) - set(expected))}')
    for key, shape in expected.items():
        value = np.asarray(arrays[key])
        if value.shape != shape:
            raise ValueError(f'{key}: expected {shape}, got {value.shape}')
        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError(f'{key}: scores must be finite real numbers')


def load_scores(path, panel, metrics=GAUSSIAN_METRICS, limit=None):
    expected = expected_scores(panel, metrics, limit)
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    validate_arrays(arrays, expected)
    return arrays
