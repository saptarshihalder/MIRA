"""Reject partial/nonfinite results and incompatible resumed experiments."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / 'experiments/lifted_cavity'
sys.path.insert(0, str(CODE))
from score_validation import expected_scores, load_scores, validate_arrays
from bounded_v4 import bind_manifest


def test_partial_or_nonfinite_scores_rejected(tmp_path):
    panel = dict(pool=dict(n=4), banks={2: np.zeros((3, 5))})
    expected = expected_scores(panel, ('nll',))
    path = tmp_path / 'scores.npz'
    np.savez(path, k2_nll=np.zeros((2, 3)))
    with pytest.raises(ValueError, match='expected'):
        load_scores(path, panel, ('nll',))
    for bad in (np.nan, np.inf, -np.inf):
        values = np.zeros((4, 3)); values[0, 0] = bad
        with pytest.raises(ValueError, match='finite'):
            validate_arrays({'k2_nll': values}, expected)
    validate_arrays({'k2_nll': np.zeros((4, 3))}, expected)
    with pytest.raises(ValueError, match='keys differ'):
        validate_arrays({}, expected)


def test_real_conditions_and_smoke_limit():
    panel = dict(pool=dict(n=9), conds={0: None, 6: None})
    assert expected_scores(panel, ('nll',), limit=4) == {'e0_nll': (4,), 'e6_nll': (4,)}
    with pytest.raises(ValueError):
        expected_scores(panel, limit=0)


def test_resume_rejects_changed_identity_and_unbound_results(tmp_path):
    root = tmp_path / 'run'
    bind_manifest(root, {'source': 'abc', 'panels': '123'}, {'commit': 'first'})
    bind_manifest(root, {'source': 'abc', 'panels': '123'}, {'commit': 'same code'})
    with pytest.raises(ValueError, match='identity changed'):
        bind_manifest(root, {'source': 'def', 'panels': '123'}, {})
    unbound = tmp_path / 'old'; unbound.mkdir(); (unbound / 'model.pt').touch()
    with pytest.raises(ValueError, match='without an identity'):
        bind_manifest(unbound, {}, {})


def test_tabpfn_density_handles_double_borders_and_float_logits():
    torch = pytest.importorskip('torch')
    from tabpfn_colab import tabpfn_nll

    class Criterion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('borders', torch.tensor([0., 1.], dtype=torch.float64))

        def mean(self, logits):
            return logits.mean(-1)

        def forward(self, logits, y):
            # This indexed assignment reproduces TabPFN's ignore_init dtype requirement.
            y[torch.isnan(y)] = self.borders[0]
            assert logits.dtype == y.dtype == self.borders.dtype
            return (y - self.mean(logits)) ** 2

    class Regressor:
        def fit(self, X, y):
            pass

        def predict(self, X, output_type):
            return dict(criterion=Criterion(), logits=torch.ones((2, 3), dtype=torch.float32), mean=np.ones(2))

    nll, mean = tabpfn_nll(Regressor(), np.zeros((2, 1)), np.array([0., 1.]), np.zeros((2, 1)), np.array([2., 3.]))
    np.testing.assert_allclose(nll, [1., 4.])
    np.testing.assert_allclose(mean, [1., 1.])
