"""Neural-process baselines (np_models.py, protocol v8): interface, query independence, width handling, training."""
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1] / 'experiments' / 'lifted_cavity'
sys.path.insert(0, str(HERE))
import np_models, pfn  # noqa: E402


def _data(P=5, B=3, Q=7, seed=0):
    return pfn.fresh_batch(np.random.default_rng(seed), B, Q, P=P)


def test_shapes_and_query_independence():
    for M in (np_models.CNP(), np_models.TNP()):
        M.eval()
        sx, sy, sm, qx, qm, qy = _data()
        mu, lv = M.core(sx, sy, sm, qx, qm)
        assert mu.shape == qy.shape and lv.shape == qy.shape and torch.isfinite(mu).all()
        qx2 = qx.clone(); qx2[:, 1:] = torch.randn_like(qx2[:, 1:])            # change every other query
        mu2, lv2 = M.core(sx, sy, sm, qx2, qm)
        assert torch.allclose(mu[:, 0], mu2[:, 0], atol=1e-5) and torch.allclose(lv[:, 0], lv2[:, 0], atol=1e-5)


def test_missing_sensor_value_is_ignored():
    for M in (np_models.CNP(), np_models.TNP()):
        M.eval()
        sx, sy, sm, qx, qm, qy = _data(P=11)
        qm[:, :, 3] = 0
        qx2 = qx.clone(); qx2[:, :, 3] = 99.
        assert torch.allclose(M.core(sx, sy, sm, qx, qm)[0], M.core(sx, sy, sm, qx2, qm)[0], atol=1e-5)


def test_np_trains_a_few_steps(tmp_path):
    for m in ('cnp', 'tnp'):
        r = subprocess.run([sys.executable, 'pfn_train.py', '--model', m, '--steps', '4', '--save-every', '2', '--warmup', '2',
                            '--batch-tasks', '4', '--out', str(tmp_path / m)], cwd=HERE, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (tmp_path / m / 'model.pt').exists()
