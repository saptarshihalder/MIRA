"""Universal pool transformer (upt.py): initialization bound, reduction to the LCT, mixture metrics, training smoke."""
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1] / 'experiments' / 'lifted_cavity'
sys.path.insert(0, str(HERE))
import lct, lct_train, models, upt  # noqa: E402
from evaluate2 import gauss_metrics  # noqa: E402


def _batch(seed=0):
    return lct_train.fresh_batch(np.random.default_rng(seed), B=4, Q=6)


def test_init_is_within_log_sigmoid_of_the_fa_closed_form_for_every_query():
    torch.manual_seed(0)
    m = upt.UPT().eval()
    b = _batch()
    with torch.no_grad():
        nll = m.nll(b)
        mu0, lv0 = models.closed_form(b['anchors'][1], b['qx'], b['qm'], b['tid'])
        fa = models.gauss_nll(mu0, lv0, b['qy'])
        _, ms, vs, *_ = m.predictive(b)
    assert torch.allclose(ms, mu0, atol=1e-5) and torch.allclose(vs, lv0, atol=1e-5)
    assert (nll <= fa - math.log(1 / (1 + math.exp(-upt.GATE_BIAS))) + 1e-5).all()


def test_closed_gate_reduces_to_the_lct_with_shared_weights():
    torch.manual_seed(1)
    l = lct.LCT().eval()
    for p in l.head.parameters():
        torch.nn.init.normal_(p, std=.05)                       # a non-trivial site head
    m = upt.UPT().eval()
    missing, unexpected = m.load_state_dict(l.state_dict(), strict=False)
    assert set(missing) == {'gate.weight', 'gate.bias'} and not unexpected
    torch.nn.init.constant_(m.gate.bias, 500.)              # e^-500: the free component carries no weight
    b = _batch(1)
    with torch.no_grad():
        mu, lv = l(b)
        assert torch.allclose(m.nll(b), models.gauss_nll(mu, lv, b['qy']), atol=1e-4)


def test_one_component_mixture_metrics_match_the_gaussian_metrics():
    rng = np.random.default_rng(3)
    mu, var, y = rng.normal(size=50), np.exp(rng.normal(size=50)), rng.normal(size=50)
    g = gauss_metrics(mu, var, y)
    w = np.stack((np.ones(50), np.zeros(50)), 1)
    h = upt.mixture_metrics(w, np.stack((mu, mu + 1), 1), np.stack((var, var), 1), y)
    for k in ('nll', 'se', 'crps'):
        assert np.allclose(g[k], h[k]), k
    assert np.array_equal(g['cov'], h['cov'])


def test_mixture_crps_matches_monte_carlo():
    rng = np.random.default_rng(4)
    w, mu, var, y = np.array([[.3, .7]]), np.array([[-1., 2.]]), np.array([[.5, 2.]]), np.array([.4])
    comp = rng.random(400000) < .3
    x = np.where(comp, rng.normal(-1, math.sqrt(.5), comp.size), rng.normal(2, math.sqrt(2.), comp.size))
    mc = np.abs(x - y[0]).mean() - .5 * np.abs(x - rng.permutation(x)).mean()
    assert abs(upt.mixture_metrics(w, mu, var, y)['crps'][0] - mc) < 1e-2


def test_upt_trains_a_few_steps(tmp_path):
    r = subprocess.run([sys.executable, 'lct_train.py', '--model', 'upt', '--steps', '6', '--save-every', '3',
                        '--warmup', '2', '--out', str(tmp_path / 'u')], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / 'u' / 'model.pt').exists()
