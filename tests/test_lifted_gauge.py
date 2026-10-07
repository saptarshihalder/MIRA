"""Contract checks on new generated inputs; no historical confirmation scores."""
import sys
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/lifted_cavity'))
import anchors
import data
import family
import models
from gauge import SignAveragedCavity, reverse_factor


@pytest.mark.parametrize('width', [5, 8])
def test_sign_permutation_padding_and_gradients(width):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    pool = data.build_pool(4, 620270 + width, P=width)
    _, batch = next(data.eval_batches(pool, family.mask_bank(1, width)))
    base = models.LiftedCavity(K=1)
    net = SignAveragedCavity(base)
    with torch.no_grad():
        initial = net(batch)
        expected = models.closed_form(batch['anchors'][1], batch['qx'], batch['qm'], batch['tid'])
        for a, b in zip(initial, expected):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    torch.nn.init.normal_(base.site[-1].weight, std=.025)
    before = batch['anchors'][1]['D'].clone()
    result = net(batch)
    for a, b in zip(result, net(reverse_factor(batch))):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(before, batch['anchors'][1]['D'], atol=0, rtol=0)
    perm = torch.arange(width - 1, -1, -1)
    other = dict(batch)
    for key in ['sx', 'sm', 'qx', 'qm']:
        other[key] = batch[key][..., perm]
    mu, cov = anchors.em_batched(other['sx'], other['sy'], other['sm'])
    other['anchors'] = {1: {k: v.float() for k, v in anchors.fa_batched(mu, cov, 1).items()}}
    for a, b in zip(result, net(other)):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    hidden = dict(batch)
    hidden['qx'] = batch['qx'] + 100 * (1 - batch['qm'])
    for a, b in zip(result, net(hidden)):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    models.gauss_nll(*result, batch['qy']).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in base.parameters())
    assert base.site[-1].weight.grad.abs().sum() > 0


def test_multi_factor_rejected():
    with pytest.raises(ValueError):
        SignAveragedCavity(models.LiftedCavity(K=2))
