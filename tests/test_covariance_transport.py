"""Source-only structural checks, not successor efficacy evidence."""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("transport_readout", Path(__file__).resolve().parents[1] /
                                            "experiments/lifted_transport/readout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example(p=5, k=2):
    generator = torch.Generator().manual_seed(91)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    n, d = 4, 8
    return [rand(n, p, d), rand(n, p), rand(n, p, k), torch.ones(n, p, dtype=torch.float64) * .4,
            rand(n, p), rand(n), torch.ones(n, dtype=torch.float64), rand(n, p),
            (rand(n, p) > -.5).double()]


def trained_shape():
    head = module.CovarianceTransportHead(8).double()
    with torch.no_grad():
        head.pair.copy_(torch.arange(64).reshape(8, 8) / 80.)
        head.local.weight.fill_(.1)
        head.local.bias.fill_(.03)
    return head


def test_zero_head_recovers_full_gaussian_anchor():
    x = example()
    mean, logvar = module.CovarianceTransportHead(8).double()(*x)
    _, a, factors, psi, mx, my, vy, qx, mask = x
    for i in range(len(my)):
        keep = mask[i].bool()
        response, noise = a[i, keep], factors[i, keep]
        covariance = noise @ noise.T + torch.diag(psi[i, keep]) + vy[i] * torch.outer(response, response)
        expected_mean = my[i] + vy[i] * response @ torch.linalg.solve(covariance, qx[i, keep] - mx[i, keep])
        expected_var = vy[i] - vy[i] ** 2 * response @ torch.linalg.solve(covariance, response)
        torch.testing.assert_close(mean[i], expected_mean, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(logvar[i].exp(), expected_var, atol=1e-10, rtol=1e-10)


def test_factor_rotation_and_sensor_permutation_preserve_nonzero_head():
    head, x = trained_shape(), example()
    expected = head(*x)
    rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
    rotated = list(x); rotated[2] = x[2] @ rotation
    for left, right in zip(expected, head(*rotated)):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = [value[:, order] if index not in (5, 6) else value for index, value in enumerate(x)]
    for left, right in zip(expected, head(*permuted)):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_missing_values_empty_queries_variable_width_and_positive_covariance():
    head = trained_shape()
    for p in (1, 5, 11):
        x = example(p=p)
        expected = head(*x)
        x[0][~x[8].bool()] = float('nan')
        x[7][~x[8].bool()] = float('nan')
        mean, logvar, covariance = head(*x, return_covariance=True)
        for left, right in zip(expected, (mean, logvar)):
            torch.testing.assert_close(left, right)
        assert bool((torch.linalg.eigvalsh(covariance) > 0).all())
        x[8].zero_()
        mean, logvar = head(*x)
        torch.testing.assert_close(mean, x[5])
        torch.testing.assert_close(logvar.exp(), x[6])


def test_covariance_correction_has_a_nonzero_gradient_at_analytic_initialization():
    x = example()
    head = module.CovarianceTransportHead(8).double()
    mean, logvar = head(*x)
    target = torch.tensor([-2., -.1, .9, 3.], dtype=torch.float64)
    loss = .5 * (logvar + (target - mean).square() * (-logvar).exp()).mean()
    loss.backward()
    assert bool(torch.isfinite(head.pair.grad).all()) and head.pair.grad.norm() > 1e-10
    assert head.local.weight.grad.norm() > 1e-10


def test_full_model_ignores_query_labels_and_masks_nan_before_attention():
    from experiments.lifted_transport.model import CovarianceTransportPFN
    torch.manual_seed(63)
    model = CovarianceTransportPFN(d=16, layers=1, heads=2, ff=32, femb=4).double().eval()
    with torch.no_grad():
        model.head.local.weight.fill_(.03)
        model.head.pair.fill_(.01)
    x = example(p=5, k=1)
    _, a, factors, psi, mx, my, vy, qx, qm = x
    batch = dict(sx=torch.randn(4, 8, 5, dtype=torch.float64), sy=torch.randn(4, 8, dtype=torch.float64),
                 sm=torch.ones(4, 8, 5, dtype=torch.float64), qx=qx, qm=qm,
                 tid=torch.arange(4), qy=torch.zeros(4, dtype=torch.float64),
                 anchors={1:dict(A=a,D=factors,psi=psi,mx=mx,my=my,vy=vy)})
    # The inherited column-feature generator uses default float32; use a standard float32 model here.
    model = model.float()
    batch = {key: (value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
             for key, value in batch.items()}
    batch['anchors'] = {1:{key:value.float() for key,value in batch['anchors'][1].items()}}
    expected = model(batch)
    batch['qy'].fill_(1e6)
    batch['qx'][~batch['qm'].bool()] = float('nan')
    for left, right in zip(expected, model(batch)):
        torch.testing.assert_close(left, right)
