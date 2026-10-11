"""CPU engineering benchmark against the identical dense Gaussian calculation.

No fitting, data panels, model selection or efficacy claims. Fixed widths,
one warmup, five samples, float64, one thread; readout only, anchors excluded.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from .model import CompositionalMixture, Query


def dense(query):
    w, noise = query.loadings, query.noise
    covariance = w @ w.transpose(-1, -2) + torch.diag_embed(noise)
    chol = torch.linalg.cholesky(covariance)
    residual = query.qx[:, None] - query.locations
    cross = query.vy.sqrt()[:, None, None] * w[..., 0]
    solved = torch.cholesky_solve(torch.stack((residual, cross), -1), chol)
    means = query.my[:, None] + (cross * solved[..., 0]).sum(-1)
    variances = query.vy[:, None] - (cross * solved[..., 1]).sum(-1)
    log_density = -.5 * (query.qx.shape[-1] * 1.8378770664093453
                         + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                         + (residual * solved[..., 0]).sum(-1))
    return means, variances, log_density


def run(out):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261118)
    rows = []
    with torch.no_grad():
        for p in (32, 128, 512):
            rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
            query = Query(rand(1, 3, p, 5) * .3, rand(1, 3, p), torch.ones(1, 3, p, dtype=torch.float64),
                          torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
                          torch.zeros(3, dtype=torch.float64), rand(1, p), torch.ones(1, p, dtype=torch.float64))
            small = lambda: query.decode(query.summarize())
            reference = lambda: dense(query)
            small(); reference()
            times = {'compositional_seconds': [], 'dense_seconds': []}
            for _ in range(5):
                for key, call in zip(times, (small, reference)):
                    start = time.perf_counter(); call()
                    times[key].append(time.perf_counter() - start)
            actual, expected = small(), reference()
            comparisons = list(zip((actual.means, actual.variances, actual.log_evidence), expected))
            for a, b in comparisons:
                torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
            rows.append(dict(sensors=p, **times,
                             dense_over_small_median=statistics.median(times['dense_seconds']) /
                             statistics.median(times['compositional_seconds']),
                             max_absolute_discrepancy=max(float((a-b).abs().max()) for a, b in comparisons)))
    result = dict(kind='engineering-only-untrained', source='constructed random tensor fixtures',
                  seed=20261118, dtype='float64', threads=1, repeats=5, components=3, latent_width=5,
                  parameter_count=sum(p.numel() for p in CompositionalMixture().parameters()),
                  python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                  rows=rows, training_steps=0, cloud_cost_usd=0,
                  limitations='Readout only; no anchor/encoder/training cost or trained-model quality comparison.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps({k: result[k] for k in ('kind', 'parameter_count', 'training_steps', 'cloud_cost_usd')}))
    for row in rows:
        print(json.dumps({k: row[k] for k in ('sensors', 'dense_over_small_median', 'max_absolute_discrepancy')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
