"""Exact Bayes oracle for the cavity_site / anchored_cavity synthetic sensor family.

Replays experiments/cavity_site.py::generate to recover each task's true parameters,
then computes, for every development task x two-sensor-deletion mask x query:
  - exact posterior log density of the true y (1-D quadrature, exact up to grid error)
  - moment-matched Gaussian oracle (best Gaussian-output predictor in KL)
  - population best linear predictor (infinite-support ridge) with its residual variance
  - complete-case support ridge (fit only on support rows where the 3 needed sensors are observed)
and compares with the repo's stored predictions.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import family as fam_lc  # support-only closed forms shared with the lifted-cavity experiments
import fa as fa_lc

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
LOG2PI = np.log(2 * np.pi)
OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'artifacts' / 'reports' / 'lifted_cavity_v1' / 'oracle_audit'
OUTDIR.mkdir(parents=True, exist_ok=True)


def replay(n, seed):
    """Exact replay of cavity_site.generate, keeping the per-task parameters."""
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n):
        a = rng.choice([-1., 1.], 5) * rng.uniform(.6, 1.8, 5)
        b = rng.normal(0, .4, 5)
        f = rng.uniform(.5, 2.5, 5)
        d = rng.normal(0, .7, 5)
        noise = rng.uniform(.15, .8, 5)
        y = rng.normal(size=96)
        u = rng.normal(size=96)
        x = y[:, None] * a + b + .4 * np.sin(y[:, None] * f) + u[:, None] * d + rng.normal(size=(96, 5)) * noise
        sm = (rng.random((48, 5)) > .2).astype(float)
        rng.integers(0, 6, 48)  # source query mask draw (unused for development scoring)
        tasks.append(dict(a=a, b=b, f=f, d=d, s=noise, x=x, y=y, sm=sm))
    return tasks


def context_stats(x, m):
    n = m.sum(0).clip(1)
    mean = (x * m).sum(0) / n
    xc = (x - mean) * m
    scale = np.sqrt((xc * xc).sum(0) / n).clip(.1)
    return mean, scale


GRID = np.linspace(-7, 7, 2801)
LOGPRIOR = -.5 * (LOG2PI + GRID ** 2)
DY = GRID[1] - GRID[0]


def exact_posterior(t, S, xq, yq):
    """Return exact log p(y_true|x_S), posterior mean and variance for all queries."""
    a, b, f, d, s = (t[k][S] for k in ('a', 'b', 'f', 'd', 's'))
    cov = np.outer(d, d) + np.diag(s ** 2)
    prec = np.linalg.inv(cov)
    logdet = np.linalg.slogdet(cov)[1]
    k = len(S)

    def loglik(yv, xs):
        mu = yv[..., None] * a + b + .4 * np.sin(yv[..., None] * f)  # (..., k)
        r = xs - mu
        return -.5 * (k * LOG2PI + logdet + np.einsum('...i,ij,...j->...', r, prec, r))

    xs = xq[:, S]
    ll = loglik(GRID[None, :], xs[:, None, :]) + LOGPRIOR[None, :]  # (nq, grid)
    mx = ll.max(1, keepdims=True)
    w = np.exp(ll - mx)
    z = w.sum(1) * DY
    logz = np.log(z) + mx[:, 0]
    post = w / (w.sum(1, keepdims=True))
    pm = (post * GRID).sum(1)
    pv = (post * (GRID - pm[:, None]) ** 2).sum(1)
    lt = loglik(yq, xs) - .5 * (LOG2PI + yq ** 2) - logz
    return lt, pm, pv


def pop_linear(t, S):
    """Population best linear predictor of y from x_S (closed form)."""
    a, b, f, d, s = (t[k] for k in ('a', 'b', 'f', 'd', 's'))
    eys = f * np.exp(-f ** 2 / 2)  # E[y sin(f y)]
    cxy = a + .4 * eys
    F1, F2 = np.meshgrid(f, f, indexing='ij')
    ess = .5 * (np.exp(-(F1 - F2) ** 2 / 2) - np.exp(-(F1 + F2) ** 2 / 2))
    cxx = np.outer(a, a) + .4 * np.outer(a, eys) + .4 * np.outer(eys, a) + .16 * ess + np.outer(d, d) + np.diag(s ** 2)
    cxx, cxy, mean = cxx[np.ix_(S, S)], cxy[S], b[S]
    beta = np.linalg.solve(cxx, cxy)
    resid = 1 - cxy @ beta
    return beta, mean, resid


def gauss_nll(mu, var, y):
    return .5 * (LOG2PI + np.log(var) + (y - mu) ** 2 / var)


def ridge_fit(design, target, penalty_diag):
    inv = np.linalg.inv(design.T @ design + np.diag(penalty_diag))
    w = inv @ design.T @ target
    lev = np.sum((design @ inv) * design, axis=1)
    loo = (target - design @ w) / (1 - lev).clip(.05)
    return w, max(float((loo ** 2).mean()), .01)


def analyse(name, dev_seed):
    out = REPO / f'artifacts/reports/{name}'
    dev = np.load(REPO / f'artifacts/local/{name}/development.npz')
    tasks = replay(64, dev_seed)
    # verify replay reproduces the stored standardized queries exactly
    maxerr = 0.
    for i, t in enumerate(tasks):
        mean, scale = context_stats(t['x'][:48], t['sm'])
        maxerr = max(maxerr, float(np.abs((t['x'][48:] - mean) / scale - dev['x'][i]).max()))
        maxerr = max(maxerr, float(np.abs(t['y'][48:] - dev['y'][i]).max()))
    masks = np.ones((10, 5))
    for i, pair in enumerate(itertools.combinations(range(5), 2)):
        masks[i, list(pair)] = 0
    rows = {k: [] for k in ('oracle', 'oracle_gauss', 'pop_linear', 'ridge_cc', 'ridge_cc_mse', 'oracle_mse', 'pop_mse', 'n_cc',
                            'em_gauss', 'fa1')}
    for t in tasks:
        xq, yq = t['x'][48:], t['y'][48:]
        mean, scale = context_stats(t['x'][:48], t['sm'])
        ssx = (t['x'][:48] - mean) / scale * t['sm']
        qxs = (xq - mean) / scale
        emu, esig = fam_lc.em_gaussian(ssx, t['y'][:48], t['sm'], 40)
        f1 = fa_lc.fit_fa(emu, esig, 1)
        for mask in masks:
            S = np.flatnonzero(mask)
            lt, pm, pv = exact_posterior(t, S, xq, yq)
            rows['oracle'].append(-lt)
            rows['oracle_gauss'].append(gauss_nll(pm, pv, yq))
            rows['oracle_mse'].append((pm - yq) ** 2)
            beta, mu0, resid = pop_linear(t, S)
            pmu = (xq[:, S] - mu0) @ beta
            rows['pop_linear'].append(gauss_nll(pmu, np.full_like(pmu, resid), yq))
            rows['pop_mse'].append((pmu - yq) ** 2)
            # complete-case ridge on standardized support rows with all of S observed
            cc = t['sm'][:, S].all(1)
            rows['n_cc'].append(np.full(48, cc.sum()))
            sx = (t['x'][:48][cc][:, S] - mean[S]) / scale[S]
            design = np.column_stack((np.ones(cc.sum()), sx))
            pen = np.ones(design.shape[1]); pen[0] = 0
            w, var = ridge_fit(design, t['y'][:48][cc], pen)
            qmu = np.column_stack((np.ones(48), (xq[:, S] - mean[S]) / scale[S])) @ w
            rows['ridge_cc'].append(gauss_nll(qmu, np.full(48, var), yq))
            rows['ridge_cc_mse'].append((qmu - yq) ** 2)
            rows['em_gauss'].append(fam_lc.gnll(*fam_lc.em_conditional(emu, esig, qxs, S, 48), yq))
            rows['fa1'].append(fam_lc.gnll(*fa_lc.lifted_poe(f1, qxs, S, 48), yq))
    res = {k: np.concatenate(v) for k, v in rows.items()}
    summary = dict(replay_max_abs_error=maxerr,
                   oracle_nll=float(res['oracle'].mean()), oracle_gauss_nll=float(res['oracle_gauss'].mean()),
                   oracle_mse=float(res['oracle_mse'].mean()),
                   pop_linear_nll=float(res['pop_linear'].mean()), pop_linear_mse=float(res['pop_mse'].mean()),
                   ridge_cc_nll=float(res['ridge_cc'].mean()), ridge_cc_mse=float(res['ridge_cc_mse'].mean()),
                   em_gauss_nll=float(res['em_gauss'].mean()), fa1_nll=float(res['fa1'].mean()),
                   mean_complete_case_rows=float(res['n_cc'].mean()))
    models = {}
    for f in sorted(out.glob('pred_*.npz')):
        z = np.load(f)
        models[f.stem[5:]] = dict(nll=float(z['nll'].mean()), mse=float(z['mse'].mean()), cov=float(z['coverage'].mean()))
        # per-row alignment check against the oracle ordering (task*480 + mask*48 + q)
    summary['models'] = models
    np.savez_compressed(Path(OUTDIR) / f'oracle_{name}.npz', **res)
    return summary


if __name__ == '__main__':
    report = {'cavity_site_v1': analyse('cavity_site_v1', 819102),
              'anchored_cavity_v1': analyse('anchored_cavity_v1', 819202)}
    Path(OUTDIR, 'oracle_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
