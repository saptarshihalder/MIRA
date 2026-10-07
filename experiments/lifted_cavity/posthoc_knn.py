"""Post-hoc diagnostic (not in any protocol): per-episode k-nearest-neighbour regression on the observed sensors.
Support and query hours of an episode are drawn from the same week, so a query often has temporally adjacent support
hours with near-identical readings; kNN measures how much a pure retrieval mechanism can exploit this.
k is chosen per episode and sensor set by leave-one-out NLL on the support (k in 1..10), predictive variance = LOO MSE.
Writes runs_posthoc/knn/cells_<panel>.npz."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from evaluate2 import gauss_metrics

KS = (1, 2, 3, 5, 7, 10)


def knn_fit_predict(Xs, ys, Xq):
    n = len(ys)
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-6
    Zs, Zq = (Xs - mu_s) / sd_s, (Xq - mu_s) / sd_s
    Dss = ((Zs[:, None] - Zs[None]) ** 2).sum(-1); np.fill_diagonal(Dss, np.inf)
    order = np.argsort(Dss, 1)
    best = None
    for k in KS:
        if k >= n:
            break
        loo = ys[order[:, :k]].mean(1); v = max(((loo - ys) ** 2).mean(), 1e-4)
        nll = .5 * np.log(2 * np.pi * v) + .5 * ((loo - ys) ** 2).mean() / v
        if best is None or nll < best[0]:
            best = (nll, k, v)
    _, k, v = best
    Dqs = ((Zq[:, None] - Zs[None]) ** 2).sum(-1)
    pred = ys[np.argsort(Dqs, 1)[:, :k]].mean(1)
    return pred, np.full(len(Xq), v)


out = Path('runs_posthoc/knn'); out.mkdir(parents=True, exist_ok=True)
(out / 'train.json').write_text(json.dumps(dict(model='knn', note=__doc__.splitlines()[0])))
for tag in sys.argv[1:]:
    panel = torch.load(f'cache/{tag}.pt', weights_only=False); pool = panel['pool']; t0 = time.time(); res = {}
    for e, qm in panel['conds'].items():
        acc = {m: np.zeros(pool['n']) for m in ('nll', 'se', 'cov', 'crps')}
        for t in range(pool['n']):
            sx, sy, sm = (pool[k][t].double().numpy() for k in ('sx', 'sy', 'sm')); qx, qy = pool['qx'][t].double().numpy(), pool['qy'][t].double().numpy()
            per = {m: np.zeros(len(qy)) for m in acc}
            pats = {}
            for q in range(len(qy)):
                pats.setdefault(qm[t, q].tobytes(), []).append(q)
            for key, idx in pats.items():
                idx = np.array(idx); S = np.flatnonzero(qm[t, idx[0]])
                cc = sm[:, S].all(1) if len(S) else np.ones(len(sy), bool)
                if len(S) == 0 or cc.sum() < 12:
                    mu, var = np.full(len(idx), sy.mean()), np.full(len(idx), sy.var(ddof=1) * (1 + 1 / len(sy)))
                else:
                    mu, var = knn_fit_predict(sx[cc][:, S], sy[cc], qx[idx][:, S])
                for m, v in gauss_metrics(mu, var, qy[idx]).items():
                    per[m][idx] = v
            for m in acc:
                acc[m][t] = per[m].mean()
        for m, v in acc.items():
            res[f'e{e}_{m}'] = v
        print(tag, 'extra', e, round(float(acc['nll'].mean()), 4), f'{time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(out / f'cells_{tag}.npz', **res)
