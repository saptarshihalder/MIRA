"""Protocol v6 per-episode baselines on real panels (cells format of eval_real.py score).

python baselines_v6.py --panel <cache>/real_metr_la_s6061.pt --out <runs_real>/metr_la/lgbm   --method lgbm
python baselines_v6.py --panel <cache>/real_metr_la_s6061.pt --out <runs_real>/metr_la/mice   --method mice

lgbm  LightGBM with native missing values, fitted on the 48 support rows (200 trees, learning rate .05,
      min_child_samples 5, otherwise defaults). LightGBM is a point predictor; its predictive is Gaussian with the
      mean squared 5-fold cross-validated residual on the support rows as variance (fixed before any v6 score).
mice  sklearn IterativeImputer (defaults, random_state 0) fitted on the support inputs, then BayesianRidge on the
      imputed support; queries are imputed with the same imputer; predictive N(mean, std^2) from BayesianRidge.
Unobserved inputs are NaN for both (support: support mask; queries: the condition's query mask).
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import torch
from evaluate2 import gauss_metrics


def lgbm_predict(Xs, ys, Xq):
    import lightgbm as lgb
    kw = dict(n_estimators=200, learning_rate=.05, min_child_samples=5, verbose=-1, random_state=0)
    folds = np.arange(len(ys)) % 5
    res = np.empty(len(ys))
    for f in range(5):
        m = lgb.LGBMRegressor(**kw).fit(Xs[folds != f], ys[folds != f])
        res[folds == f] = ys[folds == f] - m.predict(Xs[folds == f])
    m = lgb.LGBMRegressor(**kw).fit(Xs, ys)
    return m.predict(Xq), np.full(len(Xq), max(float((res ** 2).mean()), 1e-6))


def mice_predict(Xs, ys, Xq):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    keep = ~np.isnan(Xs).all(0)                                  # a column never observed in the support carries nothing
    Xs, Xq = Xs[:, keep], Xq[:, keep]
    if Xs.shape[1] == 0:
        return np.full(len(Xq), ys.mean()), np.full(len(Xq), ys.var() + 1e-6)
    imp = IterativeImputer(random_state=0).fit(Xs)
    r = BayesianRidge().fit(imp.transform(Xs), ys)
    mu, sd = r.predict(imp.transform(Xq), return_std=True)
    return mu, sd ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--method', choices=['lgbm', 'mice'], required=True)
    a = ap.parse_args()
    fn = dict(lgbm=lgbm_predict, mice=mice_predict)[a.method]
    panel = torch.load(a.panel, weights_only=False)
    pool, tag = panel['pool'], Path(a.panel).stem
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells = {}
    for e, qm in panel['conds'].items():
        acc = {m: np.zeros(pool['n']) for m in ('nll', 'se', 'cov', 'crps')}
        for t in range(pool['n']):
            sm = pool['sm'][t].numpy().astype(bool)
            Xs = np.where(sm, pool['sx'][t].numpy(), np.nan).astype(float); ys = pool['sy'][t].numpy().astype(float)
            Xq = np.where(qm[t].astype(bool), pool['qx'][t].numpy(), np.nan).astype(float); yq = pool['qy'][t].numpy().astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu, var = fn(Xs, ys, Xq)
            for m, v in gauss_metrics(mu, var, yq).items():
                acc[m][t] = v.mean()
        cells[e] = acc
        print(a.method, tag, f'e{e}', f"nll {acc['nll'].mean():.4f}", f'{time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(out / f'cells_{tag}.npz', **{f'e{e}_{m}': v for e, d in cells.items() for m, v in d.items()})
    (out / 'train.json').write_text(json.dumps(dict(model=a.method, panel=tag, seconds=time.time() - t0), indent=1))


if __name__ == '__main__':
    main()
