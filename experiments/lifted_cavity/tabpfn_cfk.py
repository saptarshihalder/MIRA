"""Protocol v15: TabPFN's K-fold cross-fit of each episode's support (the v11 two-fold cross-fit, for K folds).

    python tabpfn_cfk.py --panel <panel.pt> --out <run dir> --version v3.5 --k 4 [--stride 2 --offset 0]

For each episode and fold k (realdata.kfolds: contiguous blocks of NS / K rows), TabPFN is fitted in context on the
other K - 1 folds (natural support masks) and predicts fold k under every condition's shared cross-fit masks
(realdata.crossfit_masks, seed 20000 + extra), in one call per fold. Folds whose training target is constant are left
NaN, and so are episodes whose whole support target is constant (TabPFN undefined), as in v11. Same TabPFN settings as
tabpfn_colab.py (8 estimators, random state 0, NaN for missing inputs).
Output: <out>/pqk<K>_<tag>.npz, or pqk<K>_<tag>_o<offset>s<stride>.npz for a part (episodes offset, offset + stride, ...),
with cf_e{e}_nll and cf_e{e}_mean of shape (n, NS).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import realdata
from tabpfn_colab import make_regressor, tabpfn_nll, task_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True); ap.add_argument('--out', required=True); ap.add_argument('--version', default='v3.5')
    ap.add_argument('--k', type=int, default=4); ap.add_argument('--stride', type=int, default=1); ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None, help='smoke test: stop after this many episodes')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    tag = Path(a.panel).stem
    part = '' if a.stride == 1 else f'_o{a.offset}s{a.stride}'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / (f'pqk{a.k}_{tag}{part}.npz' if a.limit is None else f'partial_pqk{a.k}_{tag}{part}_limit{a.limit}.npz')
    if dest.exists():
        print('done already', dest, flush=True); return
    panel = torch.load(a.panel, weights_only=False); pool = panel['pool']; E = list(panel['conds'])
    NS, n = realdata.NS, pool['n']
    folds = realdata.kfolds(a.k)
    cfm = {e: realdata.crossfit_masks(pool, e, realdata.CF_SEED + e) for e in E}
    reg = make_regressor(a.device, a.version)
    res = {f'cf_e{e}_{m}': np.full((n, NS), np.nan) for e in E for m in ('nll', 'mean')}
    t0, calls, done = time.time(), 0, 0
    for t in range(a.offset, n, a.stride):
        if a.limit is not None and done >= a.limit:
            break
        done += 1
        Xs, ys, qx, qy = task_arrays(pool, t)
        if np.ptp(ys) == 0:
            continue
        sx = pool['sx'][t].numpy().astype(float)
        for tr, te in folds:
            if np.ptp(ys[tr]) == 0:
                continue
            Xb = np.concatenate([np.where(cfm[e][t, te] > 0, sx[te], np.nan) for e in E])
            l, mu = tabpfn_nll(reg, Xs[tr], ys[tr], Xb, np.tile(ys[te], len(E))); calls += 1
            for i, e in enumerate(E):
                res[f'cf_e{e}_nll'][t, te] = l.reshape(len(E), -1)[i]; res[f'cf_e{e}_mean'][t, te] = mu.reshape(len(E), -1)[i]
        if done % 100 == 0:
            print(tag, part, done, f'{time.time() - t0:.0f}s', flush=True)
    tmp = dest.with_suffix('.tmp.npz'); np.savez_compressed(tmp, **res); tmp.replace(dest)
    meta = dict(version=a.version, k=a.k, stride=a.stride, offset=a.offset, episodes=done, calls=calls, seconds=time.time() - t0)
    if torch.cuda.is_available():
        meta['gpu'] = torch.cuda.get_device_name(0)
    (out / f'pqk{a.k}_{tag}{part}.json').write_text(json.dumps(meta, indent=1))
    print(tag, part, 'done', meta, flush=True)


if __name__ == '__main__':
    main()
