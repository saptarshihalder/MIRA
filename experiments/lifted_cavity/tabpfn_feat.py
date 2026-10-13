"""Protocol v14: TabPFN with UPT-SF's prediction as one more input column (the "UPT column").

    python tabpfn_feat.py --root <runs root> --net dwd_t2m --panel-seed 9091 [--version v3.5] [--limit N]

For each episode and condition e (extra dropped sensors), TabPFN is fitted in context on the 48 labelled support rows and
predicts the 48 query rows, with one extra column:
- query rows: UPT-SF's test prediction under the condition's masks (pq e{e}_mean);
- support rows: UPT-SF's cross-fitted prediction (pq cf_e{e}_mean): each half of the support predicted from the other
  half under the shared cross-fit masks (realdata.crossfit_masks, seed 20000 + e), so no row's own label enters its
  column.
UPT-SF's predictions are averaged over seeds 1-3 (runs <model>_s{1,2,3}_ft, files pq_<tag>_tta8.npz).
Two variants, which differ only at e > 0:
- F1 (`uptcol`): support inputs keep their natural missingness, as in tabpfn_colab.py;
- F2 (`uptcol_mirror`): support inputs also get the cross-fit masks of condition e, the same masks under which their UPT
  column was computed, so context and query rows carry the same kind of missingness.
At e = 0 the cross-fit masks are the natural masks, so F1 = F2 (one call). Everything else as tabpfn_colab.py (8
estimators, random state 0, NaN for missing inputs, CRPS and 90% coverage from 102 quantiles, constant-support episodes
undefined). Outputs: runs_real/<net>/tabpfn_<version>_uptcol{,_mirror}/cells_<tag>.npz.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import realdata
from tabpfn_colab import N_ESTIMATORS, RANDOM_STATE, make_regressor, tabpfn_nll, task_arrays

SEEDS = (1, 2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--net', required=True); ap.add_argument('--panel-seed', type=int, required=True)
    ap.add_argument('--version', default='v3.5'); ap.add_argument('--model', default='upt_L_sfm'); ap.add_argument('--sfx', default='_tta8')
    ap.add_argument('--limit', type=int, default=None, help='smoke test: only the first LIMIT episodes (partial_ outputs)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    R = Path(a.root) / 'runs_real' / a.net
    tag = f'real_{a.net}_s{a.panel_seed}'
    panel = torch.load(Path(a.root) / 'panels' / f'{tag}.pt', weights_only=False); pool = panel['pool']
    E = list(panel['conds'])
    U = [np.load(R / f'{a.model}_s{s}_ft' / f'pq_{tag}{a.sfx}.npz') for s in SEEDS]
    col = {k: np.mean([u[k] for u in U], 0) for e in E for k in (f'e{e}_mean', f'cf_e{e}_mean')}
    cfm = {e: realdata.crossfit_masks(pool, e, realdata.CF_SEED + e) for e in E}
    n = pool['n'] if a.limit is None else min(a.limit, pool['n'])
    vname = 'tabpfn_' + a.version.replace('.', '_')
    outs = {v: R / f'{vname}_{v}' for v in ('uptcol', 'uptcol_mirror')}
    stem = f'cells_{tag}' if a.limit is None else f'partial_cells_{tag}_limit{a.limit}'
    if all((o / f'{stem}.npz').exists() for o in outs.values()):
        print('done already'); return
    reg = make_regressor(a.device, a.version)
    res = {v: {f'e{e}_{m}': np.full(n, np.nan) for e in E for m in ('nll', 'se', 'crps', 'cov')} for v in outs}
    und = np.zeros(n, bool); t0 = time.time(); calls = 0
    for t in range(n):
        Xs, ys, qx, qy = task_arrays(pool, t)
        if np.ptp(ys) == 0:                       # protocol v6 amendment 3: TabPFN undefined on a constant support target
            und[t] = True; continue
        sx = pool['sx'][t].numpy().astype(float)
        for e in E:
            Xq = np.where(np.asarray(panel['conds'][e][t]) > 0, qx, np.nan)
            Xq = np.concatenate([Xq, col[f'e{e}_mean'][t][:, None]], 1)
            cs = col[f'cf_e{e}_mean'][t][:, None]
            variants = {'uptcol': np.concatenate([Xs, cs], 1)}
            if e:
                variants['uptcol_mirror'] = np.concatenate([np.where(cfm[e][t] > 0, sx, np.nan), cs], 1)
            for v, S in variants.items():
                nll, mu, crps, cov = tabpfn_nll(reg, S, ys, Xq, qy, quant=True); calls += 1
                r = res[v]
                r[f'e{e}_nll'][t], r[f'e{e}_se'][t] = nll.mean(), ((mu - qy) ** 2).mean()
                r[f'e{e}_crps'][t], r[f'e{e}_cov'][t] = crps.mean(), cov.mean()
            if not e:                             # F1 = F2 at e = 0
                for m in ('nll', 'se', 'crps', 'cov'):
                    res['uptcol_mirror'][f'e0_{m}'][t] = res['uptcol'][f'e0_{m}'][t]
        if (t + 1) % 100 == 0:
            print(a.net, t + 1, n, f'{time.time() - t0:.0f}s',
                  {e: round(float(np.nanmean(res['uptcol'][f'e{e}_nll'][:t + 1])), 3) for e in E}, flush=True)
    meta = dict(model=f'{vname}+uptcol', version=a.version, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, column=f'{a.model}{a.sfx} seeds {SEEDS}',
                episodes=int(n), calls=calls, seconds=time.time() - t0, undefined=int(und.sum()))
    if torch.cuda.is_available():
        meta['gpu'] = torch.cuda.get_device_name(0)
    for v, o in outs.items():
        o.mkdir(parents=True, exist_ok=True)
        r = res[v]; r['undefined'] = und
        assert all(np.isfinite(r[k][~und]).all() for k in r if k != 'undefined'), v
        tmp = o / f'{stem}.tmp.npz'; np.savez_compressed(tmp, **r); tmp.replace(o / f'{stem}.npz')
        (o / 'train.json').write_text(json.dumps(dict(meta, variant=v), indent=1))
    print(a.net, 'done', {v: {e: round(float(np.nanmean(res[v][f'e{e}_nll'])), 4) for e in E} for v in outs}, flush=True)


if __name__ == '__main__':
    main()
