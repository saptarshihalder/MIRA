"""Protocol v11: UPT (+) TabPFN, the linear pool of a lifted-site transformer and a pretrained foundation model, with
per-episode stacking weights (Yao, Vehtari, Simpson & Gelman 2018) chosen on a two-fold cross-fit of the episode's own
labelled support rows (no test label, no other episode, nothing trained).

    python pool_eval.py --root <runs root> --net usgs_ny --panel-seed 10101 --model upt_L_sfm --sfx _tta8
Inputs: pq_<tag><sfx>.npz of runs_real/<net>/<model>_s{1,2,3}_ft and pq_<tag>.npz of runs_real/<net>/tabpfn_v2.
For each seed, episode and condition: w = argmax over GRID of the cross-fit log score
    sum_rows log(w p_UPT(y_r) + (1 - w) p_TabPFN(y_r)),
then every test query gets p = w p_UPT + (1 - w) p_TabPFN. Where TabPFN is undefined (constant training target) the
pool is UPT alone (w = 1). Output: runs_real/<net>/pool_<model>_tabpfn/cells_<tag>.npz with e{e}_nll, e{e}_se (seed
averages, as every other learned method's cells are averaged downstream: here the three seeds are stored as one run
whose values are already seed averages) and e{e}_w (mean weight on UPT per episode).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

GRID = np.linspace(.02, .98, 49)
SEEDS = (1, 2, 3)


def stack_weight(u, t):
    ok = np.isfinite(u) & np.isfinite(t)
    if not ok.any():
        return 1.
    lu, lt = -u[ok], -t[ok]
    score = [np.logaddexp(np.log(w) + lu, np.log1p(-w) + lt).sum() for w in GRID]
    return float(GRID[int(np.argmax(score))])


def pooled(w, u, t):
    """Per-query pooled NLL; queries where TabPFN is undefined keep UPT's NLL."""
    out = u.copy()
    ok = np.isfinite(t)
    if w < 1:
        out[ok] = -np.logaddexp(np.log(w) - u[ok], np.log1p(-w) - t[ok])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--net', required=True); ap.add_argument('--panel-seed', type=int, required=True)
    ap.add_argument('--model', default='upt_L_sfm'); ap.add_argument('--sfx', default='_tta8')
    ap.add_argument('--single', action='store_true', help='protocol v12: --model is one run directory (no seeds), e.g. fa1_closed')
    a = ap.parse_args()
    R = Path(a.root) / 'runs_real' / a.net
    tag = f'real_{a.net}_s{a.panel_seed}'
    pool = torch.load(Path(a.root) / 'panels' / f'{tag}.pt', weights_only=False)['pool']
    qy = pool['qy'].double().numpy()
    T = np.load(R / 'tabpfn_v2' / f'pq_{tag}.npz')
    U = [np.load(R / a.model / f'pq_{tag}{a.sfx}.npz')] if a.single else         [np.load(R / f'{a.model}_s{s}_ft' / f'pq_{tag}{a.sfx}.npz') for s in SEEDS]
    E = sorted({int(k.split('_')[0][1:]) for k in T.files if k[0] == 'e'})
    res = {}
    for e in E:
        nll, se, wts = np.zeros((len(U), pool['n'])), np.zeros((len(U), pool['n'])), np.zeros((len(U), pool['n']))
        for i, u in enumerate(U):
            for ep in range(pool['n']):
                w = stack_weight(u[f'cf_e{e}_nll'][ep], T[f'cf_e{e}_nll'][ep])
                tq = T[f'e{e}_nll'][ep]
                p = pooled(w, u[f'e{e}_nll'][ep], tq)
                m = np.where(np.isfinite(tq), w * u[f'e{e}_mean'][ep] + (1 - w) * np.nan_to_num(T[f'e{e}_mean'][ep]), u[f'e{e}_mean'][ep])
                nll[i, ep], se[i, ep], wts[i, ep] = p.mean(), ((m - qy[ep]) ** 2).mean(), w
        res[f'e{e}_nll'], res[f'e{e}_se'], res[f'e{e}_w'] = nll.mean(0), se.mean(0), wts.mean(0)
        assert np.isfinite(res[f'e{e}_nll']).all()
    out = R / f'pool_{a.model}_tabpfn'; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'cells_{tag}.npz', **res)
    (out / 'train.json').write_text(json.dumps(dict(model='pool', components=[a.model + a.sfx, 'tabpfn_v2'], grid=[float(GRID[0]), float(GRID[-1]), len(GRID)],
                                                    rule='per-episode two-fold cross-fit stacking (protocol v11)'), indent=1))
    print(a.net, {f'e{e}': (round(float(res[f'e{e}_nll'].mean()), 4), 'w', round(float(res[f'e{e}_w'].mean()), 2)) for e in E})


if __name__ == '__main__':
    main()
