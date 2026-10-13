"""Protocol v15: UPT-SF (+) TabPFN-3.5 with the stacking weight from better cross-fits (docs/LIFTED_CAVITY_PROTOCOL_V15.md).

    python pool_eval15.py --root <runs root> --net usgs_ny --panel-seed 10101 --tab tabpfn_v3_5 [--sfx-out _s2]

Variants (the pool is always p = w p_UPT + (1 - w) p_TabPFN per query; w per episode and condition, no test label):
  v13ref  2-fold cross-fit, grid G0 = {.02, ..., .98}, per-seed pools averaged over seeds (the v13 rule; must
          reproduce pool_upt_L_sfm_tabpfn_v3_5 exactly)
  P0      2-fold, grid G1, seed-ensemble partner                         (descriptive)
  P1      4-fold, G0, per seed
  P2      4-fold, G1, per seed
  P3      4-fold, G1, seed-ensemble partner
  P4      learning-curve extrapolation to 48 rows from the 2-fold (24 rows) and 4-fold (36 rows) cross-fits, G1,
          seed-ensemble partner
G1 = G0 + {.001, .003, .01} + {.99, .997, .999}. Seed-ensemble partner: the equal mixture of UPT-SF seeds 1-3 (log
density = logmeanexp of the seeds' log densities). Extrapolation, per support row and model, linear in 1/n:
l48 = l36 + 0.5 (l36 - l24), with the correction clipped to +-1 nat; where l24 is missing, l36 is used.
Where TabPFN is undefined (constant support target), the pool is the partner alone (w = 1), as in v11.
Outputs: runs_real/<net>/pool15_<variant><sfx-out>/cells_<tag>.npz with e{e}_nll, e{e}_se, e{e}_w (seed averages for
per-seed variants).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

SEEDS = (1, 2, 3)
G0 = np.linspace(.02, .98, 49)
G1 = np.unique(np.r_[[.001, .003, .01], G0, [.99, .997, .999]])
VARIANTS = {'v13ref': ('cf2', G0, False), 'P0': ('cf2', G1, True), 'P1': ('cf4', G0, False), 'P2': ('cf4', G1, False),
            'P3': ('cf4', G1, True), 'P4': ('cfx', G1, True)}


def load_k(d, stem):
    """pqk file, or the union of its parts (_o<i>s<k>), as a dict of arrays."""
    full = d / f'{stem}.npz'
    if full.exists():
        z = np.load(full); return {k: z[k] for k in z.files}
    parts = sorted(d.glob(f'{stem}_o*s*.npz'))
    if not parts:
        raise FileNotFoundError(full)
    out = None
    for p in parts:
        z = np.load(p)
        if out is None:
            out = {k: z[k].copy() for k in z.files}
        else:
            for k in z.files:
                ok = np.isfinite(z[k]); out[k][ok] = z[k][ok]
    return out


def logmeanexp(a, axis=0):
    m = np.nanmax(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.)
    return (np.log(np.mean(np.exp(a - m), axis=axis, keepdims=True)) + m).squeeze(axis)


def extrapolate(l24, l36):
    d = np.clip(0.5 * (l36 - l24), -1., 1.)
    return np.where(np.isfinite(d), l36 + d, l36)


def stack_weight(lu, lt, grid):
    ok = np.isfinite(lu) & np.isfinite(lt)
    if not ok.any():
        return 1.
    s = np.logaddexp(np.log(grid)[:, None] + lu[ok][None], np.log1p(-grid)[:, None] + lt[ok][None]).sum(1)
    return float(grid[int(np.argmax(s))])


def pooled_nll(w, lu, lt):
    """Per-query pooled NLL from log densities; queries where TabPFN is undefined keep the partner's."""
    out = -lu.copy()
    ok = np.isfinite(lt)
    if w < 1:
        out[ok] = -np.logaddexp(np.log(w) + lu[ok], np.log1p(-w) + lt[ok])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--net', required=True); ap.add_argument('--panel-seed', type=int, required=True)
    ap.add_argument('--tab', default='tabpfn_v3_5'); ap.add_argument('--model', default='upt_L_sfm'); ap.add_argument('--sfx', default='_tta8')
    ap.add_argument('--sfx-out', default=''); ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--upt-net', default=None, help='protocol v16: network folder of the UPT-SF runs (default: --net)')
    a = ap.parse_args()
    R = Path(a.root) / 'runs_real' / a.net
    tag = f'real_{a.net}_s{a.panel_seed}'
    qy = torch.load(Path(a.root) / 'panels' / f'{tag}.pt', weights_only=False)['pool']['qy'].double().numpy()
    T = np.load(R / a.tab / f'pq_{tag}.npz')
    E = sorted({int(k.split('_')[0][1:]) for k in T.files if k[0] == 'e'})
    need4 = any(VARIANTS[v][0] in ('cf4', 'cfx') for v in a.variants.split(','))
    T4 = load_k(R / a.tab, f'pqk4_{tag}') if need4 else None
    RU = Path(a.root) / 'runs_real' / (a.upt_net or a.net)
    U = [np.load(RU / f'{a.model}_s{s}_ft' / f'pq_{tag}{a.sfx}.npz') for s in SEEDS]
    U4 = [load_k(RU / f'{a.model}_s{s}_ft', f'pqk4_{tag}{a.sfx}') for s in SEEDS] if need4 else None
    n = qy.shape[0]
    for v in a.variants.split(','):
        cf, grid, ens = VARIANTS[v]
        res = {}
        for e in E:
            lt_test = -T[f'e{e}_nll']; mt = T[f'e{e}_mean']
            lt_cf = {'cf2': -T[f'cf_e{e}_nll'], 'cf4': -T4[f'cf_e{e}_nll'] if T4 else None}
            lt_cf['cfx'] = extrapolate(lt_cf['cf2'], lt_cf['cf4']) if T4 else None
            lu_test = np.stack([-u[f'e{e}_nll'] for u in U]); mu = np.stack([u[f'e{e}_mean'] for u in U])
            lu_cf = {'cf2': np.stack([-u[f'cf_e{e}_nll'] for u in U])}
            if U4:
                lu_cf['cf4'] = np.stack([-u[f'cf_e{e}_nll'] for u in U4])
            if ens:                                # one partner: the equal mixture of the seeds (mixed, then extrapolated)
                mix = {k: logmeanexp(x)[None] for k, x in lu_cf.items()}
                if 'cf4' in mix:
                    mix['cfx'] = extrapolate(mix['cf2'], mix['cf4'])
                lu_test_p, mu_p, lu_cf_p = logmeanexp(lu_test)[None], mu.mean(0)[None], mix[cf]
            else:
                if 'cf4' in lu_cf:
                    lu_cf['cfx'] = extrapolate(lu_cf['cf2'], lu_cf['cf4'])
                lu_test_p, mu_p, lu_cf_p = lu_test, mu, lu_cf[cf]
            S = len(lu_test_p)
            nll, se, wts = np.zeros((S, n)), np.zeros((S, n)), np.zeros((S, n))
            for i in range(S):
                for t in range(n):
                    w = stack_weight(lu_cf_p[i][t], lt_cf[cf][t], grid)
                    p = pooled_nll(w, lu_test_p[i][t], lt_test[t])
                    m = np.where(np.isfinite(lt_test[t]), w * mu_p[i][t] + (1 - w) * np.nan_to_num(mt[t]), mu_p[i][t])
                    nll[i, t], se[i, t], wts[i, t] = p.mean(), ((m - qy[t]) ** 2).mean(), w
            res[f'e{e}_nll'], res[f'e{e}_se'], res[f'e{e}_w'] = nll.mean(0), se.mean(0), wts.mean(0)
            assert np.isfinite(res[f'e{e}_nll']).all(), (v, e)
        out = R / f'pool15_{v}{a.sfx_out}'; out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f'cells_{tag}.npz', **res)
        (out / 'train.json').write_text(json.dumps(dict(model='pool', variant=v, crossfit=cf, grid=[float(x) for x in grid],
                                                        seed_ensemble=ens, partner=a.model + a.sfx, tabpfn=a.tab), indent=1))
        print(a.net, v, {f'e{e}': (round(float(np.nanmean(res[f'e{e}_nll'])), 4), 'w', round(float(res[f'e{e}_w'].mean()), 3)) for e in E}, flush=True)


if __name__ == '__main__':
    main()
