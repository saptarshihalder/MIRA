"""Protocol v11 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V11.md): UPT (+) TabPFN on the four fresh v10 networks.

python confirm11.py --v10 .../mira_v10 --v9 .../mira_v9 --out <json>
Statistics exactly as confirm9/confirm10 (per-episode metrics, seed averages, pooled over networks, paired 95% intervals
cluster-robust over network-windows; gain = baseline - pool, positive favours the pool). The pool's cells are already
seed averages (pool_eval.py). Episodes on which TabPFN is undefined are excluded from comparisons with TabPFN.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--v10', required=True); ap.add_argument('--v9', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
V10, V9 = Path(a.v10), Path(a.v9)
sel = json.loads((V10 / 'dev' / 'selection10.json').read_text())
M = sel['model']
NETS10 = {n: (V10, 10101) for n in ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')}
NETS9 = {n: (V9, 9091) for n in ('dwd_t2m', 'aurn_no2', 'usgs_q')}
ALL = {**NETS10, **NETS9}
POOL, TAB, UPT = (f'pool_{M}_tabpfn', ''), ('tabpfn_v2', ''), (M + '_s{}_ft', '_tta8')


def cells(net, spec, e, metric='nll'):
    root, seed = ALL[net]
    pat, sfx = spec
    load = lambda run: np.load(root / 'runs_real' / net / run / f'cells_real_{net}_s{seed}{sfx}.npz')
    if '{}' in pat:
        return np.mean([load(pat.format(s))[f'e{e}_{metric}'] for s in (1, 2, 3)], 0)
    z = load(pat)
    v = z[f'e{e}_{metric}'].astype(float)
    if 'undefined' not in z.files and not np.isfinite(v).all():
        raise ValueError(f'nonfinite cells {net} {pat}')
    return v


def keys(nets):
    out = []
    for n in nets:
        root, seed = ALL[n]
        out += [f'{n}|{k.split("|")[0]}' for k in torch.load(root / 'panels' / f'real_{n}_s{seed}.pt', weights_only=False)['pool']['keys']]
    return np.array(out)


def paired(base, new, clusters, margin=.01):
    ok = np.isfinite(base) & np.isfinite(new)
    d = (base - new)[ok]; c = clusters[ok]
    d = np.array([d[c == k].mean() for k in sorted(set(c))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), episodes=int(ok.sum()), excluded=int((~ok).sum()),
                passed=bool(m >= margin and lo > 0))


def compare(base, nets, e, metric='nll', margin=.01):
    try:
        b = np.concatenate([cells(n, base, e, metric) for n in nets]); p = np.concatenate([cells(n, POOL, e, metric) for n in nets])
        return paired(b, p, keys(nets), margin)
    except FileNotFoundError as err:
        return dict(status=f'not run ({Path(err.filename).parent.name})')


N10 = tuple(NETS10)
res = dict(selection=sel, pool=POOL[0])
res['X1_pool_vs_tabpfn_nll_e0'] = compare(TAB, N10, 0)
res['X2_pool_vs_tabpfn_nll_e3'] = compare(TAB, N10, 3)
res['X2_pool_vs_tabpfn_nll_e6'] = compare(TAB, N10, 6)
res['X3_pool_vs_uptsf_nll_e0'] = compare(UPT, N10, 0)
sec = {}
for e in (0, 3, 6):
    sec[f'pool_vs_uptsf_nll_e{e}'] = compare(UPT, N10, e)
    sec[f'pool_vs_tabpfn_se_e{e}'] = compare(TAB, N10, e, 'se', 0.)
    for n in N10:
        sec[f'{n}|pool_vs_tabpfn_nll_e{e}'] = compare(TAB, (n,), e)
        sec[f'{n}|pool_vs_uptsf_nll_e{e}'] = compare(UPT, (n,), e)
res['secondary'] = sec
desc = {}
for n in ALL:                                                   # v9 networks: development (already used), descriptive
    for e in (0, 3, 6):
        row = {}
        for nm, spec in (('pool', POOL), ('upt_sf', UPT), ('tabpfn_v2', TAB)):
            try:
                row[nm] = dict(nll=float(np.nanmean(cells(n, spec, e))), rmse=float(np.sqrt(np.nanmean(cells(n, spec, e, 'se')))))
            except (FileNotFoundError, KeyError):
                row[nm] = None
        try:
            row['w_upt'] = float(cells(n, POOL, e, 'w').mean())
        except (FileNotFoundError, KeyError):
            row['w_upt'] = None
        desc[f'{n}|e{e}'] = row
res['descriptive'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and ('gain' in v or 'status' in v):
        print(f"{k:34s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in desc.items():
    print(k, {m: (None if x is None else (round(x['nll'], 3) if isinstance(x, dict) else round(x, 2))) for m, x in v.items()})
