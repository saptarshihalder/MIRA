"""Protocol v12 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V12.md): is the lifted-site model what makes the pool with TabPFN
work? Same stacking rule, same TabPFN outputs, three partners: UPT-SF, PFN-L (matched-scale plain transformer, v9 recipe)
and the closed-form FA-Gaussian.

python confirm12.py --v10 .../mira_v10 --out <json>
Statistics as v9-v11; gain = baseline pool - UPT-SF pool (positive favours the UPT-SF pool). Episodes on which TabPFN is
undefined are excluded (there the pools are not pools).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
R = Path(a.v10)
NETS = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')
POOLS = {'uptsf': 'pool_upt_L_sfm_tabpfn', 'pfnL': 'pool_pfn_L_tabpfn', 'fa': 'pool_fa1_closed_tabpfn'}


def cells(net, run, e, metric='nll'):
    return np.load(R / 'runs_real' / net / run / f'cells_real_{net}_s10101.npz')[f'e{e}_{metric}'].astype(float)


def defined(net, e):
    return np.isfinite(cells(net, 'tabpfn_v2', e))


def keys(nets):
    out = []
    for n in nets:
        out += [f'{n}|{k.split("|")[0]}' for k in torch.load(R / 'panels' / f'real_{n}_s10101.pt', weights_only=False)['pool']['keys']]
    return np.array(out)


def paired(base, new, clusters, margin=.01):
    ok = np.isfinite(base) & np.isfinite(new)
    d = (base - new)[ok]; c = clusters[ok]
    d = np.array([d[c == k].mean() for k in sorted(set(c))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), episodes=int(ok.sum()), excluded=int((~ok).sum()),
                passed=bool(m >= margin and lo > 0))


def compare(base, new, nets, e, metric='nll', margin=.01):
    try:
        b, p = [], []
        for n in nets:
            ok = defined(n, e)
            b.append(np.where(ok, cells(n, base, e, metric), np.nan)); p.append(np.where(ok, cells(n, new, e, metric), np.nan))
        return paired(np.concatenate(b), np.concatenate(p), keys(nets), margin)
    except FileNotFoundError as err:
        return dict(status=f'not run ({Path(err.filename).parent.name})')


res = {}
for e in (0, 3, 6):
    res[f'Y1_uptsfpool_vs_pfnLpool_nll_e{e}'] = compare(POOLS['pfnL'], POOLS['uptsf'], NETS, e)
    res[f'Y2_uptsfpool_vs_fapool_nll_e{e}'] = compare(POOLS['fa'], POOLS['uptsf'], NETS, e)
sec = {}
for e in (0, 3, 6):
    for name in ('pfnL', 'fa'):
        sec[f'{name}pool_vs_tabpfn_nll_e{e}'] = compare('tabpfn_v2', POOLS[name], NETS, e)
    sec[f'uptsfpool_vs_pfnLpool_se_e{e}'] = compare(POOLS['pfnL'], POOLS['uptsf'], NETS, e, 'se', 0.)
    for n in NETS:
        sec[f'{n}|uptsfpool_vs_pfnLpool_nll_e{e}'] = compare(POOLS['pfnL'], POOLS['uptsf'], (n,), e)
        sec[f'{n}|uptsfpool_vs_fapool_nll_e{e}'] = compare(POOLS['fa'], POOLS['uptsf'], (n,), e)
res['secondary'] = sec
desc = {}
for n in NETS:
    for e in (0, 3, 6):
        ok = defined(n, e)
        row = {}
        for name, run in list(POOLS.items()) + [('tabpfn', 'tabpfn_v2')]:
            try:
                v = cells(n, run, e)[ok]
                row[name] = round(float(v.mean()), 4)
                if name != 'tabpfn':
                    row[name + '_w'] = round(float(cells(n, run, e, 'w')[ok].mean()), 3)
            except FileNotFoundError:
                row[name] = None
        desc[f'{n}|e{e}'] = row
res['descriptive_defined_episodes'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and ('gain' in v or 'status' in v):
        print(f"{k:40s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in desc.items():
    print(k, v)
