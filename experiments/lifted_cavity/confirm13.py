"""Protocol v13 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V13.md): UPT-SF (+) TabPFN-3.5 on the four v10 networks.

python confirm13.py --v10 .../mira_v10 --out <json>
Statistics as v9-v12 (per-episode metrics, seed averages, pooled over networks, paired 95% intervals cluster-robust over
network-windows; gain = baseline - pool, positive favours the pool). Episodes on which TabPFN-3.5 is undefined are
excluded from comparisons with TabPFN-3.5 and from the partner check.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
R = Path(a.v10)
NETS = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')
M = json.loads((R / 'dev' / 'selection10.json').read_text())['model']
TAB, TAB2 = ('tabpfn_v3_5', ''), ('tabpfn_v2', '')
POOL, POOL_PFN, POOL_V2 = (f'pool_{M}_tabpfn_v3_5', ''), ('pool_pfn_L_tabpfn_v3_5', ''), (f'pool_{M}_tabpfn', '')
UPT = (M + '_s{}_ft', '_tta8')


def cells(net, spec, e, metric='nll'):
    pat, sfx = spec
    load = lambda run: np.load(R / 'runs_real' / net / run / f'cells_real_{net}_s10101{sfx}.npz')
    if '{}' in pat:
        return np.mean([load(pat.format(s))[f'e{e}_{metric}'] for s in (1, 2, 3)], 0)
    z = load(pat)
    v = z[f'e{e}_{metric}'].astype(float)
    if 'undefined' not in z.files and not np.isfinite(v).all():
        raise ValueError(f'nonfinite cells {net} {pat}')
    return v


def defined(net, e):
    return np.isfinite(cells(net, TAB, e))


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


def compare(base, new, nets, e, metric='nll', margin=.01, only_defined=False):
    """base vs new; TabPFN's undefined episodes drop out through its NaN cells, or explicitly with only_defined."""
    try:
        b, p = [], []
        for n in nets:
            ok = defined(n, e) if only_defined else True
            b.append(np.where(ok, cells(n, base, e, metric), np.nan)); p.append(np.where(ok, cells(n, new, e, metric), np.nan))
        return paired(np.concatenate(b), np.concatenate(p), keys(nets), margin)
    except FileNotFoundError as err:
        return dict(status=f'not run ({Path(err.filename).parent.name})')


res = dict(model=M, pool=POOL[0], foundation_model='TabPFN-3.5 (tabpfn 9.1.0, ModelVersion.V3_5, 8 estimators)')
res['Z1_pool_vs_tabpfn35_nll_e0'] = compare(TAB, POOL, NETS, 0)
res['Z2_pool_vs_tabpfn35_nll_e3'] = compare(TAB, POOL, NETS, 3)
res['Z2_pool_vs_tabpfn35_nll_e6'] = compare(TAB, POOL, NETS, 6)
res['Z3_pool_vs_uptsf_nll_e0'] = compare(UPT, POOL, NETS, 0)
sec = {}
for e in (0, 3, 6):
    if e:
        sec[f'pool_vs_uptsf_nll_e{e}'] = compare(UPT, POOL, NETS, e)
    sec[f'pool_vs_tabpfn35_se_e{e}'] = compare(TAB, POOL, NETS, e, 'se', 0.)
    sec[f'uptsfpool_vs_pfnLpool_nll_e{e}'] = compare(POOL_PFN, POOL, NETS, e, only_defined=True)
    for n in NETS:
        sec[f'{n}|pool_vs_tabpfn35_nll_e{e}'] = compare(TAB, POOL, (n,), e)
        sec[f'{n}|pool_vs_uptsf_nll_e{e}'] = compare(UPT, POOL, (n,), e)
res['secondary'] = sec
desc = {}
for n in NETS:
    for e in (0, 3, 6):
        row = {}
        try:
            ok = defined(n, e); row['tabpfn35_undefined'] = int((~ok).sum())
        except FileNotFoundError:
            ok = None
        for name, spec in (('tabpfn35', TAB), ('tabpfn_v2', TAB2), ('upt_sf', UPT), ('pool35', POOL), ('pool_v2', POOL_V2),
                           ('pfnL_pool35', POOL_PFN)):
            try:
                v, s = cells(n, spec, e), cells(n, spec, e, 'se')
                if ok is not None:
                    v, s = v[ok], s[ok]
                row[name] = dict(nll=round(float(np.nanmean(v)), 4), rmse=round(float(np.sqrt(np.nanmean(s))), 4))
                if name.startswith('tabpfn'):
                    c, cv = cells(n, spec, e, 'crps'), cells(n, spec, e, 'cov')
                    row[name].update(crps=round(float(np.nanmean(c if ok is None else c[ok])), 4),
                                     cov90=round(float(np.nanmean(cv if ok is None else cv[ok])), 4))
                if 'pool' in name:
                    w = cells(n, spec, e, 'w')
                    row[name]['w'] = round(float((w if ok is None else w[ok]).mean()), 3)
            except (FileNotFoundError, KeyError):
                row[name] = None
        desc[f'{n}|e{e}'] = row
res['descriptive_tabpfn35_defined_episodes'] = desc
try:
    res['tabpfn35_timing'] = {n: json.loads((R / 'runs_real' / n / 'tabpfn_v3_5' / 'train.json').read_text()).get('timing') for n in NETS}
except FileNotFoundError:
    pass
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and ('gain' in v or 'status' in v):
        print(f"{k:40s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in desc.items():
    print(k, v)
