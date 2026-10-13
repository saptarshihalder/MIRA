"""Protocol v14 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V14.md): TabPFN-3.5 with the UPT column on the four v10 networks.

python confirm14.py --v10 .../mira_v10 --out <json>
Statistics as v9-v13 (per-episode metrics, pooled over networks, paired 95% intervals cluster-robust over
network-windows; gain = baseline - selected variant). Episodes on which TabPFN-3.5 is undefined are excluded.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
R = Path(a.v10)
NETS = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')
sel = json.loads((R / 'dev' / 'selection14.json').read_text())['selection']
VAR = {'F1_uptcol': 'tabpfn_v3_5_uptcol', 'F2_uptcol_mirror': 'tabpfn_v3_5_uptcol_mirror'}[sel['selected']]
TAB, POOL = 'tabpfn_v3_5', 'pool_upt_L_sfm_tabpfn_v3_5'


def cells(net, run, e, metric='nll'):
    return np.load(R / 'runs_real' / net / run / f'cells_real_{net}_s10101.npz')[f'e{e}_{metric}'].astype(float)


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


def compare(base, nets, e, metric='nll', margin=.01):
    try:
        b, p = [], []
        for n in nets:
            ok = np.isfinite(cells(n, TAB, e))
            b.append(np.where(ok, cells(n, base, e, metric), np.nan)); p.append(np.where(ok, cells(n, VAR, e, metric), np.nan))
        return paired(np.concatenate(b), np.concatenate(p), keys(nets), margin)
    except FileNotFoundError as err:
        return dict(status=f'not run ({Path(err.filename).parent.name})')


res = dict(selected=sel, run=VAR)
res['R1_uptcol_vs_tabpfn35_nll_e0'] = compare(TAB, NETS, 0)
res['R2_uptcol_vs_tabpfn35_nll_e3'] = compare(TAB, NETS, 3)
res['R2_uptcol_vs_tabpfn35_nll_e6'] = compare(TAB, NETS, 6)
sec = {}
for e in (0, 3, 6):
    sec[f'uptcol_vs_pool35_nll_e{e}'] = compare(POOL, NETS, e)
    for m in ('se', 'crps'):
        sec[f'uptcol_vs_tabpfn35_{m}_e{e}'] = compare(TAB, NETS, e, m, 0.)
    for n in NETS:
        sec[f'{n}|uptcol_vs_tabpfn35_nll_e{e}'] = compare(TAB, (n,), e)
res['secondary'] = sec
desc = {}
for n in NETS:
    for e in (0, 3, 6):
        ok = np.isfinite(cells(n, TAB, e))
        row = {}
        for name, run in (('tabpfn35', TAB), ('uptcol', VAR), ('pool35', POOL)):
            try:
                row[name] = dict(nll=round(float(cells(n, run, e)[ok].mean()), 4), rmse=round(float(np.sqrt(cells(n, run, e, 'se')[ok].mean())), 4))
                if name != 'pool35':
                    row[name].update(crps=round(float(cells(n, run, e, 'crps')[ok].mean()), 4), cov90=round(float(cells(n, run, e, 'cov')[ok].mean()), 4))
            except (FileNotFoundError, KeyError):
                row[name] = None
        desc[f'{n}|e{e}'] = row
res['descriptive'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and ('gain' in v or 'status' in v):
        print(f"{k:40s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in desc.items():
    print(k, v)
