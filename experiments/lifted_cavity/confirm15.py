"""Protocol v15 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V15.md): the selected learning-curve-aware pool vs TabPFN-3.5.

python confirm15.py --v10 .../mira_v10 --out <json>
Statistics as v9-v14 (per-episode metrics, paired 95% intervals cluster-robust over network-windows; gain = baseline -
selected pool). Episodes on which TabPFN-3.5 is undefined are excluded.
V1, V2: superiority, pooled (gain >= .01 and lower bound > 0). V3: non-inferiority on each network at each condition
(lower bound > -0.005), twelve tests, all must pass.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
R = Path(a.v10)
NETS = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')
sel = json.loads((R / 'dev' / 'selection15.json').read_text())['selection']
SEL = f"pool15_{sel['selected']}"
TAB, POOL13, REF = 'tabpfn_v3_5', 'pool_upt_L_sfm_tabpfn_v3_5', 'pool15_v13ref'
NI = -0.005


def cells(net, run, e, metric='nll'):
    return np.load(R / 'runs_real' / net / run / f'cells_real_{net}_s10101.npz')[f'e{e}_{metric}'].astype(float)


def keys(nets):
    out = []
    for n in nets:
        out += [f'{n}|{k.split("|")[0]}' for k in torch.load(R / 'panels' / f'real_{n}_s10101.pt', weights_only=False)['pool']['keys']]
    return np.array(out)


def paired(base, new, clusters, margin=.01, noninf=None):
    ok = np.isfinite(base) & np.isfinite(new)
    d = (base - new)[ok]; c = clusters[ok]
    d = np.array([d[c == k].mean() for k in sorted(set(c))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    passed = bool(lo > noninf) if noninf is not None else bool(m >= margin and lo > 0)
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), episodes=int(ok.sum()), excluded=int((~ok).sum()), passed=passed)


def compare(base, new, nets, e, metric='nll', margin=.01, noninf=None):
    b, p = [], []
    for n in nets:
        ok = np.isfinite(cells(n, TAB, e))
        b.append(np.where(ok, cells(n, base, e, metric), np.nan)); p.append(np.where(ok, cells(n, new, e, metric), np.nan))
    return paired(np.concatenate(b), np.concatenate(p), keys(nets), margin, noninf)


res = dict(selected=sel, run=SEL)
res['V1_sel_vs_tabpfn35_nll_e0'] = compare(TAB, SEL, NETS, 0)
res['V2_sel_vs_tabpfn35_nll_e3'] = compare(TAB, SEL, NETS, 3)
res['V2_sel_vs_tabpfn35_nll_e6'] = compare(TAB, SEL, NETS, 6)
v3 = {f'{n}|e{e}': compare(TAB, SEL, (n,), e, noninf=NI) for n in NETS for e in (0, 3, 6)}
res['V3_noninferior_every_network'] = dict(passed=all(v['passed'] for v in v3.values()), margin=NI, tests=v3)
sec = {}
for e in (0, 3, 6):
    sec[f'sel_vs_pool13_nll_e{e}'] = compare(POOL13, SEL, NETS, e, margin=0.)
    sec[f'sel_vs_tabpfn35_se_e{e}'] = compare(TAB, SEL, NETS, e, 'se', 0.)
    for n in NETS:
        sec[f'{n}|sel_vs_tabpfn35_nll_e{e}'] = compare(TAB, SEL, (n,), e)
res['secondary'] = sec
chk = {}
for n in NETS:
    for e in (0, 3, 6):
        chk[f'{n}|e{e}'] = float(np.nanmax(np.abs(cells(n, REF, e) - cells(n, POOL13, e))))
res['check_v13ref_equals_v13_pool_max_abs_diff'] = max(chk.values())
desc = {}
for n in NETS:
    for e in (0, 3, 6):
        ok = np.isfinite(cells(n, TAB, e))
        desc[f'{n}|e{e}'] = {nm: round(float(cells(n, run, e)[ok].mean()), 4) for nm, run in (('tabpfn35', TAB), ('pool13', POOL13), ('selected', SEL))}
        desc[f'{n}|e{e}']['w_selected'] = round(float(cells(n, SEL, e, 'w')[ok].mean()), 3)
res['descriptive'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()) + [(f'V3 {k}', v) for k, v in v3.items()]:
    if isinstance(v, dict) and 'gain' in v:
        print(f"{k:40s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}")
print('V3 all networks non-inferior:', res['V3_noninferior_every_network']['passed'])
print('v13ref reproduces the v13 pool, max |diff| =', res['check_v13ref_equals_v13_pool_max_abs_diff'])
for k, v in desc.items():
    print(k, v)
