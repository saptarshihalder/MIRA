"""Protocol v16 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V16.md): the frozen v15 pool (P3) vs TabPFN-3.5 on new targets
and days after TabPFN-3.5's release.

python confirm16.py --v10 .../mira_v10 --out <json>
Statistics as v9-v15 (per-episode metrics, paired 95% intervals cluster-robust over network-windows; gain = baseline -
pool). Episodes on which TabPFN-3.5 is undefined are excluded from comparisons with TabPFN-3.5. UPT-SF alone: the
per-episode mean of its per-query NLL (pq_ files), averaged over seeds 1-3, as its cells would be.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
R = Path(a.v10)
NETS = {'usgs_ny_f': 'usgs_ny', 'usgs_wa_f': 'usgs_wa', 'dwd_p_f': 'dwd_p', 'aurn_pm25_f': 'aurn_pm25'}
SEED, TAB, SEL, REF, NI = 16161, 'tabpfn_v3_5', 'pool15_P3', 'pool15_v13ref', -0.005


def cells(net, run, e, metric='nll'):
    if run == 'upt':
        return np.mean([np.load(R / 'runs_real' / NETS[net] / f'upt_L_sfm_s{s}_ft' / f'pq_real_{net}_s{SEED}_tta8.npz')[f'e{e}_nll'].mean(1)
                        for s in (1, 2, 3)], 0) if metric == 'nll' else None
    return np.load(R / 'runs_real' / net / run / f'cells_real_{net}_s{SEED}.npz')[f'e{e}_{metric}'].astype(float)


def keys(nets):
    out = []
    for n in nets:
        out += [f'{n}|{k.split("|")[0]}' for k in torch.load(R / 'panels' / f'real_{n}_s{SEED}.pt', weights_only=False)['pool']['keys']]
    return np.array(out)


def paired(base, new, clusters, margin=.01, noninf=None):
    ok = np.isfinite(base) & np.isfinite(new)
    d = (base - new)[ok]; c = clusters[ok]
    d = np.array([d[c == k].mean() for k in sorted(set(c))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    passed = bool(lo > noninf) if noninf is not None else bool(m >= margin and lo > 0)
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), episodes=int(ok.sum()), excluded=int((~ok).sum()), passed=passed)


def compare(base, new, nets, e, metric='nll', margin=.01, noninf=None, defined_only=True):
    b, p = [], []
    for n in nets:
        ok = np.isfinite(cells(n, TAB, e)) if defined_only else True
        b.append(np.where(ok, cells(n, base, e, metric), np.nan)); p.append(np.where(ok, cells(n, new, e, metric), np.nan))
    return paired(np.concatenate(b), np.concatenate(p), keys(nets), margin, noninf)


N = tuple(NETS)
res = {}
res['F1_pool_vs_tabpfn35_nll_e0'] = compare(TAB, SEL, N, 0)
res['F2_pool_vs_tabpfn35_nll_e3'] = compare(TAB, SEL, N, 3)
res['F2_pool_vs_tabpfn35_nll_e6'] = compare(TAB, SEL, N, 6)
sec = {}
for e in (0, 3, 6):
    sec[f'pool_vs_uptsf_nll_e{e}'] = compare('upt', SEL, N, e, defined_only=False)
    sec[f'pool_vs_v13rule_nll_e{e}'] = compare(REF, SEL, N, e, margin=0.)
    sec[f'pool_vs_tabpfn35_se_e{e}'] = compare(TAB, SEL, N, e, 'se', 0.)
    for n in N:
        sec[f'{n}|pool_vs_tabpfn35_nll_e{e}'] = compare(TAB, SEL, (n,), e)
        sec[f'{n}|noninf_vs_tabpfn35_e{e}'] = compare(TAB, SEL, (n,), e, noninf=NI)
res['secondary'] = sec
res['noninferior_every_network'] = all(sec[f'{n}|noninf_vs_tabpfn35_e{e}']['passed'] for n in N for e in (0, 3, 6))
desc = {}
for n in N:
    for e in (0, 3, 6):
        ok = np.isfinite(cells(n, TAB, e))
        desc[f'{n}|e{e}'] = dict(episodes=int(ok.size), tabpfn35_undefined=int((~ok).sum()),
                                 **{nm: round(float(cells(n, run, e)[ok].mean()), 4) for nm, run in
                                    (('tabpfn35', TAB), ('uptsf', 'upt'), ('pool_v13rule', REF), ('pool_P3', SEL))},
                                 w_P3=round(float(cells(n, SEL, e, 'w')[ok].mean()), 3))
res['descriptive'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and 'gain' in v:
        print(f"{k:40s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}")
print('non-inferior on every network:', res['noninferior_every_network'])
for k, v in desc.items():
    print(k, v)
