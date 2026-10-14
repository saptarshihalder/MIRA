"""Protocol v9 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V9.md) on the three 2026 networks, plus development descriptives.

python confirm9.py --v9 /content/drive/MyDrive/mira_v9 --v6 .../mira_v6 --v7 .../mira_v7 --out <json>
Gains are baseline - UPT* (positive favours UPT*), per episode, seed-averaged (seeds 1-3) for learned models, pooled over
the networks; paired 95% intervals are cluster-robust over network-windows. UPT* is the variant chosen by select9.py.
Standardized metrics divide by the episode's support-target standard deviation: CRPS / sigma, squared error / sigma^2.
Episodes on which TabPFN is undefined (constant support target) are excluded from comparisons with TabPFN and counted.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--v9', required=True); ap.add_argument('--v6', required=True); ap.add_argument('--v7', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
V9 = Path(a.v9)
NETS9 = {'dwd_t2m': (V9, 9091), 'aurn_no2': (V9, 9091), 'usgs_q': (V9, 9091)}
DEV = {'metr_la': (Path(a.v6), 6061), 'pems_bay': (Path(a.v6), 6061), 'intel': (Path(a.v6), 6061),
       'pems03': (Path(a.v7), 7071), 'pems08': (Path(a.v7), 7071), 'air_cn': (Path(a.v7), 7071)}
ALL = {**NETS9, **DEV}
SEEDS = (1, 2, 3)
sel = json.loads((V9 / 'dev' / 'selection.json').read_text())
STAR = (sel['model'] + '_s{}_ft', sel['cells_suffix'])
SIG = {}


def sigma(net):
    if net not in SIG:
        root, seed = ALL[net]
        SIG[net] = torch.load(root / 'panels' / f'real_{net}_s{seed}.pt', weights_only=False)['pool']['sy'].double().std(1).clamp_min(.1).numpy()
    return SIG[net]


def cells(net, run, sfx, e, metric):
    root, seed = ALL[net]
    z = np.load(root / 'runs_real' / net / run / f'cells_real_{net}_s{seed}{sfx}.npz')
    v = z[f'e{e}_{metric}'].astype(float)
    if 'undefined' in z.files:                                   # TabPFN: NaN exactly where flagged (v6 amendment 3)
        assert np.isfinite(v[~z['undefined']]).all() and np.isnan(v[z['undefined']]).all()
        return v
    if not np.isfinite(v).all():
        raise ValueError(f'nonfinite cells: {net}/{run}{sfx}')
    return v


def one(spec, net, e, metric):
    pat, sfx = spec
    base = 'crps' if metric == 'crps_std' else 'se' if metric == 'se_std' else metric
    v = np.mean([cells(net, pat.format(s), sfx, e, base) for s in SEEDS], 0) if '{}' in pat else cells(net, pat, sfx, e, base)
    return v / sigma(net) if metric == 'crps_std' else v / sigma(net) ** 2 if metric == 'se_std' else v


def pooled(spec, nets, e, metric='nll'):
    return np.concatenate([one(spec, n, e, metric) for n in nets])


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


def compare(base, new, nets, e, metric='nll', margin=.01):
    try:
        return paired(pooled(base, nets, e, metric), pooled(new, nets, e, metric), keys(nets), margin)
    except FileNotFoundError as err:
        return dict(status=f'not run ({Path(err.filename).parent.name}/{Path(err.filename).name})')


N9 = tuple(NETS9)
TAB = ('tabpfn_v2', '')
TNP, CNP, KRIG = ('tnp_s{}_ft', ''), ('cnp_s{}_ft', ''), ('ignnk_cal_s{}', '')
pfnl = [('pfn_L_s{}_ft', STAR[1])] + ([('pfn_L_crps_s{}_ft', STAR[1])] if sel['crps_objective'] else [])
avail = []
for c in pfnl:
    try:
        avail.append((pooled(c, N9, 0).mean(), c))
    except FileNotFoundError:
        pass
PFNL = min(avail)[1] if avail else pfnl[0]                       # the stronger PFN-L variant (conservative for UPT*)
res = dict(selection=sel, pfnL_comparator=PFNL[0] + PFNL[1])
res['V1_star_vs_tabpfn_nll_e0'] = compare(TAB, STAR, N9, 0)
res['V2_star_vs_tabpfn_nll_e3'] = compare(TAB, STAR, N9, 3)
res['V2_star_vs_tabpfn_nll_e6'] = compare(TAB, STAR, N9, 6)
res['V3_star_vs_pfnL_nll_e0'] = compare(PFNL, STAR, N9, 0)
res['V4_star_vs_tnp_nll_e0'] = compare(TNP, STAR, N9, 0)
res['V5_star_vs_krigcal_nll_e0'] = compare(KRIG, STAR, N9, 0)
res['V6_star_vs_tabpfn_crps_e0'] = compare(TAB, STAR, N9, 0, 'crps_std', 0.)
sec = {}
for e in (0, 3, 6):
    for name, b in (('tabpfn', TAB), ('pfnL', PFNL), ('tnp', TNP), ('cnp', CNP), ('krigcal', KRIG), ('upt', ('upt_s{}_ft', '')),
                    ('pfn', ('pfn_s{}_ft', ''))):
        sec[f'star_vs_{name}_nll_e{e}'] = compare(b, STAR, N9, e)
        sec[f'star_vs_{name}_crps_e{e}'] = compare(b, STAR, N9, e, 'crps_std', 0.)
        sec[f'star_vs_{name}_se_e{e}'] = compare(b, STAR, N9, e, 'se_std', 0.)
    sec[f'uptL_vs_tabpfn_nll_e{e}'] = compare(TAB, ('upt_L_s{}_ft', ''), N9, e)
for n in N9:
    for name, b in (('tabpfn', TAB), ('pfnL', PFNL), ('tnp', TNP), ('krigcal', KRIG)):
        sec[f'{n}|star_vs_{name}_nll_e0'] = compare(b, STAR, (n,), 0)
        sec[f'{n}|star_vs_{name}_crps_e0'] = compare(b, STAR, (n,), 0, 'crps_std', 0.)
res['secondary'] = sec

METHODS = {'star': STAR, 'upt_L': ('upt_L_s{}_ft', ''), 'upt_L_tta8': ('upt_L_s{}_ft', '_tta8'),
           'upt_L_crps': ('upt_L_crps_s{}_ft', ''), 'upt_L_crps_tta8': ('upt_L_crps_s{}_ft', '_tta8'),
           'pfn_L': ('pfn_L_s{}_ft', ''), 'pfn_L_tta8': ('pfn_L_s{}_ft', '_tta8'),
           'pfn_L_crps': ('pfn_L_crps_s{}_ft', ''), 'pfn_L_crps_tta8': ('pfn_L_crps_s{}_ft', '_tta8'),
           'upt': ('upt_s{}_ft', ''), 'pfn': ('pfn_s{}_ft', ''), 'tnp': TNP, 'cnp': CNP, 'ignnk': ('ignnk_s{}', ''), 'ignnk_cal': KRIG,
           'tabpfn_v2': TAB}


def describe(nets):
    desc = {}
    for n in nets:
        root, seed = ALL[n]
        refs = torch.load(root / 'panels' / f'real_{n}_s{seed}.pt', weights_only=False)['refs']
        for e in (0, 3, 6):
            row = {}
            for m, spec in METHODS.items():
                try:
                    nll = one(spec, n, e, 'nll'); se = one(spec, n, e, 'se'); ses = one(spec, n, e, 'se_std')
                    r = dict(nll=float(np.nanmean(nll)), rmse=float(np.sqrt(np.nanmean(se))), rmse_std=float(np.sqrt(np.nanmean(ses))))
                    try:
                        r.update(crps_std=float(np.nanmean(one(spec, n, e, 'crps_std'))), cov90=float(np.nanmean(one(spec, n, e, 'cov'))))
                    except KeyError:
                        pass
                    row[m] = r
                except (FileNotFoundError, KeyError):
                    row[m] = None
            for k, v in refs[e].items():
                row[k] = dict(nll=float(np.nanmean(v['nll'])))
                if np.isfinite(v['se']).any():
                    row[k].update(rmse=float(np.sqrt(np.nanmean(v['se']))), rmse_std=float(np.sqrt(np.nanmean(v['se'] / sigma(n) ** 2))),
                                  crps_std=float(np.nanmean(v['crps'] / sigma(n))), cov90=float(np.nanmean(v['cov'])))
            desc[f'{n}|e{e}'] = row
    return desc


res['descriptive_v9'] = describe(N9)
res['descriptive_dev'] = describe(tuple(DEV))
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and ('gain' in v or 'status' in v):
        print(f"{k:34s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in res['descriptive_v9'].items():
    print(k, {m: (None if x is None else round(x['nll'], 3)) for m, x in v.items()})
