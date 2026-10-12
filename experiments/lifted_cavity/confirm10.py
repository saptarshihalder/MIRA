"""Protocol v10 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V10.md) on the four fresh 2026 networks, plus descriptives on all
thirteen networks.

python confirm10.py --v10 .../mira_v10 --v9 .../mira_v9 --v6 .../mira_v6 --v7 .../mira_v7 --out <json>
Statistics exactly as confirm9.py: per-episode metrics, seed averages (seeds 1-3) for learned models, pooled over the
networks, paired 95% intervals cluster-robust over network-windows, gain = baseline - UPT-SF (positive favours UPT-SF);
standardized metrics divide by the episode's support-target standard deviation (clamped at .1, as in v9). Episodes on
which TabPFN is undefined are excluded from comparisons with TabPFN and counted.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
for k in ('v10', 'v9', 'v6', 'v7', 'out'):
    ap.add_argument(f'--{k}', required=True)
a = ap.parse_args()
V10, V9 = Path(a.v10), Path(a.v9)
NETS10 = {n: (V10, 10101) for n in ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')}
OLD = {**{n: (Path(a.v6), 6061) for n in ('metr_la', 'pems_bay', 'intel')}, **{n: (Path(a.v7), 7071) for n in ('pems03', 'pems08', 'air_cn')},
       **{n: (V9, 9091) for n in ('dwd_t2m', 'aurn_no2', 'usgs_q')}}
ALL = {**NETS10, **OLD}
SEEDS = (1, 2, 3)
sel = json.loads((V10 / 'dev' / 'selection10.json').read_text())
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
    if 'undefined' in z.files:
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


N10 = tuple(NETS10)
TAB = ('tabpfn_v2', '')
TNP, KRIG, UPTSTAR = ('tnp_s{}_ft', ''), ('ignnk_cal_s{}', ''), ('upt_L_crps_s{}_ft', '_tta8')
avail = []
for c in (('pfn_L_sf_s{}_ft', '_tta8'), ('pfn_L_sfn_s{}_ft', '_tta8')):
    try:
        avail.append((pooled(c, N10, 0).mean(), c))
    except FileNotFoundError:
        pass
PFNL = min(avail)[1] if avail else ('pfn_L_sf_s{}_ft', '_tta8')    # the stronger PFN-L-SF variant (conservative)
res = dict(selection=sel, pfnL_comparator=PFNL[0] + PFNL[1])
res['W1_sf_vs_tabpfn_nll_e0'] = compare(TAB, STAR, N10, 0)
res['W2_sf_vs_tabpfn_nll_e3'] = compare(TAB, STAR, N10, 3)
res['W2_sf_vs_tabpfn_nll_e6'] = compare(TAB, STAR, N10, 6)
res['W3_sf_vs_pfnLsf_nll_e0'] = compare(PFNL, STAR, N10, 0)
res['W4_sf_vs_tabpfn_crps_e0'] = compare(TAB, STAR, N10, 0, 'crps_std', 0.)
res['W5_sf_vs_tnp_nll_e0'] = compare(TNP, STAR, N10, 0)
res['W6_sf_vs_krigcal_nll_e0'] = compare(KRIG, STAR, N10, 0)
res['W7_sf_vs_uptstar_nll_e0'] = compare(UPTSTAR, STAR, N10, 0)
sec = {}
for e in (0, 3, 6):
    for name, b in (('tabpfn', TAB), ('pfnLsf', PFNL), ('tnp', TNP), ('krigcal', KRIG), ('uptstar', UPTSTAR)):
        sec[f'sf_vs_{name}_nll_e{e}'] = compare(b, STAR, N10, e)
        sec[f'sf_vs_{name}_crps_e{e}'] = compare(b, STAR, N10, e, 'crps_std', 0.)
        sec[f'sf_vs_{name}_se_e{e}'] = compare(b, STAR, N10, e, 'se_std', 0.)
for n in N10:
    for name, b in (('tabpfn', TAB), ('pfnLsf', PFNL), ('tnp', TNP), ('krigcal', KRIG), ('uptstar', UPTSTAR)):
        for e in (0, 3, 6):
            sec[f'{n}|sf_vs_{name}_nll_e{e}'] = compare(b, STAR, (n,), e)
        sec[f'{n}|sf_vs_{name}_crps_e0'] = compare(b, STAR, (n,), 0, 'crps_std', 0.)
res['secondary'] = sec
OLDN = tuple(OLD)                                                      # all thirteen networks, descriptive
for e in (0, 3, 6):
    res[f'all13_sf_vs_tabpfn_nll_e{e}'] = compare(TAB, STAR, N10 + OLDN, e)

METHODS = {'sf': STAR, 'upt_L_sf': ('upt_L_sf_s{}_ft', '_tta8'), 'upt_L_sfm': ('upt_L_sfm_s{}_ft', '_tta8'),
           'upt_star': UPTSTAR, 'upt_L': ('upt_L_s{}_ft', ''), 'pfn_L_sf': ('pfn_L_sf_s{}_ft', '_tta8'),
           'pfn_L_sfn': ('pfn_L_sfn_s{}_ft', '_tta8'), 'pfn_L': ('pfn_L_s{}_ft', '_tta8'), 'tnp': TNP, 'ignnk_cal': KRIG,
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
            desc[f'{n}|e{e}'] = row
    return desc


res['descriptive_v10'] = describe(N10)
res['descriptive_old'] = describe(OLDN)
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in list(res.items()) + list(sec.items()):
    if isinstance(v, dict) and ('gain' in v or 'status' in v):
        print(f"{k:36s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:3d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in res['descriptive_v10'].items():
    print(k, {m: (None if x is None else round(x['nll'], 3)) for m, x in v.items()})
