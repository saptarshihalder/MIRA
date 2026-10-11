"""Protocol v7 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V7.md), computed mechanically from saved cells.

python confirm7.py --root /content/drive/MyDrive/mira_v7
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

NETS = ('pems03', 'pems08', 'air_cn')
SEEDS = (1, 2, 3)
ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', default=None)
a = ap.parse_args()
R = Path(a.root)
KEYS = {n: np.array([f'{n}|{k.split("|")[0]}' for k in torch.load(R / 'panels' / f'real_{n}_s7071.pt', weights_only=False)['pool']['keys']])
        for n in NETS}


def paired(base, new, clusters, margin=.01, noninf=None):
    d = base - new
    if len(clusters) != len(d):
        raise ValueError('Episode and cluster counts differ.')
    d = np.array([d[clusters == k].mean() for k in sorted(set(clusters))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    passed = bool(lo > noninf) if noninf is not None else bool(m >= margin and lo > 0)
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), passed=passed)


def cells(net, run, e, sfx=''):
    z = np.load(R / 'runs_real' / net / run / f'cells_real_{net}_s7071{sfx}.npz')
    v = z[f'e{e}_nll']
    if 'undefined' in z.files:                                     # TabPFN: NaN exactly where flagged (amendment 3 of v6)
        assert np.isfinite(v[~z['undefined']]).all() and np.isnan(v[z['undefined']]).all()
        return v
    if not np.isfinite(v).all():
        raise ValueError(f'nonfinite cells: {net}/{run}{sfx}')
    return v


def model(m, e, sfx=''):
    return np.concatenate([np.mean([cells(n, f'{m}_s{s}_ft', e, sfx) for s in SEEDS], 0) for n in NETS])


def tab(e):
    return np.concatenate([cells(n, 'tabpfn_v2', e) for n in NETS])


keys = np.concatenate([KEYS[n] for n in NETS])
res = {}
for q, e in (('Q1', 3), ('Q2', 6)):
    t, u = tab(e), model('upt', e); ok = np.isfinite(t)
    res[f'{q}_upt_vs_tabpfn_e{e}'] = dict(paired(t[ok], u[ok], keys[ok]), excluded_undefined=int((~ok).sum()))
res['Q3_upt_vs_pfn_e0'] = paired(model('pfn', 0), model('upt', 0), keys)
res['Q4_upt64_noninf_upt_e0'] = paired(model('upt', 0), model('upt', 0, '_f64'), keys, noninf=-.01)
res['headline_Q1_Q2'] = res['Q1_upt_vs_tabpfn_e3']['passed'] and res['Q2_upt_vs_tabpfn_e6']['passed']
try:
    for e in (0, 3, 6):
        t, u = tab(e), model('upt_L', e); ok = np.isfinite(t)
        res[f'S1_uptL_vs_tabpfn_e{e}'] = paired(t[ok], u[ok], keys[ok])
    res['S2_uptL_vs_upt_e0'] = paired(model('upt', 0), model('upt_L', 0), keys)
except FileNotFoundError as err:
    res['S1_S2'] = f'not run ({err.filename})'
desc = {}
for n in NETS:
    for e in (0, 3, 6):
        row = {}
        for m, sfx in (('pfn', ''), ('upt', ''), ('upt', '_f64'), ('upt_L', '')):
            try:
                v = np.mean([cells(n, f'{m}_s{s}_ft', e, sfx) for s in SEEDS], 0)
                row[m + sfx] = float(v.mean()); row[f'{m}{sfx}_episodes_nll_gt10'] = int((v > 10).sum())
            except FileNotFoundError:
                row[m + sfx] = None
        t = cells(n, 'tabpfn_v2', e); row['tabpfn_v2'] = float(np.nanmean(t)); row['tabpfn_undefined'] = int(np.isnan(t).sum())
        refs = torch.load(R / 'panels' / f'real_{n}_s7071.pt', weights_only=False)['refs'][e]
        row.update({k: float(np.nanmean(v['nll'])) for k, v in refs.items()})
        desc[f'{n}|e{e}'] = row
res['descriptive'] = desc
out = Path(a.out or R / 'confirm_v7.json'); out.write_text(json.dumps(res, indent=1))
for k, v in res.items():
    if k[:2] in ('Q1', 'Q2', 'Q3', 'Q4', 'S1', 'S2') and isinstance(v, dict):
        print(f"{k:28s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:4d} -> {'PASS' if v['passed'] else 'FAIL'}")
    elif k.startswith('S1_S2'):
        print(k, v)
print('headline (Q1 and Q2 pass):', res['headline_Q1_Q2'])
for k, v in desc.items():
    print(k, {m: (round(x, 4) if isinstance(x, float) else x) for m, x in v.items()})
