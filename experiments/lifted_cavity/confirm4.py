"""Protocol v4 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V4.md), computed mechanically from saved cells.

python confirm4.py --runs runs --real runs_real --cache cache --out runs/confirm_v4.json
Learned v4 models are averaged over their three training seeds per task before any comparison.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs'); ap.add_argument('--cache', default='cache'); ap.add_argument('--real', default='runs_real')
ap.add_argument('--out', default='runs/confirm_v4.json')
a = ap.parse_args()
R, C, RR = Path(a.runs), Path(a.cache), Path(a.real)
H1, H3 = 'v2_s20261401_n256_p5_nl0.4_sr48', 'v2_s20261403_n128_p16_nl0.4_sr48'
NEW = ('beijing_pm10', 'beijing_so2', 'beijing_o3')
SEEDS = (1, 2, 3)


def syn(name, tag, k=2):
    if name == 'tabpfn_v2':
        return np.load(R / 'tabpfn_v2' / f'cells_{tag}.npz')[f'k{k}_nll'].mean(1)
    return np.mean([np.load(R / f'{name}_s{s}' / f'cells_{tag}.npz')[f'k{k}_nll'] for s in SEEDS], 0).mean(1)


def real(name, e):
    """Per-episode NLL pooled over the three new pollutants, with week keys for clustering."""
    vals, weeks = [], []
    for ds in NEW:
        tag = f'real_{ds}_s4041'
        if name == 'tabpfn_v2':
            v = np.load(R / 'tabpfn_v2' / f'cells_{tag}.npz')[f'e{e}_nll']
        else:
            v = np.mean([np.load(RR / ds / f'{name}_s{s}_ft' / f'cells_{tag}.npz')[f'e{e}_nll'] for s in SEEDS], 0)
        keys = torch.load(C / f'{tag}.pt', weights_only=False)['pool']['keys']
        vals.append(v); weeks += [k.split('|')[0] for k in keys]
    return np.concatenate(vals), weeks


def paired(base, new, clusters=None, margin=.01, noninf=None):
    d = base - new
    if clusters is not None:
        d = np.array([d[[c == k for c in clusters]].mean() for k in sorted(set(clusters))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    passed = bool(lo > noninf) if noninf is not None else bool(m >= margin and lo > 0)
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), passed=passed)


res = {}
res['E8_H1_k2_lctL_vs_pfnL'] = paired(syn('pfn_L', H1), syn('lct_L', H1))
res['E9_H3_k2_lctL_vs_pfnL'] = paired(syn('pfn_L', H3), syn('lct_L', H3))
lct0, wk = real('lct_L', 0); pfn0, _ = real('pfn_L', 0)
lct6, wk6 = real('lct_L', 6); pfn6, _ = real('pfn_L', 6)
res['E10_bjnew_nat_lctLft_noninf_pfnLft'] = paired(pfn0, lct0, wk, noninf=-.02)
res['E11_bjnew_plus6_lctLft_vs_pfnLft'] = paired(pfn6, lct6, wk6)
tab0, _ = real('tabpfn_v2', 0)
res['E12_bjnew_nat_lctLft_vs_tabpfn'] = paired(tab0, lct0, wk)
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(f"{k:40s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:4d} -> {'PASS' if v['passed'] else 'FAIL'}")
