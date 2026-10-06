"""Protocol v2 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V2.md), computed mechanically from saved cells.

python confirm2.py [--runs runs] [--cache cache] [--real runs_real] [--out runs/confirm_v2.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs'); ap.add_argument('--cache', default='cache'); ap.add_argument('--real', default='runs_real')
ap.add_argument('--out', default='runs/confirm_v2.json')
a = ap.parse_args()
R, C, RR = Path(a.runs), Path(a.cache), Path(a.real)
F1, F3 = 'v2_s20261201_n256_p5_nl0.4_sr48', 'v2_s20261203_n128_p16_nl0.4_sr48'
BJ = 'real_beijing_s2027'
MARGIN = .01


def model_cells(name, seeds, tag, k, metric='nll'):
    return np.mean([np.load(R / f'{name}_s{s}' / f'cells_{tag}.npz')[f'k{k}_{metric}'] for s in seeds], 0).mean(1)


def ref_cells(tag, name, k, metric='nll'):
    return torch.load(C / f'{tag}.pt', weights_only=False)['refs'][k][name][metric].mean(1)


def paired(base, new, clusters=None):
    d = base - new
    if clusters is not None:
        keys = sorted(set(clusters))
        d = np.array([d[[c == k for c in clusters]].mean() for k in keys])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(gain=float(m), lo=float(m - 1.96 * se), hi=float(m + 1.96 * se), n=int(len(d)),
                passed=bool(m >= MARGIN and m - 1.96 * se > 0))


res = {}
lift_f1 = model_cells('lift1', (1, 2, 3), F1, 2)
res['E1_F1_k2_lift1_vs_nlfa'] = paired(ref_cells(F1, 'nlfa', 2), lift_f1)
res['E2_F1_k2_lift1_vs_pfn'] = paired(model_cells('pfn', (1,), F1, 2), lift_f1)
lift_f3 = model_cells('lift1', (1, 2, 3), F3, 2)
res['E3a_F3_k2_lift1_vs_fa1'] = paired(ref_cells(F3, 'fa1', 2), lift_f3)
res['E3b_F3_k2_lift1_vs_pfn'] = paired(model_cells('pfn', (1,), F3, 2), lift_f3)
bj = torch.load(C / f'{BJ}.pt', weights_only=False)
weeks = [k.split('|')[0] for k in bj['pool']['keys']]
lift_bj = np.mean([np.load(RR / 'beijing' / f'lift1_s{s}_ft' / f'cells_{BJ}.npz')['e0_nll'] for s in (1, 2, 3)], 0)
res['E4a_beijing_nat_lift1ft_vs_blr'] = paired(bj['refs'][0]['blr_cc']['nll'], lift_bj, weeks)
res['E4b_beijing_nat_lift1ft_vs_pfnft'] = paired(np.load(RR / 'beijing' / 'pfn_s1_ft' / f'cells_{BJ}.npz')['e0_nll'], lift_bj, weeks)
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(f"{k:38s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:4d} -> {'PASS' if v['passed'] else 'FAIL'}")
