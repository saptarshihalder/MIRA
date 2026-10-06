"""Protocol v3 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V3.md), computed mechanically from saved cells."""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs'); ap.add_argument('--cache', default='cache'); ap.add_argument('--real', default='runs_real')
ap.add_argument('--out', default='runs/confirm_v3.json')
a = ap.parse_args()
R, C, RR = Path(a.runs), Path(a.cache), Path(a.real)
G1, G3, BJ = 'v2_s20261301_n256_p5_nl0.4_sr48', 'v2_s20261303_n128_p16_nl0.4_sr48', 'real_beijing_no2_s3031'


def cells(name, seeds, tag, k=2):
    return np.mean([np.load(R / f'{name}_s{s}' / f'cells_{tag}.npz')[f'k{k}_nll'] for s in seeds], 0).mean(1)


def paired(base, new, clusters=None, margin=.01, noninf=None):
    d = base - new
    if clusters is not None:
        d = np.array([d[[c == k for c in clusters]].mean() for k in sorted(set(clusters))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    passed = bool(lo > noninf) if noninf is not None else bool(m >= margin and lo > 0)
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), passed=passed)


res = {}
res['E5_G1_k2_lct_vs_pfn'] = paired(cells('pfn', (1,), G1), cells('lct', (1,), G1))
res['E6_G3_k2_lct_vs_pfn'] = paired(cells('pfn', (1,), G3), cells('lct', (1,), G3))
bj = torch.load(C / f'{BJ}.pt', weights_only=False)
weeks = [k.split('|')[0] for k in bj['pool']['keys']]
ld = lambda nm: np.load(RR / 'beijing_no2' / nm / f'cells_{BJ}.npz')['e0_nll']
lct = ld('lct_s1_ft')
res['E7a_bjno2_lctft_vs_lift1ft'] = paired(np.mean([ld(f'lift1_s{s}_ft') for s in (1, 2, 3)], 0), lct, weeks)
res['E7b_bjno2_lctft_noninf_pfnft'] = paired(ld('pfn_s1_ft'), lct, weeks, noninf=-.02)
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(f"{k:34s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:4d} -> {'PASS' if v['passed'] else 'FAIL'}")
