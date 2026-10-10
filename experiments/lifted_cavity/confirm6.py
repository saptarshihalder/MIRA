"""Protocol v6 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V6.md), computed mechanically from saved cells and predictions.

python confirm6.py --root /content/drive/MyDrive/mira_v6
Expects <root>/runs_real/<net>/<run>/cells_real_<net>_s6061.npz, <root>/preds/preds_real_<net>_s6061.npz (site_pool.py
predict for pfn_s1_ft and pfn_s2_ft) and <root>/runs/<model>_s<k>/cells_<P4 panel>.npz.
"""
import argparse, json
from pathlib import Path
import numpy as np
from site_pool import pooled

NETS = ('metr_la', 'pems_bay', 'intel')
SEEDS = (1, 2, 3)
P4 = 'v2_s20261603_n128_p16_nl0.4_sr48'
ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', default=None)
a = ap.parse_args()
R = Path(a.root)


def paired(base, new, clusters=None, margin=.01, noninf=None):
    d = base - new
    if clusters is not None:
        if len(clusters) != len(d):
            raise ValueError('Episode and cluster counts differ.')
        d = np.array([d[clusters == k].mean() for k in sorted(set(clusters))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    passed = bool(lo > noninf) if noninf is not None else bool(m >= margin and lo > 0)
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), passed=passed)


def cells(net, run, e=0, metric='nll'):
    z = np.load(R / 'runs_real' / net / run / f'cells_real_{net}_s6061.npz')
    v = z[f'e{e}_{metric}']
    if not np.isfinite(v).all():
        raise ValueError(f'nonfinite cells: {net}/{run}')
    return v


def seed_avg(net, model, e=0, metric='nll'):
    return np.mean([cells(net, f'{model}_s{s}_ft', e, metric) for s in SEEDS], 0)


def real(fn):
    """Concatenate a per-episode quantity over the networks, with network-day cluster keys."""
    vals, keys = [], []
    for net in NETS:
        P = np.load(R / 'preds' / f'preds_real_{net}_s6061.npz')
        v = fn(net, dict(P))
        vals.append(v); keys += [f'{net}|{w}' for w in P['weeks']]
    return np.concatenate(vals), np.array(keys)


res = {}
upt, days = real(lambda n, P: seed_avg(n, 'upt'))
pfn, _ = real(lambda n, P: seed_avg(n, 'pfn'))
pfn_pool, _ = real(lambda n, P: pooled(P, 'pfn_s1_ft', 'pfn_s2_ft', 0, 'lin')[0])
tab, _ = real(lambda n, P: cells(n, 'tabpfn_v2'))
res['P1_real_upt_vs_pfn'] = paired(pfn, upt, days)
res['P2_real_upt_noninf_pfn2pool'] = paired(pfn_pool, upt, days, noninf=-.01)
res['P3_real_upt_vs_tabpfn'] = paired(tab, upt, days)
syn = lambda m: np.mean([np.load(R / 'runs' / f'{m}_s{s}' / f'cells_{P4}.npz')['k2_nll'] for s in SEEDS], 0).mean(1)
res['P4_syn16_upt_vs_pfn'] = paired(syn('pfn'), syn('upt'))
res['headline_P1_P2_P4'] = all(res[k]['passed'] for k in ('P1_real_upt_vs_pfn', 'P2_real_upt_noninf_pfn2pool', 'P4_syn16_upt_vs_pfn'))

import torch
desc = {}
for net in NETS:
    refs = torch.load(R / 'panels' / f'real_{net}_s6061.pt', weights_only=False)['refs']
    for e in (0, 3, 6):
        row = {}
        for m in ('pfn', 'lct', 'upt', 'lift1'):
            row[m] = float(seed_avg(net, m, e).mean())
        for b in ('tabpfn_v2', 'lgbm', 'mice'):
            row[b] = float(cells(net, b, e).mean())
        row.update({k: float(np.nanmean(v['nll'])) for k, v in refs[e].items()})
        desc[f'{net}|e{e}'] = row
res['descriptive'] = desc
out = Path(a.out or R / 'confirm_v6.json'); out.write_text(json.dumps(res, indent=1))
for k, v in res.items():
    if k.startswith('P'):
        print(f"{k:30s} gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:5d} -> {'PASS' if v['passed'] else 'FAIL'}")
print('headline (P1, P2, P4 all pass):', res['headline_P1_P2_P4'])
for k, v in desc.items():
    print(k, {m: round(x, 4) for m, x in v.items()})
