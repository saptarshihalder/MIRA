"""Protocol v8 endpoints (docs/LIFTED_CAVITY_PROTOCOL_V8.md) over the six real networks of v6 and v7.

python confirm8.py --v6 /content/drive/MyDrive/mira_v6 --v7 /content/drive/MyDrive/mira_v7 --out <json>
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--v6', required=True); ap.add_argument('--v7', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
NETS = {'metr_la': (Path(a.v6), 6061), 'pems_bay': (Path(a.v6), 6061), 'intel': (Path(a.v6), 6061),
        'pems03': (Path(a.v7), 7071), 'pems08': (Path(a.v7), 7071), 'air_cn': (Path(a.v7), 7071)}
SEEDS = (1, 2, 3)
METHODS = {'upt': 'upt_s{}_ft', 'pfn': 'pfn_s{}_ft', 'tnp': 'tnp_s{}_ft', 'cnp': 'cnp_s{}_ft', 'ignnk': 'ignnk_s{}', 'ignnk_cal': 'ignnk_cal_s{}'}


def cells(net, run, e, metric='nll'):
    root, seed = NETS[net]
    v = np.load(root / 'runs_real' / net / run / f'cells_real_{net}_s{seed}.npz')[f'e{e}_{metric}']
    if not np.isfinite(v).all():
        raise ValueError(f'nonfinite cells: {net}/{run}')
    return v


def method(m, net, e, metric='nll'):
    return np.mean([cells(net, METHODS[m].format(s), e, metric) for s in SEEDS], 0)


def keys(net):
    root, seed = NETS[net]
    return np.array([f'{net}|{k.split("|")[0]}' for k in torch.load(root / 'panels' / f'real_{net}_s{seed}.pt', weights_only=False)['pool']['keys']])


def paired(base, new, clusters, margin=.01):
    d = base - new
    d = np.array([d[clusters == k].mean() for k in sorted(set(clusters))])
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = m - 1.96 * se, m + 1.96 * se
    return dict(gain=float(m), lo=float(lo), hi=float(hi), n=int(len(d)), passed=bool(m >= margin and lo > 0))


K = {n: keys(n) for n in NETS}
allk = np.concatenate([K[n] for n in NETS])
res = {}
for e in (0, 3, 6):
    upt = np.concatenate([method('upt', n, e) for n in NETS])
    for rid, b in (('R1', 'tnp'), ('R2', 'cnp'), ('R3', 'ignnk_cal')):
        try:                                                     # an endpoint whose inputs are not finished yet is 'not run'
            res[f'{rid}_upt_vs_{b}_e{e}'] = paired(np.concatenate([method(b, n, e) for n in NETS]), upt, allk)
        except FileNotFoundError as err:
            res[f'{rid}_upt_vs_{b}_e{e}'] = dict(status=f'not run ({Path(err.filename).parent.name})')
desc = {}
for n in NETS:
    for e in (0, 3, 6):
        row = {}
        for m in METHODS:
            try:
                nll, se, crps, cov = (method(m, n, e, x) for x in ('nll', 'se', 'crps', 'cov'))
                row[m] = dict(nll=float(nll.mean()), rmse=float(np.sqrt(se.mean())), crps=float(crps.mean()), cov90=float(cov.mean()))
            except (FileNotFoundError, KeyError):
                row[m] = None
        desc[f'{n}|e{e}'] = row
res['descriptive'] = desc
Path(a.out).write_text(json.dumps(res, indent=1))
for k, v in res.items():
    if k[0] == 'R':
        print(f"{k:24s} " + (f"gain {v['gain']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']:4d} -> {'PASS' if v['passed'] else 'FAIL'}"
                             if 'gain' in v else v['status']))
for k, v in desc.items():
    print(k, {m: (None if x is None else round(x['nll'], 3)) for m, x in v.items()})
