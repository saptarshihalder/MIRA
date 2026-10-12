"""Exploratory look at the v10 development cells (v9 networks, already used): where does UPT-SF lose to TabPFN v2?

python dev_check10.py --v9 /content/drive/MyDrive/mira_v9 [--net usgs_q]
"""
import argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--v9', required=True); ap.add_argument('--net', default='usgs_q')
a = ap.parse_args()
R, n = a.v9, a.net
pool = torch.load(f'{R}/panels/real_{n}_s9091.pt', weights_only=False)['pool']
sd = pool['sy'].double().std(1).numpy()


def L(m, sfx='_tta8'):
    return np.mean([np.load(f'{R}/runs_real/{n}/{m}_s{s}_ft/cells_real_{n}_s9091{sfx}.npz')['e0_nll'] for s in (1, 2, 3)], 0)


M = {'F (sfm)': L('upt_L_sfm'), 'E (sf)': L('upt_L_sf'), 'UPT*': L('upt_L_crps'), 'PFN-L-SF': L('pfn_L_sf'), 'PFN-L': L('pfn_L')}
T = np.load(f'{R}/runs_real/{n}/tabpfn_v2/cells_real_{n}_s9091.npz')['e0_nll']; ok = np.isfinite(T)
print('undefined for TabPFN:', int((~ok).sum()), 'their support sd:', np.round(sd[~ok], 5))
for k, v in M.items():
    print(f'{k:10s} mean {v.mean():8.3f}  mean(defined) {v[ok].mean():8.3f}  median {np.median(v[ok]):7.3f}  max {v.max():9.1f}  '
          f'worst-5 sd {np.round(sd[np.argsort(-v)[:5]], 4)}')
print(f'{"TabPFN":10s} mean(defined) {T[ok].mean():8.3f}  median {np.median(T[ok]):7.3f}  max {T[ok].max():9.1f}')
for lo, hi in ((0, .02), (.02, .1), (.1, 9)):
    m = ok & (sd >= lo) & (sd < hi)
    print(f'sd [{lo}, {hi}) n={int(m.sum())}: ' + '  '.join(f'{k} {v[m].mean():.3f}' for k, v in M.items()) + f'  TabPFN {T[m].mean():.3f}'
          f'  F<TabPFN in {(M["F (sfm)"][m] < T[m]).mean():.2f} of episodes')
