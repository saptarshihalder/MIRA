"""Provisional per-network look at whatever v10/v11 cells already exist (descriptive only; the endpoints are confirm10.py
and confirm11.py once every network is done).

python provisional_v10.py --root /content/drive/MyDrive/mira_v10
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True)
a = ap.parse_args()
R = Path(a.root) / 'runs_real'
M = {'UPT-SF': ('upt_L_sfm_s{}_ft', '_tta8'), 'PFN-L-SF': ('pfn_L_sf_s{}_ft', '_tta8'), 'PFN-L': ('pfn_L_s{}_ft', '_tta8'),
     'TNP-D': ('tnp_s{}_ft', ''), 'Krig.cal': ('ignnk_cal_s{}', ''), 'TabPFN': ('tabpfn_v2', ''), 'POOL': ('pool_upt_L_sfm_tabpfn', '')}
for net in ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25'):
    tag = f'cells_real_{net}_s10101'
    print(f'== {net}')
    for e in (0, 3, 6):
        row = []
        for name, (pat, sfx) in M.items():
            files = [R / net / pat.format(s) / f'{tag}{sfx}.npz' for s in (1, 2, 3)] if '{}' in pat else [R / net / pat / f'{tag}{sfx}.npz']
            have = [f for f in files if f.exists()]
            if not have:
                row.append(f'{name} --'); continue
            v = np.mean([np.load(f)[f'e{e}_nll'] for f in have], 0)
            row.append(f'{name} {np.nanmean(v):+.3f}' + ('' if len(have) == len(files) else f'({len(have)}/{len(files)})'))
        print(f'  +{e}: ' + ' | '.join(row))
