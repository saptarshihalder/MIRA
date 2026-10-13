"""Protocol v14 on Colab: check the inputs exist, and that the TabPFN-3.5 license token works (booleans only).

    python check_inputs14.py --v9 /content/drive/MyDrive/mira_v9 --v10 /content/drive/MyDrive/mira_v10
"""
import argparse, os, subprocess
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--v9', required=True); ap.add_argument('--v10', required=True)
a = ap.parse_args()
for root, nets, seed in ((Path(a.v9), ('dwd_t2m', 'aurn_no2', 'usgs_q'), 9091), (Path(a.v10), ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25'), 10101)):
    for n in nets:
        R = root / 'runs_real' / n
        have = dict(panel=(root / 'panels' / f'real_{n}_s{seed}.pt').exists(),
                    upt_pq=sum((R / f'upt_L_sfm_s{s}_ft' / f'pq_real_{n}_s{seed}_tta8.npz').exists() for s in (1, 2, 3)),
                    pfnL_pq=sum((R / f'pfn_L_s{s}_ft' / f'pq_real_{n}_s{seed}_tta8.npz').exists() for s in (1, 2, 3)),
                    upt_cells=sum((R / f'upt_L_sfm_s{s}_ft' / f'cells_real_{n}_s{seed}_tta8.npz').exists() for s in (1, 2, 3)),
                    tabpfn35=(R / 'tabpfn_v3_5' / f'cells_real_{n}_s{seed}.npz').exists())
        print(n, have, flush=True)
tok = os.environ.get('TABPFN_TOKEN', '')
print('token in env:', len(tok) > 20)
if tok:
    from tabpfn import browser_auth as ba
    from tabpfn.settings import settings
    lic = ba._get_license_name('tabpfn_3_5')
    print('token valid:', ba.verify_token(tok, settings.tabpfn.auth_api_url), '| license accepted:',
          ba.check_license_accepted(tok, settings.tabpfn.auth_api_url, lic))
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(),
      '| cpus', os.cpu_count())
