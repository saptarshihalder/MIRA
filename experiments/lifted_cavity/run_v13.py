"""Protocol v13 on Colab (CPU is enough): TabPFN-3.5's Modal outputs (committed under artifacts/v13_tabpfn35) into the
Drive runs, the pools with UPT-SF and PFN-L, then confirm13.

    python run_v13.py --root /content/drive/MyDrive/mira_v10
"""
import argparse, shutil, sys
from pathlib import Path
from colab_v6 import sh, parallel

NETS = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')
SEED = 10101
ART = Path(__file__).resolve().parents[2] / 'artifacts' / 'v13_tabpfn35'

ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--par', type=int, default=4)
a = ap.parse_args()
R = Path(a.root); L = R / 'logs'
for n in NETS:
    dst = R / 'runs_real' / n / 'tabpfn_v3_5'; dst.mkdir(parents=True, exist_ok=True)
    for f in (ART / n).iterdir():
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)
    assert (dst / f'cells_real_{n}_s{SEED}.npz').exists() and (dst / f'pq_real_{n}_s{SEED}.npz').exists(), n
pools = []
for n in NETS:
    for m in ('upt_L_sfm', 'pfn_L'):
        pools.append((f'pool35_{n}_{m}', lambda n=n, m=m: (R / 'runs_real' / n / f'pool_{m}_tabpfn_v3_5' / f'cells_real_{n}_s{SEED}.npz').exists(),
                      [sys.executable, 'pool_eval.py', '--root', R, '--net', n, '--panel-seed', SEED, '--model', m, '--sfx', '_tta8',
                       '--tab', 'tabpfn_v3_5'], L / f'pool35_{n}_{m}.log'))
parallel(pools, a.par)
sh([sys.executable, 'confirm13.py', '--v10', R, '--out', R / 'confirm_v13.json'], L / 'confirm13.log')
print((L / 'confirm13.log').read_text()[-6000:])
