"""Protocol v12 on Colab: per-query PFN-L and FA-Gaussian on the v10 panels, their pools with TabPFN, confirm12.

    python run_v12.py --root /content/drive/MyDrive/mira_v10
"""
import argparse, sys
from pathlib import Path
from colab_v6 import sh, parallel

NETS = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')
SEED = 10101

ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--par', type=int, default=4)
a = ap.parse_args()
R = Path(a.root); L = R / 'logs'
panel = lambda n: R / 'panels' / f'real_{n}_s{SEED}.pt'
items = []
for n in NETS:
    for s in (1, 2, 3):
        run = R / 'runs_real' / n / f'pfn_L_s{s}_ft'
        items.append((f'pq_{n}_pfn_L_s{s}', lambda run=run, n=n: (run / f'pq_real_{n}_s{SEED}_tta8.npz').exists(),
                      [sys.executable, 'eval_real.py', 'score', '--panel', panel(n), '--runs', run, '--device', 'cuda', '--tta', 8, '--per-query'],
                      L / f'pq_{n}_pfn_L_s{s}.log'))
    out = R / 'runs_real' / n / 'fa1_closed'
    items.append((f'pq_{n}_fa1', lambda out=out, n=n: (out / f'pq_real_{n}_s{SEED}.npz').exists(),
                  [sys.executable, 'pq_closed.py', '--panel', panel(n), '--out', out], L / f'pq_{n}_fa1.log'))
parallel(items, a.par)
pools = []
for n in NETS:
    for m, extra in (('pfn_L', ['--sfx', '_tta8']), ('fa1_closed', ['--sfx', '', '--single'])):
        pools.append((f'pool_{n}_{m}', lambda n=n, m=m: (R / 'runs_real' / n / f'pool_{m}_tabpfn' / f'cells_real_{n}_s{SEED}.npz').exists(),
                      [sys.executable, 'pool_eval.py', '--root', R, '--net', n, '--panel-seed', SEED, '--model', m] + extra,
                      L / f'pool_{n}_{m}.log'))
parallel(pools, a.par)
sh([sys.executable, 'confirm12.py', '--v10', R, '--out', R / 'confirm_v12.json'], L / 'confirm12.log')
print((L / 'confirm12.log').read_text()[-5000:])
