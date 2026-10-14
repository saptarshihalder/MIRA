"""Protocol v17 on Colab (docs/LIFTED_CAVITY_PROTOCOL_V17.md): UPT-SF on the three fresh v9-network panels, the pools,
then confirm17 (combined with v16's four networks).

    python run_v17.py --v9 /content/drive/MyDrive/mira_v9 --v10 /content/drive/MyDrive/mira_v10 upt     (UPT-SF outputs)
    python run_v17.py --v9 ... --v10 ... final   (TabPFN-3.5 outputs in from the repo, pools, confirm17)
Panels come from the repo (artifacts/v17_tabpfn35/panels/, built locally) and are checked against panel_digests.json.
"""
import argparse, json, shutil, sys
from pathlib import Path
from colab_v6 import sh, parallel

NETS = {'dwd_t2m_f': 'dwd_t2m', 'aurn_no2_f': 'aurn_no2', 'usgs_q_f': 'usgs_q'}
SEED, M = 17171, 'upt_L_sfm'
ART = Path(__file__).resolve().parents[2] / 'artifacts' / 'v17_tabpfn35'

ap = argparse.ArgumentParser(); ap.add_argument('--v9', required=True); ap.add_argument('--v10', required=True)
ap.add_argument('--par', type=int, default=8); ap.add_argument('stage', choices=('upt', 'final'))
a = ap.parse_args()
R, L = Path(a.v9), Path(a.v10) / 'logs'
import panel_digest
want = json.loads((ART / 'panel_digests.json').read_text())
for n in NETS:
    src, dst = ART / 'panels' / f'real_{n}_s{SEED}.pt', R / 'panels' / f'real_{n}_s{SEED}.pt'
    if not dst.exists():
        shutil.copy2(src, dst)
    got = panel_digest.digest(dst)
    assert got == want[n], (n, got, want[n])
    if a.stage == 'final':
        tab = R / 'runs_real' / n / 'tabpfn_v3_5'; tab.mkdir(parents=True, exist_ok=True)
        for f in (ART / n).iterdir():
            if not (tab / f.name).exists():
                shutil.copy2(f, tab / f.name)
print('panels verified' + ('; TabPFN-3.5 outputs in place' if a.stage == 'final' else ''), flush=True)
items = []
for n, base in NETS.items():
    panel = R / 'panels' / f'real_{n}_s{SEED}.pt'
    for s in (1, 2, 3):
        run = R / 'runs_real' / base / f'{M}_s{s}_ft'
        items.append((f'pq17_{n}_s{s}', lambda run=run, n=n: (run / f'pq_real_{n}_s{SEED}_tta8.npz').exists(),
                      [sys.executable, 'eval_real.py', 'score', '--panel', panel, '--runs', run, '--device', 'cuda', '--tta', 8, '--per-query'],
                      L / f'pq17_{n}_s{s}.log'))
        items.append((f'cfk17_{n}_s{s}', lambda run=run, n=n: (run / f'pqk4_real_{n}_s{SEED}_tta8.npz').exists(),
                      [sys.executable, 'eval_real.py', 'score', '--panel', panel, '--runs', run, '--device', 'cuda', '--tta', 8, '--cf-k', 4],
                      L / f'cfk17_{n}_s{s}.log'))
parallel(items, a.par)
if a.stage == 'upt':
    sys.exit(0)
parallel([(f'pool17_{n}', lambda n=n: (R / 'runs_real' / n / 'pool15_P3' / f'cells_real_{n}_s{SEED}.npz').exists(),
           [sys.executable, 'pool_eval15.py', '--root', R, '--net', n, '--panel-seed', SEED, '--tab', 'tabpfn_v3_5', '--upt-net', base,
            '--variants', 'v13ref,P3'], L / f'pool17_{n}.log') for n, base in NETS.items()], a.par)
sh([sys.executable, 'confirm17.py', '--v10', a.v10, '--v9', R, '--out', Path(a.v10) / 'confirm_v17.json'], L / 'confirm17.log')
print((L / 'confirm17.log').read_text()[-6000:])
