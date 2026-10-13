"""Protocol v16 on Colab (docs/LIFTED_CAVITY_PROTOCOL_V16.md): UPT-SF on the fresh panels, the pools, confirm16.

    python run_v16.py --v10 /content/drive/MyDrive/mira_v10 [--par 8]
1. The fresh panels: copied from the repo (artifacts/v16_tabpfn35/panels/, built locally) into <v10>/panels, after
   checking their content digests against artifacts/v16_tabpfn35/panel_digests.json.
2. TabPFN-3.5's outputs (Modal; artifacts/v16_tabpfn35/<net>/) into <v10>/runs_real/<net>/tabpfn_v3_5.
3. UPT-SF (v10's upt_L_sfm fine-tunes, seeds 1-3, tta 8): per-query outputs and the 4-fold cross-fit of each panel.
4. pool_eval15.py (variants v13ref and P3) and confirm16.py.
"""
import argparse, json, shutil, sys
from pathlib import Path
from colab_v6 import sh, parallel

NETS = {'usgs_ny_f': 'usgs_ny', 'usgs_wa_f': 'usgs_wa', 'dwd_p_f': 'dwd_p', 'aurn_pm25_f': 'aurn_pm25'}
SEED, M = 16161, 'upt_L_sfm'
ART = Path(__file__).resolve().parents[2] / 'artifacts' / 'v16_tabpfn35'

ap = argparse.ArgumentParser(); ap.add_argument('--v10', required=True); ap.add_argument('--par', type=int, default=8)
ap.add_argument('stage', choices=('upt', 'final'), help='upt: panels + UPT-SF outputs; final: TabPFN outputs in, pools, confirm16')
a = ap.parse_args()
R = Path(a.v10); L = R / 'logs'
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
        items.append((f'pq16_{n}_s{s}', lambda run=run, n=n: (run / f'pq_real_{n}_s{SEED}_tta8.npz').exists(),
                      [sys.executable, 'eval_real.py', 'score', '--panel', panel, '--runs', run, '--device', 'cuda', '--tta', 8, '--per-query'],
                      L / f'pq16_{n}_s{s}.log'))
        items.append((f'cfk16_{n}_s{s}', lambda run=run, n=n: (run / f'pqk4_real_{n}_s{SEED}_tta8.npz').exists(),
                      [sys.executable, 'eval_real.py', 'score', '--panel', panel, '--runs', run, '--device', 'cuda', '--tta', 8, '--cf-k', 4],
                      L / f'cfk16_{n}_s{s}.log'))
parallel(items, a.par)
if a.stage == 'upt':
    sys.exit(0)
parallel([(f'pool16_{n}', lambda n=n: (R / 'runs_real' / n / 'pool15_P3' / f'cells_real_{n}_s{SEED}.npz').exists(),
           [sys.executable, 'pool_eval15.py', '--root', R, '--net', n, '--panel-seed', SEED, '--tab', 'tabpfn_v3_5', '--upt-net', base,
            '--variants', 'v13ref,P3'], L / f'pool16_{n}.log') for n, base in NETS.items()], a.par)
sh([sys.executable, 'confirm16.py', '--v10', R, '--out', R / 'confirm_v16.json'], L / 'confirm16.log')
print((L / 'confirm16.log').read_text()[-6000:])
