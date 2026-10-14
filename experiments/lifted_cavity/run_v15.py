"""Protocol v15 on Colab (docs/LIFTED_CAVITY_PROTOCOL_V15.md).

    python run_v15.py --v9 /content/drive/MyDrive/mira_v9 --v10 /content/drive/MyDrive/mira_v10 inputs    4-fold cross-fits
    python run_v15.py --v9 ... --v10 ... dev        development pools on the v9 networks, then the selection
    python run_v15.py --v9 ... --v10 ... confirm    the selected pool and v13ref on v10, then confirm15.py
TABPFN_TOKEN must be in the environment for `inputs` (read from the Colab secret by the launching cell; never printed).
TabPFN-3.5's 4-fold cross-fit on v10 comes from Modal (artifacts/v15_tabpfn35_cf4/<net>/, copied in by `confirm`).
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from colab_v6 import sh, parallel

N9, S9 = ('dwd_t2m', 'aurn_no2', 'usgs_q'), 9091
N10, S10 = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25'), 10101
M, K = 'upt_L_sfm', 4
CANDIDATES = ('P1', 'P2', 'P3', 'P4')
ART = Path(__file__).resolve().parents[2] / 'artifacts' / 'v15_tabpfn35_cf4'

ap = argparse.ArgumentParser(); ap.add_argument('--v9', required=True); ap.add_argument('--v10', required=True)
ap.add_argument('--par', type=int, default=8); ap.add_argument('stage', choices=('inputs', 'dev', 'confirm'))
a = ap.parse_args()
V9, V10 = Path(a.v9), Path(a.v10)
L = V10 / 'logs'


def upt_cfk(R, nets, seed):
    out = []
    for n in nets:
        for s in (1, 2, 3):
            run = R / 'runs_real' / n / f'{M}_s{s}_ft'
            out.append((f'cfk_{n}_s{s}', lambda run=run, n=n: (run / f'pqk{K}_real_{n}_s{seed}_tta8.npz').exists(),
                        [sys.executable, 'eval_real.py', 'score', '--panel', R / 'panels' / f'real_{n}_s{seed}.pt', '--runs', run,
                         '--device', 'cuda', '--tta', 8, '--cf-k', K], L / f'cfk_{n}_s{s}.log'))
    return out


def tab_cfk_dev():
    out = []
    for n in N9:
        d = V9 / 'runs_real' / n / 'tabpfn_v3_5_s2'
        out.append((f'tabcfk_{n}', lambda d=d, n=n: (d / f'pqk{K}_real_{n}_s{S9}_o0s2.npz').exists(),
                    [sys.executable, 'tabpfn_cfk.py', '--panel', V9 / 'panels' / f'real_{n}_s{S9}.pt', '--out', d, '--version', 'v3.5',
                     '--k', K, '--stride', 2, '--offset', 0, '--device', 'cuda'], L / f'tabcfk_{n}.log'))
    return out


def cells(R, n, s, run):
    return np.load(R / 'runs_real' / n / run / f'cells_real_{n}_s{s}.npz')


if a.stage == 'inputs':
    parallel(tab_cfk_dev() + upt_cfk(V9, N9, S9) + upt_cfk(V10, N10, S10), a.par)
elif a.stage == 'dev':
    parallel(tab_cfk_dev() + upt_cfk(V9, N9, S9), a.par)
    parallel([(f'pool15dev_{n}', lambda n=n: all((V9 / 'runs_real' / n / f'pool15_{v}_s2' / f'cells_real_{n}_s{S9}.npz').exists()
                                                   for v in ('v13ref', 'P0') + CANDIDATES),
               [sys.executable, 'pool_eval15.py', '--root', V9, '--net', n, '--panel-seed', S9, '--tab', 'tabpfn_v3_5_s2', '--sfx-out', '_s2'],
               L / f'pool15dev_{n}.log') for n in N9], a.par)
    rep = {}
    for e in (0, 3, 6):
        row = {}
        for name, run in [('tabpfn35', 'tabpfn_v3_5_s2')] + [(v, f'pool15_{v}_s2') for v in ('v13ref', 'P0') + CANDIDATES]:
            vals = []
            for n in N9:
                ok = np.isfinite(cells(V9, n, S9, 'tabpfn_v3_5_s2')[f'e{e}_nll'].astype(float))   # scored, TabPFN-defined
                z = cells(V9, n, S9, run)
                v = z[f'e{e}_nll'].astype(float)[ok]; vals.append(v)
                row[f'{name}|{n}'] = round(float(v.mean()), 4)
                if name != 'tabpfn35':
                    row[f'{name}|{n}|w'] = round(float(z[f'e{e}_w'][ok].mean()), 3)
            row[name] = round(float(np.concatenate(vals).mean()), 4)
        rep[f'e{e}'] = row
    score = {v: float(np.mean([rep[f'e{e}'][v] for e in (0, 3, 6)])) for v in CANDIDATES}
    sel = min(score, key=score.get)
    go = all(rep[f'e{e}'][sel] < rep[f'e{e}']['tabpfn35'] for e in (0, 3, 6))
    rep['selection'] = dict(selected=sel, mean_nll=score, tabpfn35_mean_nll=float(np.mean([rep[f'e{e}']['tabpfn35'] for e in (0, 3, 6)])),
                            go_to_confirmation=bool(go))
    (V10 / 'dev').mkdir(exist_ok=True)
    (V10 / 'dev' / 'selection15.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
else:
    sel = json.loads((V10 / 'dev' / 'selection15.json').read_text())['selection']
    assert sel['go_to_confirmation'], 'protocol v15: the development selection rule did not allow confirmation'
    for n in N10:
        dst = V10 / 'runs_real' / n / 'tabpfn_v3_5'
        for f in (ART / n).iterdir():
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    parallel(upt_cfk(V10, N10, S10), a.par)
    parallel([(f'pool15_{n}', lambda n=n: all((V10 / 'runs_real' / n / f'pool15_{v}' / f'cells_real_{n}_s{S10}.npz').exists()
                                                for v in ('v13ref', sel['selected'])),
               [sys.executable, 'pool_eval15.py', '--root', V10, '--net', n, '--panel-seed', S10, '--tab', 'tabpfn_v3_5',
                '--variants', f"v13ref,{sel['selected']}"], L / f'pool15_{n}.log') for n in N10], a.par)
    sh([sys.executable, 'confirm15.py', '--v10', V10, '--out', V10 / 'confirm_v15.json'], L / 'confirm15.log')
    print((L / 'confirm15.log').read_text()[-6000:])
