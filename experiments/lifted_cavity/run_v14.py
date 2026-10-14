"""Protocol v14 on Colab (docs/LIFTED_CAVITY_PROTOCOL_V14.md): TabPFN-3.5 combined with UPT-SF beyond a linear pool.

    python run_v14.py --v9 /content/drive/MyDrive/mira_v9 --v10 /content/drive/MyDrive/mira_v10 pqv      UPT-SF mean + variance
    python run_v14.py --v9 ... --v10 ... dev        variants on the v9 networks (every 2nd episode), then the selection
    python run_v14.py --v9 ... --v10 ... confirm    the selected variant on the v10 networks, then confirm14.py
TABPFN_TOKEN must be in the environment (read from the Colab secret by the launching cell; never printed).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from colab_v6 import sh, parallel

N9, S9 = ('dwd_t2m', 'aurn_no2', 'usgs_q'), 9091
N10, S10 = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25'), 10101
M = 'upt_L_sfm'
VARIANTS = ('uptcol', 'uptcol2', 'uptres', 'uptsite')
GROUPS = (('alone', 'uptsite'), ('uptcol',), ('uptcol2',), ('uptres',))     # one process each, per network
STRIDE = 2

ap = argparse.ArgumentParser(); ap.add_argument('--v9', required=True); ap.add_argument('--v10', required=True)
ap.add_argument('--par', type=int, default=8); ap.add_argument('stage', choices=('pqv', 'dev', 'confirm'))
a = ap.parse_args()
V9, V10 = Path(a.v9), Path(a.v10)
L = V10 / 'logs'


def cells(R, n, s, run):
    return R / 'runs_real' / n / run / f'cells_real_{n}_s{s}.npz'


def pqv_items(R, nets, seed):
    out = []
    for n in nets:
        for s in (1, 2, 3):
            run = R / 'runs_real' / n / f'{M}_s{s}_ft'
            out.append((f'pqv_{n}_s{s}', lambda run=run, n=n: (run / f'pqv_real_{n}_s{seed}_tta8.npz').exists(),
                        [sys.executable, 'eval_real.py', 'score', '--panel', R / 'panels' / f'real_{n}_s{seed}.pt', '--runs', run, '--device', 'cuda',
                         '--tta', 8, '--per-query', '--pq-var'], L / f'pqv_{n}_s{s}.log'))
    return out


def feat_items(R, nets, seed, groups, stride):
    out = []
    for n in nets:
        for g in groups:
            runs = [('tabpfn_v3_5' if v == 'alone' else f'tabpfn_v3_5_{v}') + (f'_s{stride}' if stride > 1 else '') for v in g]
            out.append((f'v14_{n}_{"+".join(g)}', lambda R=R, n=n, runs=runs: all(cells(R, n, seed, r).exists() for r in runs),
                        [sys.executable, 'tabpfn_feat.py', '--root', R, '--net', n, '--panel-seed', seed, '--variants', ','.join(g),
                         '--stride', stride, '--version', 'v3.5', '--model', M], L / f'v14_{n}_{"+".join(g)}_s{stride}.log'))
    return out


def nll(path, e):
    return np.load(path)[f'e{e}_nll'].astype(float)


if a.stage == 'pqv':
    parallel(pqv_items(V9, N9, S9) + pqv_items(V10, N10, S10), a.par)
elif a.stage == 'dev':
    parallel(pqv_items(V9, N9, S9), a.par)
    parallel(feat_items(V9, N9, S9, GROUPS, STRIDE), a.par)
    tab = f'tabpfn_v3_5_s{STRIDE}'
    parallel([(f'pool35dev_{n}', lambda n=n: cells(V9, n, S9, f'pool_{M}_{tab}').exists(),
               [sys.executable, 'pool_eval.py', '--root', V9, '--net', n, '--panel-seed', S9, '--model', M, '--sfx', '_tta8', '--tab', tab],
               L / f'pool35dev_{n}.log') for n in N9], a.par)
    rep = {}
    for e in (0, 3, 6):
        row = {}
        for name, run in [('tabpfn35', tab)] + [(v, f'tabpfn_v3_5_{v}_s{STRIDE}') for v in VARIANTS] + [('pool35', f'pool_{M}_{tab}')]:
            vals = []
            for n in N9:
                ok = np.isfinite(nll(cells(V9, n, S9, tab), e))          # scored (every 2nd) and TabPFN-defined episodes
                v = nll(cells(V9, n, S9, run), e)[ok]; vals.append(v); row[f'{name}|{n}'] = round(float(v.mean()), 4)
            row[name] = round(float(np.concatenate(vals).mean()), 4)
        rep[f'e{e}'] = row
    score = {v: float(np.mean([rep[f'e{e}'][v] for e in (0, 3, 6)])) for v in VARIANTS}
    sel = min(score, key=score.get)
    go = all(rep[f'e{e}'][sel] < rep[f'e{e}']['tabpfn35'] for e in (0, 3, 6))
    rep['selection'] = dict(selected=sel, mean_nll=score, tabpfn35_mean_nll=float(np.mean([rep[f'e{e}']['tabpfn35'] for e in (0, 3, 6)])),
                            go_to_confirmation=bool(go))
    (V10 / 'dev').mkdir(exist_ok=True)
    (V10 / 'dev' / 'selection14.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
else:
    sel = json.loads((V10 / 'dev' / 'selection14.json').read_text())['selection']
    assert sel['go_to_confirmation'], 'protocol v14: the development selection rule did not allow confirmation'
    parallel(pqv_items(V10, N10, S10), a.par)
    parallel(feat_items(V10, N10, S10, ((sel['selected'],),), 1), a.par)
    sh([sys.executable, 'confirm14.py', '--v10', V10, '--out', V10 / 'confirm_v14.json'], L / 'confirm14.log')
    print((L / 'confirm14.log').read_text()[-6000:])
