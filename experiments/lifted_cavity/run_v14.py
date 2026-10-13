"""Protocol v14 on Colab (docs/LIFTED_CAVITY_PROTOCOL_V14.md): TabPFN-3.5 with UPT-SF's prediction as an input column.

    python run_v14.py --v9 /content/drive/MyDrive/mira_v9 --v10 /content/drive/MyDrive/mira_v10 dev
    python run_v14.py --v9 ... --v10 ... confirm          (only if dev's selection rule allows it)
TABPFN_TOKEN must be in the environment (read from the Colab secret by the launching cell; never printed).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from colab_v6 import sh, parallel

N9, S9 = ('dwd_t2m', 'aurn_no2', 'usgs_q'), 9091
N10, S10 = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25'), 10101
M = 'upt_L_sfm'

ap = argparse.ArgumentParser(); ap.add_argument('--v9', required=True); ap.add_argument('--v10', required=True)
ap.add_argument('--par', type=int, default=6); ap.add_argument('stage', choices=('dev', 'confirm'))
a = ap.parse_args()
V9, V10 = Path(a.v9), Path(a.v10)
L = V10 / 'logs'
alone = lambda R, n, s: R / 'runs_real' / n / 'tabpfn_v3_5' / f'cells_real_{n}_s{s}.npz'
feat = lambda R, n, s, v='uptcol': R / 'runs_real' / n / f'tabpfn_v3_5_{v}' / f'cells_real_{n}_s{s}.npz'


def items(R, nets, seed, with_alone):
    out = []
    for n in nets:
        if with_alone:
            out.append((f'tabpfn35_{n}', lambda R=R, n=n: alone(R, n, seed).exists(),
                        [sys.executable, 'tabpfn_colab.py', '--panel', R / 'panels' / f'real_{n}_s{seed}.pt', '--out', R / 'runs_real' / n / 'tabpfn_v3_5',
                         '--version', 'v3.5', '--allow-constant-support', '--crps', '--device', 'cuda'], L / f'tabpfn35_{n}.log'))
        out.append((f'uptcol_{n}', lambda R=R, n=n: feat(R, n, seed).exists() and feat(R, n, seed, 'uptcol_mirror').exists(),
                    [sys.executable, 'tabpfn_feat.py', '--root', R, '--net', n, '--panel-seed', seed, '--version', 'v3.5', '--model', M],
                    L / f'uptcol_{n}.log'))
    return out


def nll(path, e):
    z = np.load(path)
    return z[f'e{e}_nll'].astype(float)


def upt_pq(R, nets, seed):
    """UPT-SF's per-query and cross-fit outputs (eval_real.py --per-query, as in v11), where they do not exist yet."""
    out = []
    for n in nets:
        for s in (1, 2, 3):
            run = R / 'runs_real' / n / f'{M}_s{s}_ft'
            out.append((f'pq_{n}_{M}_s{s}', lambda run=run, n=n: (run / f'pq_real_{n}_s{seed}_tta8.npz').exists(),
                        [sys.executable, 'eval_real.py', 'score', '--panel', R / 'panels' / f'real_{n}_s{seed}.pt', '--runs', run, '--device', 'cuda',
                         '--tta', 8, '--per-query'], L / f'pq_{n}_{M}_s{s}.log'))
    return out


if a.stage == 'dev':
    parallel(upt_pq(V9, N9, S9), a.par)
    parallel(items(V9, N9, S9, True), a.par)
    pools = [(f'pool35_{n}', lambda n=n: (V9 / 'runs_real' / n / f'pool_{M}_tabpfn_v3_5' / f'cells_real_{n}_s{S9}.npz').exists(),
              [sys.executable, 'pool_eval.py', '--root', V9, '--net', n, '--panel-seed', S9, '--model', M, '--sfx', '_tta8', '--tab', 'tabpfn_v3_5'],
              L / f'pool35_{n}.log') for n in N9]
    parallel(pools, a.par)
    rep = {}
    for e in (0, 3, 6):
        row = {}
        for name, f in (('tabpfn35', lambda n: alone(V9, n, S9)), ('F1_uptcol', lambda n: feat(V9, n, S9)),
                        ('F2_uptcol_mirror', lambda n: feat(V9, n, S9, 'uptcol_mirror')),
                        ('pool35', lambda n: V9 / 'runs_real' / n / f'pool_{M}_tabpfn_v3_5' / f'cells_real_{n}_s{S9}.npz')):
            vals = []
            for n in N9:
                ok = np.isfinite(nll(alone(V9, n, S9), e))
                v = nll(f(n), e)[ok]; vals.append(v); row[f'{name}|{n}'] = round(float(v.mean()), 4)
            row[name] = round(float(np.concatenate(vals).mean()), 4)
        rep[f'e{e}'] = row
    score = {v: np.mean([rep[f'e{e}'][v] for e in (0, 3, 6)]) for v in ('F1_uptcol', 'F2_uptcol_mirror')}
    sel = min(score, key=score.get)
    go = all(rep[f'e{e}'][sel] < rep[f'e{e}']['tabpfn35'] for e in (0, 3, 6))
    rep['selection'] = dict(selected=sel, mean_nll=score, go_to_confirmation=bool(go))
    (V10 / 'dev').mkdir(exist_ok=True)
    (V10 / 'dev' / 'selection14.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
else:
    sel = json.loads((V10 / 'dev' / 'selection14.json').read_text())['selection']
    assert sel['go_to_confirmation'], 'protocol v14: the development selection rule did not allow confirmation'
    parallel(items(V10, N10, S10, False), a.par)
    sh([sys.executable, 'confirm14.py', '--v10', V10, '--out', V10 / 'confirm_v14.json'], L / 'confirm14.log')
    print((L / 'confirm14.log').read_text()[-6000:])
