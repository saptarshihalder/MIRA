"""Protocol v8 baselines on Google Colab (docs/LIFTED_CAVITY_PROTOCOL_V8.md): neural processes and kriging.

    python colab_v8.py --root /content/drive/MyDrive/mira_v8 --v6 .../mira_v6 --v7 .../mira_v7 all
Stages: source (CNP/TNP s1-s3) finetune score krig synth confirm. Outputs for each network go into the v6 / v7 roots'
runs_real/<net>/ next to UPT, so every comparison reads one place. Finished items are skipped; training resumes.
"""
import argparse, json, sys, time
from pathlib import Path
from colab_v6 import CODE, sh, parallel, done_train

SEEDS = (1, 2, 3)
NPS = ('cnp', 'tnp')
RECIPE = ['--steps', '30000', '--batch-tasks', '16', '--queries', '16', '--lr', '1e-3', '--warmup', '1000',
          '--d', '64', '--layers', '4', '--heads', '4', '--ff', '128', '--device', 'cuda']          # = runs/pfn_s1


def nets(a):
    v6, v7 = Path(a.v6), Path(a.v7)
    return {'metr_la': (v6, 6061, 'data_v6'), 'pems_bay': (v6, 6061, 'data_v6'), 'intel': (v6, 6061, 'data_v6'),
            'pems03': (v7, 7071, 'data_v7'), 'pems08': (v7, 7071, 'data_v7'), 'air_cn': (v7, 7071, 'data_v7')}


def source(root, a):
    parallel([(f'train_{m}_s{s}', lambda m=m, s=s: (root / 'runs' / f'{m}_s{s}').exists() and done_train(root / 'runs' / f'{m}_s{s}'),
               [sys.executable, 'pfn_train.py', '--model', m, *RECIPE, '--seed', s, '--out', root / 'runs' / f'{m}_s{s}'],
               root / 'logs' / f'train_{m}_s{s}.log') for m in NPS for s in SEEDS], 6)


def finetune(root, a):
    items = []
    for n, (r, seed, data) in nets(a).items():
        for m in NPS:
            for s in SEEDS:
                out = r / 'runs_real' / n / f'{m}_s{s}_ft'
                items.append((f'ft_{n}_{m}_s{s}', lambda out=out: (out / 'model.pt').exists() and not (out / 'ckpt.pt').exists(),
                              [sys.executable, 'finetune_real.py', '--dataset', n, '--init', root / 'runs' / f'{m}_s{s}', '--out', out,
                               '--pool-cache', r / 'pools', '--v6', r / data, '--device', 'cuda', '--ckpt-every', '500'],
                              root / 'logs' / f'ft_{n}_{m}_s{s}.log'))
    parallel(items)


def score(root, a):
    items = []
    for n, (r, seed, data) in nets(a).items():
        for m in NPS:
            for s in SEEDS:
                run = r / 'runs_real' / n / f'{m}_s{s}_ft'
                items.append((f'score_{n}_{m}_s{s}', lambda run=run, n=n, seed=seed: (run / f'cells_real_{n}_s{seed}.npz').exists(),
                              [sys.executable, 'eval_real.py', 'score', '--panel', r / 'panels' / f'real_{n}_s{seed}.pt', '--runs', run,
                               '--device', 'cuda'], root / 'logs' / f'score_{n}_{m}_s{s}.log'))
    parallel(items)


def krig(root, a):
    items = []
    for n, (r, seed, data) in nets(a).items():
        for s in SEEDS:
            out = r / 'runs_real' / n / f'ignnk_s{s}'
            items.append((f'krig_{n}_s{s}', lambda out=out, n=n, seed=seed: (out / f'cells_real_{n}_s{seed}.npz').exists() and
                          (out.parent / f'ignnk_cal_s{out.name[-1]}' / f'cells_real_{n}_s{seed}.npz').exists(),
                          [sys.executable, 'kriging.py', '--dataset', n, '--data', r / data, '--panel', r / 'panels' / f'real_{n}_s{seed}.pt',
                           '--out', out, '--seed', s, '--device', 'cuda'], root / 'logs' / f'krig_{n}_s{s}.log'))
    parallel(items)


def synth(root, a):
    tag = 'v2_s20261603_n128_p16_nl0.4_sr48'
    parallel([(f'syn_{m}_s{s}', lambda m=m, s=s: (root / 'runs' / f'{m}_s{s}' / f'cells_{tag}.npz').exists(),
               [sys.executable, 'evaluate2.py', 'model', '--run', root / 'runs' / f'{m}_s{s}', '--panel', Path(a.v6) / 'panels' / f'{tag}.pt',
                '--device', 'cuda'], root / 'logs' / f'syn_{m}_s{s}.log') for m in NPS for s in SEEDS])


def confirm(root, a):
    sh([sys.executable, 'confirm8.py', '--v6', a.v6, '--v7', a.v7, '--out', root / 'confirm_v8.json'], root / 'logs' / 'confirm8.log')
    print((root / 'logs' / 'confirm8.log').read_text()[-6000:])


STAGES = dict(source=source, finetune=finetune, score=score, krig=krig, synth=synth, confirm=confirm)
ORDER = ['krig', 'source', 'synth', 'finetune', 'score', 'confirm']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--v6', required=True); ap.add_argument('--v7', required=True)
    ap.add_argument('stage', nargs='+', choices=list(STAGES) + ['all'])
    a = ap.parse_args()
    r = Path(a.root); r.mkdir(parents=True, exist_ok=True)
    for st in (ORDER if a.stage == ['all'] else a.stage):
        print(f'== {st}', time.strftime('%H:%M:%S'), flush=True)
        STAGES[st](r, a)
