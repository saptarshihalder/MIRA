"""Protocol v7 on Google Colab (docs/LIFTED_CAVITY_PROTOCOL_V7.md). Source checkpoints come from the v6 root.

    python colab_v7.py --root /content/drive/MyDrive/mira_v7 --v6 /content/drive/MyDrive/mira_v6 sourceL   # UPT-L s1-s3
    python colab_v7.py --root ... --v6 ... all              # data panels pools finetune score tabpfn confirm pack
    python colab_v7.py --root ... --v6 ... finetuneL scoreL confirm pack                     # after sourceL finished
Every finished item is skipped; training and fine-tuning resume from Drive checkpoints. Logs: <root>/logs.
"""
import argparse, json, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path
from colab_v6 import CODE, sh, parallel, done_train

NETS = ('pems03', 'pems08', 'air_cn')
SEED_PANEL = 7071
SMALL = ('pfn', 'upt')
SEEDS = (1, 2, 3)
HF = 'https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/'
ZIPS = {'pems03': 'traffic/pems03.zip', 'pems08': 'traffic/pems08.zip', 'air_quality': 'air/air_quality.zip'}
RECIPE_L = ['--steps', '40000', '--batch-tasks', '32', '--queries', '16', '--lr', '5e-4', '--warmup', '2000',
            '--d', '128', '--layers', '8', '--heads', '8', '--ff', '512', '--device', 'cuda', '--model', 'upt',
            '--prefetch', '4', '--ckpt-every', '500']                       # = protocol v4 PFN-L / LCT-L recipe


def panel(root, n):
    return root / 'panels' / f'real_{n}_s{SEED_PANEL}.pt'


def data(root, v6):
    d = root / 'data_v7'; d.mkdir(parents=True, exist_ok=True)
    for name, path in ZIPS.items():
        if not (d / name).exists():
            z = d / f'{name}.zip'
            urllib.request.urlretrieve(HF + path, z)
            zipfile.ZipFile(z).extractall(d / name)
    print('data ok', sorted(p.name for p in d.iterdir()))


def panels(root, v6):
    (root / 'panels').mkdir(parents=True, exist_ok=True)
    parallel([(f'panel_{n}', lambda n=n: panel(root, n).exists(),
               [sys.executable, 'eval_real.py', 'build', '--dataset', n, '--seed', SEED_PANEL, '--v6', root / 'data_v7',
                '--out', panel(root, n)], root / 'logs' / f'panel_{n}.log') for n in NETS])


def pools(root, v6):
    code = ('import argparse, json, sys, finetune_real as f; d = json.loads(sys.argv[1]); '
            'f.source_pool(argparse.Namespace(seed=11, episodes=4000, period=None, airq=None, beijing=None, gas=None, **d))')
    parallel([(f'pool_{n}', lambda n=n: (root / 'pools' / f'src_{n}_s11_n4000.pt').exists(),
               [sys.executable, '-c', code, json.dumps(dict(dataset=n, pool_cache=str(root / 'pools'), v6=str(root / 'data_v7')))],
               root / 'logs' / f'pool_{n}.log') for n in NETS])


def _ft_items(root, runs_dir, models):
    items = []
    for n in NETS:
        for m in models:
            for s in SEEDS:
                out = root / 'runs_real' / n / f'{m}_s{s}_ft'
                items.append((f'ft_{n}_{m}_s{s}', lambda out=out: (out / 'model.pt').exists() and not (out / 'ckpt.pt').exists(),
                              [sys.executable, 'finetune_real.py', '--dataset', n, '--init', runs_dir / f'{m}_s{s}', '--out', out,
                               '--pool-cache', root / 'pools', '--v6', root / 'data_v7', '--device', 'cuda', '--ckpt-every', '500'],
                              root / 'logs' / f'ft_{n}_{m}_s{s}.log'))
    return items


def finetune(root, v6):
    for m in SMALL:
        for s in SEEDS:
            assert done_train(v6 / 'runs' / f'{m}_s{s}'), f'missing v6 source checkpoint {m}_s{s}'
    parallel(_ft_items(root, v6 / 'runs', SMALL))


def _score_items(root, models, f64=()):
    items = []
    for n in NETS:
        for m in models:
            for s in SEEDS:
                run = root / 'runs_real' / n / f'{m}_s{s}_ft'
                for sfx in [''] + (['_f64'] if m in f64 else []):
                    items.append((f'score_{n}_{m}_s{s}{sfx}', lambda run=run, n=n, sfx=sfx: (run / f'cells_real_{n}_s{SEED_PANEL}{sfx}.npz').exists(),
                                  [sys.executable, 'eval_real.py', 'score', '--panel', panel(root, n), '--runs', run, '--device', 'cuda']
                                  + (['--site-float64'] if sfx else []), root / 'logs' / f'score_{n}_{m}_s{s}{sfx}.log'))
    return items


def score(root, v6):
    parallel(_score_items(root, SMALL, f64=('upt',)))


def tabpfn(root, v6):
    parallel([(f'tabpfn_{n}', lambda n=n: (root / 'runs_real' / n / 'tabpfn_v2' / f'cells_real_{n}_s{SEED_PANEL}.npz').exists(),
               [sys.executable, 'tabpfn_colab.py', '--panel', panel(root, n), '--out', root / 'runs_real' / n / 'tabpfn_v2',
                '--allow-constant-support'], root / 'logs' / f'tabpfn_{n}.log') for n in NETS], 1)


def sourceL(root, v6):
    parallel([(f'train_upt_L_s{s}', lambda s=s: (root / 'runs' / f'upt_L_s{s}').exists() and done_train(root / 'runs' / f'upt_L_s{s}', 40000),
               [sys.executable, 'lct_train.py', *RECIPE_L, '--seed', s, '--out', root / 'runs' / f'upt_L_s{s}'],
               root / 'logs' / f'train_upt_L_s{s}.log') for s in SEEDS], 3)


def finetuneL(root, v6):
    parallel(_ft_items(root, root / 'runs', ('upt_L',)), 3)


def scoreL(root, v6):
    parallel(_score_items(root, ('upt_L',)))


def confirm(root, v6):
    sh([sys.executable, 'confirm7.py', '--root', root], root / 'logs' / 'confirm7.log')
    print((root / 'logs' / 'confirm7.log').read_text()[-6000:])


def pack(root, v6):
    z = root / 'mira_v7_results.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(root.rglob('*')):
            if p.is_file() and (p.suffix in ('.json', '.log') or p.name.startswith('cells_') or (p.name == 'model.pt' and 'runs' in p.parts)):
                f.write(p, p.relative_to(root))
    print(z, z.stat().st_size)


STAGES = dict(data=data, panels=panels, pools=pools, finetune=finetune, score=score, tabpfn=tabpfn, confirm=confirm,
              pack=pack, sourceL=sourceL, finetuneL=finetuneL, scoreL=scoreL)
ORDER = ['data', 'panels', 'pools', 'finetune', 'score', 'tabpfn', 'confirm', 'pack']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--v6', required=True)
    ap.add_argument('stage', nargs='+', choices=list(STAGES) + ['all'])
    a = ap.parse_args()
    r, v6 = Path(a.root), Path(a.v6); r.mkdir(parents=True, exist_ok=True)
    for st in (ORDER if a.stage == ['all'] else a.stage):
        print(f'== {st}', time.strftime('%H:%M:%S'), flush=True)
        STAGES[st](r, v6)
