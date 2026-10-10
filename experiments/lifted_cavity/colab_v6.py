"""Protocol v6 GPU work on Google Colab (docs/LIFTED_CAVITY_PROTOCOL_V6.md, docs/OCT20_PLAN.md).

    python colab_v6.py --root /content/drive/MyDrive/mira_v6 source      # batch 1: UPT s1-s3, LCT s3, PFN s3
    python colab_v6.py --root /content/drive/MyDrive/mira_v6 all         # batch 2: data .. confirm, in order
    python colab_v6.py --root /content/drive/MyDrive/mira_v6 <stage>     # data panels pools finetune score baselines
                                                                         # preds confirm pack status
Re-run the same command after any disconnect: every finished item is skipped (its output exists), and training and
fine-tuning resume from their Drive checkpoints. Logs: <root>/logs/<item>.log.
"""
import argparse, json, shutil, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parent
REPO_RUNS = CODE.parents[1] / 'artifacts' / 'reports' / 'lifted_cavity_paper' / 'runs'
RECIPE = ['--steps', '30000', '--batch-tasks', '16', '--queries', '16', '--lr', '1e-3', '--warmup', '1000',
          '--d', '64', '--layers', '4', '--heads', '4', '--ff', '128', '--device', 'cuda']   # = runs/lct_s1, runs/pfn_s1
JOBS = [('upt_s1', 'lct_train.py', ['--model', 'upt', '--seed', '1', '--prefetch', '4']),
        ('upt_s2', 'lct_train.py', ['--model', 'upt', '--seed', '2', '--prefetch', '4']),
        ('upt_s3', 'lct_train.py', ['--model', 'upt', '--seed', '3', '--prefetch', '4']),
        ('lct_s3', 'lct_train.py', ['--model', 'lct', '--seed', '3', '--prefetch', '4']),
        ('pfn_s3', 'pfn_train.py', ['--seed', '3'])]
NETS = ('metr_la', 'pems_bay', 'intel')
MODELS = ('pfn', 'lct', 'upt', 'lift1')
SEEDS = (1, 2, 3)
SEED_PANEL = 6061
SYN = [('20261603', '128', '16'), ('20261601', '256', '5')]          # P4 panel; 5-sensor ablation panel
HF = 'https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/'
INTEL = 'https://db.csail.mit.edu/labdata/'
PAR = 6


def sh(cmd, log, cwd=CODE):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'a') as f:
        f.write(f'\n$ {" ".join(map(str, cmd))}\n'); f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f'{log.name} failed ({r.returncode}); see {log}')


def parallel(items, n=PAR):
    """items: (name, done_fn, cmd, log). Runs the unfinished ones, n at a time; every failure is reported at the end."""
    todo = [it for it in items if not it[1]()]
    print(f'{len(items) - len(todo)} done, {len(todo)} to run', flush=True)
    errs = []

    def go(it):
        name, _, cmd, log = it
        try:
            t = time.time(); sh(cmd, log); print('ok', name, f'{time.time() - t:.0f}s', flush=True)
        except Exception as e:                                   # keep going; report all failures at the end
            errs.append(str(e)); print('FAILED', name, e, flush=True)
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(go, todo))
    if errs:
        raise RuntimeError(f'{len(errs)} item(s) failed: ' + '; '.join(errs))


def done_train(out, steps=30000):
    return (out / 'model.pt').exists() and not (out / 'ckpt.pt').exists() and \
        json.loads((out / 'train.json').read_text()).get('steps_done', steps) == steps


# ------------------------------------------------------------------ batch 1
def source(root):
    procs = []
    for name, script, extra in JOBS:
        out = root / 'runs' / name
        if out.exists() and done_train(out):
            print('done', name); continue
        out.mkdir(parents=True, exist_ok=True)
        log = open(root / f'{name}.log', 'a')
        procs.append((name, subprocess.Popen([sys.executable, script, *RECIPE, *extra, '--out', str(out)], cwd=CODE,
                                             stdout=log, stderr=subprocess.STDOUT)))
        print('started', name, flush=True)
    t0 = time.time()
    while procs:
        time.sleep(60)
        for name, p in list(procs):
            if p.poll() is not None:
                print(name, 'exit', p.returncode, f'{(time.time() - t0) / 60:.0f} min', flush=True); procs.remove((name, p))
        status(root, quiet=True)
    status(root)


def status(root, quiet=False):
    rows = []
    for name, *_ in JOBS:
        f = root / 'runs' / name / 'train.json'
        m = json.loads(f.read_text()) if f.exists() else {}
        rows.append(f"{name}: {m.get('steps_done', 0)}/30000 ema {m['trace'][-1]['ema_loss']:.4f}" if m.get('trace') else f'{name}: -')
    print(' | '.join(rows), flush=True)
    if not quiet:
        for net in NETS:
            d = root / 'runs_real' / net
            if d.exists():
                print(net, sorted(p.name for p in d.iterdir() if any(p.glob('cells_*.npz'))))


# ------------------------------------------------------------------ batch 2
def data(root):
    import numpy as np, pandas as pd, urllib.request
    d = root / 'data_v6'; d.mkdir(parents=True, exist_ok=True)
    for name in ('metr_la', 'pems_bay'):
        if not (d / f'{name}.npz').exists():
            z = d / f'{name}.zip'
            urllib.request.urlretrieve(HF + f'{name}.zip', z)
            zipfile.ZipFile(z).extractall(d / name)
            df = pd.read_hdf(d / name / f'{name}.h5')
            np.savez_compressed(d / f'{name}.npz', X=df.values.astype('float32'), ts=df.index.values.astype('datetime64[s]'),
                                ids=np.array([str(c) for c in df.columns]))
    for f in ('data.txt.gz', 'mote_locs.txt'):
        dst = d / ('intel_data.txt.gz' if f == 'data.txt.gz' else f)
        if not dst.exists():
            urllib.request.urlretrieve(INTEL + f, dst)
    print('data ok', sorted(p.name for p in d.iterdir()))


def panels(root):
    P, L = root / 'panels', root / 'logs'
    P.mkdir(parents=True, exist_ok=True)
    items = [(f'panel_{n}', lambda n=n: (P / f'real_{n}_s{SEED_PANEL}.pt').exists(),
              [sys.executable, 'eval_real.py', 'build', '--dataset', n, '--seed', SEED_PANEL, '--v6', root / 'data_v6',
               '--out', P / f'real_{n}_s{SEED_PANEL}.pt'], L / f'panel_{n}.log') for n in NETS]
    items += [(f'syn_{s}', lambda s=s, t=t, p=p: (P / f'v2_s{s}_n{t}_p{p}_nl0.4_sr48.pt').exists(),
               [sys.executable, 'evaluate2.py', 'panel', '--seed', s, '--tasks', t, '--sensors', p, '--cache', P],
               L / f'syn_{s}.log') for s, t, p in SYN]
    parallel(items)


def pools(root):
    """Source pools once per network (fine-tuning reuses them via --pool-cache; the pool depends only on its key)."""
    code = ('import argparse, json, sys, finetune_real as f; d = json.loads(sys.argv[1]); '
            'f.source_pool(argparse.Namespace(seed=11, episodes=4000, period=None, airq=None, beijing=None, gas=None, **d))')
    items = []
    for n in NETS:
        key = root / 'pools' / f'src_{n}_s11_n4000.pt'
        args = json.dumps(dict(dataset=n, pool_cache=str(root / 'pools'), v6=str(root / 'data_v6')))
        items.append((f'pool_{n}', lambda key=key: key.exists(), [sys.executable, '-c', code, args], root / 'logs' / f'pool_{n}.log'))
    parallel(items)


def seed_runs(root):
    """Source checkpoints: v2/v3 seeds from the repository, the new seeds from batch 1."""
    for m in MODELS:
        for s in SEEDS:
            dst = root / 'runs' / f'{m}_s{s}'
            if not (dst / 'model.pt').exists():
                src = REPO_RUNS / f'{m}_s{s}'
                assert (src / 'model.pt').exists(), f'missing source checkpoint {m}_s{s}'
                dst.mkdir(parents=True, exist_ok=True)
                for f in ('model.pt', 'train.json'):
                    shutil.copy2(src / f, dst / f)


def finetune(root):
    seed_runs(root)
    items = []
    for n in NETS:
        for m in MODELS:
            for s in SEEDS:
                out = root / 'runs_real' / n / f'{m}_s{s}_ft'
                items.append((f'ft_{n}_{m}_s{s}', lambda out=out: (out / 'model.pt').exists() and not (out / 'ckpt.pt').exists(),
                              [sys.executable, 'finetune_real.py', '--dataset', n, '--init', root / 'runs' / f'{m}_s{s}', '--out', out,
                               '--pool-cache', root / 'pools', '--v6', root / 'data_v6', '--device', 'cuda', '--ckpt-every', '500'],
                              root / 'logs' / f'ft_{n}_{m}_s{s}.log'))
    parallel(items)


def score(root):
    items = []
    for n in NETS:
        panel = root / 'panels' / f'real_{n}_s{SEED_PANEL}.pt'
        for m in MODELS:
            for s in SEEDS:
                run = root / 'runs_real' / n / f'{m}_s{s}_ft'
                items.append((f'score_{n}_{m}_s{s}', lambda run=run, n=n: (run / f'cells_real_{n}_s{SEED_PANEL}.npz').exists(),
                              [sys.executable, 'eval_real.py', 'score', '--panel', panel, '--runs', run, '--device', 'cuda'],
                              root / 'logs' / f'score_{n}_{m}_s{s}.log'))
    for s_, t, p in SYN:
        tag = f'v2_s{s_}_n{t}_p{p}_nl0.4_sr48'
        for m in MODELS:
            for s in SEEDS:
                run = root / 'runs' / f'{m}_s{s}'
                items.append((f'syn_{tag}_{m}_s{s}', lambda run=run, tag=tag: (run / f'cells_{tag}.npz').exists(),
                              [sys.executable, 'evaluate2.py', 'model', '--run', run, '--panel', root / 'panels' / f'{tag}.pt',
                               '--device', 'cuda'], root / 'logs' / f'syn_{tag}_{m}_s{s}.log'))
    parallel(items)


def baselines(root):
    items = []
    for n in NETS:
        panel = root / 'panels' / f'real_{n}_s{SEED_PANEL}.pt'
        for meth in ('lgbm', 'mice'):
            out = root / 'runs_real' / n / meth
            items.append((f'{meth}_{n}', lambda out=out, n=n: (out / f'cells_real_{n}_s{SEED_PANEL}.npz').exists(),
                          [sys.executable, 'baselines_v6.py', '--panel', panel, '--out', out, '--method', meth],
                          root / 'logs' / f'{meth}_{n}.log'))
    parallel(items)
    tab = [(f'tabpfn_{n}', lambda n=n: (root / 'runs_real' / n / 'tabpfn_v2' / f'cells_real_{n}_s{SEED_PANEL}.npz').exists(),
            [sys.executable, 'tabpfn_colab.py', '--panel', root / 'panels' / f'real_{n}_s{SEED_PANEL}.pt',
             '--out', root / 'runs_real' / n / 'tabpfn_v2'], root / 'logs' / f'tabpfn_{n}.log') for n in NETS]
    parallel(tab, 1)                                              # one GPU job at a time


def preds(root):
    items = []
    for n in NETS:
        runs = ','.join(str(root / 'runs_real' / n / r) for r in ('pfn_s1_ft', 'pfn_s2_ft', 'lct_s1_ft'))
        items.append((f'preds_{n}', lambda n=n: (root / 'preds' / f'preds_real_{n}_s{SEED_PANEL}.npz').exists(),
                      [sys.executable, 'site_pool.py', 'predict', '--panel', root / 'panels' / f'real_{n}_s{SEED_PANEL}.pt',
                       '--runs', runs, '--preds', root / 'preds', '--threads', '2'], root / 'logs' / f'preds_{n}.log'))
    parallel(items)


def confirm(root):
    sh([sys.executable, 'confirm6.py', '--root', root], root / 'logs' / 'confirm6.log')
    print((root / 'logs' / 'confirm6.log').read_text()[-6000:])


def pack(root):
    z = root / 'mira_v6_results.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(root.rglob('*')):
            if p.is_file() and (p.suffix in ('.json', '.log') or p.name.startswith(('cells_', 'preds_')) or
                                (p.name == 'model.pt' and 'runs' in p.parts)):
                f.write(p, p.relative_to(root))
    print(z, z.stat().st_size)


STAGES = dict(source=source, status=status, data=data, panels=panels, pools=pools, finetune=finetune, score=score,
              baselines=baselines, preds=preds, confirm=confirm, pack=pack)
ORDER = ['data', 'panels', 'pools', 'finetune', 'score', 'baselines', 'preds', 'confirm', 'pack']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('stage', choices=list(STAGES) + ['all'])
    a = ap.parse_args()
    r = Path(a.root); r.mkdir(parents=True, exist_ok=True)
    for st in (ORDER if a.stage == 'all' else [a.stage]):
        print(f'== {st}', time.strftime('%H:%M:%S'), flush=True)
        STAGES[st](r)
