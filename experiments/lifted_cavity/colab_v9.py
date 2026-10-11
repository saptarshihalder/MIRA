"""Protocol v9 on Google Colab (docs/LIFTED_CAVITY_PROTOCOL_V9.md).

    D=/content/drive/MyDrive
    python colab_v9.py --root $D/mira_v9 --v4 $D/mira_v4 --v6 $D/mira_v6 --v7 $D/mira_v7 --v8 $D/mira_v8 <stages>
Chain A (v9 networks): data panels pools ft9 krig9 tabpfn9 wait score9 confirm pack
Chain B (development, then the selected variant): ftdev scoredev select ft9var
Every finished item is skipped and fine-tuning resumes from Drive checkpoints, so relaunching after a disconnect resumes.
Source checkpoints (never retrained here): pfn/upt from v6, UPT-L from v7, PFN-L from v4, CNP/TNP from v8.
"""
import argparse, json, sys, time
from pathlib import Path
from colab_v6 import sh, parallel, done_train

N9 = ('dwd_t2m', 'aurn_no2', 'usgs_q')
DEV6, DEV7 = ('metr_la', 'pems_bay', 'intel'), ('pems03', 'pems08', 'air_cn')
SEED9 = 9091
SEEDS = (1, 2, 3)
TTA = ('pfn', 'upt', 'upt_L', 'pfn_L', 'upt_L_crps', 'pfn_L_crps')      # models with random feature embeddings
BASE9 = ('pfn', 'upt', 'cnp', 'tnp', 'upt_L', 'pfn_L')
VARIANT = {'upt_L_crps': ('upt_L', ['--crps', '1']), 'pfn_L_crps': ('pfn_L', ['--crps', '1'])}
PAR_A, PAR_B = 7, 5


def nets(a):
    """network -> (root, panel seed, data folder)"""
    d = {n: (Path(a.root), SEED9, Path(a.root) / 'data_v9') for n in N9}
    d.update({n: (Path(a.v6), 6061, Path(a.v6) / 'data_v6') for n in DEV6})
    d.update({n: (Path(a.v7), 7071, Path(a.v7) / 'data_v7') for n in DEV7})
    return d


def src(a, m, s):
    base = VARIANT.get(m, (m, []))[0]
    root = {'pfn': a.v6, 'upt': a.v6, 'upt_L': a.v7, 'pfn_L': a.v4, 'cnp': a.v8, 'tnp': a.v8}[base]
    return Path(root) / 'runs' / f'{base}_s{s}'


def panel(a, n):
    root, seed, _ = nets(a)[n]
    return root / 'panels' / f'real_{n}_s{seed}.pt'


def logs(a):
    p = Path(a.root) / 'logs'; p.mkdir(parents=True, exist_ok=True)
    return p


def ft_items(a, pairs):
    items = []
    for n, m in pairs:
        root, _, data = nets(a)[n]
        for s in SEEDS:
            out = root / 'runs_real' / n / f'{m}_s{s}_ft'
            items.append((f'ft_{n}_{m}_s{s}', lambda out=out: (out / 'model.pt').exists() and not (out / 'ckpt.pt').exists(),
                          [sys.executable, 'finetune_real.py', '--dataset', n, '--init', src(a, m, s), '--out', out,
                           '--pool-cache', root / 'pools', '--v6', data, '--device', 'cuda', '--ckpt-every', '500']
                          + VARIANT.get(m, (None, []))[1], logs(a) / f'ft_{n}_{m}_s{s}.log'))
    return items


def score_items(a, pairs, ttas=(1, 8)):
    items = []
    for n, m in pairs:
        root, seed, _ = nets(a)[n]
        for s in SEEDS:
            run = root / 'runs_real' / n / f'{m}_s{s}_ft'
            for k in (ttas if m in TTA else (1,)):
                sfx = f'_tta{k}' if k > 1 else ''
                items.append((f'score_{n}_{m}_s{s}{sfx}', lambda run=run, n=n, seed=seed, sfx=sfx: (run / f'cells_real_{n}_s{seed}{sfx}.npz').exists(),
                              [sys.executable, 'eval_real.py', 'score', '--panel', panel(a, n), '--runs', run, '--device', 'cuda', '--tta', k],
                              logs(a) / f'score_{n}_{m}_s{s}{sfx}.log'))
    return items


def wait_for(paths, what, hours=12):
    t0 = time.time()
    while not all(Path(p).exists() for p in paths):
        if time.time() - t0 > hours * 3600:
            raise TimeoutError(what)
        if int(time.time() - t0) % 1800 < 60:
            print('waiting for', what, time.strftime('%H:%M:%S'), flush=True)
        time.sleep(60)


# ------------------------------------------------------------------ chain A
def data(a):
    sh([sys.executable, 'fetch_v9.py', '--out', Path(a.root) / 'data_v9', *N9], logs(a) / 'fetch_v9.log')
    print((logs(a) / 'fetch_v9.log').read_text()[-3000:])


def panels(a):
    (Path(a.root) / 'panels').mkdir(parents=True, exist_ok=True)
    parallel([(f'panel_{n}', lambda n=n: panel(a, n).exists(),
               [sys.executable, 'eval_real.py', 'build', '--dataset', n, '--seed', SEED9, '--v6', Path(a.root) / 'data_v9',
                '--out', panel(a, n)], logs(a) / f'panel_{n}.log') for n in N9])


def pools(a):
    code = ('import argparse, json, sys, finetune_real as f; d = json.loads(sys.argv[1]); '
            'f.source_pool(argparse.Namespace(seed=11, episodes=4000, period=None, airq=None, beijing=None, gas=None, **d))')
    r = Path(a.root)
    parallel([(f'pool_{n}', lambda n=n: (r / 'pools' / f'src_{n}_s11_n4000.pt').exists(),
               [sys.executable, '-c', code, json.dumps(dict(dataset=n, pool_cache=str(r / 'pools'), v6=str(r / 'data_v9')))],
               logs(a) / f'pool_{n}.log') for n in N9])


def ft9(a):
    for m in BASE9:
        for s in SEEDS:
            assert done_train(src(a, m, s), 40000 if m.endswith('_L') else 30000), f'missing source checkpoint {src(a, m, s)}'
    order = ('upt_L', 'pfn_L', 'upt', 'pfn', 'tnp', 'cnp')                     # long runs first
    parallel(ft_items(a, [(n, m) for m in order for n in N9]), PAR_A)


def krig9(a):
    items = []
    for n in N9:
        root, seed, data_ = nets(a)[n]
        for s in SEEDS:
            out = root / 'runs_real' / n / f'ignnk_s{s}'
            items.append((f'krig_{n}_s{s}', lambda out=out, n=n, seed=seed, s=s: (out / f'cells_real_{n}_s{seed}.npz').exists() and
                          (out.parent / f'ignnk_cal_s{s}' / f'cells_real_{n}_s{seed}.npz').exists(),
                          [sys.executable, 'kriging.py', '--dataset', n, '--data', data_, '--panel', panel(a, n), '--out', out,
                           '--seed', s, '--device', 'cuda'], logs(a) / f'krig_{n}_s{s}.log'))
    parallel(items, PAR_A)


def tabpfn9(a):
    r = Path(a.root)
    parallel([(f'tabpfn_{n}', lambda n=n: (r / 'runs_real' / n / 'tabpfn_v2' / f'cells_real_{n}_s{SEED9}.npz').exists(),
               [sys.executable, 'tabpfn_colab.py', '--panel', panel(a, n), '--out', r / 'runs_real' / n / 'tabpfn_v2',
                '--allow-constant-support', '--crps'], logs(a) / f'tabpfn_{n}.log') for n in N9], 1)


def wait(a):
    dev = Path(a.root) / 'dev'
    wait_for([dev / 'selection.json'], 'variant selection (chain B)')
    if json.loads((dev / 'selection.json').read_text())['crps_objective']:
        wait_for([dev / 'ft9var.done'], 'variant fine-tuning (chain B)')


def score9(a):
    sel = json.loads((Path(a.root) / 'dev' / 'selection.json').read_text())
    models = BASE9 + (tuple(VARIANT) if sel['crps_objective'] else ())
    parallel(score_items(a, [(n, m) for n in N9 for m in models]), PAR_A)


def confirm(a):
    out = Path(a.root) / 'confirm_v9.json'
    sh([sys.executable, 'confirm9.py', '--v9', a.root, '--v6', a.v6, '--v7', a.v7, '--out', out], logs(a) / 'confirm9.log')
    print((logs(a) / 'confirm9.log').read_text()[-8000:])


def pack(a):
    import zipfile
    r = Path(a.root); z = r / 'mira_v9_results.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(r.rglob('*')):
            if p.is_file() and (p.suffix in ('.json', '.log') or p.name.startswith('cells_')):
                f.write(p, p.relative_to(r))
    print(z, z.stat().st_size)


# ------------------------------------------------------------------ chain B
def ftdev(a):
    for m in ('upt_L', 'pfn_L'):
        for s in SEEDS:
            assert done_train(src(a, m, s), 40000), f'missing source checkpoint {src(a, m, s)}'
    pairs = [(n, 'upt_L_crps') for n in DEV7] + [(n, 'upt_L') for n in DEV6] + [(n, 'pfn_L') for n in DEV6 + DEV7]
    parallel(ft_items(a, pairs), PAR_B)


def scoredev(a):
    pairs = [(n, 'upt_L_crps') for n in DEV7] + [(n, 'upt_L') for n in DEV6 + DEV7] + [(n, 'pfn_L') for n in DEV6 + DEV7]
    parallel(score_items(a, pairs), PAR_B)


def select(a):
    out = Path(a.root) / 'dev' / 'selection.json'
    if not out.exists():
        sh([sys.executable, 'select9.py', '--v7', a.v7, '--out', out], logs(a) / 'select9.log')
    print(out.read_text())


def ft9var(a):
    dev = Path(a.root) / 'dev'
    if json.loads((dev / 'selection.json').read_text())['crps_objective']:
        wait_for([Path(a.root) / 'pools' / f'src_{n}_s11_n4000.pt' for n in N9], 'v9 source pools (chain A)')
        parallel(ft_items(a, [(n, m) for m in VARIANT for n in N9]), PAR_B)
    (dev / 'ft9var.done').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))


STAGES = dict(data=data, panels=panels, pools=pools, ft9=ft9, krig9=krig9, tabpfn9=tabpfn9, wait=wait, score9=score9, confirm=confirm,
              pack=pack, ftdev=ftdev, scoredev=scoredev, select=select, ft9var=ft9var)
CHAIN = dict(A=['data', 'panels', 'pools', 'ft9', 'krig9', 'tabpfn9', 'wait', 'score9', 'confirm', 'pack'],
             B=['ftdev', 'scoredev', 'select', 'ft9var'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('root', 'v4', 'v6', 'v7', 'v8'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('stage', nargs='+', choices=list(STAGES) + list(CHAIN))
    a = ap.parse_args()
    Path(a.root).mkdir(parents=True, exist_ok=True)
    failed = []
    for st in [s for x in a.stage for s in CHAIN.get(x, [x])]:
        print(f'== {st}', time.strftime('%H:%M:%S'), flush=True)
        try:
            STAGES[st](a)
        except RuntimeError as e:            # failed items of a stage: later stages still run on what finished
            if st in ('data', 'panels', 'pools', 'select'):
                raise
            failed.append(st); print('STAGE FAILED', st, e, flush=True)
    if failed:
        raise SystemExit(f'stages with failed items: {failed} (relaunch to retry them)')
