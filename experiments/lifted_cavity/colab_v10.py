"""Protocol v10 on Google Colab (docs/LIFTED_CAVITY_PROTOCOL_V10.md).

    D=/content/drive/MyDrive
    python colab_v10.py --root $D/mira_v10 --v9 $D/mira_v9 --v8 $D/mira_v8 --v7 $D/mira_v7 --v6 $D/mira_v6 <chain or stages>
Chain D (development on the v9 networks): ftdev scoredev select
Chain F (fresh v10 networks): data panels pools krig ft ftsel score pool confirm confirm11 pack
Chain T (TabPFN v2 on the fresh panels, four at a time; also writes the v11 per-query and cross-fit outputs): tabpfn
Chain O (the selected model and PFN-L-SF on the six v6/v7 networks, descriptive; then the v11 pool on the v9 networks,
descriptive): ftold scoreold devpool confirm
Every finished item is skipped and fine-tuning resumes from Drive checkpoints, so relaunching after a disconnect resumes.
Source checkpoints (never retrained): UPT-L (v7), PFN-L (v9, amendment 1 of v9), TNP-D (v8).
"""
import argparse, json, sys, time
from pathlib import Path
from colab_v6 import sh, parallel, done_train

N10 = ('usgs_ny', 'usgs_wa', 'dwd_p', 'aurn_pm25')
N9 = ('dwd_t2m', 'aurn_no2', 'usgs_q')
DEV6, DEV7 = ('metr_la', 'pems_bay', 'intel'), ('pems03', 'pems08', 'air_cn')
SEED10 = 10101
SEEDS = (1, 2, 3)
SF = ['--yz', '--lv-bound', '10']
VARIANT = {'upt_L_sf': ('upt_L', ['--crps', '1'] + SF), 'upt_L_sfm': ('upt_L', ['--crps', '1'] + SF + ['--free-k', '3']),
           'pfn_L_sf': ('pfn_L', ['--crps', '1'] + SF), 'pfn_L_sfn': ('pfn_L', SF), 'upt_L_crps': ('upt_L', ['--crps', '1']),
           'tnp': ('tnp', [])}
TTA = {'upt_L_sf', 'upt_L_sfm', 'pfn_L_sf', 'pfn_L_sfn', 'upt_L_crps'}
PAR = 6


def nets_(a):
    return nets(a)


def nets(a):
    d = {n: (Path(a.root), SEED10, Path(a.root) / 'data_v10') for n in N10}
    d.update({n: (Path(a.v9), 9091, Path(a.v9) / 'data_v9') for n in N9})
    d.update({n: (Path(a.v6), 6061, Path(a.v6) / 'data_v6') for n in DEV6})
    d.update({n: (Path(a.v7), 7071, Path(a.v7) / 'data_v7') for n in DEV7})
    return d


def src(a, m, s):
    base = VARIANT[m][0]
    return Path({'upt_L': a.v7, 'pfn_L': a.v9, 'tnp': a.v8}[base]) / 'runs' / f'{base}_s{s}'


def panel(a, n):
    root, seed, _ = nets(a)[n]
    return root / 'panels' / f'real_{n}_s{seed}.pt'


def logs(a):
    p = Path(a.root) / 'logs'; p.mkdir(parents=True, exist_ok=True)
    return p


def selection(a):
    return json.loads((Path(a.root) / 'dev' / 'selection10.json').read_text())


def ft_items(a, pairs):
    items = []
    for n, m in pairs:
        root, _, data = nets(a)[n]
        for s in SEEDS:
            out = root / 'runs_real' / n / f'{m}_s{s}_ft'
            items.append((f'ft_{n}_{m}_s{s}', lambda out=out: (out / 'model.pt').exists() and not (out / 'ckpt.pt').exists(),
                          [sys.executable, 'finetune_real.py', '--dataset', n, '--init', src(a, m, s), '--out', out,
                           '--pool-cache', root / 'pools', '--v6', data, '--device', 'cuda', '--ckpt-every', '500'] + VARIANT[m][1],
                          logs(a) / f'ft_{n}_{m}_s{s}.log'))
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


def wait_for(paths, what, hours=14):
    t0 = time.time()
    while not all(Path(p).exists() for p in paths):
        if time.time() - t0 > hours * 3600:
            raise TimeoutError(what)
        if int(time.time() - t0) % 1800 < 60:
            print('waiting for', what, time.strftime('%H:%M:%S'), flush=True)
        time.sleep(60)


def check_sources(a, models):
    for m in models:
        for s in SEEDS:
            p = src(a, m, s)
            assert done_train(p, 40000 if VARIANT[m][0].endswith('_L') else 30000), f'missing source checkpoint {p}'


# ------------------------------------------------------------------ chain D
def ftdev(a):
    check_sources(a, ('upt_L_sf', 'pfn_L_sf'))
    parallel(ft_items(a, [(n, m) for m in ('upt_L_sf', 'upt_L_sfm', 'pfn_L_sf') for n in N9]), PAR)


def scoredev(a):
    parallel(score_items(a, [(n, m) for m in ('upt_L_sf', 'upt_L_sfm', 'pfn_L_sf') for n in N9]), PAR)


def select(a):
    out = Path(a.root) / 'dev' / 'selection10.json'
    if not out.exists():
        sh([sys.executable, 'select10.py', '--v9', a.v9, '--out', out], logs(a) / 'select10.log')
    print(out.read_text())


# ------------------------------------------------------------------ chain F
def data(a):
    import shutil                         # amendment 3: a snapshot committed to the repository is used as is
    snap, out = Path(__file__).resolve().parents[2] / 'data_snapshots' / 'v10', Path(a.root) / 'data_v10'
    out.mkdir(parents=True, exist_ok=True)
    for f in (sorted(snap.glob('*')) if snap.exists() else []):
        if not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    sh([sys.executable, 'fetch_v10.py', '--out', out, *N10], logs(a) / 'fetch_v10.log')
    print((logs(a) / 'fetch_v10.log').read_text()[-3000:])


def panels(a):
    (Path(a.root) / 'panels').mkdir(parents=True, exist_ok=True)
    parallel([(f'panel_{n}', lambda n=n: panel(a, n).exists(),
               [sys.executable, 'eval_real.py', 'build', '--dataset', n, '--seed', SEED10, '--v6', Path(a.root) / 'data_v10',
                '--out', panel(a, n)], logs(a) / f'panel_{n}.log') for n in N10])


def pools(a):
    code = ('import argparse, json, sys, finetune_real as f; d = json.loads(sys.argv[1]); '
            'f.source_pool(argparse.Namespace(seed=11, episodes=4000, period=None, airq=None, beijing=None, gas=None, **d))')
    r = Path(a.root)
    parallel([(f'pool_{n}', lambda n=n: (r / 'pools' / f'src_{n}_s11_n4000.pt').exists(),
               [sys.executable, '-c', code, json.dumps(dict(dataset=n, pool_cache=str(r / 'pools'), v6=str(r / 'data_v10')))],
               logs(a) / f'pool_{n}.log') for n in N10])


def krig(a):
    items = []
    for n in N10:
        root, seed, data_ = nets(a)[n]
        for s in SEEDS:
            out = root / 'runs_real' / n / f'ignnk_s{s}'
            items.append((f'krig_{n}_s{s}', lambda out=out, n=n, seed=seed, s=s: (out / f'cells_real_{n}_s{seed}.npz').exists() and
                          (out.parent / f'ignnk_cal_s{s}' / f'cells_real_{n}_s{seed}.npz').exists(),
                          [sys.executable, 'kriging.py', '--dataset', n, '--data', data_, '--panel', panel(a, n), '--out', out,
                           '--seed', s, '--device', 'cuda'], logs(a) / f'krig_{n}_s{s}.log'))
    parallel(items, PAR)


def ft(a):
    models = ('pfn_L_sf', 'pfn_L_sfn', 'upt_L_crps', 'tnp')
    check_sources(a, models)
    parallel(ft_items(a, [(n, m) for m in models for n in N10]), PAR)


def ftsel(a):
    wait_for([Path(a.root) / 'dev' / 'selection10.json'], 'selection (chain D)')
    parallel(ft_items(a, [(n, selection(a)['model']) for n in N10]), PAR)


def pq_items(a, nets, m):
    items = []
    for n in nets:
        root, seed, _ = nets_(a)[n]
        for s in SEEDS:
            run = root / 'runs_real' / n / f'{m}_s{s}_ft'
            items.append((f'pq_{n}_{m}_s{s}', lambda run=run, n=n, seed=seed: (run / f'pq_real_{n}_s{seed}_tta8.npz').exists(),
                          [sys.executable, 'eval_real.py', 'score', '--panel', panel(a, n), '--runs', run, '--device', 'cuda', '--tta', 8,
                           '--per-query'], logs(a) / f'pq_{n}_{m}_s{s}.log'))
    return items


def score(a):
    models = ('pfn_L_sf', 'pfn_L_sfn', 'upt_L_crps', 'tnp', selection(a)['model'])
    parallel(score_items(a, [(n, m) for n in N10 for m in models]) + pq_items(a, N10, selection(a)['model']), PAR)
    wait_for([Path(a.root) / 'runs_real' / n / 'tabpfn_v2' / f'cells_real_{n}_s{SEED10}.npz' for n in N10], 'TabPFN (chain T)')


def pool(a, nets=N10):
    items = []
    for n in nets:
        root, seed, _ = nets_(a)[n]
        m = selection(a)['model']
        items.append((f'pool_{n}', lambda root=root, n=n, seed=seed, m=m: (root / 'runs_real' / n / f'pool_{m}_tabpfn' / f'cells_real_{n}_s{seed}.npz').exists(),
                      [sys.executable, 'pool_eval.py', '--root', root, '--net', n, '--panel-seed', seed, '--model', m, '--sfx', '_tta8'],
                      logs(a) / f'pool_{n}.log'))
    parallel(items, PAR)


def confirm11(a):
    out = Path(a.root) / 'confirm_v11.json'
    sh([sys.executable, 'confirm11.py', '--v10', a.root, '--v9', a.v9, '--out', out], logs(a) / 'confirm11.log')
    print((logs(a) / 'confirm11.log').read_text()[-6000:])


def devpool(a):
    """v11 descriptive: the pool on the v9 networks (already used): TabPFN per-query + cross-fit, UPT-SF per-query."""
    r = Path(a.v9)
    parallel([(f'tabpfn_pq_{n}', lambda n=n: (r / 'runs_real' / n / 'tabpfn_v2' / f'pq_real_{n}_s9091.npz').exists(),
               [sys.executable, 'tabpfn_colab.py', '--panel', panel(a, n), '--out', r / 'runs_real' / n / 'tabpfn_v2', '--per-query'],
               logs(a) / f'tabpfn_pq_{n}.log') for n in N9] + pq_items(a, N9, selection(a)['model']), 4)
    pool(a, N9)


def confirm(a):
    out = Path(a.root) / 'confirm_v10.json'
    sh([sys.executable, 'confirm10.py', '--v10', a.root, '--v9', a.v9, '--v6', a.v6, '--v7', a.v7, '--out', out], logs(a) / 'confirm10.log')
    print((logs(a) / 'confirm10.log').read_text()[-8000:])


def pack(a):
    import zipfile
    r = Path(a.root); z = r / 'mira_v10_results.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(r.rglob('*')):
            if p.is_file() and (p.suffix in ('.json', '.log') or p.name.startswith('cells_')):
                f.write(p, p.relative_to(r))
    print(z, z.stat().st_size)


# ------------------------------------------------------------------ chain T
def tabpfn(a):
    wait_for([panel(a, n) for n in N10], 'fresh panels (chain F)')
    r = Path(a.root)
    parallel([(f'tabpfn_{n}', lambda n=n: (r / 'runs_real' / n / 'tabpfn_v2' / f'cells_real_{n}_s{SEED10}.npz').exists(),
               [sys.executable, 'tabpfn_colab.py', '--panel', panel(a, n), '--out', r / 'runs_real' / n / 'tabpfn_v2',
                '--allow-constant-support', '--crps'], logs(a) / f'tabpfn_{n}.log') for n in N10], 4)


# ------------------------------------------------------------------ chain O
def ftold(a):
    wait_for([Path(a.root) / 'dev' / 'selection10.json'], 'selection (chain D)')
    parallel(ft_items(a, [(n, m) for m in (selection(a)['model'], 'pfn_L_sf') for n in DEV6 + DEV7]), 4)


def scoreold(a):
    parallel(score_items(a, [(n, m) for m in (selection(a)['model'], 'pfn_L_sf') for n in DEV6 + DEV7], ttas=(8,)), 4)


STAGES = dict(ftdev=ftdev, scoredev=scoredev, select=select, data=data, panels=panels, pools=pools, krig=krig, ft=ft, ftsel=ftsel,
              score=score, pool=pool, confirm=confirm, confirm11=confirm11, pack=pack, tabpfn=tabpfn, ftold=ftold, scoreold=scoreold,
              devpool=devpool)
CHAIN = dict(D=['ftdev', 'scoredev', 'select'],
             F=['data', 'panels', 'pools', 'krig', 'ft', 'ftsel', 'score', 'pool', 'confirm', 'confirm11', 'pack'],
             T=['tabpfn'], O=['ftold', 'scoreold', 'devpool', 'confirm'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('root', 'v9', 'v8', 'v7', 'v6'):
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
