"""Protocol v4 on a GPU (docs/LIFTED_CAVITY_PROTOCOL_V4.md): Google Colab or any CUDA machine.

    python colab_v4.py --root /content/drive/MyDrive/mira_v4 --panels /content/mira_panels

One command runs everything. Re-run the same command after any disconnect: all state lives under --root (Google
Drive), finished items are skipped, and training and fine-tuning resume from their last checkpoint.

Order:
1. data: Beijing CSVs, SHA-256 verified.
2. panels: verify the frozen evaluation panels.
3. smoke: about 10 minutes of GPU checks, plus a time estimate.
4. main: for seed 1, 2, 3: train, score and fine-tune PFN-L then LCT-L; TabPFN v2 on the v4 panels after seed 1.
5. confirm: the endpoints E8-E12 (confirm4.py).
6. extra: descriptive scores on the earlier panels.
7. pack: <root>/mira_v4_results.zip, small, without weights.

A single stage: python colab_v4.py ... smoke|main|confirm|extra|pack|status
"""
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path
from score_validation import GAUSSIAN_METRICS, TABPFN_METRICS, load_scores

CODE = Path(__file__).resolve().parent

BEIJING_URL = ('https://raw.githubusercontent.com/sofiaalban/MP2beijing/626e097b16cdf32436a3d5724ff7d91964c89728/'
               'PRSA_Data_20130301-20170228/')
BEIJING = {   # UCI Beijing Multi-Site Air Quality (CC BY 4.0); hashes as used by every earlier run
    'Aotizhongxin': '8f8457efef51dea61dd3c38ebbccd9fa25f5a6ba2a23e90e5b500b25015f4f27',
    'Changping': 'dbac77681385b631d093823afdeea3ec96d2d234442afd5f8fb91bcc526d5863',
    'Dingling': 'bc2dcda191a790a5cbc1af71f142b7cdf96759e500c0dde07dbe5c826d54dfa0',
    'Dongsi': 'f56e7cf80603f10d627bcd51f7f3a36b108ed6aec898dc92376ee4d30387f9ac',
    'Guanyuan': 'bbe31877c38679a71ba8df202ffb6c1f531f2aadf1d6180efd28fc25c112bb13',
    'Gucheng': '589cff73621c521a28557f4c093c8df47825606232f614d07b52970fc2e2b336',
    'Huairou': 'e0dd971c8b5d87cf0ebf548edd1b9e0c94ca6dc1af13ffee2e487732b4270544',
    'Nongzhanguan': '0a348fe79a8923ab84fb093a50f61eb64cb083c70f5a934971c99de28fdba749',
    'Shunyi': '8d27b386d14db242a1dd0183809841685e34e1a7a354499238532c5e12f82949',
    'Tiantan': '1865d4b500b55e5d8587c233e68126e9af6cbebc800a951022c778da1c3a0e2c',
    'Wanliu': 'f3e0954a64c31937a86a0aa51e8df7f99626fc04ea46c09ae91c9a20317f47af',
    'Wanshouxigong': 'f77c09bff941ce907aefd0281df52bffc25604f16205f55c65e00ca3d39e0800'}

H1, H3 = 'v2_s20261401_n256_p5_nl0.4_sr48', 'v2_s20261403_n128_p16_nl0.4_sr48'
OLD_SYN = ['v2_s20261201_n256_p5_nl0.4_sr48', 'v2_s20261202_n128_p8_nl0.4_sr48', 'v2_s20261203_n128_p16_nl0.4_sr48',
           'v2_s20261204_n128_p5_nl0.8_sr48', 'v2_s20261205_n128_p5_nl0.0_sr48', 'v2_s20261206_n128_p5_nl0.4_sr16',
           'v2_s20261207_n128_p5_nl0.4_sr96', 'v2_s20261301_n256_p5_nl0.4_sr48', 'v2_s20261303_n128_p16_nl0.4_sr48']
NEW = ('beijing_pm10', 'beijing_so2', 'beijing_o3')
V4_REAL = [f'real_{d}_s4041' for d in NEW]
OLD_REAL = ['real_beijing_s2027', 'real_airq_co_s2027', 'real_airq_no2_s2027', 'real_beijing_no2_s3031', 'real_beijing_co_s3031']
PANELS = [H1, H3] + V4_REAL + OLD_SYN + OLD_REAL

SEEDS = (1, 2, 3)
MODELS = (('pfn_L', 'pfn_train.py'), ('lct_L', 'lct_train.py'))
ARCH = ['--d', '128', '--layers', '8', '--heads', '8', '--ff', '512']
RECIPE = ['--steps', '40000', '--batch-tasks', '32', '--queries', '16', '--lr', '5e-4', '--warmup', '2000'] + ARCH
STEPS = 40000
FT = ['--ckpt-every', '500']              # fine-tuning: the protocol v2 recipe = finetune_real.py defaults
EXTRA_ARGS = {'pfn_L': [], 'lct_L': ['--prefetch', '4']}   # prefetch changes speed only (same batches, same order)

FAILED = []


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sh(label, args, env=None, expect_rc=0):
    """Run one step, streaming its output here and to <root>/logs/<label>.log. True if it exits with expect_rc."""
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u'] + [str(x) for x in args]
    with open(LOGS / f'{label}.log', 'a') as f:
        f.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M:%S")} {" ".join(cmd)}\n'); f.flush()
        p = subprocess.Popen(cmd, cwd=CODE, env=dict(ENV, **(env or {})), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        for line in p.stdout:
            print(f'[{label}] {line}', end='', flush=True); f.write(line); f.flush()
        rc = p.wait()
    if rc != expect_rc:
        FAILED.append(label)
        print(f'!! {label} failed (exit {rc}); log: {LOGS / (label + ".log")}', flush=True)
    return rc == expect_rc


def ok_npz(path, source_panel, metrics=GAUSSIAN_METRICS):
    """Only complete, finite scores for every panel condition can be skipped."""
    if not path.exists():
        return False
    try:
        load_scores(path, source_panel, metrics)
    except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile) as e:
        print(f'invalid scores {path}: {e}; recomputation required', flush=True)
        return False
    return True


def score_step(label, args, dest, source_panel, metrics=GAUSSIAN_METRICS):
    if ok_npz(dest, source_panel, metrics):
        return True
    if not sh(label, args):
        return False
    if not ok_npz(dest, source_panel, metrics):
        FAILED.append(label)
        return False
    return True


def has_ckpt(run):
    return (run / 'ckpt.pt').exists() or (run / 'ckpt.prev.pt').exists()


def bind_checkpoint(run):
    """Record the final checkpoint BEFORE scoring; refuse silent replacement."""
    identity = {name: sha256(run / name) for name in ('model.pt', 'train.json')}
    p = run / 'checkpoint_identity.json'
    if p.exists() and json.loads(p.read_text()) != identity:
        raise ValueError(f'Completed checkpoint changed in {run}; preserve it and use a new run.')
    if not p.exists():
        p.write_text(json.dumps(identity, indent=1))
    return True


def train_done(run, steps=None):
    j = run / 'train.json'
    return (j.exists() and (run / 'model.pt').exists() and not has_ckpt(run)
            and json.loads(j.read_text()).get('steps_done') == (steps or STEPS) and bind_checkpoint(run))


def ft_done(run):
    j = run / 'train.json'
    return (j.exists() and (run / 'model.pt').exists() and not has_ckpt(run)
            and 'finetune' in json.loads(j.read_text()) and bind_checkpoint(run))


def panel(tag):
    return PANELS_DIR / f'{tag}.pt'


# ---------------------------------------------------------------- stages

def stage_data():
    d = DATA / 'beijing'; d.mkdir(parents=True, exist_ok=True)
    for st, h in BEIJING.items():
        name = f'PRSA_Data_{st}_20130301-20170228.csv'; dest = d / name
        if dest.exists() and sha256(dest) == h:
            continue
        tmp = d / (name + '.part')
        urllib.request.urlretrieve(BEIJING_URL + name, tmp)
        if not tmp.exists() or sha256(tmp) != h:
            raise SystemExit(f'{name}: download failed or SHA-256 mismatch')
        tmp.replace(dest)
    print('data: 12 Beijing files verified', flush=True)


def stage_panels():
    sums = dict(line.split()[::-1] for line in (PANELS_DIR / 'SHA256SUMS').read_text().splitlines() if line.strip())
    for tag in PANELS:
        f = f'{tag}.pt'
        if f not in sums or not panel(tag).exists() or sha256(panel(tag)) != sums[f]:
            raise SystemExit(f'panel {f} missing or corrupt in {PANELS_DIR}; re-clone the lifted-cavity-panels branch')
    print(f'panels: {len(PANELS)} verified', flush=True)


def smoke_panels(folder):
    """Independent synthetic fixtures and Beijing SOURCE episodes; never evaluation panels."""
    import torch
    import data, evaluate2, realdata
    from finetune_real import load_raw
    syn = folder / 'smoke_source_synthetic.pt'
    real = folder / 'smoke_source_beijing.pt'
    torch.save(dict(pool=data.build_pool(4, 90401, P=16), banks=evaluate2.banks(16, 90401),
                    meta=dict(smoke=True, split='generated_source')), syn)
    raw = load_raw('beijing_pm10', dict(beijing=DATA / 'beijing'))
    pool = realdata.to_pool(realdata.episodes('beijing_pm10', raw, 'source', 90402, n_source=4))
    conds = {e: realdata.condition_masks(pool, e, 90403 + e) for e in (0, 3, 6)}
    torch.save(dict(pool=pool, conds=conds, meta=dict(smoke=True, split='source')), real)
    return syn, real


def stage_smoke():
    rep_path = ROOT / 'smoke' / 'report.json'
    if rep_path.exists() and json.loads(rep_path.read_text()).get('source_only') is True:
        print('smoke: passed earlier', json.loads(rep_path.read_text()).get('estimate_hours'), flush=True)
        return True
    import torch
    d = ROOT / 'smoke'; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    rep = dict(device=DEV, gpu=torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu', torch=torch.__version__,
               python=platform.python_version(), source_only=True)
    syn_panel, real_panel = smoke_panels(d)
    ok = True
    for name, script in MODELS:
        # 1. resume on this device: stop after a checkpoint, then continue
        r = d / f'{name}_resume'
        args = [script, '--seed', '1', '--out', r, '--device', DEV] + SMOKE_RECIPE + ['--steps', '40', '--ckpt-every', '20',
                                                                                   '--save-every', '40'] + EXTRA_ARGS[name]
        ok &= sh(f'smoke_{name}', args, env={'MIRA_STOP_AT': '30'}, expect_rc=1)     # stops early on purpose
        ok &= sh(f'smoke_{name}', args) and train_done(r, 40)
        # 2. speed: 500 steps of the real recipe
        t = d / name
        ok &= sh(f'smoke_{name}', [script, '--seed', '1', '--out', t, '--device', DEV] + SMOKE_RECIPE +
                 ['--steps', '500', '--save-every', '500', '--ckpt-every', '250'] + EXTRA_ARGS[name])
        if not ok:
            break
        tr = json.loads((t / 'train.json').read_text())['trace']
        rep[f'{name}_sec_per_step'] = (tr[-1]['seconds'] - tr[0]['seconds']) / (tr[-1]['step'] - tr[0]['step'])
        # 3. scoring a synthetic panel
        t0 = time.time()
        ok &= score_step(f'smoke_{name}', ['evaluate2.py', 'model', '--run', t, '--panel', syn_panel, '--device', DEV],
                         t / f'cells_{syn_panel.stem}.npz', syn_panel)
        rep[f'{name}_score_synthetic_source_sec'] = time.time() - t0
        # 4. fine-tuning (builds the cached source pool used later) with a resume, then scoring a real panel
        f = d / f'{name}_ft'
        ftargs = ['finetune_real.py', '--dataset', 'beijing_pm10', '--init', t, '--out', f, '--device', DEV,
                  '--beijing', DATA / 'beijing', '--pool-cache', POOLS] + SMOKE_FT
        t0 = time.time()
        ok &= sh(f'smoke_{name}', ftargs, env={'MIRA_STOP_AT': '15'}, expect_rc=1)
        ok &= sh(f'smoke_{name}', ftargs) and ft_done(f)
        rep[f'{name}_ft_pool_and_20_steps_sec'] = time.time() - t0
        t0 = time.time()
        ok &= score_step(f'smoke_{name}', ['eval_real.py', 'score', '--panel', real_panel, '--runs', f, '--device', DEV],
                         f / f'cells_{real_panel.stem}.npz', real_panel)
        rep[f'{name}_score_real_sec'] = time.time() - t0
        if not ok:
            break
    if ok:          # 5. TabPFN v2: download, a few tasks of a synthetic and a real panel
        t0 = time.time()
        ok &= sh('smoke_tabpfn', ['tabpfn_colab.py', '--panel', syn_panel, real_panel, '--out', d / 'tabpfn',
                                  '--device', DEV])
        ok &= all(ok_npz(d / 'tabpfn' / f'cells_{p.stem}.npz', p, TABPFN_METRICS)
                  for p in (syn_panel, real_panel))
        if ok:
            tim = json.loads((d / 'tabpfn' / 'train.json').read_text())['timing']
            rep['tabpfn_sec_per_call'] = sum(v['seconds'] for v in tim.values()) / sum(v['calls'] for v in tim.values())
        rep['tabpfn_smoke_sec'] = time.time() - t0
    if not ok:
        raise SystemExit(f'SMOKE TEST FAILED. Send the logs in {LOGS} (smoke_*.log).')
    n_tab = 1792 * 4 + 6 * 718 + 2 * 23      # TabPFN calls: synthetic tasks x 4 bank sizes + real episodes
    rep['estimate_hours'] = dict(
        train=round(len(SEEDS) * STEPS * (rep['pfn_L_sec_per_step'] + rep['lct_L_sec_per_step']) / 3600, 1),
        tabpfn=round(n_tab * rep['tabpfn_sec_per_call'] / 3600, 1))
    rep_path.write_text(json.dumps(rep, indent=1))
    print('smoke: passed', json.dumps(rep, indent=1), flush=True)
    return True


def train_one(name, script, s):
    run = RUNS / f'{name}_s{s}'
    if train_done(run):
        return True
    return sh(f'train_{name}_s{s}', [script, '--seed', s, '--out', run, '--device', DEV, '--ckpt-every', '1000']
              + RECIPE + EXTRA_ARGS[name]) and train_done(run)


def score_syn(name, s, tags):
    run = RUNS / f'{name}_s{s}'
    if not train_done(run):
        return False
    good = True
    for tag in tags:
        good &= score_step(f'score_{name}_s{s}', ['evaluate2.py', 'model', '--run', run, '--panel', panel(tag), '--device', DEV],
                           run / f'cells_{tag}.npz', panel(tag))
    return good


def finetune(name, s):
    init = RUNS / f'{name}_s{s}'
    if not train_done(init):
        return False
    good = True
    for ds, tag in zip(NEW, V4_REAL):
        run = REAL / ds / f'{name}_s{s}_ft'
        if not ft_done(run):
            if not sh(f'ft_{name}_s{s}_{ds}', ['finetune_real.py', '--dataset', ds, '--init', init, '--out', run,
                                                '--device', DEV, '--beijing', DATA / 'beijing', '--pool-cache', POOLS] + FT):
                good = False; continue
        good &= score_step(f'ft_{name}_s{s}_{ds}', ['eval_real.py', 'score', '--panel', panel(tag), '--runs', run, '--device', DEV],
                           run / f'cells_{tag}.npz', panel(tag))
    return good


def tabpfn(tags):
    todo = [t for t in tags if not ok_npz(RUNS / 'tabpfn_v2' / f'cells_{t}.npz', panel(t), TABPFN_METRICS)]
    if todo:
        if sh('tabpfn_v2', ['tabpfn_colab.py', '--panel'] + [panel(t) for t in todo] + ['--out', RUNS / 'tabpfn_v2',
                                                                                      '--version', 'v2', '--device', DEV]):
            if not all(ok_npz(RUNS / 'tabpfn_v2' / f'cells_{t}.npz', panel(t), TABPFN_METRICS) for t in todo):
                FAILED.append('tabpfn_v2_scores')


def stage_main():
    for s in SEEDS:
        for name, script in MODELS:
            if train_one(name, script, s):
                score_syn(name, s, [H1, H3])
                finetune(name, s)
        if s == SEEDS[0]:
            tabpfn([H1, H3] + V4_REAL)
            stage_pack()


def v4_complete(seeds=SEEDS):
    need = [(RUNS / f'{n}_s{s}' / f'cells_{t}.npz', t, GAUSSIAN_METRICS)
            for n, _ in MODELS for s in seeds for t in (H1, H3)]
    need += [(REAL / ds / f'{n}_s{s}_ft' / f'cells_{t}.npz', t, GAUSSIAN_METRICS)
             for n, _ in MODELS for s in seeds for ds, t in zip(NEW, V4_REAL)]
    need += [(RUNS / 'tabpfn_v2' / f'cells_{t}.npz', t, TABPFN_METRICS) for t in V4_REAL]
    return all(ok_npz(p, panel(t), metrics) for p, t, metrics in need)


def stage_confirm(seeds=SEEDS):
    seeds = list(seeds)
    if seeds != list(SEEDS):
        raise ValueError('Protocol v4 confirmation requires all three seeds; no early endpoint calculations')
    if not v4_complete(seeds):
        print(f'confirm: inputs for seeds {seeds} incomplete; skipped', flush=True); return
    out = RUNS / ('confirm_v4.json' if seeds == list(SEEDS) else f'early_v4_seeds{"".join(map(str, seeds))}.json')
    sh('confirm', ['confirm4.py', '--runs', RUNS, '--real', REAL, '--cache', PANELS_DIR, '--out', out, '--seeds'] + seeds)


def stage_extra():
    for s in SEEDS:
        for name, _ in MODELS:
            score_syn(name, s, OLD_SYN)
    tabpfn(PANELS)


def stage_pack():
    import torch
    z = ROOT / 'mira_v4_results.zip'; tmp = ROOT / 'mira_v4_results.zip.part'
    info = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), device=DEV, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, failed=FAILED,
                commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CODE, capture_output=True, text=True).stdout.strip())
    try:
        import tabpfn as tp
        info['tabpfn'] = tp.__version__
    except Exception:
        pass
    n = 0
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('manifest.json', json.dumps(info, indent=1))
        for p in (ROOT / 'run_identity.json', ROOT / 'sessions.jsonl'):
            if p.exists():
                zf.write(p, p.name)
        for base in (RUNS, REAL, ROOT / 'smoke', LOGS):
            if not base.exists():
                continue
            for p in sorted(base.rglob('*')):
                if p.is_file() and (p.suffix in ('.json', '.log')
                                    or (p.name.startswith('cells_') and p.suffix == '.npz' and '.tmp' not in p.name)):
                    zf.write(p, p.relative_to(ROOT)); n += 1
    tmp.replace(z)
    print(f'pack: {n} files -> {z} ({z.stat().st_size / 1e6:.1f} MB). Download it from Drive and attach it in the chat.',
          flush=True)


def stage_status():
    rows = []
    for s in SEEDS:
        for name, _ in MODELS:
            run = RUNS / f'{name}_s{s}'
            j = run / 'train.json'
            steps = json.loads(j.read_text()).get('steps_done', 0) if j.exists() else 0
            syn = sum((run / f'cells_{t}.npz').exists() for t in [H1, H3] + OLD_SYN)
            ft = sum((REAL / ds / f'{name}_s{s}_ft' / f'cells_{t}.npz').exists() for ds, t in zip(NEW, V4_REAL))
            rows.append(f'{name}_s{s}: trained {steps}/{STEPS}, synthetic panels {syn}/{2 + len(OLD_SYN)}, '
                        f'fine-tuned+scored {ft}/3')
    tab = sum((RUNS / 'tabpfn_v2' / f'cells_{t}.npz').exists() for t in PANELS)
    rows.append(f'tabpfn_v2: {tab}/{len(PANELS)} panels')
    rows.append(f'endpoints: {"done" if (RUNS / "confirm_v4.json").exists() else "pending"}')
    print('\n'.join(rows), flush=True)
    if (RUNS / 'confirm_v4.json').exists():
        print((RUNS / 'confirm_v4.json').read_text(), flush=True)


def main():
    global ROOT, DATA, RUNS, REAL, LOGS, POOLS, PANELS_DIR, DEV, ENV, SMOKE_RECIPE, SMOKE_FT, RECIPE, STEPS, FT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stages', nargs='*', default=['all'],
                    choices=['all', 'data', 'panels', 'smoke', 'main', 'confirm', 'extra', 'pack', 'status'])
    ap.add_argument('--root', required=True, help='results folder, e.g. /content/drive/MyDrive/mira_v4')
    ap.add_argument('--panels', required=True, help='clone of the lifted-cavity-panels branch')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--test', action='store_true', help='tiny models and steps: tests this runner on a CPU, results meaningless')
    a = ap.parse_args()
    ROOT = Path(a.root).resolve(); PANELS_DIR = Path(a.panels).resolve()
    DATA, RUNS, REAL, LOGS, POOLS = ROOT / 'data', ROOT / 'runs', ROOT / 'runs_real', ROOT / 'logs', ROOT / 'cache' / 'src'
    ROOT.mkdir(parents=True, exist_ok=True)
    import torch
    DEV = a.device
    if DEV == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('No GPU. In Colab: Runtime > Change runtime type > GPU, then run all cells again.')
    ENV = dict(os.environ, PYTHONUNBUFFERED='1', MIRA_BEIJING_DIR=str(DATA / 'beijing'), MIRA_LC_CACHE=str(PANELS_DIR))
    SMOKE_RECIPE = RECIPE[2:]                   # the recipe without --steps
    SMOKE_FT = ['--steps', '20', '--ckpt-every', '10']
    if a.test:
        ARCH_T = ['--d', '16', '--layers', '1', '--heads', '2', '--ff', '32']
        RECIPE = ['--steps', '60', '--batch-tasks', '4', '--queries', '4', '--lr', '5e-4', '--warmup', '10'] + ARCH_T
        STEPS = 60; SMOKE_RECIPE = RECIPE[2:]
        FT = ['--steps', '10', '--episodes', '64', '--ckpt-every', '500']; SMOKE_FT = ['--steps', '20', '--ckpt-every', '10', '--episodes', '64']
    stages = ['data', 'panels', 'smoke', 'main', 'confirm', 'pack', 'extra', 'pack'] if 'all' in a.stages else a.stages
    if 'status' not in stages:
        stage_status()
    for st in stages:
        print(f'\n##### {st} #####', flush=True)
        dict(data=stage_data, panels=stage_panels, smoke=stage_smoke, main=stage_main, confirm=stage_confirm,
             extra=stage_extra, pack=stage_pack, status=stage_status)[st]()
    print('\n##### status #####'); stage_status()
    if FAILED:
        print('FAILED steps (re-run the same command; if they fail again, send the zip):', sorted(set(FAILED)), flush=True)
        sys.exit(1)
    print('ALL DONE' if (RUNS / 'confirm_v4.json').exists() and 'all' in a.stages else 'stage(s) done', flush=True)


if __name__ == '__main__':
    main()
