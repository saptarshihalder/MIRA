"""Protocol v6 GPU work on Google Colab (docs/OCT20_PLAN.md). Stage 'source' trains on the synthetic prior only, so
it scores nothing and may run before the v6 protocol is frozen.

    python colab_v6.py --root /content/drive/MyDrive/mira_v6 source    # UPT s1-s3, LCT s3, PFN s3 (shared recipe)
    python colab_v6.py --root /content/drive/MyDrive/mira_v6 status
    python colab_v6.py --root /content/drive/MyDrive/mira_v6 pack      # <root>/mira_v6_source.zip: weights + metadata

Re-run the same command after any disconnect: finished runs are skipped and unfinished ones resume from their
Drive checkpoints (resume.py).
"""
import argparse, json, subprocess, sys, time, zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parent
RECIPE = ['--steps', '30000', '--batch-tasks', '16', '--queries', '16', '--lr', '1e-3', '--warmup', '1000',
          '--d', '64', '--layers', '4', '--heads', '4', '--ff', '128', '--device', 'cuda']   # = runs/lct_s1, runs/pfn_s1
JOBS = [('upt_s1', 'lct_train.py', ['--model', 'upt', '--seed', '1', '--prefetch', '4']),
        ('upt_s2', 'lct_train.py', ['--model', 'upt', '--seed', '2', '--prefetch', '4']),
        ('upt_s3', 'lct_train.py', ['--model', 'upt', '--seed', '3', '--prefetch', '4']),
        ('lct_s3', 'lct_train.py', ['--model', 'lct', '--seed', '3', '--prefetch', '4']),
        ('pfn_s3', 'pfn_train.py', ['--seed', '3'])]


def done(out):
    return (out / 'model.pt').exists() and not (out / 'ckpt.pt').exists() and \
        json.loads((out / 'train.json').read_text()).get('steps_done') == 30000


def source(root):
    procs = []
    for name, script, extra in JOBS:
        out = root / 'runs' / name
        if out.exists() and done(out):
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


def pack(root):
    z = root / 'mira_v6_source.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for name, *_ in JOBS:
            out = root / 'runs' / name
            assert done(out), f'{name} unfinished'
            for fn in ('model.pt', 'train.json'):
                f.write(out / fn, f'runs/{name}/{fn}')
    print(z, z.stat().st_size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('stage', choices=['source', 'status', 'pack'])
    a = ap.parse_args()
    r = Path(a.root); r.mkdir(parents=True, exist_ok=True)
    dict(source=source, status=status, pack=pack)[a.stage](r)
