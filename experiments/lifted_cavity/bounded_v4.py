"""Run a pinned v4 stage with immutable input identity and a hard wall-time cap.

Linux/Colab only; stopping is not an automatic retry. Checkpoints remain in --root.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def bind_manifest(root, identity, metadata):
    dest = root / 'run_identity.json'
    if dest.exists():
        if json.loads(dest.read_text())['identity'] != identity:
            raise ValueError('Run identity changed; use a NEW root. Existing checkpoints are preserved.')
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refusing to adopt an existing root without an identity manifest.')
    root.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(identity=identity, metadata=metadata), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--panels', required=True)
    ap.add_argument('--minutes', type=float, required=True)
    ap.add_argument('stages', nargs='+', choices=['data', 'panels', 'smoke', 'main', 'confirm', 'pack', 'status'])
    a = ap.parse_args()
    if os.name != 'posix' or not 0 < a.minutes <= 720:
        raise SystemExit('Requires Linux and a wall-time bound in (0, 720] minutes.')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; no CPU fallback for this run.')
    root, panels = Path(a.root).resolve(), Path(a.panels).resolve()
    sources = sorted(CODE.glob('*.py')) + [REPO / 'docs/LIFTED_CAVITY_PROTOCOL_V4.md']
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'tabpfn')}
    if packages['tabpfn'] != '9.1.0':
        raise SystemExit('Protocol v4 requires tabpfn==9.1.0.')
    identity = dict(schema=1, source={str(p.relative_to(REPO)): digest(p) for p in sources},
                    panels={p.name: digest(p) for p in sorted(panels.glob('*.pt'))},
                    panel_manifest=digest(panels / 'SHA256SUMS'), packages=packages,
                    device=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    bind_manifest(root, identity, dict(commit=commit, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    lock = root / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('RUNNING.lock exists. Verify the earlier process stopped before removing this lock.')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    cmd = [sys.executable, '-u', str(CODE / 'colab_v4.py'), '--root', str(root), '--panels', str(panels)] + a.stages
    start = time.time(); proc = None; rc = 1; timed_out = False
    try:
        proc = subprocess.Popen(cmd, cwd=CODE, start_new_session=True)
        try:
            rc = proc.wait(timeout=a.minutes * 60)
        except subprocess.TimeoutExpired:
            timed_out = True; rc = 124
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        with open(root / 'sessions.jsonl', 'a') as f:
            f.write(json.dumps(dict(start_unix=start, seconds=time.time()-start, minutes_cap=a.minutes,
                                   stages=a.stages, exit_code=rc, timed_out=timed_out, automatic_retries=0)) + '\n')
        lock.unlink(missing_ok=True)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
