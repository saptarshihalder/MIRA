"""One explicitly reserved bounded recursive JEPA A100 engineering worker."""
from pathlib import Path
import modal

LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[1] if LOCAL else Path('/opt/mira')


def worker_factory():
    def worker(protocol_sha256):
        import hashlib
        import io
        import json
        import subprocess
        import sys
        import tempfile
        import time
        import zipfile
        from pathlib import Path
        root = Path('/opt/mira')
        protocol = root / 'configs/spectral_v1.json'
        if hashlib.sha256(protocol.read_bytes()).hexdigest() != protocol_sha256:
            raise ValueError('Frozen recursive JEPA protocol hash mismatch')
        plan = json.loads(protocol.read_text())
        if plan['child_timeout_seconds'] != 540 or plan['gpu_timeout_seconds'] != 600 or plan['automatic_retries'] != 0:
            raise ValueError('Keep committed600second bound and zero retries')
        for relative, digest in plan['file_sha256'].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen input changed: ' + relative)
        workspace = Path(tempfile.mkdtemp(prefix='mira_recursive_jepa_'))
        started = time.monotonic()
        with (workspace / 'runner.log').open('w') as log:
            try:
                result = subprocess.run([sys.executable, str(root / 'experiments/spectral_v1/run.py'),
                    '--config', str(protocol), '--out', str(workspace / 'results'), '--device', 'cuda'],
                    cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=540)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        seconds = time.monotonic() - started
        runtime = dict(seconds=seconds, exit_code=code, requested_gpu='A100-40GB',
            cpu_physical_cores=2, memory_gib=8, timeout_seconds=600, startup_timeout_seconds=60,
            retries=0, min_containers=0, max_containers=1,
            active_compute_estimate_usd=seconds * .00062696,
            charge_status='Resource estimate excludes load/build/provider charges; $.80 reservation retained')
        (workspace / 'wrapper_runtime.json').write_text(json.dumps(runtime, indent=2))
        print((workspace / 'runner.log').read_text()[-2000:], flush=True)
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in workspace.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(workspace))
        return dict(archive=memory.getvalue(), **runtime)
    return worker


if LOCAL:
    import sys
    sys.path.insert(0, str(ROOT))
    from infra.modal_mechanism import image
    app = modal.App('mira-spectral-v1')
    image = image.add_local_dir(ROOT / 'experiments/spectral_v1', '/opt/mira/experiments/spectral_v1')
    image = image.add_local_dir(ROOT / 'experiments/bridge_v1', '/opt/mira/experiments/bridge_v1')
    image = image.add_local_file(ROOT / 'configs/spectral_v1.json', '/opt/mira/configs/spectral_v1.json')
    image = image.add_local_file(ROOT / 'docs/SPECTRAL_PROTOCOL.md', '/opt/mira/docs/SPECTRAL_PROTOCOL.md')
    image = image.add_local_file(ROOT / 'infra/modal_spectral_v1.py', '/opt/mira/infra/modal_spectral_v1.py')
    run_training = app.function(image=image, gpu='A100-40GB', cpu=(2, 2), memory=(8192, 8192),
        timeout=600, startup_timeout=60, min_containers=0, max_containers=1, scaledown_window=2,
        retries=0, serialized=True, name='train_spectral')(worker_factory())
