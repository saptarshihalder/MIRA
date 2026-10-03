"""One bounded A100 architecture pilot; no idle GPU, retries or schedule."""
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
        protocol = root / 'configs/bridge_v2.json'
        if hashlib.sha256(protocol.read_bytes()).hexdigest() != protocol_sha256:
            raise ValueError('Protocol identity mismatch')
        plan = json.loads(protocol.read_text())
        for relative, digest in plan['file_sha256'].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen source changed: ' + relative)
        workspace = Path(tempfile.mkdtemp(prefix='mira_bridge_v2_'))
        start = time.monotonic()
        with (workspace / 'runner.log').open('w') as log:
            try:
                result = subprocess.run([sys.executable, str(root / 'experiments/bridge_v2/run.py'),
                    '--config', str(protocol), '--out', str(workspace / 'results'), '--device', 'cuda'],
                    cwd=root, stdout=log, stderr=subprocess.STDOUT,
                    timeout=plan['child_timeout_seconds'])
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        seconds = time.monotonic() - start
        runtime = dict(seconds=seconds, exit_code=code, requested_gpu='A100-40GB',
                       cpu_physical_cores=2, memory_gib=8, timeout_seconds=600,
                       startup_timeout_seconds=60, retries=0,
                       active_compute_estimate_usd=seconds * .00062696,
                       charge_status='Resource estimate excludes load/build/provider charges; $.80 provision retained')
        (workspace / 'wrapper_runtime.json').write_text(json.dumps(runtime, indent=2))
        print((workspace / 'runner.log').read_text()[-2500:], flush=True)
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
    app = modal.App('mira-bridge-v2-a100')
    image = image.add_local_dir(ROOT / 'experiments/bridge_v2', '/opt/mira/experiments/bridge_v2')
    image = image.add_local_dir(ROOT / 'experiments/bridge_v1', '/opt/mira/experiments/bridge_v1')
    image = image.add_local_dir(ROOT / 'configs', '/opt/mira/configs')
    image = image.add_local_file(ROOT / 'docs/BRIDGE_V2_PROTOCOL.md', '/opt/mira/docs/BRIDGE_V2_PROTOCOL.md')
    image = image.add_local_file(ROOT / 'infra/modal_bridge_v2.py', '/opt/mira/infra/modal_bridge_v2.py')
    run_training = app.function(image=image, gpu='A100-40GB', cpu=(2, 2), memory=(8192, 8192),
        timeout=600, startup_timeout=60, min_containers=0, max_containers=1,
        scaledown_window=2, retries=0, serialized=True,
        name='train_bridge')(worker_factory())
