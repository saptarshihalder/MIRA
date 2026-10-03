"""Manually invoked deployed A100 app; no schedule, endpoint or idle GPU."""
from pathlib import Path
import modal

LOCAL=modal.is_local()
ROOT=Path(__file__).resolve().parents[1] if LOCAL else Path('/opt/mira')


def worker_factory():
    def worker(protocol_sha256):
        import hashlib
        import io
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import time
        import zipfile
        root=Path('/opt/mira')
        protocol=root/'configs/large_native_v2.json'
        if hashlib.sha256(protocol.read_bytes()).hexdigest()!=protocol_sha256:
            raise ValueError('Deployed protocol differs from reserved protocol')
        plan=json.loads(protocol.read_text())
        for relative,digest in plan['file_sha256'].items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen input changed: '+relative)
        workspace=Path(tempfile.mkdtemp(prefix='mira_large_a100_'))
        start=time.monotonic()
        with (workspace/'runner.log').open('w') as log:
            try:
                result=subprocess.run([sys.executable,str(root/'experiments/policy_adapter_v1/large_native_v2.py'),
                                       '--out',str(workspace/'results'),'--config',str(protocol)],cwd=root,stdout=log,
                                      stderr=subprocess.STDOUT,timeout=1320)
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=124
        runtime=dict(seconds=time.monotonic()-start,exit_code=code,requested_gpu='A100-40GB',
                     cpu_physical_cores=4,memory_gib=32,timeout_seconds=1500,retries=0,
                     active_compute_estimate_usd=(time.monotonic()-start)*.00070644,
                     charge_status='Estimate excludes load, build and other provider charges; $1.50 repair reservation retained')
        (workspace/'wrapper_runtime.json').write_text(json.dumps(runtime,indent=2))
        print((workspace/'runner.log').read_text()[-4000:],flush=True)
        memory=io.BytesIO()
        with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in workspace.rglob('*'):
                if path.is_file():
                    archive.write(path,path.relative_to(workspace))
        return dict(archive=memory.getvalue(),**runtime)
    return worker


if LOCAL:
    import sys
    sys.path.insert(0,str(ROOT))
    from infra.modal_mechanism import image
    app=modal.App('mira-large-native-a100-repair')
    image=image.add_local_dir(ROOT/'experiments/policy_adapter_v1','/opt/mira/experiments/policy_adapter_v1')
    image=image.add_local_dir(ROOT/'configs','/opt/mira/configs')
    image=image.add_local_file(ROOT/'infra/modal_large_native_v2.py','/opt/mira/infra/modal_large_native_v2.py')
    image=image.add_local_file(ROOT/'infra/modal_large_native_v2.py','/opt/mira/infra/modal_large_native_v2.py')
    image=image.add_local_dir(ROOT/'artifacts/runs/large_native_data','/opt/mira/artifacts/runs/large_native_data')
    run_evaluation=app.function(image=image,gpu='A100-40GB',cpu=(4,4),memory=(32768,32768),
        timeout=1500,startup_timeout=180,min_containers=0,max_containers=1,scaledown_window=2,
        retries=0,serialized=True,name='evaluate_large_native')(worker_factory())
