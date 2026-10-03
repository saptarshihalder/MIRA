"""Separate one-call phase explicitly authorized by the user's GPU instruction."""
from pathlib import Path
import modal

LOCAL=modal.is_local()
ROOT=Path(__file__).resolve().parents[1] if LOCAL else Path('/opt/mira')


def worker_factory():
    def worker():
        import io
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import time
        import zipfile
        root=Path('/opt/mira')
        workspace=Path(tempfile.mkdtemp(prefix='mira_native_gpu_'))
        out=workspace/'results'
        start=time.monotonic()
        with (workspace/'runner.log').open('w') as log:
            try:
                result=subprocess.run([sys.executable,str(root/'experiments/policy_adapter_v1/gpu_native.py'),
                    '--out',str(out)],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=780)
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=124
        (workspace/'wrapper_runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,exit_code=code,gpu='T4')))
        memory=io.BytesIO()
        with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in workspace.rglob('*'):
                if path.is_file(): archive.write(path,path.relative_to(workspace))
        return dict(archive=memory.getvalue(),exit_code=code,seconds=time.monotonic()-start)
    return worker


def main(run_id:str='policy_native_gpu_v1'):
    import hashlib
    import io
    import json
    import zipfile
    from infra.modal_mechanism import finish_call
    protocol=ROOT/'configs/policy_gpu_native_v1.json'
    plan=json.loads(protocol.read_text())
    for relative,digest in plan['file_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest: raise ValueError('Frozen input changed: '+relative)
    target=ROOT/'artifacts/runs'/run_id
    if target.exists(): raise ValueError('Preserve existing run')
    ledger=ROOT/'artifacts/manifests/compute_ledger.json'
    entries=json.loads(ledger.read_text())
    phase=[e for e in entries if e.get('phase')=='policy_native_gpu']
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    if len(phase)!=1 or phase[0]['run_id']!=run_id or phase[0]['status']!='reserved' or phase[0]['protocol_sha256']!=digest:
        raise ValueError('One-call phase requires a matching reservation before Modal launch; no retry')
    updates=dict(status='failed')
    try:
        payload=remote_worker.remote()
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload['archive'])) as archive:
            if any(not (target/p.filename).resolve().is_relative_to(target.resolve()) for p in archive.infolist()): raise ValueError('Unsafe archive')
            archive.extractall(target)
        updates.update(status='completed' if payload['exit_code']==0 else 'failed',exit_code=payload['exit_code'],
            active_seconds=payload['seconds'],results=str(target))
        print(json.dumps(updates))
        if payload['exit_code']: raise SystemExit(payload['exit_code'])
    except BaseException as error:
        updates['error']=repr(error)
        raise
    finally: finish_call(ledger,run_id,updates)


if LOCAL:
    import sys
    sys.path.insert(0,str(ROOT))
    from infra.modal_mechanism import image
    app=modal.App('mira-policy-native-gpu')
    image=image.add_local_dir(ROOT/'experiments/policy_adapter_v1','/opt/mira/experiments/policy_adapter_v1')
    image=image.add_local_dir(ROOT/'configs','/opt/mira/configs')
    image=image.add_local_dir(ROOT/'artifacts/reports/policy_mixture_v1','/opt/mira/artifacts/reports/policy_mixture_v1')
    image=image.add_local_file(ROOT/'artifacts/runs/diabetes_native_raw/diabetic_data.csv','/opt/mira/artifacts/runs/diabetes_native_raw/diabetic_data.csv')
    remote_worker=app.function(image=image,gpu='T4',cpu=(2,2),memory=(8192,8192),timeout=900,max_containers=1,
        scaledown_window=2,retries=0,serialized=True,name='train_native_mixtures')(worker_factory())
    main=app.local_entrypoint()(main)
