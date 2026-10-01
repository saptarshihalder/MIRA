"""Separate one-call GPU repair; original two failed reservations remain consumed.

V2 placed runner.log inside the runner output before invocation, triggering its
old-output preservation guard. This repair places that log in a parent workspace
and gives the runner a previously nonexistent child. A frozen repair protocol
records the human's October 2 GPU instruction before the separate $.50 reservation.
"""
from pathlib import Path
import modal

IS_LOCAL=modal.is_local()
ROOT=Path(__file__).resolve().parents[1] if IS_LOCAL else Path('/opt/mira')


def make_worker():
    """By-value worker without local SDK or infrastructure globals."""
    def worker(protocol_relative: str,model: str,cache_volume):
        import io
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import time
        import zipfile

        root=Path('/opt/mira')
        protocol=(root/protocol_relative).resolve()
        if not protocol.is_relative_to(root.resolve()) or model not in ('tabpfn:v2','tabicl:v2'):
            raise ValueError('Invalid frozen remote protocol/backbone')
        workspace=Path(tempfile.mkdtemp(prefix='mira_compiler_v3_'))
        out=workspace/'results'  # Remains nonexistent until the runner creates it.
        environment=os.environ.copy()
        environment['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('src','.','experiments/mask_compiler'))
        command=[sys.executable,str(root/'experiments/mask_compiler/run_backbone.py'),'--protocol',str(protocol),
                 '--model',model,'--cache-root','/cache','--out',str(out)]
        started=time.monotonic()
        try:
            with (workspace/'runner.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=str(root),env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=780)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
        except Exception as error:
            code=1
            (workspace/'wrapper_error.json').write_text(json.dumps({'error':repr(error)}),encoding='utf-8')
        seconds=time.monotonic()-started
        try:
            cache_volume.commit()
        except Exception as error:
            code=code or 1
            (workspace/'cache_commit_error.json').write_text(json.dumps({'error':repr(error)}),encoding='utf-8')
        (workspace/'modal_runtime.json').write_text(json.dumps({'seconds':seconds,'exit_code':code,'command':command,
            'gpu':'T4','max_seconds':900,'wrapper':'modal_compiler_v3','remote_root':str(root),
            'phase':'mask_compiler_gpu_repair',
            'estimated_active_compute_usd':seconds*(.000164+2*.0000131+8*.00000222),
            'invoice_status':'estimate; excludes startup/build/provider rounding'},indent=2),encoding='utf-8')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in workspace.rglob('*'):
                if path.is_file():
                    relative=path.relative_to(out) if path.is_relative_to(out) else path.relative_to(workspace)
                    archive.write(path,relative)
        return {'archive':buffer.getvalue(),'seconds':seconds,'exit_code':code}
    return worker


cloud_worker=make_worker()


def pilot_digest(entries):
    import hashlib
    import json
    old=[entry for entry in entries if entry.get('phase')=='mask_compiler_backbone']
    return hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_repair(protocol,root):
    import json
    import sys
    sys.path.insert(0,str(root/'src'))
    sys.path.insert(0,str(root/'experiments/mask_compiler'))
    from run_backbone import load_protocol,contained,sha
    plan,digest=load_protocol(protocol,root)
    repair=plan.get('gpu_repair',{})
    authorization=repair.get('authorization',{})
    request=authorization.get('request','')
    if (authorization.get('origin')!='human_user' or authorization.get('date')!='2026-10-02' or
            not isinstance(request,str) or 'gpu' not in request.lower() or 'modal' not in request.lower()):
        raise ValueError('Frozen human October 2 GPU/Modal instruction required')
    if any(repair.get(key)!=value for key,value in {'phase':'mask_compiler_gpu_repair','model':'tabicl:v2',
             'max_calls':1,'reservation_usd':.5}.items()):
        raise ValueError('Repair permits only one bounded $.50 TabICL call')
    if not isinstance(repair.get('failed_pilot_ledger_sha256'),str) or len(repair['failed_pilot_ledger_sha256'])!=64:
        raise ValueError('Freeze the complete original failed pilot ledger entries')
    files=plan['file_sha256']
    if 'infra/modal_compiler_v3.py' not in files:
        raise ValueError('Freeze the repaired wrapper before paid compute')
    uploaded=('src/','scripts/','tests/','experiments/mask_compiler/','infra/','configs/','artifacts/manifests/',
              'artifacts/reports/trained_compiler_v1/','artifacts/reports/matched_compiler_v1/','artifacts/data/natural_compiler/')
    if any(path!='pyproject.toml' and not path.startswith(uploaded) for path in files):
        raise ValueError('Frozen hash map includes files outside the uploaded remote layout')
    base_file=contained(root,repair.get('base_protocol_file',''))
    checksum=repair.get('base_protocol_sha256')
    if (sha(base_file)!=checksum or base_file.with_suffix('.sha256').read_text().strip()!=checksum or
            files.get(repair['base_protocol_file'])!=checksum):
        raise ValueError('Historical v2 protocol checksum differs')
    base=json.loads(base_file.read_text())
    if not base.get('frozen') or plan['protocol_version']==base.get('protocol_version'):
        raise ValueError('Require a distinct frozen repair protocol')
    keys=('models','ensembles','synthetic','real','views','compiler_checkpoint','linear_checkpoint','deepsets_checkpoint',
          'checkpoint_sha256','decision_gate','budgets')
    if any(plan.get(key)!=base.get(key) for key in keys):
        raise ValueError('GPU repair cannot change the experimental plan, weights, checkpoints or initial budgets')
    if any(files.get(path)!=checksum for path,checksum in base['file_sha256'].items()):
        raise ValueError('Historical executed source/data hashes must remain unchanged')
    return plan,digest


def validate_repair_ledger(entries,plan):
    old=[entry for entry in entries if entry.get('phase')=='mask_compiler_backbone']
    if (len(old)!=2 or any(entry.get('status')!='failed' or entry.get('reserved_usd')!=.5 for entry in old) or
            pilot_digest(entries)!=plan['gpu_repair']['failed_pilot_ledger_sha256']):
        raise ValueError('Keep both original failed pilot reservations unchanged and charged')
    repair=[entry for entry in entries if entry.get('phase')=='mask_compiler_gpu_repair']
    if repair:
        raise ValueError('GPU repair allowance exhausted: one call/$.50 including failure or crash; no retry')


def reserve_repair(ledger,budget,run_id,model,plan,digest):
    import json
    import math
    from datetime import datetime,timezone
    from infra.modal_mechanism import ledger_lock,atomic_ledger_write,validate_run_id
    validate_run_id(run_id)
    with ledger_lock(ledger):
        entries=json.loads(ledger.read_text()) if ledger.exists() else []
        validate_repair_ledger(entries,plan)
        if model!='tabicl:v2' or any(entry.get('run_id')==run_id for entry in entries):
            raise ValueError('Only a fresh TabICL repair run ID is permitted')
        values=[float(entry['reserved_usd']) for entry in entries]
        cap=float(budget['total_cap_usd'])
        if cap!=26 or any(not math.isfinite(value) or value<0 for value in values) or sum(values)+.5>cap-3+1e-9:
            raise ValueError('Global $26 cap reached or invalid; preserve the $3 reserve')
        entries.append({'run_id':run_id,'phase':'mask_compiler_gpu_repair','model':model,'reserved_usd':.5,
                        'status':'reserved','max_seconds':900,'protocol_sha256':digest,'protocol_version':plan['protocol_version'],
                        'failed_pilot_ledger_sha256':plan['gpu_repair']['failed_pilot_ledger_sha256'],
                        'authorization':plan['gpu_repair']['authorization'],'date':datetime.now(timezone.utc).isoformat()})
        atomic_ledger_write(ledger,entries)


def main(run_id:str,model:str,protocol_file:str='configs/compiler_gpu_repair_v1.json'):
    if not IS_LOCAL:
        raise RuntimeError('Local entrypoint cannot run in a cloud container')
    import io
    import json
    import zipfile
    from infra.modal_mechanism import finish_call,validate_run_id
    from run_backbone import natural_episodes
    validate_run_id(run_id)
    protocol=Path(protocol_file).resolve()
    if not protocol.is_relative_to(ROOT.resolve()):
        raise ValueError('Protocol must remain inside the uploaded repository')
    plan,digest=load_repair(protocol,ROOT)
    list(natural_episodes(plan,ROOT))
    target=ROOT/'artifacts/runs'/run_id
    if target.exists():
        raise ValueError('Preserve previous output; use a new run ID')
    ledger=ROOT/'artifacts/manifests/compute_ledger.json'
    budget=json.loads((ROOT/'configs/budget.json').read_text())
    reserve_repair(ledger,budget,run_id,model,plan,digest)
    updates={'status':'failed'}
    try:
        payload=run_compiler.remote(str(protocol.relative_to(ROOT)).replace('\\','/'),model,cache)
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload['archive'])) as archive:
            if any(not (target/member.filename).resolve().is_relative_to(target.resolve()) for member in archive.infolist()):
                raise ValueError('Unsafe returned archive path')
            archive.extractall(target)
        updates.update(status='completed' if payload['exit_code']==0 else 'failed',active_seconds=payload['seconds'],
                       exit_code=payload['exit_code'],results=str(target))
        print(json.dumps({'run_id':run_id,'model':model,**updates}))
        if payload['exit_code']:
            raise SystemExit(payload['exit_code'])
    except BaseException as error:
        updates['error']=repr(error)
        raise
    finally:
        finish_call(ledger,run_id,updates)


if IS_LOCAL:
    import sys
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/'experiments/mask_compiler'))
    from infra.modal_mechanism import cache,image
    app=modal.App('mira-mask-compiler-gpu-repair')
    compiler_image=image
    for directory in ('experiments/mask_compiler','infra','configs','tests','artifacts/manifests',
                      'artifacts/reports/trained_compiler_v1','artifacts/reports/matched_compiler_v1','artifacts/data/natural_compiler'):
        compiler_image=compiler_image.add_local_dir(ROOT/directory,'/opt/mira/'+directory)
    run_compiler=app.function(image=compiler_image,gpu='T4',cpu=(2,2),memory=(8192,8192),timeout=900,
        max_containers=1,scaledown_window=2,retries=0,serialized=True,name='run_compiler',volumes={'/cache':cache})(cloud_worker)
    main=app.local_entrypoint()(main)
