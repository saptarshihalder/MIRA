"""Local-only hydration, serialization, amendment, and bounded-call checks."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/mask_compiler"))
from run_backbone import load_protocol


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def wrapper(monkeypatch):
    fake = SimpleNamespace(is_local=lambda:False)
    monkeypatch.setitem(sys.modules,"modal",fake)
    spec = importlib.util.spec_from_file_location("compiler_v2_remote_test",ROOT/"infra/modal_compiler_v2.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def ledger_helpers(monkeypatch):
    spec = importlib.util.spec_from_file_location("compiler_ledger_fixture",ROOT/"tests/test_modal_budget.py")
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    base = fixture.wrapper.__wrapped__(monkeypatch)
    monkeypatch.setitem(sys.modules,"infra.modal_mechanism",base)
    return base


def amendment(tmp_path):
    required = ["experiments/mask_compiler/run_backbone.py","infra/modal_compiler.py","infra/modal_mechanism.py",
                "pyproject.toml","experiments/mask_compiler/feasibility.py","experiments/mask_compiler/train_selector.py",
                "experiments/mask_compiler/matched_training.py","experiments/mask_compiler/compiler.npz",
                "experiments/mask_compiler/linear.npz","src/mira/data.py"]
    files = {}
    for relative in required:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"historical freeze")
        files[relative]=sha(p)
    previous = {"frozen":True,"protocol_version":"v1","models":["tabpfn:v2","tabicl:v2"],"ensembles":4,
        "synthetic":{"orders":[2,4],"gammas":[0,.9],"seeds":[74000,74001,74002],"context":128,"queries":256},
        "real":[],"views":["native","identity","trained","heuristic","matched_linear","cv_basis","random_basis"],
        "compiler_checkpoint":"experiments/mask_compiler/compiler.npz","linear_checkpoint":"experiments/mask_compiler/linear.npz",
        "checkpoint_sha256":{"pin.ckpt":"test"},
        "file_sha256":files}
    (tmp_path/"configs").mkdir()
    old = tmp_path/"configs/v1.json";old.write_text(json.dumps(previous));old.with_suffix(".sha256").write_text(sha(old))
    directory=tmp_path/"artifacts/manifests";directory.mkdir(parents=True)
    log=directory/"failed.log";log.write_text("Container hydration: ModuleNotFoundError: No module named 'infra'")
    evidence={"run_id":"first","protocol_sha256":sha(old),"stage":"container_hydration","model_invocations":0,
        "prediction_files":0,"app_id":"ap-test","app_stopped":True,"reserved_usd":.5,"reservation_consumed":True,
        "log_file":str(log.relative_to(tmp_path)).replace('\\','/'),"log_sha256":sha(log)}
    proof=directory/"failure.json";proof.write_text(json.dumps(evidence))
    plan=json.loads(json.dumps(previous));plan["protocol_version"]="v2"
    plan["predecessor"]={"run_id":"first","protocol_file":"configs/v1.json","protocol_sha256":sha(old),
        "evidence_file":"artifacts/manifests/failure.json","evidence_sha256":sha(proof)}
    new_source=tmp_path/"infra/modal_compiler_v2.py";new_source.write_bytes(b"new hydration-safe wrapper")
    for p in (old,proof,log,new_source):
        plan["file_sha256"][str(p.relative_to(tmp_path)).replace('\\','/')]=sha(p)
    protocol=tmp_path/"configs/v2.json"
    write_freeze(protocol,plan)
    (tmp_path/"configs/budget.json").write_text('{"total_cap_usd":26}')
    return protocol,plan,evidence


def write_freeze(path,plan):
    path.write_text(json.dumps(plan));path.with_suffix(".sha256").write_text(sha(path))


def test_actual_sdk_remote_layout_import_and_serialization_have_no_local_imports(tmp_path):
    python=ROOT.parent/".venv-modal/Scripts/python.exe"
    if not python.exists():
        pytest.skip("Local Modal SDK runtime unavailable")
    module=tmp_path/"root/modal_compiler_v2.py";module.parent.mkdir();shutil.copyfile(ROOT/"infra/modal_compiler_v2.py",module)
    code=r'''
import builtins,importlib.util,sys
import modal
from modal._serialization import serialize,deserialize
modal.is_local=lambda:False
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=='infra' or name.startswith('infra.') or name in ('run_backbone','feasibility') or name.startswith('mira'):
        raise AssertionError('Remote hydration imported local module: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
modal.App=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Remote hydration constructed an App'))
spec=importlib.util.spec_from_file_location('hydration_probe',sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
assert module.ROOT.as_posix()=='/opt/mira' and not hasattr(module,'app')
payload=serialize(module.cloud_worker)
del sys.modules[spec.name]
restored=deserialize(payload,None)
assert '<locals>' in restored.__qualname__ and restored.__closure__ is None
assert not any(name in restored.__globals__ for name in ('ROOT','cache','image','run_backbone','infra'))
print('remote top-level import and actual Modal serialization passed; bytes='+str(len(payload)))
'''
    result=subprocess.run([str(python),"-c",code,str(module)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert "passed" in result.stdout


def test_worker_uses_remote_paths_unique_outputs_and_archives_timeout(wrapper,monkeypatch):
    calls=[]
    def fake(command,**kwargs):
        calls.append((command,kwargs))
        out=Path(command[-1]);(out/"partial.json").write_text('{"completed_cells":1}')
        kwargs["stdout"].write("retained runner log\n")
        if len(calls)==2:
            raise subprocess.TimeoutExpired(command,780)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,"run",fake)
    commits=[];cache=SimpleNamespace(commit=lambda:commits.append(True))
    payloads=[wrapper.cloud_worker("configs/v2.json","tabicl:v2",cache) for _ in range(2)]
    assert calls[0][0][-1]!=calls[1][0][-1] and len(commits)==2
    assert payloads[0]["exit_code"]==0 and payloads[1]["exit_code"]==124
    for command,kwargs in calls:
        assert command[1]==str(Path('/opt/mira')/'experiments/mask_compiler/run_backbone.py')
        assert kwargs['cwd']==str(Path('/opt/mira')) and kwargs['timeout']==780
        assert 'experiments' in kwargs['env']['PYTHONPATH']
    for payload in payloads:
        with zipfile.ZipFile(io.BytesIO(payload['archive'])) as archive:
            assert {'partial.json','runner.log','modal_runtime.json'}<=set(archive.namelist())


def test_amendment_preserves_historical_experiment_and_hashes(wrapper,tmp_path):
    path,plan,evidence=amendment(tmp_path)
    assert wrapper.load_successor(path,tmp_path)[2]==evidence
    plan['views'][0],plan['views'][1]=plan['views'][1],plan['views'][0]
    write_freeze(path,plan)
    with pytest.raises(ValueError,match='cannot change experiments'):
        wrapper.load_successor(path,tmp_path)


def test_claimed_inference_or_existing_predictions_reject_amendment(wrapper,tmp_path):
    path,plan,evidence=amendment(tmp_path)
    result=tmp_path/'artifacts/runs/first/predictions';result.mkdir(parents=True);(result/'actual.npz').write_bytes(b'prediction')
    with pytest.raises(ValueError,match='inference artifacts'):
        wrapper.load_successor(path,tmp_path)
    (result/'actual.npz').unlink()
    evidence['model_invocations']=1
    proof=tmp_path/plan['predecessor']['evidence_file'];proof.write_text(json.dumps(evidence))
    plan['predecessor']['evidence_sha256']=sha(proof);plan['file_sha256'][plan['predecessor']['evidence_file']]=sha(proof)
    write_freeze(path,plan)
    with pytest.raises(ValueError,match='zero inference'):
        wrapper.load_successor(path,tmp_path)


def test_consumed_predecessor_cost_and_other_backbone_only(wrapper,ledger_helpers,tmp_path):
    path,plan,evidence=amendment(tmp_path)
    ledger=tmp_path/'artifacts/manifests/ledger.json'
    first={'run_id':'first','phase':'mask_compiler_backbone','model':'tabpfn:v2','reserved_usd':.5,'status':'failed',
           'protocol_sha256':evidence['protocol_sha256'],'failure_stage':'container_hydration','app_id':evidence['app_id'],
           'app_stopped':True,'model_invocations':0}
    ledger.write_text(json.dumps([first]))
    with pytest.raises(ValueError,match='no retry'):
        wrapper.reserve_successor(ledger,{'total_cap_usd':26},'retry','tabpfn:v2',plan,sha(path),evidence)
    wrapper.reserve_successor(ledger,{'total_cap_usd':26},'second','tabicl:v2',plan,sha(path),evidence)
    entries=json.loads(ledger.read_text());assert entries[0]==first and sum(e['reserved_usd'] for e in entries)==1
    with pytest.raises(ValueError,match='two calls'):
        wrapper.reserve_successor(ledger,{'total_cap_usd':26},'third','tabicl:v2',plan,sha(path),evidence)


def test_global_cap_and_wrong_cost_proof_rejected(wrapper,ledger_helpers,tmp_path):
    path,plan,evidence=amendment(tmp_path);ledger=tmp_path/'ledger.json'
    first={'run_id':'first','phase':'mask_compiler_backbone','model':'tabpfn:v2','reserved_usd':.5,'status':'failed',
           'protocol_sha256':evidence['protocol_sha256'],'failure_stage':'container_hydration','app_id':evidence['app_id'],
           'app_stopped':True,'model_invocations':0}
    ledger.write_text(json.dumps([first]))
    with pytest.raises(ValueError,match='Global compute cap'):
        wrapper.reserve_successor(ledger,{'total_cap_usd':3.5},'second','tabicl:v2',plan,sha(path),evidence)
    first['status']='completed';ledger.write_text(json.dumps([first]))
    with pytest.raises(ValueError,match='charged predecessor'):
        wrapper.reserve_successor(ledger,{'total_cap_usd':26},'second','tabicl:v2',plan,sha(path),evidence)
