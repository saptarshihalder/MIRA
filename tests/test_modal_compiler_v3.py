"""Exercise the repaired parent worker against the actual frozen runner."""
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

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/mask_compiler'))
from run_backbone import run
from mira.data import generate_episode

spec=importlib.util.spec_from_file_location('compiler_v3_local_tests',ROOT/'infra/modal_compiler_v3.py')
wrapper=importlib.util.module_from_spec(spec)
prior_modal=sys.modules.get('modal')
sys.modules['modal']=SimpleNamespace(is_local=lambda:False)
try:
    spec.loader.exec_module(wrapper)
finally:
    if prior_modal is None:
        del sys.modules['modal']
    else:
        sys.modules['modal']=prior_modal


class Estimator:
    classes_=np.array([0,1])
    def fit(self,X,y):
        return self
    def predict_proba(self,X):
        return np.column_stack([np.full(len(X),.25),np.full(len(X),.75)])
    def get_params(self,deep=False):
        return {'mock':True}


def episode():
    ep=generate_episode('pairwise',74009,.9,context=32,queries=16)
    return {'episode_id':'integration','kind':'synthetic','dataset':None,'fold':None,'order':2,'gamma':.9,'seed':74009,
        'support':ep.support,'query':ep.query,'native_context':ep.support.observations.values,
        'native_query':ep.query.observations.values,'selected_columns':np.arange(4),
        'targets':ep.targets.labels,'oracle':ep.targets.oracle,'groups':None}


def unpack(payload):
    with zipfile.ZipFile(io.BytesIO(payload['archive'])) as archive:
        return {name:archive.read(name) for name in archive.namelist()}


def test_worker_integration_actual_runner_has_fresh_child_and_flat_archive(monkeypatch):
    calls=[];factory_calls=[];commits=[]
    original_subprocess_run=subprocess.run
    def factory(*args):
        factory_calls.append(args)
        return Estimator()
    plan={'protocol_version':'local_integration_only','views':['native','identity','random_basis'],
          'file_sha256':{},'checkpoint_sha256':{}}
    def subprocess_fixture(command,**kwargs):
        if not isinstance(command,(list,tuple)) or len(command)<2 or 'run_backbone.py' not in str(command[1]):
            return original_subprocess_run(command,**kwargs)
        out=Path(command[-1]);calls.append(out)
        assert not out.exists() and (out.parent/'runner.log').exists()
        assert Path(kwargs['stdout'].name).parent==out.parent
        result=run(plan,'local_test','tabicl:v2',out,factory=factory,selectors={},episodes=[episode()],
                   capture_lock=False,check_cache=False)
        assert result['status']=='complete' and result['completed_cells']==3
        kwargs['stdout'].write('Actual runner fixture completed.\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',subprocess_fixture)
    cache=SimpleNamespace(commit=lambda:commits.append(True))
    payloads=[wrapper.cloud_worker('configs/unexecuted_v3.json','tabicl:v2',cache) for _ in range(2)]
    assert calls[0]!=calls[1] and len(factory_calls)==6 and len(commits)==2
    for payload in payloads:
        files=unpack(payload)
        assert payload['exit_code']==0 and {'manifest.json','results.json','runner.log','modal_runtime.json'}<=set(files)
        assert not any(name.startswith('results/') for name in files)
        assert json.loads(files['manifest.json'])['completed_cells']==3
        assert 'episodes/integration.npz' in files and 'predictions/integration_identity.npz' in files
        with np.load(io.BytesIO(files['predictions/integration_identity.npz']),allow_pickle=False) as prediction:
            assert np.all(prediction['probability']==.75)


def test_worker_actual_runner_failure_preserves_case_and_skips_rest(monkeypatch):
    called=[]
    original_subprocess_run=subprocess.run
    def fail(*args):
        called.append(args)
        raise RuntimeError('fixture checkpoint unavailable')
    def subprocess_fixture(command,**kwargs):
        if not isinstance(command,(list,tuple)) or len(command)<2 or 'run_backbone.py' not in str(command[1]):
            return original_subprocess_run(command,**kwargs)
        plan={'protocol_version':'local_only','views':['identity','random_basis'],'file_sha256':{},'checkpoint_sha256':{}}
        result=run(plan,'local_test','tabicl:v2',Path(command[-1]),factory=fail,selectors={},episodes=[episode()],
                   capture_lock=False,check_cache=False)
        return SimpleNamespace(returncode=0 if result['status']=='complete' else 1)
    monkeypatch.setattr(subprocess,'run',subprocess_fixture)
    payload=wrapper.cloud_worker('configs/unexecuted_v3.json','tabicl:v2',SimpleNamespace(commit=lambda:None))
    files=unpack(payload)
    assert payload['exit_code']==1 and len(called)==1
    assert {'episodes/integration.npz','episodes/integration.json','errors.json','runner.log','modal_runtime.json'}<=set(files)
    assert 'checkpoint unavailable' in files['errors.json'].decode()
    assert json.loads(files['manifest.json'])['completed_cells']==0


def test_preexisting_child_still_triggers_real_preservation_guard(monkeypatch):
    def subprocess_fixture(command,**kwargs):
        out=Path(command[-1]);out.mkdir();(out/'prior.txt').write_bytes(b'original preserved')
        run({'protocol_version':'local_only','views':['native'],'file_sha256':{},'checkpoint_sha256':{}},
            'local_test','tabicl:v2',out,selectors={},episodes=[episode()],capture_lock=False,check_cache=False)
        raise AssertionError('Nonempty-output preservation guard was bypassed')
    monkeypatch.setattr(subprocess,'run',subprocess_fixture)
    payload=wrapper.cloud_worker('configs/unexecuted_v3.json','tabicl:v2',SimpleNamespace(commit=lambda:None))
    files=unpack(payload)
    assert payload['exit_code']==1 and files['prior.txt']==b'original preserved'
    assert 'Preserve old outputs' in files['wrapper_error.json'].decode() and 'runner.log' in files


def test_timeout_preserves_partial_child_and_parent_log(monkeypatch):
    def timeout(command,**kwargs):
        out=Path(command[-1]);out.mkdir();(out/'partial.json').write_text('{"completed_cells":1}')
        kwargs['stdout'].write('before timeout\n')
        raise subprocess.TimeoutExpired(command,780)
    monkeypatch.setattr(subprocess,'run',timeout)
    payload=wrapper.cloud_worker('configs/unexecuted_v3.json','tabicl:v2',SimpleNamespace(commit=lambda:None))
    files=unpack(payload)
    assert payload['exit_code']==124 and {'partial.json','runner.log','modal_runtime.json'}<=set(files)


@pytest.fixture
def ledger_helpers(monkeypatch):
    spec=importlib.util.spec_from_file_location('compiler_v3_ledger_fixture',ROOT/'tests/test_modal_budget.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    base=fixture.wrapper.__wrapped__(monkeypatch)
    monkeypatch.setitem(sys.modules,'infra.modal_mechanism',base)
    return base


def repair_fixture(tmp_path):
    spec=importlib.util.spec_from_file_location('compiler_v2_protocol_fixture',ROOT/'tests/test_modal_compiler_v2.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    base_path,base,_=fixture.amendment(tmp_path)
    old=[{'phase':'mask_compiler_backbone','run_id':'first','model':'tabpfn:v2','reserved_usd':.5,'status':'failed'},
         {'phase':'mask_compiler_backbone','run_id':'second','model':'tabicl:v2','reserved_usd':.5,'status':'failed'}]
    ledger=tmp_path/'artifacts/manifests/compute_ledger.json';ledger.write_text(json.dumps(old))
    plan=json.loads(json.dumps(base));plan['protocol_version']='gpu_repair_v1'
    plan['gpu_repair']={'authorization':{'origin':'human_user','date':'2026-10-02','request':'use gpu man, use modal and colab pro'},
        'phase':'mask_compiler_gpu_repair','model':'tabicl:v2','max_calls':1,'reservation_usd':.5,
        'base_protocol_file':'configs/v2.json','base_protocol_sha256':fixture.sha(base_path),
        'failed_pilot_ledger_sha256':wrapper.pilot_digest(old)}
    source=tmp_path/'infra/modal_compiler_v3.py';source.write_bytes(b'GPU repair wrapper')
    plan['file_sha256']['infra/modal_compiler_v3.py']=fixture.sha(source)
    plan['file_sha256']['configs/v2.json']=fixture.sha(base_path)
    path=tmp_path/'configs/repair.json';fixture.write_freeze(path,plan)
    return path,plan,ledger,old


def test_repair_freeze_requires_human_gpu_request_and_unchanged_experiment(tmp_path):
    path,plan,ledger,old=repair_fixture(tmp_path)
    assert wrapper.load_repair(path,tmp_path)[0]==plan
    plan['gpu_repair']['authorization']['origin']='model_inference'
    path.write_text(json.dumps(plan));path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='human October 2'):
        wrapper.load_repair(path,tmp_path)
    plan['gpu_repair']['authorization']['origin']='human_user'
    plan['views'][0],plan['views'][1]=plan['views'][1],plan['views'][0]
    path.write_text(json.dumps(plan));path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='cannot change the experimental plan'):
        wrapper.load_repair(path,tmp_path)


def test_one_new_half_dollar_phase_preserves_both_failed_pilot_entries(tmp_path,ledger_helpers):
    path,plan,ledger,old=repair_fixture(tmp_path)
    with pytest.raises(ValueError,match='fresh TabICL'):
        wrapper.reserve_repair(ledger,{'total_cap_usd':26},'wrong','tabpfn:v2',plan,'new')
    wrapper.reserve_repair(ledger,{'total_cap_usd':26},'repair','tabicl:v2',plan,'new')
    entries=json.loads(ledger.read_text())
    assert entries[:2]==old and len(entries)==3 and sum(entry['reserved_usd'] for entry in entries)==1.5
    assert entries[-1]['phase']=='mask_compiler_gpu_repair' and entries[-1]['reserved_usd']==.5
    with pytest.raises(ValueError,match='allowance exhausted'):
        wrapper.reserve_repair(ledger,{'total_cap_usd':26},'retry','tabicl:v2',plan,'new')


@pytest.mark.parametrize('status',['reserved','failed','completed'])
def test_any_prior_repair_status_consumes_entire_allowance(tmp_path,ledger_helpers,status):
    path,plan,ledger,old=repair_fixture(tmp_path)
    ledger.write_text(json.dumps(old+[{'phase':'mask_compiler_gpu_repair','reserved_usd':.5,'status':status}]))
    before=ledger.read_bytes()
    with pytest.raises(ValueError,match='allowance exhausted'):
        wrapper.reserve_repair(ledger,{'total_cap_usd':26},'new','tabicl:v2',plan,'new')
    assert ledger.read_bytes()==before


def test_altered_failed_entries_or_global_reserve_block_repair(tmp_path,ledger_helpers):
    path,plan,ledger,old=repair_fixture(tmp_path)
    altered=json.loads(json.dumps(old));altered[0]['extra']='modified history';ledger.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='unchanged and charged'):
        wrapper.reserve_repair(ledger,{'total_cap_usd':26},'repair','tabicl:v2',plan,'new')
    ledger.write_text(json.dumps(old+[{'run_id':'other','phase':'other','reserved_usd':22}]))
    with pytest.raises(ValueError,match=r'preserve the \$3 reserve'):
        wrapper.reserve_repair(ledger,{'total_cap_usd':26},'repair','tabicl:v2',plan,'new')


def test_local_failure_consumes_repair_and_never_retries(tmp_path,ledger_helpers,monkeypatch):
    path,plan,ledger,old=repair_fixture(tmp_path)
    monkeypatch.setattr(wrapper,'IS_LOCAL',True);monkeypatch.setattr(wrapper,'ROOT',tmp_path)
    calls=[]
    def remote(*args):
        calls.append(args)
        raise RuntimeError('mock remote unavailable')
    monkeypatch.setattr(wrapper,'run_compiler',SimpleNamespace(remote=remote),raising=False)
    monkeypatch.setattr(wrapper,'cache',SimpleNamespace(),raising=False)
    with pytest.raises(RuntimeError,match='mock remote unavailable'):
        wrapper.main('repair','tabicl:v2',str(path))
    entries=json.loads(ledger.read_text())
    assert entries[:2]==old and entries[-1]['status']=='failed' and len(calls)==1
    with pytest.raises(ValueError,match='allowance exhausted'):
        wrapper.main('another','tabicl:v2',str(path))
    assert len(calls)==1


def test_historical_freezes_unchanged():
    for version in (1,2):
        protocol=ROOT/f'configs/trained_compiler_validation_v{version}.json'
        if not protocol.exists():
            continue
        assert protocol.with_suffix('.sha256').read_text().strip()==hashlib.sha256(protocol.read_bytes()).hexdigest()
        plan=json.loads(protocol.read_text())
        assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==checksum for path,checksum in plan['file_sha256'].items())


def test_actual_modal_sdk_hydration_and_by_value_serialization(tmp_path):
    python=ROOT.parent/'.venv-modal/Scripts/python.exe'
    if not python.exists():
        pytest.skip('Local Modal SDK runtime unavailable')
    source=tmp_path/'root/modal_compiler_v3.py';source.parent.mkdir();shutil.copyfile(ROOT/'infra/modal_compiler_v3.py',source)
    code=r'''
import builtins,importlib.util,sys,modal
from modal._serialization import serialize,deserialize
modal.is_local=lambda:False
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=='infra' or name.startswith('infra.') or name in ('run_backbone','feasibility') or name.startswith('mira'):
        raise AssertionError('Remote hydration imported local module: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
modal.App=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Remote App registration'))
spec=importlib.util.spec_from_file_location('hydration_v3',sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
assert module.ROOT.as_posix()=='/opt/mira' and not hasattr(module,'app')
payload=serialize(module.cloud_worker);del sys.modules[spec.name]
restored=deserialize(payload,None)
assert restored.__closure__ is None and '<locals>' in restored.__qualname__
assert not any(name in restored.__globals__ for name in ('ROOT','cache','image','infra','run_backbone'))
print('Actual Modal hydration/serialization passed')
'''
    result=subprocess.run([str(python),'-c',code,str(source)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
