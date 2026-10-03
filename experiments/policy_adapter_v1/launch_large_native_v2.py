"""One reserved invocation of the deployed app; preserves failures and outputs."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import modal

ROOT=Path(__file__).resolve().parents[2]
RUN_ID='large_native_a100_v2'
PHASE='large_native_a100_repair'


def main():
    ledger=ROOT/'artifacts/manifests/compute_ledger.json'
    policy=json.loads((ROOT/'artifacts/manifests/compute_policy.json').read_text())
    entries=json.loads(ledger.read_text())
    reserved=sum(float(entry['reserved_usd']) for entry in entries)
    if reserved+policy['reproduction_reserve_usd']>policy['total_cap_usd']:
        raise ValueError('Latest total cap and reproduction reserve exceeded')
    phase=[entry for entry in entries if entry.get('phase')==PHASE]
    protocol=ROOT/'configs/large_native_v2.json'
    digest=hashlib.sha256(protocol.read_bytes()).hexdigest()
    if len(phase)!=1 or phase[0]['run_id']!=RUN_ID or phase[0]['status']!='reserved' or phase[0]['protocol_sha256']!=digest:
        raise ValueError('Requires one matching unused reservation; no automatic retry')
    plan=json.loads(protocol.read_text())
    for relative,expected in plan['file_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen input changed: '+relative)
    target=ROOT/'artifacts/runs'/RUN_ID
    if target.exists():
        raise ValueError('Preserve existing run')
    target.mkdir(parents=True)
    phase[0].update(status='invoking',remote_invocations_authorized=1)
    ledger.write_text(json.dumps(entries,indent=2))
    try:
        function=modal.Function.from_name('mira-large-native-a100-repair','evaluate_large_native')
        with modal.enable_output():
            payload=function.remote(digest)
        with zipfile.ZipFile(io.BytesIO(payload.pop('archive'))) as archive:
            if any(not (target/item.filename).resolve().is_relative_to(target.resolve()) for item in archive.infolist()):
                raise ValueError('Unsafe result archive')
            archive.extractall(target)
        phase[0].update(payload,status='completed' if payload['exit_code']==0 else 'failed',
                        remote_function_calls=1,results=str(target),actual_charged_usd=None)
        print(json.dumps(phase[0]),flush=True)
        if payload['exit_code']:
            raise SystemExit(payload['exit_code'])
    except BaseException as error:
        phase[0].update(status='failed',error=repr(error),remote_function_calls=phase[0].get('remote_function_calls','unverified; do not retry'))
        raise
    finally:
        ledger.write_text(json.dumps(entries,indent=2))


if __name__=='__main__':
    main()
