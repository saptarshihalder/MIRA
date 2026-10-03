"""Preserve packaging failure and freeze final manual call within latest cap."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    source=ROOT/'configs/large_native_v2.json'
    plan=json.loads(source.read_text())
    for relative,digest in plan['file_sha256'].items():
        assert checksum(ROOT/relative)==digest,relative
    launcher=(ROOT/'experiments/policy_adapter_v1/launch_large_native_v2.py').read_text()
    launcher=launcher.replace('large_native_a100_v2','large_native_a100_v3').replace('large_native_a100_repair','large_native_a100_final')
    launcher=launcher.replace('large_native_v2.json','large_native_v3.json').replace('mira-large-native-a100-repair','mira-large-native-a100-final')
    (ROOT/'experiments/policy_adapter_v1/launch_large_native_v3.py').write_text(launcher,encoding='utf-8',newline='\n')
    target=ROOT/'configs/large_native_v3.json'
    assert not target.exists(),'Preserve final protocol'
    plan.update(phase='large_native_a100_final',paid_phase_cap_usd=.4,timeout_seconds=300,
                startup_timeout_seconds=60,cpu_physical_cores=2,memory_gib=8,
                parent_repair_protocol_sha256=checksum(source),
                packaging_repair='Mount full infra source tree at /opt/mira/infra. Same data, numerical repair, seeds, models and evaluation; no scientific selection.')
    for relative in ['infra/modal_large_native_v3.py','experiments/policy_adapter_v1/launch_large_native_v3.py']:
        plan['file_sha256'][relative]=checksum(ROOT/relative)
    ledger=ROOT/'artifacts/manifests/compute_ledger.json'
    entries=json.loads(ledger.read_text())
    assert not any(entry.get('phase')==plan['phase'] for entry in entries)
    policy=json.loads((ROOT/'artifacts/manifests/compute_policy.json').read_text())
    assert sum(entry['reserved_usd'] for entry in entries)+.4+policy['reproduction_reserve_usd']<=policy['total_cap_usd']
    for entry in entries:
        if entry.get('phase')=='large_native_a100_repair':
            entry.update(remote_function_calls=1,failure='Cloud input validation found missing mounted v1 wrapper; no child training invocation',
                         provider_charge_status='Failed GPU startup counted; reservation retained')
    target.write_text(json.dumps(plan,indent=2),encoding='utf-8',newline='\n')
    entries.append(dict(run_id='large_native_a100_v3',phase=plan['phase'],reserved_usd=.4,status='reserved',
                        date='2026-10-03',model='Same large native CUDA evaluation; packaging and numerical repair',
                        protocol_sha256=checksum(target),max_calls=1,gpu='A100-40GB',timeout_seconds=300,
                        startup_timeout_seconds=60,retries=0,actual_charged_usd=None,
                        authorization='Existing human GPU request and conditional extension, within latest $20 cap and protected $3 reserve'))
    ledger.write_text(json.dumps(entries,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(protocol_sha256=checksum(target),total_reserved_usd=sum(entry['reserved_usd'] for entry in entries))))
