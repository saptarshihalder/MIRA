"""Preserve failed protocol; authorize one bounded numerical repair under $20."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    wrapper=(ROOT/'infra/modal_large_native.py').read_text()
    wrapper=wrapper.replace('large_native_v1.json','large_native_v2.json').replace("large_native.py'","large_native_v2.py'")
    wrapper=wrapper.replace("'--out',str(workspace/'results')","'--out',str(workspace/'results'),'--config',str(protocol)")
    wrapper=wrapper.replace('timeout=3300','timeout=1320').replace('timeout_seconds=3600','timeout_seconds=1500').replace('timeout=3600','timeout=1500')
    wrapper=wrapper.replace('$4 retained','$1.50 repair reservation retained').replace('mira-large-native-a100','mira-large-native-a100-repair')
    wrapper=wrapper.replace("    image=image.add_local_dir(ROOT/'configs','/opt/mira/configs')",
        "    image=image.add_local_dir(ROOT/'configs','/opt/mira/configs')\n    image=image.add_local_file(ROOT/'infra/modal_large_native_v2.py','/opt/mira/infra/modal_large_native_v2.py')")
    (ROOT/'infra/modal_large_native_v2.py').write_text(wrapper,encoding='utf-8',newline='\n')
    launcher=(ROOT/'experiments/policy_adapter_v1/launch_large_native.py').read_text()
    launcher=launcher.replace('large_native_a100_v1','large_native_a100_v2').replace("PHASE='large_native_a100'","PHASE='large_native_a100_repair'")
    launcher=launcher.replace('large_native_v1.json','large_native_v2.json').replace('mira-large-native-a100','mira-large-native-a100-repair')
    launcher=launcher.replace("remote_function_calls='unverified; do not retry'","remote_function_calls=phase[0].get('remote_function_calls','unverified; do not retry')")
    (ROOT/'experiments/policy_adapter_v1/launch_large_native_v2.py').write_text(launcher,encoding='utf-8',newline='\n')
    old_protocol=ROOT/'configs/large_native_v1.json'
    plan=json.loads(old_protocol.read_text())
    for relative,digest in plan['file_sha256'].items():
        if checksum(ROOT/relative)!=digest:
            raise ValueError('Preserve original frozen inputs: '+relative)
    target=ROOT/'configs/large_native_v2.json'
    if target.exists():
        raise ValueError('Preserve existing repair protocol')
    plan.update(phase='large_native_a100_repair',paid_phase_cap_usd=1.5,timeout_seconds=1500,
                parent_protocol_sha256=checksum(old_protocol),
                numerical_repair='Same ridge calibration objective; safeguarded Newton solver and 1e-7 stationarity certificate. All data, partitions, seeds, primary model and hyperparameters unchanged. Partial v1 development remains used.')
    for relative in ['experiments/policy_adapter_v1/large_native_v2.py',
                     'experiments/policy_adapter_v1/launch_large_native_v2.py','infra/modal_large_native_v2.py']:
        plan['file_sha256'][relative]=checksum(ROOT/relative)
    ledger=ROOT/'artifacts/manifests/compute_ledger.json'
    entries=json.loads(ledger.read_text())
    if any(entry.get('phase')==plan['phase'] for entry in entries):
        raise ValueError('Repair already reserved')
    policy=json.loads((ROOT/'artifacts/manifests/compute_policy.json').read_text())
    if sum(entry['reserved_usd'] for entry in entries)+1.5+policy['reproduction_reserve_usd']>policy['total_cap_usd']:
        raise ValueError('Latest cap exceeded')
    for entry in entries:
        if entry.get('phase')=='large_native_a100':
            entry.update(remote_function_calls=1,app_id='ap-29sdBNwyGIJse79T72LRNz',
                         failure='Calibration baseline L-BFGS-B ABNORMAL line search; partial CUDA checkpoints/predictions preserved')
    target.write_text(json.dumps(plan,indent=2),encoding='utf-8',newline='\n')
    entries.append(dict(run_id='large_native_a100_v2',phase=plan['phase'],reserved_usd=1.5,status='reserved',
                        date='2026-10-03',model='Same large native CUDA protocol; numerical calibration repair',
                        protocol_sha256=checksum(target),max_calls=1,gpu='A100-40GB',timeout_seconds=1500,
                        startup_timeout_seconds=180,retries=0,actual_charged_usd=None,
                        authorization='Human requested Modal code execution and allowed extension if needed; cap $20 and reproduction reserve remain binding'))
    ledger.write_text(json.dumps(entries,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(dict(protocol_sha256=checksum(target),total_reserved_usd=sum(entry['reserved_usd'] for entry in entries))))
