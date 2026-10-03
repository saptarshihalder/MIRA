"""Freeze prepared inputs, inspect partitions without scores, and reserve cost."""
import hashlib
import json
from pathlib import Path
import large_native as large

ROOT=large.ROOT


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    prepared=ROOT/'artifacts/runs/large_native_data/manifest.json'
    manifest=json.loads(prepared.read_text())
    manifest['brfss_schema_sha256']=checksum(ROOT/'configs/brfss_common_variables_v1.json')
    prepared.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    protocol=ROOT/'configs/large_native_v1.json'
    if protocol.exists():
        raise ValueError('Preserve the frozen protocol')
    plan=dict(large.DEFAULTS,tasks=manifest['tasks'],training_seeds=[171001,171002,171003],
              primary_model='native_scratch',phase='large_native_a100',paid_phase_cap_usd=4.,
              max_remote_calls=1,gpu='A100-40GB',timeout_seconds=3600,startup_timeout_seconds=180,
              retries=0,total_cap_usd=20.,reproduction_reserve_usd=3.,
              raw_sha256=manifest['raw_sha256'],raw_records=manifest['raw_records'],
              scope='User-authorized larger native development; two domains, three correlated tasks, no confirmation',
              protocol_document='docs/LARGE_NATIVE_PROTOCOL.md')
    paths=['experiments/policy_adapter_v1/'+name for name in ['pilot.py','native.py','mixture.py',
            'large_native.py','prepare_large_native.py','build_brfss_schema.py','launch_large_native.py']]
    paths+=['infra/modal_large_native.py','configs/brfss_common_variables_v1.json',
            'configs/brfss_layout_2024.json','configs/brfss_layout_2025.json',
            'artifacts/runs/large_native_data/manifest.json']
    paths+=['artifacts/runs/large_native_data/'+task['npz'] for task in plan['tasks']]
    plan['file_sha256']={relative:checksum(ROOT/relative) for relative in paths}
    boundaries=[]
    for task in plan['tasks']:
        data=large.load_data(ROOT/'artifacts/runs/large_native_data'/task['npz'],task['columns'])
        parts=large.partitions(data,plan,plan['training_seeds'][0])
        boundaries.append(dict(task=task['name'],rows=len(data['y']),source_fit=len(parts['fit']),
                               meta_episodes=len(parts['training']),validation_groups=len(parts['validation']),
                               development_groups=len(parts['development']),omitted=parts['omitted']))
    plan['metadata_only_partition_check']=boundaries
    policy=json.loads((ROOT/'artifacts/manifests/compute_policy.json').read_text())
    ledger=ROOT/'artifacts/manifests/compute_ledger.json'
    entries=json.loads(ledger.read_text())
    if any(entry.get('phase')==plan['phase'] for entry in entries):
        raise ValueError('Paid phase already exists')
    if sum(float(entry['reserved_usd']) for entry in entries)+4.+policy['reproduction_reserve_usd']>policy['total_cap_usd']:
        raise ValueError('Latest compute cap exceeded')
    protocol.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    entries.append(dict(run_id='large_native_a100_v1',phase=plan['phase'],reserved_usd=4.,status='reserved',
                        date='2026-10-03',model='CUDA XGBoost plus native neural/linear/calibration controls',
                        protocol_sha256=checksum(protocol),max_calls=1,gpu='A100-40GB',
                        timeout_seconds=3600,retries=0,actual_charged_usd=None,
                        authorization='Human requested larger 2026-release datasets and powerful live Modal GPU; latest total cap $20'))
    ledger.write_text(json.dumps(entries,indent=2),encoding='utf-8')
    (ROOT/'artifacts/manifests/large_native_source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(protocol_sha256=checksum(protocol),partitions=boundaries,total_reserved_usd=sum(e['reserved_usd'] for e in entries))))
