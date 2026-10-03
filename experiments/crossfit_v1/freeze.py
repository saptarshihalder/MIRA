"""Reserve and freeze one bounded engineering call; never consume confirmation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [f'experiments/crossfit_v1/{name}' for name in
         ('model.py', 'run.py', 'launch.py', 'freeze.py')]
FILES += ['infra/modal_crossfit_v1.py', 'docs/CROSSFIT_PROTOCOL.md',
          'experiments/spectral_v1/model.py', 'experiments/bridge_v1/run.py', 'experiments/bridge_v1/model.py']


def main():
    protocol = ROOT / 'configs/crossfit_v1.json'
    plan = json.loads(protocol.read_text())
    ledger = ROOT / 'artifacts/manifests/compute_ledger.json'
    entries = json.loads(ledger.read_text())
    if any(entry['run_id'] == plan['run_id'] for entry in entries):
        raise ValueError('Reservation already exists; do not automatically retry')
    if plan['updates'] != 500 or plan['automatic_retries'] != 0:
        raise ValueError('Engineering bounds changed')
    seeds = set(plan['train_seeds'] + plan['validation_seeds'] + plan['development_seeds'])
    if seeds & set(range(98000, 98020)):
        raise ValueError('Protected confirmation seed')
    policy = json.loads((ROOT / 'artifacts/manifests/compute_policy.json').read_text())
    provision = float(plan['gpu_reservation_usd'])
    total = sum(float(entry['reserved_usd']) for entry in entries) + provision
    if provision < .8 or total + policy['reproduction_reserve_usd'] > policy['total_cap_usd']:
        raise ValueError('Insufficient budget; preserve reproduction reserve')
    # A10040GB+2CPU+8GiB <=662 resource seconds plus $.38 contingency in $.80.
    plan['file_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in FILES}
    protocol.write_text(json.dumps(plan, indent=2) + '\n', newline='\n')
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    entry = dict(run_id=plan['run_id'], phase='crossfit_development',
        date=datetime.now(timezone.utc).isoformat(), status='reserved', reserved_usd=provision,
        protocol_sha256=digest, gpu='A100-40GB', max_calls=1, retries=0,
        timeout_seconds=600, startup_timeout_seconds=60, child_timeout_seconds=540,
        cpu_physical_cores=2, memory_gib=8, actual_charged_usd=None,
        authorization='Human requested recursive label embedding architecture and GPU where needed; cap20/reserve3 preserved',
        maximum_resource_compute_usd=.4151,
        charge_status='Worst-case provision includes provider contingency; invoice unverified')
    entries.append(entry)
    ledger.write_text(json.dumps(entries, indent=2) + '\n', newline='\n')
    manifest = dict(frozen_at_utc=entry['date'], protocol_sha256=digest, files=plan['file_sha256'],
        engineering_only=True, confirmation_seeds_untouched='98000-98019',
        total_provisions_usd=total, reproduction_reserve_usd=policy['reproduction_reserve_usd'])
    (ROOT / 'artifacts/manifests/crossfit_v1_freeze.json').write_text(
        json.dumps(manifest, indent=2) + '\n', newline='\n')
    print(json.dumps(dict(protocol_sha256=digest, provision=provision, total_provisions=total)))


if __name__ == '__main__':
    main()
