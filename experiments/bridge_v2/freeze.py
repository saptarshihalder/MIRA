"""Freeze one architecture gate and reserve its worst-case GPU provision."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
FILES = ['experiments/bridge_v2/model.py', 'experiments/bridge_v2/run.py',
         'experiments/bridge_v2/launch.py', 'experiments/bridge_v2/freeze.py',
         'infra/modal_bridge_v2.py', 'docs/BRIDGE_V2_PROTOCOL.md',
         'experiments/bridge_v1/run.py', 'experiments/bridge_v1/model.py']


def main():
    protocol = ROOT / 'configs/bridge_v2.json'
    plan = json.loads(protocol.read_text())
    ledger = ROOT / 'artifacts/manifests/compute_ledger.json'
    entries = json.loads(ledger.read_text())
    if any(e['run_id'] == 'bridge_v2_a100' for e in entries):
        raise ValueError('Existing reservation/run must be preserved')
    plan['file_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}
    protocol.write_text(json.dumps(plan, indent=2) + '\n', newline='\n')
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    policy = json.loads((ROOT / 'artifacts/manifests/compute_policy.json').read_text())
    provision = float(plan['gpu_reservation_usd'])
    if sum(float(e['reserved_usd']) for e in entries) + provision + policy['reproduction_reserve_usd'] > policy['total_cap_usd']:
        raise ValueError('Insufficient headroom; preserve reproduction reserve')
    # A10040GB+2CPU+8GiB maximum660secs+2secs scale-down costs <=$.416,
    # leaving >$.38 within $.80 for image/load/provider contingencies.
    if provision < .8 or plan['automatic_retries'] != 0:
        raise ValueError('Unexpected bounds')
    entry = dict(run_id='bridge_v2_a100', phase='bridge_architecture_development',
        date=datetime.now(timezone.utc).isoformat(), status='reserved', reserved_usd=provision,
        model='Support regularization1857/global1/type1857/target-only1857',
        protocol_sha256=digest, gpu='A100-40GB', max_calls=1, retries=0,
        timeout_seconds=600, startup_timeout_seconds=60, child_timeout_seconds=540,
        cpu_physical_cores=2, memory_gib=8, actual_charged_usd=None,
        authorization='Human requested implementing MIRA-Bridge and GPU wherever needed; totalcap20 and reproduction reserve3 preserved',
        maximum_resource_compute_usd=.4151,
        charge_status='Worst-case resource plus >$.38 contingency reserved before deployment/dispatch')
    entries.append(entry)
    ledger.write_text(json.dumps(entries, indent=2) + '\n', newline='\n')
    (ROOT / 'artifacts/manifests/bridge_v2_freeze.json').write_text(json.dumps(dict(
        frozen_at_utc=entry['date'], protocol_sha256=digest, files=plan['file_sha256'],
        development_seeds=plan['development_seeds'], confirmation_seeds_untouched='98000-98019',
        budget_reserved_usd=sum(float(e['reserved_usd']) for e in entries),
        reproduction_reserve_usd=3, total_cap_usd=20), indent=2) + '\n', newline='\n')
    print(json.dumps(dict(protocol_sha256=digest, provision=provision,
                         total_provisions=sum(float(e['reserved_usd']) for e in entries))))


if __name__ == '__main__':
    main()
