"""Dispatch exactly one frozen, prepaid call; preserve outputs on failure."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import modal

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'bridge_v2_a100'


def main():
    ledger = ROOT / 'artifacts/manifests/compute_ledger.json'
    policy = json.loads((ROOT / 'artifacts/manifests/compute_policy.json').read_text())
    entries = json.loads(ledger.read_text())
    if sum(float(e['reserved_usd']) for e in entries) + policy['reproduction_reserve_usd'] > policy['total_cap_usd']:
        raise ValueError('Budget cap/reproduction reserve exceeded')
    entry = [e for e in entries if e['run_id'] == RUN_ID]
    protocol = ROOT / 'configs/bridge_v2.json'
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    if len(entry) != 1 or entry[0]['status'] != 'reserved' or entry[0]['protocol_sha256'] != digest:
        raise ValueError('Requires one matching unused reservation; never automatically retry')
    for relative, expected in json.loads(protocol.read_text())['file_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source mismatch: ' + relative)
    target = ROOT / 'artifacts/runs' / RUN_ID
    if target.exists():
        raise ValueError('Existing run must be preserved')
    target.mkdir(parents=True)
    entry[0].update(status='invoking', remote_invocations_authorized=1)
    ledger.write_text(json.dumps(entries, indent=2) + '\n', newline='\n')
    try:
        function = modal.Function.from_name('mira-bridge-v2-a100', 'train_bridge')
        with modal.enable_output():
            payload = function.remote(digest)
        with zipfile.ZipFile(io.BytesIO(payload.pop('archive'))) as archive:
            if any(not (target / item.filename).resolve().is_relative_to(target.resolve()) for item in archive.infolist()):
                raise ValueError('Unsafe result archive')
            archive.extractall(target)
        entry[0].update(payload, status='completed' if payload['exit_code'] == 0 else 'failed',
                        remote_function_calls=1, results=str(target), actual_charged_usd=None)
        print(json.dumps(entry[0]), flush=True)
        if payload['exit_code']:
            raise SystemExit(payload['exit_code'])
    except BaseException as error:
        entry[0].update(status='failed', error=repr(error),
                        remote_function_calls=entry[0].get('remote_function_calls', 'unverified; do not retry'))
        raise
    finally:
        ledger.write_text(json.dumps(entries, indent=2) + '\n', newline='\n')


if __name__ == '__main__':
    main()
