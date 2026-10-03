"""One prepaid invocation, preserving failures and rejecting source drift."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import modal

ROOT = Path(__file__).resolve().parents[2]


def main():
    protocol = ROOT / 'configs/spectral_v1.json'
    plan = json.loads(protocol.read_text())
    policy = json.loads((ROOT / 'artifacts/manifests/compute_policy.json').read_text())
    ledger = ROOT / 'artifacts/manifests/compute_ledger.json'
    entries = json.loads(ledger.read_text())
    if sum(float(e['reserved_usd']) for e in entries) + policy['reproduction_reserve_usd'] > policy['total_cap_usd']:
        raise ValueError('Budget cap/reproduction reserve exceeded')
    match = [entry for entry in entries if entry['run_id'] == plan['run_id']]
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    if len(match) != 1 or match[0]['status'] != 'reserved' or match[0]['protocol_sha256'] != digest:
        raise ValueError('One matching unused reservation required; never retry')
    for name, expected in plan['file_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: ' + name)
    target = ROOT / 'artifacts/runs' / plan['run_id']
    if target.exists():
        raise ValueError('Preserve existing run')
    target.mkdir(parents=True)
    match[0].update(status='invoking', remote_invocations_authorized=1)
    ledger.write_text(json.dumps(entries, indent=2) + '\n', newline='\n')
    try:
        function = modal.Function.from_name('mira-spectral-v1', 'train_spectral')
        with modal.enable_output():
            payload = function.remote(digest)
        with zipfile.ZipFile(io.BytesIO(payload.pop('archive'))) as archive:
            if any(not (target / item.filename).resolve().is_relative_to(target.resolve())
                   for item in archive.infolist()):
                raise ValueError('Unsafe archive path')
            archive.extractall(target)
        match[0].update(payload, status='completed' if payload['exit_code'] == 0 else 'failed',
                        remote_function_calls=1, results=str(target), actual_charged_usd=None)
        print(json.dumps(match[0]), flush=True)
        if payload['exit_code']:
            raise SystemExit(payload['exit_code'])
    except BaseException as error:
        match[0].update(status='failed', error=repr(error),
            remote_function_calls=match[0].get('remote_function_calls', 'unverified; do not retry'))
        raise
    finally:
        ledger.write_text(json.dumps(entries, indent=2) + '\n', newline='\n')


if __name__ == '__main__':
    main()
