"""Tighten closed-call provisions from provider resource reports; never call them invoices."""
import json
from decimal import Decimal
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = ROOT / 'artifacts/manifests'
    resources = json.loads((manifest / 'modal_billing_resources_20261003.json').read_text(encoding='utf-8-sig'))
    apps = json.loads((manifest / 'modal_apps_before_bridge.json').read_text(encoding='utf-8-sig'))
    if any(int(a['tasks']) for a in apps):
        raise ValueError('Outstanding tasks: preserve original provisions')
    totals = {}
    for row in resources:
        totals[row['object_id']] = totals.get(row['object_id'], Decimal(0)) + Decimal(row['cost'])
    ledger_path = manifest / 'compute_ledger.json'
    entries = json.loads(ledger_path.read_text())
    before = sum(Decimal(str(e['reserved_usd'])) for e in entries)
    changes = []
    # Only these two oversized, closed failure provisions change. All older
    # provisions, the successful final provision and reproduction reserve stay.
    for entry in entries:
        if entry['run_id'] not in ('large_native_a100_v1', 'large_native_a100_v2'):
            continue
        app_id = entry['app_id']
        app = next(a for a in apps if a['app_id'] == app_id)
        if entry['status'] != 'failed' or app['state'] != 'stopped' or app_id not in totals:
            raise ValueError('Cannot reconcile active/unmetered call')
        cost = totals[app_id]
        if cost >= Decimal('.1'):
            raise ValueError('Provider cost changed: reassess provision')
        prior = entry['reserved_usd']
        entry.setdefault('original_reserved_usd', prior)
        entry.update(reserved_usd=.5, provider_reported_resource_cost_usd=float(cost),
                     actual_charged_usd=None,
                     provider_cost_status='Resource-level billing report reconciled; final invoice unverified; $.50 retained per stopped failure including pending-charge contingency',
                     reconciliation_manifest='artifacts/manifests/bridge_budget_reconciliation.json')
        changes.append(dict(run_id=entry['run_id'], previous_provision=prior,
                            provider_resource_cost=float(cost), retained_provision=.5))
    after = sum(Decimal(str(e['reserved_usd'])) for e in entries)
    record = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
                  report_start='2026-09-30', report_end_exclusive='2026-10-04',
                  method='Modal billing report --show-resources --json and app list --json; BrowserOS cost-summary cross-check',
                  provider_reported_all_apps_usd=float(sum(totals.values())),
                  invoice_verified=False, changes=changes,
                  previous_provisions_usd=float(before), provisions_after_usd=float(after),
                  reproduction_reserve_usd=3, total_cap_usd=20,
                  rationale='Stopped failures have no outstanding tasks and metered CPU, memory and GPU costs below $.04/$.025. Retain $.50 each (over ten times reported costs) for pending attribution; release only unused job-duration provision. Older calls remain conservatively reserved.')
    ledger_path.write_text(json.dumps(entries, indent=2) + '\n', newline='\n')
    (manifest / 'bridge_budget_reconciliation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
