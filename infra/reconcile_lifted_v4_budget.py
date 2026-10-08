"""Conservative account-level reconciliation; no cloud execution or invoice claim."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'artifacts/manifests'

def read(name):
    return json.loads((M / name).read_text(encoding='utf-8-sig'))

def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))

def main():
    apps = read('modal_apps_20261009.json')
    if any(int(a['tasks']) for a in apps):
        raise ValueError('Outstanding tasks; do not release historical provisions')
    summaries = {month: read(file) for month, file in [
        ('2026-09', 'modal_summary_202609.json'), ('2026-10', 'modal_summary_20261009.json')]}
    gross = sum(Decimal(s['metered_cost']) for s in summaries.values())
    if gross > Decimal('3'):
        raise ValueError('Reassess historical contingency against increased metering')
    ledger = read('compute_ledger.json')
    if any(e['run_id'] == 'historical_modal_account_20261009' for e in ledger):
        raise ValueError('Already reconciled; never duplicate reservations')
    previous = sum(Decimal(str(e.get('reserved_usd', 0))) for e in ledger)
    released = []
    for e in ledger:
        if e.get('reserved_usd', 0):
            e.setdefault('original_reserved_usd', e['reserved_usd'])
            released.append(e['run_id'])
            e['reserved_usd'] = 0
            e['reservation_accounted_by'] = 'historical_modal_account_20261009'
    ledger.append(dict(run_id='historical_modal_account_20261009', date='2026-10-09',
        status='metered_account_reconciled_with_contingency', reserved_usd=3.5,
        metered_gross_usd=float(gross), provider_billed_so_far_usd=sum(float(s['billed_cost']) for s in summaries.values()),
        final_invoice_verified=False, cloud_calls=0,
        reason='Count entire September and October account gross, including failures, storage and unrelated usage; retain $3.50 while final invoices remain open.',
        evidence='artifacts/manifests/lifted_v4_budget_reconciliation.json'))
    ledger.append(dict(run_id='lifted_v4_modal_source_smoke', date='2026-10-09',
        status='reserved_not_launched', reserved_usd=1.0, gpu='A100-40GB',
        timeout_seconds=1200, startup_timeout_seconds=120, automatic_retries=0,
        reason='Source-only fixed recipe smoke before any external scores; includes build/startup/egress contingency.'))
    record = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
        summaries=summaries, historical_gross_usd=float(gross), historical_provision_usd=3.5,
        previous_provisions_usd=float(previous), replaced_run_ids=released,
        final_invoice_verified=False, smoke_reservation_usd=1.0,
        conditional_main_max_reservation_usd=11.5, reproduction_reserve_usd=3,
        total_cap_usd=20, worst_case_all_phases_usd=19,
        rationale='Provider billing summaries now give actual metered and billed-so-far amounts; keep gross usage plus >$0.91 pending-cost contingency, not zero cost from credits. All listed tasks are zero. Main needs passing smoke and a separate pre-call reservation.')
    assert sum(e.get('reserved_usd', 0) for e in ledger) + 3 <= 20
    write(M / 'lifted_v4_budget_reconciliation.json', record)
    write(M / 'compute_ledger.json', ledger)
    budget_path = ROOT / 'configs/budget.json'
    budget = json.loads(budget_path.read_text(encoding='utf-8'))
    budget['total_cap_usd'] = 20.0
    write(budget_path, budget)
    print(json.dumps({k: record[k] for k in ['historical_gross_usd','historical_provision_usd','smoke_reservation_usd','worst_case_all_phases_usd']}))

if __name__ == '__main__':
    main()
