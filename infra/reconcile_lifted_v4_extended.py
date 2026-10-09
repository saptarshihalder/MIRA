"""One finite time-budget amendment after source-only timing, before main fit."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'artifacts/manifests'

def write(path, obj, indent=2):
    path.write_bytes((json.dumps(obj, indent=indent) + '\n').encode())

def main():
    ledger_path = M / 'compute_ledger.json'
    ledger = json.loads(ledger_path.read_bytes())
    run_id = 'lifted_v4_modal_main_l40s_extended'
    if any(e['run_id'] == run_id for e in ledger):
        raise ValueError('Already reserved; no repeated amendment')
    apps = json.loads(subprocess.check_output([sys.executable, '-m', 'modal', 'app', 'list', '--json']))
    if any(int(a['tasks']) for a in apps):
        raise ValueError('Outstanding tasks; reconciliation refused')
    summary = json.loads(subprocess.check_output([sys.executable, '-m', 'modal', 'billing', 'summary', '--for', '2026-10', '--json']))
    september = json.loads((M / 'modal_summary_202609.json').read_bytes())
    gross = Decimal(summary['metered_cost']) + Decimal(september['metered_cost'])
    if gross > Decimal('3.25'):
        raise ValueError('Retain at least USD0.25 pending-cost contingency within historical USD3.50')
    runtime = json.loads((ROOT / 'artifacts/reports/lifted_v4_launch/l40s_smoke_runtime.json').read_bytes())
    gate = json.loads((ROOT / 'artifacts/reports/lifted_v4_launch/l40s_smoke_gate.json').read_bytes())
    if runtime['status'] != 'completed' or runtime['exit_code'] != 0 or gate['passed'] is not True:
        raise ValueError('Source engineering smoke must pass')
    estimate = gate['execution_estimate']['estimated_seconds']
    if not 0 < estimate <= 21000:
        raise ValueError('Source estimate must fit explicit 350-minute extension')
    smoke = next(e for e in ledger if e['run_id'] == 'lifted_v4_modal_source_smoke_l40s')
    if smoke['status'] != 'running' or smoke['reserved_usd'] != 1:
        raise ValueError('Unexpected source call state')
    smoke.update(status='source_smoke_completed_original_time_gate_failed', reserved_usd=0,
        original_reserved_usd=1, reservation_accounted_by='historical_modal_account_20261009',
        seconds=runtime['seconds'], main_fits_original_cap=False, main_estimate_seconds=estimate,
        evidence='artifacts/reports/lifted_v4_launch/l40s_smoke_gate.json',
        scientific_training_updates=0)
    account = next(e for e in ledger if e['run_id'] == 'historical_modal_account_20261009')
    account.update(metered_gross_usd=float(gross), updated_evidence='artifacts/manifests/modal_summary_after_l40s_source.json')
    ledger.append(dict(run_id=run_id, date='2026-10-09', status='reserved_not_launched',
        reserved_usd=13.25, gpu='L40S', function_timeout_seconds=21300,
        startup_timeout_seconds=120, child_minutes=350, automatic_retries=0,
        reason='One explicit source-timing-based engineering extension; unchanged frozen v4 scientific protocol, all three seeds and all endpoints.'))
    total = sum(Decimal(str(e.get('reserved_usd', 0))) for e in ledger) + Decimal('3')
    if total > Decimal('20'):
        raise ValueError('Cap or protected reproduction reserve exceeded')
    record = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), historical_gross_usd=float(gross),
        historical_provision_usd=3.5, main_reservation_usd=13.25, reproduction_reserve_usd=3,
        worst_case_all_phases_usd=float(total), final_invoice_verified=False,
        original_main_time_gate=False, source_estimated_seconds=estimate, new_child_seconds=21000,
        scientific_recipe_changed=False, external_evaluation_seen=False,
        rationale='Release the completed source provision only into account-wide gross metering. Retain historical contingency. Amend wall-time before scientific main, without changing models, steps, seeds, panels, margins or checkpoint rules.')
    write(M / 'modal_apps_after_l40s_source.json', apps)
    write(M / 'modal_summary_after_l40s_source.json', summary)
    write(M / 'lifted_v4_extended_budget_reconciliation.json', record)
    write(ledger_path, ledger, indent=1)
    print(json.dumps(record))

if __name__ == '__main__':
    main()
