"""Summarize verified development reports without rerunning inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
catalog = json.loads((ROOT / "configs/day3_matrix.json").read_text())
records, inputs, table, cells = [], {}, [], 0
for entry in catalog["entries"]:
    name = "day3_" + entry["name"]
    run = ROOT / "artifacts/runs" / name
    manifest = json.loads((run / "manifest.json").read_text())
    report_path = ROOT / "artifacts/reports" / name / "summary.json"
    report = json.loads(report_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_cells"] == manifest["expected_cells"] == report["completed_cells"]
    assert not manifest["unresolved_failed_cells"]
    inputs[str(report_path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    cells += report["completed_cells"]
    for effect in report["effects"]:
        records.append({"run": name, **effect})
        if effect["gamma"] == .9 and effect["comparison"] in {"native_to_indicators", "imputed_to_indicators", "native_to_width_control"}:
            table.append(f"| {entry['name']} | {effect['family']} | {effect['model']} | {effect['comparison']} | {effect['mean']:.5f} [{effect['ci_low']:.5f}, {effect['ci_high']:.5f}] |")
ledger = json.loads((ROOT / "artifacts/manifests/compute_ledger.json").read_text())
summary = {"new_completed_cells": cells, "prior_matrix_cells": 270,
           "total_development_matrix_cells": cells + 270, "smoke_cells_excluded": 12,
           "reserved_usd": sum(e["reserved_usd"] for e in ledger),
           "input_report_sha256": inputs, "effects": records}
out = ROOT / "artifacts/reports"
(out / "day3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out / "day3_summary.md").write_text(
    "# Day 1–3 development summary\n\n"
    f"Seven additional runs completed {cells} cells; total development matrix cells: {cells + 270} (12 smoke cells excluded). "
    "Reports were regenerated from saved probabilities and data with hash/row-identity checks. "
    "Three independently generated tasks per condition; all intervals below are exploratory, unadjusted paired 95% t intervals. "
    "No confirmation seeds were used.\n\n"
    "The 16-column width control is a stress test; the true/shuffled-indicator comparisons use matched nuisance width. "
    "Baseline missing rate .1 changes oracle headroom and is not an equal-signal comparison with .5.\n\n"
    "## Effects at gamma .9 (nats; positive favors the second representation)\n\n"
    "| Run | Family | Predictor | Comparison | Mean [interval] |\n"
    "|---|---|---|---|---|\n" + "\n".join(table) + "\n\n"
    f"Conservative reservations: ${summary['reserved_usd']:.2f} of $26; $3 reproduction reserve retained. "
    "Reservations include failures and are not invoices. Full gamma-zero results and losses remain in each linked run report and day3_summary.json.\n",
    encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("new_completed_cells", "total_development_matrix_cells", "reserved_usd")}))
