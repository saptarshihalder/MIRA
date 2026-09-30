"""Compare numerical rerun against preserved handoff and independently recompute summaries."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "reference" / "mira_pilot"
FIELDS = ["split", "family", "seed", "protocol", "new_labels", "context_size", "method"]


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    old = rows(ORIGINAL / "results_strong" / "task_results.csv")
    new = rows(HERE / "results" / "task_results.csv")
    keyed = lambda rr: {tuple(r[k] for k in FIELDS): float(r["expected_nll"]) for r in rr}
    a, b = keyed(old), keyed(new)
    assert len(a) == len(old) and len(b) == len(new), "Duplicate result keys"
    assert set(a) == set(b), "Matrix cells differ"
    delta = np.array([abs(a[k]-b[k]) for k in a])
    by_method = {method: max(abs(a[k]-b[k]) for k in a if k[-1] == method) for method in sorted({k[-1] for k in a})}
    original = json.loads((ORIGINAL / "results_strong" / "summary.json").read_text())
    rerun = json.loads((HERE / "results" / "summary.json").read_text())
    key = lambda r: (r["family"], r["protocol"], r["new_labels"])
    sm_old = {key(r): r for r in original["summary"]}
    sm_new = {key(r): r for r in rerun["summary"]}
    assert set(sm_old) == set(sm_new)
    numeric = [k for k in next(iter(sm_old.values())) if k not in {"family", "protocol", "new_labels", "selected_simple"}]
    summary_delta = max(abs(sm_old[k][f]-sm_new[k][f]) for k in sm_old for f in numeric)
    changed = [list(k) for k in sm_old if sm_old[k]["selected_simple"] != sm_new[k]["selected_simple"]]
    recompute_error = 0.
    for record in sm_new.values():
        cell = [r for r in new if (r["family"], r["protocol"], int(r["new_labels"])) == key(record)]
        methods = sorted({r["method"] for r in cell} - {"mechanism_oracle", "finite_prior_reference"})
        mean = lambda split, method: np.array([float(r["expected_nll"]) for r in cell if r["split"] == split and r["method"] == method])
        chosen = min(methods, key=lambda m: mean("development", m).mean())
        assert chosen == record["selected_simple"]
        diff = mean("evaluation", chosen) - mean("evaluation", "mechanism_oracle")
        recompute_error = max(recompute_error, abs(float(diff.mean())-record["oracle_gap"]),
            abs(float(diff.std(ddof=1)/np.sqrt(len(diff)))-record["oracle_gap_se"]))
    old_math = json.loads((ORIGINAL / "math_checks.json").read_text())
    new_math = json.loads((HERE / "math_checks.json").read_text())
    math_delta = {k: abs(old_math[k]-new_math[k]) for k in old_math if isinstance(old_math[k], (int, float))}
    result = {"row_count": len(new), "matrix_keys_identical": True, "max_absolute_raw_nll_difference": float(delta.max()),
        "raw_rows_differing_above_1e_12": int(np.sum(delta > 1e-12)), "max_difference_by_method": by_method,
        "summary_cells": len(sm_new), "changed_method_selections": changed, "max_absolute_summary_difference": summary_delta,
        "independent_summary_recompute_error": recompute_error, "math_check_absolute_differences": math_delta,
        "elapsed_seconds": rerun["metadata"]["elapsed_seconds"], "fresh_32_label_gaps": [r for r in rerun["summary"] if r["protocol"] == "fresh_target" and r["new_labels"] == 32],
        "hashes": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("results/*")) if p.is_file()}}
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"fresh_32_label_gaps", "max_difference_by_method", "hashes"}}, indent=2))


if __name__ == "__main__":
    main()
