"""Read-only saved-data audit; writes only its JSON/Markdown report."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("uc", "yc", "mc", "uq", "yq", "mq", "oracle", "base", "support_row_ids", "query_row_ids")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    catalog = json.loads((ROOT / "configs/day3_matrix.json").read_text())
    runs, data_paths, errors = {}, {}, []
    for entry in catalog["entries"]:
        name = "day3_" + entry["name"]
        directory = ROOT / "artifacts/runs" / name
        if not (directory / "manifest.json").exists():
            runs[name] = {"status": "pending", "tasks": []}
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        configuration = manifest["configuration"]
        paths = {path.name: path for path in (directory / "data").glob("*.npz")}
        data_paths[name] = paths
        expected = {f"{family}_seed{seed}_g{gamma:.12g}.npz" for family in configuration["families"]
                    for seed in configuration["seeds"] for gamma in configuration["gammas"]}
        ledger = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
        ledger_hashes = {}
        for row in ledger:
            ledger_hashes.setdefault(Path(row["data_file"]).name, set()).add(row["data_sha256"])
        tasks = []
        for filename, path in sorted(paths.items()):
            mechanism = json.loads(path.with_suffix(".json").read_text())
            record = {"file": filename, "sha256": digest(path), "strata": {}}
            record["mechanism_parameters_match_manifest"] = all(mechanism[key] == configuration[key] for key in
                ("beta", "missing_rate", "value_distribution", "collision_probability", "quantization_step"))
            theoretical_rates = np.full(mechanism["dimensions"], mechanism["missing_rate"])
            if len(mechanism["active"]) == 1 and mechanism["value_dependent"]:
                r = mechanism["missing_rate"]
                theoretical_rates[mechanism["active"][0]] += (r * (1 - r) / max(r, 1 - r)
                    * mechanism["gamma"] * mechanism["sign"] * np.tanh(mechanism["beta"] / 2))
            record["theoretical_marginal_missing_fraction_by_column"] = theoretical_rates.tolist()
            with np.load(path, allow_pickle=False) as data:
                support_ids, query_ids = data["support_row_ids"], data["query_row_ids"]
                unique_disjoint = (support_ids.shape == data["yc"].shape and query_ids.shape == data["yq"].shape
                                   and len(np.unique(support_ids)) == len(support_ids)
                                   and len(np.unique(query_ids)) == len(query_ids)
                                   and not np.intersect1d(support_ids, query_ids).size)
                record["ids_unique_disjoint"] = bool(unique_disjoint)
                support_observed = np.isfinite(data["xc"])
                counts = support_observed.sum(axis=0)
                means = np.divide(np.nansum(data["xc"], axis=0), counts,
                                  out=np.zeros(data["xc"].shape[1]), where=counts > 0)
                record["reference_support_column_means"] = means.tolist()
                for suffix in ("c", "q"):
                    x, mask = data["x" + suffix], data["m" + suffix]
                    finite = np.isfinite(x)
                    observed = int(finite.sum())
                    record["strata"][suffix] = {
                        "observed_cells": observed, "observed_zeros": int(np.sum((x == 0) & finite)),
                        "zero_fraction_among_observed": float(np.sum((x == 0) & finite) / observed) if observed else None,
                        "observed_values_exactly_equal_to_reference_support_mean": int(np.sum((x == means) & finite)),
                        "empirical_missing_fraction": float(mask.mean()),
                        "per_column_missing_fraction": mask.mean(axis=0).tolist(),
                        "nan_pattern_matches_mask": bool(np.array_equal(np.isnan(x), mask.astype(bool))),
                    }
                r = mechanism["missing_rate"]
                z = (data["mq"].astype(float)[:, mechanism["active"]] - r) / max(r, 1 - r)
                f = mechanism["sign"] * z.prod(axis=1)
                if mechanism["value_dependent"]:
                    f *= data["uq"]
                signal = mechanism["gamma"] * f
                oracle = expit(mechanism["beta"] * data["uq"] + np.log1p(signal) - np.log1p(-signal))
                record["oracle_max_abs_error"] = float(np.max(np.abs(oracle - data["oracle"])))
                record["base_max_abs_error"] = float(np.max(np.abs(expit(mechanism["beta"] * data["uq"]) - data["base"])))
            record["ledger_hash_matches"] = ledger_hashes.get(filename) == {record["sha256"]}
            if (not unique_disjoint or not record["mechanism_parameters_match_manifest"]
                    or not record["ledger_hash_matches"] or record["oracle_max_abs_error"] > 1e-12
                    or record["base_max_abs_error"] > 1e-12
                    or not all(value["nan_pattern_matches_mask"] for value in record["strata"].values())):
                errors.append({"run": name, "task": filename, "error": "Saved-data contract failed"})
            tasks.append(record)
        runs[name] = {"status": manifest["status"], "completed_cells": manifest["completed_cells"],
                      "expected_cells": manifest["expected_cells"], "expected_tasks": len(expected),
                      "all_expected_task_files": set(paths) == expected, "configuration": configuration,
                      "manifest_sha256": digest(directory / "manifest.json"),
                      "source_sha256": manifest["source_sha256"], "environment_sha256": manifest["environment_sha256"],
                      "tasks": tasks}
        distribution = configuration["value_distribution"]
        runs[name]["theoretical_zero_fraction_among_observed"] = {
            "gaussian": 0., "zero_collision": 1., "partial_collision": configuration["collision_probability"],
            "quantized": math.erf(configuration["quantization_step"] / (2 * math.sqrt(2))),
        }[distribution]
        if manifest["status"] == "complete" and (set(paths) != expected or len(ledger) != manifest["expected_cells"]):
            errors.append({"run": name, "error": "Completed run is missing requested data/results"})
    comparisons = []
    groups = [("gaussian_imputed_controls", "zero_collision_imputed_controls"),
              ("partial_collision_010", "partial_collision_050"),
              ("partial_collision_010", "quantized_step1"),
              ("partial_collision_050", "quantized_step1")]
    for left, right in groups:
        left, right = "day3_" + left, "day3_" + right
        comparison = {"left": left, "right": right, "matched_task_count": 0, "status": "pending"}
        if left in data_paths and right in data_paths:
            first, second = data_paths[left], data_paths[right]
            comparison["source_hashes_equal"] = runs[left]["source_sha256"] == runs[right]["source_sha256"]
            comparison["environment_hashes_equal"] = runs[left]["environment_sha256"] == runs[right]["environment_sha256"]
            configurations = [dict(runs[name]["configuration"]) for name in (left, right)]
            for configuration in configurations:
                for key in ("value_distribution", "collision_probability", "quantization_step"):
                    configuration.pop(key)
            comparison["configuration_equal_except_value_parameters"] = configurations[0] == configurations[1]
            if not comparison["configuration_equal_except_value_parameters"]:
                errors.append({"comparison": [left, right], "error": "Configurations differ beyond value-control parameters"})
            if set(first) != set(second):
                comparison["status"] = "unpaired seed/task panels"
                errors.append({"comparison": [left, right], "error": comparison["status"]})
            else:
                comparison["status"] = "passed"
                for filename in sorted(first):
                    with np.load(first[filename], allow_pickle=False) as a, np.load(second[filename], allow_pickle=False) as b:
                        mismatches = [field for field in FIELDS if not np.array_equal(a[field], b[field])]
                    if mismatches:
                        comparison["status"] = "failed"
                        errors.append({"comparison": [left, right], "task": filename, "mismatched_fields": mismatches})
                    else:
                        comparison["matched_task_count"] += 1
        comparisons.append(comparison)
    complete = all(run["status"] == "complete" for run in runs.values())
    report = {"status": "passed" if complete and not errors else ("failed" if errors else "pending"),
              "completed_runs": sum(run["status"] == "complete" for run in runs.values()), "expected_runs": 7,
              "saved_task_files": sum(len(run["tasks"]) for run in runs.values()),
              "distinct_seed_family_ids": len({(seed, family) for run in runs.values() if "configuration" in run
                  for seed in run["configuration"]["seeds"] for family in run["configuration"]["families"]}),
              "paired_fields": list(FIELDS), "comparisons": comparisons, "runs": runs, "errors": errors,
              "rate_semantics": "missing_rate is the reference independent Bernoulli rate, not an assertion that all empirical/marginal rates equal it",
              "method": "Independent direct checks on saved NPZ/JSON; no generator import, model calls, or paid services."}
    output = ROOT / "artifacts/reports/day3_data_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Day 3 saved-data audit", "", f"Status: **{report['status']}**; {report['completed_runs']}/7 runs complete.", "",
             f"{report['saved_task_files']} materialized task files cover {report['distinct_seed_family_ids']} distinct seed/family draws; gamma and value variants repeat paired underlying draws.", "",
             "U, labels, masks, oracle/base probabilities, and support/query IDs are compared exactly across paired value variants.", "",
             "| Run | Tasks | Baseline r | Observed zeros / observed cells | Empirical missing fraction |",
             "|---|---:|---:|---:|---:|"]
    for name, run in runs.items():
        if not run["tasks"]:
            lines.append(f"| {name} | pending | — | — | — |")
            continue
        strata = [value for task in run["tasks"] for value in task["strata"].values()]
        observed = sum(value["observed_cells"] for value in strata)
        zeros = sum(value["observed_zeros"] for value in strata)
        missing = np.mean([value["empirical_missing_fraction"] for value in strata])
        lines.append(f"| {name} | {len(run['tasks'])} | {run['configuration']['missing_rate']:g} | {zeros}/{observed} ({zeros/observed:.4%}) | {missing:.4%} |")
    lines += ["", "The missing-fraction summary above averages support/query task fractions equally; per-column/task values and exact zero counts are in JSON.", ""]
    for comparison in comparisons:
        lines.append(f"- {comparison['left']} versus {comparison['right']}: **{comparison['status']}**, {comparison['matched_task_count']} exact task pairs.")
    lines += ["", report["rate_semantics"] + ". Value-dependent outcome modulation can alter marginal rates; finite draws fluctuate as well.",
              "Observed-zero counts describe nuisance values only; U stays observed. Counts across gamma variants are descriptive, not independent samples for an uncertainty calculation.",
              "A finite observed zero does not necessarily equal its fitted imputation constant. JSON separately counts observed values exactly equal to the support-only float64 column mean; this reference does not instrument a third-party encoder's arithmetic.",
              "The audit checks saved prediction/evaluation separation through disjoint row IDs; it does not certify every third-party model implementation.",
              "No confirmation data or paid services were used."]
    if errors:
        lines += ["", "Failures: " + json.dumps(errors)]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "completed_runs": report["completed_runs"], "errors": len(errors), "comparisons": comparisons}))


if __name__ == "__main__":
    audit()
