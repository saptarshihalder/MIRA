"""Evaluate only the frozen primary decision; preserve descriptive controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
plan_path = ROOT / "configs/confirmation_pairwise_v1.json"
plan = json.loads(plan_path.read_text())
digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
assert digest == plan_path.with_suffix(".sha256").read_text().strip()
run = ROOT / "artifacts/runs/confirmation_pairwise_v1"
report = ROOT / "artifacts/reports/confirmation_pairwise_v1"
manifest = json.loads((run / "manifest.json").read_text())
summary = json.loads((report / "summary.json").read_text())
assert manifest["status"] == "complete" and summary["completed_cells"] == 360
assert not manifest["unresolved_failed_cells"] and not summary["cold_start_cells"]
assert summary["confidence"] == plan["primary"]["confidence_per_contrast"]
paired = json.loads((report / "paired_comparisons.json").read_text())
primary = []
for comparison in plan["primary"]["comparisons"]:
    effect = next(e for e in summary["effects"] if e["model"] == "tabpfn:v2"
                  and e["gamma"] == .9 and e["comparison"] == comparison)
    rows = [e for e in paired if e["model"] == "tabpfn:v2" and e["gamma"] == .9
            and e["comparison"] == comparison]
    assert sorted(e["seed"] for e in rows) == list(range(60000, 60020))
    gains = np.array([e["gain_nats"] for e in rows])
    mean, se = gains.mean(), gains.std(ddof=1) / np.sqrt(len(gains))
    half = t.ppf((1 + summary["confidence"]) / 2, len(gains)-1) * se
    assert np.allclose([mean, mean-half, mean+half], [effect[k] for k in ("mean", "ci_low", "ci_high")], atol=1e-12)
    primary.append(effect)
# Independently check the exact posterior and information boundary on each task.
tasks = list((run / "data").glob("*.npz"))
assert len(tasks) == 40
for path in tasks:
    mechanism = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as data:
        assert not set(data["support_row_ids"]) & set(data["query_row_ids"])
        assert np.array_equal(np.isnan(data["xq"]), data["mq"].astype(bool))
        f = mechanism["sign"] * np.prod(2*data["mq"][:, mechanism["active"]].astype(float)-1, axis=1)
        z = mechanism["gamma"] * f
        oracle = expit(mechanism["beta"] * data["uq"] + np.log1p(z) - np.log1p(-z))
        assert np.allclose(oracle, data["oracle"], rtol=0, atol=1e-14)
success = all(e["ci_low"] > 0 for e in primary) and primary[0]["mean"] >= plan["primary"]["practical_threshold_nats"]
result = {"frozen_protocol_sha256": digest, "primary_success": success,
          "primary": primary, "audited_tasks": len(tasks), "confirmation_cells": 360,
          "audit": "passed: primary intervals recomputed, seeds/IDs/NaN masks/exact query oracle verified",
          "scope": plan["scope"], "secondary": plan["secondary"]}
(report / "decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
lines = ["# Frozen pairwise confirmation", "", f"Primary decision: **{'passed' if success else 'not established'}**.", "",
         "Twenty fresh independent tasks; 360 cells including zero-signal and secondary-backbone controls. "
         "Two prespecified TabPFN v2 contrasts at gamma .9 use 97.5% two-sided task-level t intervals, giving "
         "Bonferroni familywise coverage of at least95%. The plan was committed before evaluation. "
         "No outcome-based stopping or selection occurred.", "",
         "| Primary comparison | Gain (nats) | 97.5% interval |", "|---|---:|---| "]
for e in primary:
    lines.append(f"| {e['comparison']} | {e['mean']:.6f} | [{e['ci_low']:.6f}, {e['ci_high']:.6f}] |")
lines += ["", result["audit"], "", "Scope: " + result["scope"], "",
          "Gamma-zero and other backbone results remain descriptive and are in report.md/summary.json. "
          "The untouched confirmation status now applies to no further pairwise rerun of these same seeds. "
          "This does not establish real-data benefit, a current-backbone result, prior causality or adapter superiority."]
(report / "decision.md").write_text("\n".join(lines).replace("least95", "least 95")+"\n", encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.3), layout="constrained")
selected = [next(e for e in summary["effects"] if e["model"] == "tabpfn:v2"
            and e["gamma"] == gamma and e["comparison"] == comparison)
            for gamma in (0, .9) for comparison in plan["primary"]["comparisons"]]
means = np.array([e["mean"] for e in selected])
errors = np.array([[e["mean"]-e["ci_low"] for e in selected], [e["ci_high"]-e["mean"] for e in selected]])
ax.errorbar(range(4), means, yerr=errors, fmt="o", capsize=5, color="#215b8f")
ax.axhline(0, color="black", linewidth=.8)
ax.set_xticks(range(4), ["Native → true\nγ=0", "Shuffled → true\nγ=0", "Native → true\nγ=.9", "Shuffled → true\nγ=.9"])
ax.set_ylabel("Expected log-loss gain (nats)")
ax.set_title("TabPFN v2: Gaussian pairwise masks, 20 fresh tasks")
ax.text(.02, .98, "97.5% task-level intervals; controls descriptive", transform=ax.transAxes, va="top", fontsize=9)
figures = ROOT / "artifacts/figures"
fig.savefig(figures / "confirmation_pairwise.png", dpi=180)
fig.savefig(figures / "confirmation_pairwise.svg")
print(json.dumps({"primary_success": success, "primary": [{k:e[k] for k in ('comparison','mean','ci_low','ci_high')} for e in primary], "audited_tasks":len(tasks)}))
