"""Regenerate paired task-level effects from saved probabilities and data."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from .artifacts import atomic_json, file_sha256, load_rows
from .data import EvaluationTargets
from .metrics import HEADROOM_EPSILON, evaluate

COMPARISONS = (
    ("native", "native_indicators", "native_to_indicators"),
    ("native", "native_shuffled", "native_to_shuffled"),
    ("native_shuffled", "native_indicators", "shuffled_to_indicators"),
    ("imputed", "imputed_indicators", "imputed_to_indicators"),
    ("native", "native_width_control", "native_to_width_control"),
)


def mean_interval(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    """Paired Student-t interval over independently generated task draws."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or not 0 < confidence < 1:
        raise ValueError("Finite values and confidence in (0,1) required")
    n = len(array)
    if n == 0:
        return {"n_tasks": 0, "mean": None, "se": None, "ci_low": None, "ci_high": None}
    mean = float(np.mean(array))
    if n == 1:
        return {"n_tasks": 1, "mean": mean, "se": None, "ci_low": None, "ci_high": None}
    se = float(np.std(array, ddof=1) / np.sqrt(n))
    half_width = float(t.ppf((1 + confidence) / 2, n - 1)) * se
    return {"n_tasks": n, "mean": mean, "se": se, "ci_low": mean - half_width, "ci_high": mean + half_width}


def _verified_rows(out: Path) -> list[dict[str, Any]]:
    rows = load_rows(out)
    # Cache truth once for each task. Never use query labels in a transform.
    truth: dict[str, EvaluationTargets] = {}
    data_hashes: dict[str, str] = {}
    for row in rows:
        data_path, prediction_path = out / row["data_file"], out / row["prediction_file"]
        if row["data_file"] not in data_hashes:
            data_hashes[row["data_file"]] = file_sha256(data_path)
        if data_hashes[row["data_file"]] != row["data_sha256"]:
            raise ValueError(f"Data artifact changed: {data_path}")
        if file_sha256(prediction_path) != row["prediction_sha256"]:
            raise ValueError(f"Prediction artifact changed: {prediction_path}")
        if row["task_id"] not in truth:
            with np.load(data_path, allow_pickle=False) as data:
                support_ids, query_ids = data["support_row_ids"], data["query_row_ids"]
                if (support_ids.shape != data["yc"].shape or query_ids.shape != data["yq"].shape
                        or len(np.unique(support_ids)) != len(support_ids)
                        or len(np.unique(query_ids)) != len(query_ids)
                        or np.intersect1d(support_ids, query_ids).size):
                    raise ValueError(f"Support/query row IDs are not disjoint: {data_path}")
                truth[row["task_id"]] = EvaluationTargets(data["yq"], data["oracle"], data["base"])
        with np.load(prediction_path, allow_pickle=False) as prediction:
            metrics = evaluate(prediction["probability"], truth[row["task_id"]])
        for name, value in metrics.items():
            saved = row[name]
            if (value is None) != (saved is None) or (value is not None and not np.isclose(value, saved, rtol=1e-10, atol=1e-10)):
                raise ValueError(f"Saved metric disagrees with predictions: {row['cell_id']} {name}")
        row.update(metrics)
    return rows


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        cells[(row["protocol"], row["family"], row["gamma"], row["model"], row["seed"])][row["mode"]] = row
    paired = []
    for (protocol, family, gamma, model, seed), modes in sorted(cells.items()):
        for baseline, comparison, name in COMPARISONS:
            if baseline not in modes or comparison not in modes:
                continue
            a, b = modes[baseline], modes[comparison]
            if a["data_sha256"] != b["data_sha256"]:
                raise ValueError("Paired comparison does not share the identical generated data")
            gain = a["expected_nll"] - b["expected_nll"]
            gap = a["expected_nll"] - a["oracle_nll"]
            paired.append({
                "protocol": protocol, "family": family, "gamma": gamma, "model": model,
                "seed": seed, "comparison": name, "baseline_mode": baseline,
                "comparison_mode": comparison, "gain_nats": gain,
                "empirical_gain_nats": a["empirical_nll"] - b["empirical_nll"],
                "baseline_to_oracle_gap": gap,
                "fraction_baseline_gap": gain / gap if gap > HEADROOM_EPSILON else None,
                "cold_start": a["cold_start"] or b["cold_start"],
            })
    return paired


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(out: Path, report_out: Path | None = None, confidence: float = 0.95) -> dict[str, Any]:
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0,1)")
    out = out.expanduser().resolve()
    destination = report_out.expanduser().resolve() if report_out else out / "report"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = _verified_rows(out)
    paired = paired_comparisons(rows)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in paired:
        grouped[(pair["protocol"], pair["family"], pair["gamma"], pair["model"], pair["comparison"])].append(pair)
    effects = []
    for (protocol, family, gamma, model, comparison), values in sorted(grouped.items()):
        expected = mean_interval([row["gain_nats"] for row in values], confidence)
        empirical = mean_interval([row["empirical_gain_nats"] for row in values], confidence)
        fractions = [row["fraction_baseline_gap"] for row in values if row["fraction_baseline_gap"] is not None]
        effects.append({
            "protocol": protocol, "family": family, "gamma": gamma, "model": model,
            "comparison": comparison, **expected,
            "empirical_gain_mean": empirical["mean"], "empirical_gain_ci_low": empirical["ci_low"],
            "empirical_gain_ci_high": empirical["ci_high"],
            "mean_baseline_to_oracle_gap": float(np.mean([row["baseline_to_oracle_gap"] for row in values])),
            "fraction_mean": float(np.mean(fractions)) if fractions else None,
            "fraction_valid_tasks": len(fractions), "cold_start_tasks": sum(row["cold_start"] for row in values),
        })
    losses: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        losses[(row["protocol"], row["family"], row["gamma"], row["model"], row["mode"])].append(row)
    loss_summary = []
    for (protocol, family, gamma, model, mode), values in sorted(losses.items()):
        loss_summary.append({"protocol": protocol, "family": family, "gamma": gamma,
                             "model": model, "mode": mode,
                             **mean_interval([row["expected_nll"] for row in values], confidence),
                             "empirical_nll": float(np.mean([row["empirical_nll"] for row in values])),
                             "oracle_nll": float(np.mean([row["oracle_nll"] for row in values])),
                             "analytic_base_nll": float(np.mean([row["analytic_base_nll"] for row in values])),
                             "seconds_total": float(sum(row["seconds"] for row in values)),
                             "cold_start_tasks": sum(row["cold_start"] for row in values)})
    summary = {
        "schema_version": 1, "run_status": manifest["status"], "completed_cells": len(rows),
        "expected_cells": manifest["expected_cells"], "confidence": confidence,
        "interval_method": "paired Student-t across independent tasks within family/gamma/model; no multiplicity correction",
        "limitations": ["Synthetic tasks are not independent real datasets.",
                        "Small task counts produce unstable intervals; n=1 has no interval.",
                        "Fractions require baseline-to-oracle gap > 1e-4 and retain negative gains.",
                        "Shuffled indicators use unlabeled support/query batches independently.",
                        "Elapsed model-cell time includes preprocessing/fit/inference but excludes setup and checkpoint hashing.",
                        "Version comparisons do not identify a causal pretraining-prior effect."],
        "effects": effects, "losses": loss_summary,
        "cold_start_cells": sum(row["cold_start"] for row in rows),
        "checkpoint_identity_status": manifest.get("checkpoint_identity_status"),
    }
    atomic_json(destination / "summary.json", summary)
    atomic_json(destination / "paired_comparisons.json", paired)
    _write_csv(destination / "paired_effects.csv", effects)
    _write_csv(destination / "losses.csv", loss_summary)
    lines = ["# Saved-prediction mechanism report", "",
             f"Run status: **{manifest['status']}**. Verified {len(rows)}/{manifest['expected_cells']} requested cells.", "",
             f"{confidence:.0%} paired Student-t intervals across independent task draws. Positive gain favors the second representation.", "",
             "| Family | Gamma | Model | Comparison | Tasks | Gain (nats) | Interval |",
             "|---|---:|---|---|---:|---:|---|"]
    for effect in effects:
        interval = "undefined (one task)" if effect["ci_low"] is None else f"[{effect['ci_low']:.6f}, {effect['ci_high']:.6f}]"
        lines.append(f"| {effect['family']} | {effect['gamma']:g} | {effect['model']} | {effect['comparison']} | "
                     f"{effect['n_tasks']} | {effect['mean']:.6f} | {interval} |")
    if not effects:
        lines += ["", "No completed paired comparison is available."]
    lines += ["", "## Interpretation limits", ""] + [f"- {item}" for item in summary["limitations"]]
    lines += [f"- Cold-start cells: {summary['cold_start_cells']}; these are Laplace fallbacks, not backbone inference.",
              f"- Checkpoint identity: {summary['checkpoint_identity_status']}.",
              "- Development results inform configuration selection; confirmation results must remain separate."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def cli() -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--out", required=True, type=Path, help="Mechanism run directory")
    command.add_argument("--report-out", type=Path)
    command.add_argument("--confidence", type=float, default=0.95)
    args = command.parse_args()
    summary = report(args.out, args.report_out, args.confidence)
    print(f"Verified {summary['completed_cells']}/{summary['expected_cells']} cells; {len(summary['effects'])} paired effect groups")


if __name__ == "__main__":
    cli()
