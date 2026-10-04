"""Read-only replay of the frozen Wine benchmark; writes audit only after pass.

No benchmark/preparation module is imported, no models are fitted, and no new
scenarios are evaluated. Tree pickles are trusted local artifacts whose hashes
must match both saved manifests and the training report before unpickling.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import csv
import hashlib
import io
import itertools
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/wine_acquisition_v1"
COLORS = ("red", "white")
SEEDS = (813001, 813002, 813003)
METHODS = ("masked_depth2", "joint_depth2", "masked_greedy", "masked_static", "masked_stop", "tree_depth2", "tree_greedy", "tree_static")
FEATURES = ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")
TEMPERATURES = (.5, .75, 1., 1.5, 2.)
AVAILABILITY = (tuple(range(11)), tuple(j for j in range(11) if j not in (0, 1)), tuple(j for j in range(11) if j not in (4, 5)), tuple(j for j in range(11) if j not in (8, 9)))
COSTS = (np.full(11, .01), np.full(11, .03), .01 + .01 * (np.arange(11) % 3))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, message):
    require(np.array_equal(actual, expected), message)


def near(actual, expected, message, atol=1e-12, rtol=1e-12):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False), message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def path_inside(base, relative):
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), "Manifest path escaped root")
    return path


def check_hashes(base, manifest):
    for relative, expected in manifest.items():
        require(digest(path_inside(base, relative)) == expected, f"Hash mismatch: {relative}")


def verify_summary(report):
    path = OUT / "summary.csv"
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ["domain", "method", "mean_nll_plus_cost"], "Summary CSV schema")
        rows = list(reader)
    actual = {(row["domain"], row["method"]): float(row["mean_nll_plus_cost"]) for row in rows}
    expected = {(color, method): report["means"][color][method] for color in COLORS for method in METHODS}
    require(len(rows) == 16 and set(actual) == set(expected), "Summary must retain all 16 domain/method means")
    for key, value in actual.items():
        near(value, expected[key], "Summary CSV versus frozen report")
    return len(rows)


def finalize_supplementary():
    """Verify root-added summary after a successful full replay, without rescoring."""
    manifest = read_json(OUT / "manifest.json")
    check_hashes(OUT, manifest)
    freeze = read_json(ROOT / "artifacts/manifests/wine_acquisition_freeze.json")
    require(len(freeze) == 8, "Eight frozen inputs")
    check_hashes(ROOT, freeze)
    audit = read_json(OUT / "audit.json")
    require(audit["status"] == "PASS" and audit["checks"]["score_cells_replayed"] == 576, "Completed full replay required")
    require(audit["report_sha256"] == digest(OUT / "report.json"), "Report changed after full replay")
    actual = {p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"}
    require(actual - set(manifest) <= {"runtime.json", "summary.csv"} and not (set(manifest) - actual), "Unexpected post-audit output mutation")
    report = read_json(OUT / "report.json")
    summary_rows = verify_summary(report)
    require(summary_rows == 16, "Expected root-generated summary")
    audit["supplementary_verification"] = {
        "summary_rows_checked": summary_rows,
        "summary_sha256": digest(OUT / "summary.csv"),
        "runtime_sha256": digest(OUT / "runtime.json"),
        "verifier_sha256": digest(Path(__file__)),
        "note": "Root added summary.csv during the first successful full replay. Existing output hashes and all eight frozen inputs were rechecked; summary matched report means. No score replay was repeated.",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for name in ("audit.json", "runtime.json", "summary.csv"):
        manifest[name] = digest(OUT / name)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": "PASS", "summary_rows_checked": summary_rows, "manifest_artifacts": len(manifest), "additional_score_replays": 0}))


def rebuild_raw(data_audit):
    data, global_groups = {}, {}
    for color in COLORS:
        source = data_audit["sources"][color]
        source_path = path_inside(ROOT, source["path"])
        require(digest(source_path) == source["sha256"], "Source provenance hash")
        rows = list(csv.reader(io.StringIO(source_path.read_bytes().decode("utf-8-sig")), delimiter=";"))
        same(rows[0], list(FEATURES) + ["quality"], "Source schema")
        values = np.asarray(rows[1:], dtype=np.float64)
        require(values.shape[1] == 12 and np.isfinite(values).all(), "Raw shape/finite values")
        full_x = values[:, :11]
        quality = values[:, 11]
        require(((quality == np.floor(quality)) & (quality >= 0) & (quality <= 10)).all(), "Raw quality scale")
        y = (quality >= 6).astype(np.uint8)
        keys, partitions = [], []
        for row, label in zip(full_x, quality):
            canonical = np.array(row, dtype="<f8")
            canonical[canonical == 0] = 0.0
            payload = canonical.tobytes()
            key = hashlib.sha256(payload).hexdigest()
            scaled = int(key, 16) * 10
            partition = 0 if scaled < 6 * 2**256 else (1 if scaled < 8 * 2**256 else 2)
            entry = global_groups.setdefault(key, {"bytes": payload, "partitions": set(), "colors": set(), "labels": set(), "n": 0})
            require(entry["bytes"] == payload, "Feature hash collision")
            entry["partitions"].add(partition)
            entry["colors"].add(color)
            entry["labels"].add(float(label))
            entry["n"] += 1
            keys.append(key)
            partitions.append(partition)
        partition = np.asarray(partitions, dtype=np.uint8)
        thresholds = np.quantile(full_x[partition == 0], [1/3, 2/3], axis=0, method="linear").T
        x = np.column_stack([np.searchsorted(thresholds[j], full_x[:, j], side="right") for j in range(11)]).astype(np.uint8)
        rebuilt = {"X": x, "y": y, "row_ids": np.asarray([f"winequality-{color}.csv:{i+2}" for i in range(len(y))]),
                   "groups": np.asarray(keys, dtype="U64"), "partition": partition, "thresholds": thresholds, "feature_names": np.asarray(FEATURES)}
        output = data_audit["outputs"][color]
        path = path_inside(ROOT, output["path"])
        require(digest(path) == output["sha256"], "Prepared artifact hash")
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == set(rebuilt), "Prepared NPZ field set")
            for name in rebuilt:
                same(saved[name], rebuilt[name], f"Raw reconstruction: {color}/{name}")
        require(output["rows"] == len(y), "Prepared row count")
        require(output["unique_feature_groups"] == len(set(keys)), "Prepared group count")
        for p, name in enumerate(("fit", "validation", "development")):
            require(output["partition_rows"][name] == int((partition == p).sum()), "Partition row count")
        data[color] = rebuilt
    overlap = sum(len(v["partitions"]) > 1 for v in global_groups.values())
    require(overlap == 0, "Global feature group crosses partitions")
    actual = {
        "total_rows": sum(len(d["y"]) for d in data.values()),
        "global_unique_feature_groups": len(global_groups),
        "duplicate_groups": sum(v["n"] > 1 for v in global_groups.values()),
        "duplicate_rows_beyond_first": sum(v["n"] - 1 for v in global_groups.values()),
        "cross_color_feature_groups": sum(len(v["colors"]) > 1 for v in global_groups.values()),
        "cross_partition_group_overlap": overlap,
    }
    for key, value in actual.items():
        require(data_audit["audit"][key] == value, f"Raw audit count: {key}")
    require(all(len(v["partitions"]) == 1 for v in global_groups.values() if len(v["labels"]) > 1), "Conflicting-label group split")
    return data, actual


def state_space():
    result = [np.full(11, -1, dtype=np.int64)]
    for width in (1, 2):
        for subset in itertools.combinations(range(11), width):
            for values in itertools.product(range(3), repeat=width):
                row = np.full(11, -1, dtype=np.int64)
                row[list(subset)] = values
                result.append(row)
    states = np.asarray(result)
    index = {tuple(row): i for i, row in enumerate(states)}
    children = {}
    for i, state in enumerate(states):
        if sum(state >= 0) >= 2:
            continue
        for feature in np.flatnonzero(state < 0):
            child_ids = []
            for value in range(3):
                child = state.copy()
                child[feature] = value
                child_ids.append(index[tuple(child)])
            children[i, int(feature)] = np.asarray(child_ids)
    require(len(states) == 529 and len(index) == 529, "State enumeration")
    return states, index, children


def mixture_log_class(checkpoint, states):
    # Reconstruct the normalized mixture directly from parameter tensors.
    with torch.no_grad():
        cat = torch.as_tensor(checkpoint["cat"]).log_softmax(-1)
        weight = torch.as_tensor(checkpoint["weight"]).flatten().log_softmax(0).reshape(2, 16)
        s = torch.as_tensor(states, dtype=torch.long)
        terms = torch.nn.functional.one_hot(s.clamp_min(0), 3) * (s >= 0)[:, :, None]
        score = torch.einsum("ndv,ckdv->nck", terms.float(), cat) + weight
        return score.logsumexp(-1)


def calibrate(probability, temperature):
    logits = np.log(np.clip(probability, 1e-8, 1.0)) / temperature
    return softmax(logits, axis=1)


def tree_input(states):
    return np.column_stack((np.maximum(states, 0) / 2, states >= 0))


def prediction_replay(folder, data, seed, training, states, children):
    temperatures = read_json(folder / "temperatures.json")
    require(set(temperatures) == {"joint", "masked", "tree"}, "Temperature model set")
    val = data["partition"] == 1
    vx = np.tile(data["X"][val].astype(np.int64), (4, 1))
    vy = np.tile(data["y"][val].astype(np.int64), 4)
    rng = np.random.default_rng(seed + 700)
    mask = rng.random(vx.shape) < rng.random((len(vx), 1))
    vs = np.where(mask, vx, -1)
    predictions, branch = {}, {}
    with np.load(folder / "state_predictions.npz", allow_pickle=False) as saved:
        require(set(saved.files) == {"states", "joint", "masked", "tree"}, "Prediction artifact fields")
        same(saved["states"], states, "Saved state enumeration")
        for mode in ("joint", "masked", "tree"):
            record = training[mode]
            checkpoint_path = folder / ("tree.pkl" if mode == "tree" else f"{mode}.npz")
            require(digest(checkpoint_path) == record["sha256"], "Checkpoint training-record hash")
            if mode == "tree":
                # These are locally produced trusted pickles, already hash-checked.
                with checkpoint_path.open("rb") as f:
                    tree = pickle.load(f)
                same(tree.classes_, [0, 1], "Tree class order")
                q = tree.predict_proba(tree_input(states))
                vp = tree.predict_proba(tree_input(vs))
            else:
                with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                    require(set(checkpoint.files) == {"cat", "weight"}, "Mixture parameter fields")
                    require(checkpoint["cat"].shape == (2, 16, 11, 3) and checkpoint["weight"].shape == (2, 16), "Mixture shape")
                    log_class = mixture_log_class(checkpoint, states).double().numpy()
                    vp = mixture_log_class(checkpoint, vs).softmax(-1).numpy()
                q = softmax(log_class, axis=1)
                log_mass = np.logaddexp(log_class[:, 0], log_class[:, 1])
                branch[mode] = {key: softmax(log_mass[child]) for key, child in children.items()}
            losses = [float((-np.log(np.clip(calibrate(vp, t)[np.arange(len(vy)), vy], 1e-8, 1))).mean()) for t in TEMPERATURES]
            near(losses, record["validation_nll"], "Existing validation calibration replay", atol=1e-7, rtol=1e-7)
            chosen = TEMPERATURES[int(np.argmin(losses))]
            require(chosen == temperatures[mode] == record["temperature"], "Frozen selected temperature")
            predictions[mode] = calibrate(q, chosen)
            near(predictions[mode], saved[mode], "529-state calibrated prediction replay")
    return predictions, branch


def rebuild_policy(prediction, transitions, children, availability, costs, kind):
    entropy = -(prediction * np.log(np.clip(prediction, 1e-8, 1))).sum(axis=1)
    actions = np.full(529, -1, dtype=np.int16)
    if kind == "stop":
        return actions, np.asarray([], dtype=np.int16)
    if kind == "static":
        options = [(entropy[0], ())]
        for feature in availability:
            c = children[0, feature]
            options.append((costs[feature] + transitions[0, feature] @ entropy[c], (feature,)))
        for first, second in itertools.combinations(availability, 2):
            c = children[0, first]
            expectation = sum(transitions[0, first][v] * (transitions[int(s), second] @ entropy[children[int(s), second]]) for v, s in enumerate(c))
            options.append((costs[first] + costs[second] + expectation, (first, second)))
        return actions, np.asarray(min(options)[1], dtype=np.int16)
    values = entropy.copy()
    # Reverse enumeration processes two-reveal states before one-reveal states.
    for state in reversed(range(529)):
        for feature in availability:
            c = children.get((state, feature))
            if c is None:
                continue
            future = entropy if kind == "greedy" else values
            candidate = costs[feature] + transitions[state, feature] @ future[c]
            if candidate < values[state]:
                values[state] = candidate
                actions[state] = feature
    return actions, np.asarray([], dtype=np.int16)


def replay_cell(saved, data, prediction, action, fixed, kind, availability, costs, index):
    dev = data["partition"] == 2
    rows = np.flatnonzero(dev)
    same(saved["row_ids"], data["row_ids"][dev], "Score row identity")
    same(saved["groups"], data["groups"][dev], "Score row groups")
    require(saved["path"].shape == (len(rows), 2), "Acquisition path shape")
    same(saved["action"], action, "All-state policy decision replay")
    same(saved["fixed"], fixed, "Static policy replay")
    require(np.isin(saved["path"], np.arange(-1, 11)).all(), "Path action range")
    terminal_ids, paid = [], np.zeros(len(rows))
    for n, source_row in enumerate(rows):
        state = np.full(11, -1, dtype=np.int64)
        stopped = False
        for step in range(2):
            state_id = index[tuple(state)]
            decision = int(fixed[step]) if kind == "static" and step < len(fixed) else (-1 if kind == "static" else int(action[state_id]))
            if stopped:
                decision = -1
            require(int(saved["path"][n, step]) == decision, "Saved path disagrees with policy")
            if decision == -1:
                stopped = True
                continue
            require(decision in availability and state[decision] < 0, "Unavailable or repeated acquisition")
            # Only the selected coordinate crosses the evaluator/policy boundary.
            state[decision] = int(data["X"][source_row, decision])
            paid[n] += costs[decision]
        terminal_ids.append(index[tuple(state)])
    probability = prediction[np.asarray(terminal_ids)]
    near(saved["probability"], probability, "Selected-feature probability replay")
    near(saved["paid"], paid, "Actual acquisition cost")
    labels = data["y"][dev]
    row_nll = -np.log(np.clip(probability[np.arange(len(rows)), labels], 1e-8, 1.0))
    score = row_nll + paid
    near(saved["score"], score, "Every per-row NLL-plus-cost")
    return {"n": len(rows), "nll": float(row_nll.mean()), "paid": float(paid.mean()), "risk": float(score.mean())}


def main():
    if not (OUT / "report.json").is_file():
        print("Auditor ready; report.json is not yet present. No polling or scoring performed.")
        return
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    limiter = threadpool_limits(limits=1)
    freeze = read_json(ROOT / "artifacts/manifests/wine_acquisition_freeze.json")
    require(len(freeze) == 8, "Expected eight frozen inputs")
    check_hashes(ROOT, freeze)
    output_manifest = read_json(OUT / "manifest.json")
    check_hashes(OUT, output_manifest)
    actual_files = {p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"}
    # Root's post-run runtime provenance and mechanical summary are authorized.
    require(actual_files - set(output_manifest) <= {"runtime.json", "summary.csv"} and not (set(output_manifest) - actual_files), "Output manifest does not cover every experiment artifact")
    supplementary = {name: digest(OUT / name) for name in ("runtime.json", "summary.csv") if (OUT / name).is_file()}
    fitted = read_json(OUT / "fitted_before_scoring.json")
    expected_fitted = {f"{color}_{seed}/{name}" for color in COLORS for seed in SEEDS for name in ("joint.npz", "masked.npz", "tree.pkl", "temperatures.json", "state_predictions.npz")}
    require(set(fitted) == expected_fitted, "Expected 30 prescoring artifacts")
    check_hashes(OUT, fitted)
    report = read_json(OUT / "report.json")
    summary_rows_checked = verify_summary(report)
    require(report["cloud_calls"] == 0, "Cloud-call record")
    data, raw_checks = rebuild_raw(read_json(ROOT / "artifacts/manifests/wine_acquisition_data_audit.json"))
    states, index, children = state_space()
    training = {(r["color"], r["seed"], r["model"]): r for r in report["training"]}
    expected_training = set(itertools.product(COLORS, SEEDS, ("joint", "masked", "tree")))
    require(len(report["training"]) == 18 and set(training) == expected_training, "All 18 fitted models retained")
    record_keys = ("color", "seed", "availability", "cost", "method")
    records = {tuple(r[k] for k in record_keys): r for r in report["records"]}
    expected_cells = set(itertools.product(COLORS, SEEDS, range(4), range(3), METHODS))
    require(len(report["records"]) == 576 and set(records) == expected_cells, "All 576 scenario/method cells retained")
    require(len(records) == len(report["records"]), "Duplicate score records")
    replayed, expected_artifacts = [], set(expected_fitted)
    for color in COLORS:
        for seed in SEEDS:
            folder = OUT / f"{color}_{seed}"
            local_training = {mode: training[color, seed, mode] for mode in ("joint", "masked", "tree")}
            predictions, transitions = prediction_replay(folder, data[color], seed, local_training, states, children)
            for ai, availability in enumerate(AVAILABILITY):
                for ci, costs in enumerate(COSTS):
                    for method in METHODS:
                        model, kind = method.split("_")
                        base = "joint" if model == "joint" else "masked"
                        action, fixed = rebuild_policy(predictions[model], transitions[base], children, availability, costs, kind)
                        path = folder / f"a{ai}_c{ci}_{method}.npz"
                        expected_artifacts.add(path.relative_to(OUT).as_posix())
                        with np.load(path, allow_pickle=False) as saved:
                            require(set(saved.files) == {"row_ids", "groups", "probability", "paid", "path", "score", "action", "fixed"}, "Score artifact fields")
                            summary = replay_cell(saved, data[color], predictions[model], action, fixed, kind, availability, costs, index)
                        key = (color, seed, ai, ci, method)
                        for field, value in summary.items():
                            near(value, records[key][field], f"Reported {field} mean")
                        replayed.append(dict(zip(record_keys, key), **summary))
    expected_artifacts.update(("report.json", "fitted_before_scoring.json"))
    require(set(output_manifest) - {"audit.json", "runtime.json", "summary.csv"} == expected_artifacts, "Unexpected or omitted experiment artifacts")
    means = {color: {method: float(np.mean([r["risk"] for r in replayed if r["color"] == color and r["method"] == method])) for method in METHODS} for color in COLORS}
    gates = {}
    for color in COLORS:
        require(set(report["means"][color]) == set(METHODS), "All method means retained")
        for method in METHODS:
            near(means[color][method], report["means"][color][method], "Report aggregate risk")
        gains = [means[color][m] - means[color]["masked_depth2"] for m in METHODS if m != "masked_depth2"]
        replicated = all(np.mean([r["risk"] for r in replayed if r["color"] == color and r["seed"] == seed and r["method"] == m]) > np.mean([r["risk"] for r in replayed if r["color"] == color and r["seed"] == seed and r["method"] == "masked_depth2"]) for seed in SEEDS for m in METHODS if m != "masked_depth2")
        gates[color] = {"minimum_gain": min(gains), "positive_all_seeds": bool(replicated), "passed": bool(min(gains) >= .005 and replicated)}
        near(gates[color]["minimum_gain"], report["gates"][color]["minimum_gain"], "Minimum-gain gate")
        for field in ("positive_all_seeds", "passed"):
            require(gates[color][field] == report["gates"][color][field], "Gate decision")
    benchmark_passed = all(g["passed"] for g in gates.values())
    require(benchmark_passed == report["passed"], "Overall gate")
    audit = {
        "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": digest(Path(__file__)), "freeze_sha256": digest(ROOT / "artifacts/manifests/wine_acquisition_freeze.json"),
        "report_sha256": digest(OUT / "report.json"), "cpu_threads": 1, "seconds": time.monotonic() - start,
        "checks": {"freeze_hashes": 8, "prescoring_hashes": len(fitted), "output_hashes_before_audit": len(output_manifest),
                   "checkpoint_training_hashes": 18, "raw_reconstruction": raw_checks, "train_only_terciles": True,
                   "mixture_checkpoints_reloaded": 12, "trusted_tree_checkpoints_reloaded": 6,
                   "states_per_model": len(states), "all_calibrated_state_predictions_reproduced": True,
                   "existing_validation_masks_and_temperature_choices_reproduced": True,
                   "all_state_policy_decisions_rederived": True, "score_cells_replayed": len(replayed),
                   "per_row_score_replays": sum(r["n"] for r in replayed), "selected_features_only_in_path_replay": True,
                   "all_row_identities_costs_scores_and_feasibility_verified": True,
                   "all_results_retained": True, "means_and_gates_recomputed": True},
        "recomputed_means": means, "recomputed_gates": gates, "benchmark_gate_passed": benchmark_passed,
        "tolerances": {"default_atol": 1e-12, "default_rtol": 1e-12, "validation_loss_atol": 1e-7, "validation_loss_rtol": 1e-7},
        "limitations": ["Replay verifies persisted artifacts and frozen source; it cannot independently prove historical execution order from hashes alone.",
                        "No new fitting, tuning, scenarios, or outcome-dependent exclusions were performed.",
                        "This audit does not establish producer/batch independence, real measurement costs, novelty, or external validity."],
        "audit_process_note": "Initial audit attempt stopped at manifest coverage because root added runtime.json after the run. Auditor allowlist was corrected for that provenance file only; no scientific retry or source/model change occurred.",
        "runtime_provenance_sha256": supplementary.get("runtime.json"),
        "summary_rows_checked": summary_rows_checked,
        "unverified_decision_rules": [],
    }
    audit_path = OUT / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    # Preserve every preexisting entry; include the authorized runtime provenance.
    output_manifest.update(supplementary)
    output_manifest["audit.json"] = digest(audit_path)
    (OUT / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    limiter.restore_original_limits()
    print(json.dumps({"audit": "PASS", "cells": len(replayed), "models": 18, "benchmark_gate_passed": benchmark_passed, "seconds": audit["seconds"]}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize-supplementary"]:
        finalize_supplementary()
    else:
        require(not sys.argv[1:], "Unknown arguments")
        main()
