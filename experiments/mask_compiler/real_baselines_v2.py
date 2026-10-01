"""CPU-only repair: unchanged baseline models on frozen three-fold native data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_backbone import ROOT, load_protocol, natural_episodes, sha
from mira.panel_data import load_panel, validate_splits
from mira.panel_runner import logistic
from mira.representations import PreparedInputs
from mira.artifacts import atomic_json, capture_environment
from mira.metrics import expected_nll


def validate_three_folds(panel, splits, declared):
    """Check complete outer partitions, group isolation, and canonical ID binding."""
    records = splits["folds"]
    if splits.get("n_splits") != 3 or len(records) != 3 or sorted(declared) != [0, 1, 2] or sorted(r["fold"] for r in records) != [0, 1, 2]:
        raise ValueError("Require the frozen three native folds")
    universe, all_held = set(range(len(panel.y))), []
    checks = []
    for record in records:
        support, query, training, held = [record[key] for key in
            ("support_indices", "query_indices", "training_indices", "heldout_indices")]
        for indices in (support, query, training, held):
            if len(indices) != len(set(indices)) or not set(indices) <= universe:
                raise ValueError("Invalid/duplicate canonical split indices")
        if set(training) & set(held) or set(training) | set(held) != universe:
            raise ValueError("Outer partitions must cover disjoint canonical rows")
        if not set(support) <= set(training) or not set(query) <= set(held):
            raise ValueError("Support/query escaped frozen outer partitions")
        if set(panel.group_ids[training]) & set(panel.group_ids[held]):
            raise ValueError("Source/identical-X groups leaked across outer partitions")
        if len(support) != splits["support_size"] or not 0 < len(query) <= splits["query_cap"]:
            raise ValueError("Support size/query cap differs")
        if set(panel.y[support]) != {0, 1} or set(panel.y[query]) != {0, 1}:
            raise ValueError("Both classes required in support and evaluator partitions")
        for key, actual in (("support_row_ids", panel.row_ids[support]), ("query_row_ids", panel.row_ids[query]),
                            ("support_group_ids", panel.group_ids[support]), ("query_group_ids", panel.group_ids[query])):
            if record[key] != actual.tolist():
                raise ValueError("Canonical IDs differ from saved indices")
        all_held.extend(held)
        checks.append({"fold": record["fold"], "support": len(support), "queries": len(query),
                       "support_groups": len(set(panel.group_ids[support])), "query_groups": len(set(panel.group_ids[query]))})
    if len(all_held) != len(universe) or set(all_held) != universe:
        raise ValueError("Every canonical row must be held out exactly once")
    return checks


def infer(inputs, groups, kind, seed):
    """Only support features/labels/groups and query features enter inference."""
    if not isinstance(inputs, PreparedInputs):
        raise TypeError("Restricted support/query inputs required")
    groups = np.asarray(groups)
    if groups.shape != inputs.labels.shape or set(inputs.labels) != {0, 1}:
        raise ValueError("Support labels/groups must align and cover both classes")
    model = logistic(seed, inputs.labels, groups) if kind == "logistic" else HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1., early_stopping=False, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(inputs.context, inputs.labels)
        probability = model.predict_proba(inputs.query)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid issued probability")
    return probability, {"warnings": [str(w.message) for w in caught],
                         "chosen_C": model.best_params_["clf__C"] if kind == "logistic" else None}


def run(protocol, out):
    if out.exists():
        raise ValueError("Preserve prior outputs; choose a new directory")
    out.mkdir(parents=True)
    rows, failures = [], []
    start = time.monotonic()
    manifest = {"status": "running", "paid_usd": 0, "source_sha256": sha(__file__), "protocol_sha256": sha(protocol),
                "repair": "Only replace the erroneous five-fold validation with independent complete three-fold checks; baseline models/CV are unchanged.",
                "scope": "Full numeric/native-mask inputs; categorical codes numeric. Support-only group CV logistic; fixed HistGB. Empirical natural-data scores; no oracle/clinical claim."}
    atomic_json(out / "manifest.json", manifest)
    try:
        plan, digest = load_protocol(protocol)
        manifest["protocol_sha256"] = digest
        manifest["environment"] = capture_environment(out)
        manifest["dependencies_sha256"] = {relative: expected for relative, expected in plan["file_sha256"].items()
            if relative.startswith("src/mira/") or relative in ("experiments/mask_compiler/run_backbone.py", "experiments/mask_compiler/real_baselines.py")}
        manifest["data_sha256"] = {relative: expected for relative, expected in plan["file_sha256"].items()
            if any(relative.startswith(spec["directory"] + "/") for spec in plan["real"])}
        audit, reproduced = {}, []
        for spec in plan["real"]:
            directory = ROOT / spec["directory"]
            panel = load_panel(directory)
            splits = json.loads((directory / "splits.json").read_text())
            audit[directory.name] = validate_three_folds(panel, splits, spec["folds"])
            try:
                validate_splits(panel, splits)
            except ValueError as error:
                reproduced.append({"dataset": directory.name, "error": repr(error), "traceback": traceback.format_exc()})
        atomic_json(out / "v1_failed_manifest.json", {"status": "failed_before_fitting", "paid_usd": 0,
            "predictions": 0, "errors": reproduced, "evidence": "Replay of the frozen v1 validator on unchanged native data; original failed output directory is retained.",
            "protocol_sha256": digest, "v1_source_sha256": sha(HERE / "real_baselines.py")})
        atomic_json(out / "split_audit.json", audit)
        episodes = list(natural_episodes(plan))
        if len(episodes) != 6:
            raise ValueError("Require the six frozen natural episodes")
        with threadpool_limits(limits=1):
            for episode in episodes:
                xc, xq = episode["native_context"], episode["native_query"]
                context_labels = episode["support"].labels
                for mode in ("native", "identity"):
                    c = xc if mode == "native" else np.column_stack([xc, np.isnan(xc).astype(np.uint8)])
                    q = xq if mode == "native" else np.column_stack([xq, np.isnan(xq).astype(np.uint8)])
                    inputs = PreparedInputs(c.copy(), context_labels.copy(), q.copy(), mode)
                    for kind in ("logistic", "histgb"):
                        try:
                            p, fitting = infer(inputs, episode["groups"], kind, 92000 + episode["fold"])
                            path = out / f"{episode['dataset']}_fold{episode['fold']}_{kind}_{mode}.npz"
                            np.savez_compressed(path, p=p, query_row_ids=episode["query_ids"], support_row_ids=episode["support_ids"],
                                                query_group_ids=episode["query_groups"], support_group_ids=episode["groups"])
                            # Evaluator accesses query outcomes only after probabilities are persisted.
                            targets = episode["targets"]
                            rows.append({"episode_id": f"{episode['dataset']}_fold{episode['fold']}", "backbone_episode_id": episode["episode_id"],
                                "dataset": episode["dataset"], "fold": episode["fold"], "kind": "natural", "model": kind, "mode": mode,
                                "queries": len(p), "empirical_nll": expected_nll(p, targets), "brier": float(np.mean((p - targets) ** 2)),
                                "auroc": float(roc_auc_score(targets, p)), "prediction_file": path.name, "prediction_sha256": sha(path), **fitting})
                        except Exception as error:
                            failures.append({"episode": episode["episode_id"], "model": kind, "mode": mode,
                                             "error": repr(error), "traceback": traceback.format_exc()})
                        atomic_json(out / "results.json", rows)
                        atomic_json(out / "errors.json", failures)
        load_protocol(protocol)  # Bracket CPU fitting with complete frozen source/data verification.
        manifest.update(status="complete" if len(rows) == 24 and not failures else "incomplete", predictions=len(rows),
                        expected_predictions=24, seconds=time.monotonic() - start)
        summary = []
        for dataset in sorted({r["dataset"] for r in rows}):
            for model in ("logistic", "histgb"):
                for mode in ("native", "identity"):
                    cell = [r for r in rows if (r["dataset"], r["model"], r["mode"]) == (dataset, model, mode)]
                    if cell:
                        summary.append({"dataset": dataset, "model": model, "mode": mode, "folds": len(cell),
                            "queries": sum(r["queries"] for r in cell), "query_weighted_nll": float(np.average([r["empirical_nll"] for r in cell], weights=[r["queries"] for r in cell]))})
        atomic_json(out / "summary.json", summary)
    except Exception as error:
        failures.append({"stage": "run", "error": repr(error), "traceback": traceback.format_exc()})
        manifest.update(status="failed", predictions=len(rows), seconds=time.monotonic() - start)
        atomic_json(out / "errors.json", failures)
        raise
    finally:
        atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/trained_compiler_validation_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/reports/natural_compiler_baselines_v2")
    args = parser.parse_args()
    run(args.protocol, args.out)
