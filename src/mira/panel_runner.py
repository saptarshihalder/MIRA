"""Frozen real-covariate experiment; query labels stay in mask generation/scoring."""
from __future__ import annotations
import argparse
import gc
import json
import shutil
import time
import warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from .artifacts import atomic_json, atomic_npz, append_result, capture_environment, checkpoint_hashes, file_sha256, source_hashes
from .metrics import expected_nll
from .models import ModelSettings, create_model, predict_binary
from .representations import PreparedInputs
from .panel_data import load_panel, validate_splits


def impose(values, native_mask, labels, rate, gamma, active, seed):
    """Evaluator-only retrospective mechanism; never a learner argument."""
    if not 0 < rate < 1 or not 0 <= gamma < 1:
        raise ValueError("Invalid mask parameters")
    probability = np.full(values.shape, rate, dtype=float)
    probability[:, active] *= 1 + gamma * (2*np.asarray(labels)-1)
    if probability.max() > 1:
        raise ValueError("Invalid class-conditional mask probability")
    imposed = np.random.default_rng(seed).random(values.shape) < probability
    union = np.asarray(native_mask, bool) | imposed
    masked = np.array(values, dtype=float, copy=True)
    masked[union] = np.nan
    return masked, imposed, union


def prepare(xc, yc, xq, mc, mq, mode, seed):
    """Only observable query values/masks; no query target parameter."""
    xc, xq = xc.copy(), xq.copy()
    if mode == "native_indicators":
        xc, xq = np.column_stack([xc, mc]), np.column_stack([xq, mq])
    elif mode == "native_shuffled":
        rc, rq = np.random.default_rng(seed+1), np.random.default_rng(seed+2)
        xc = np.column_stack([xc, np.column_stack([rc.permutation(mc[:, j]) for j in range(mc.shape[1])])])
        xq = np.column_stack([xq, np.column_stack([rq.permutation(mq[:, j]) for j in range(mq.shape[1])])])
    elif mode != "native":
        raise ValueError("Unknown representation")
    return PreparedInputs(xc.astype(np.float32), np.array(yc, copy=True), xq.astype(np.float32), mode)


def logistic(seed, labels, groups):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline([("impute", SimpleImputer(keep_empty_features=True)),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels, groups))
    if any(set(groups[a]) & set(groups[b]) or len(np.unique(labels[a])) != 2 or len(np.unique(labels[b])) != 2 for a,b in folds):
        raise ValueError("Insufficient group/class coverage for declared inner CV")
    return GridSearchCV(pipeline, {"clf__C": [.01, .1, 1, 10]},
        cv=folds, scoring="neg_log_loss", n_jobs=1, error_score="raise")


def run(args, factory=create_model):
    plan_path, out = Path(args.protocol_file), Path(args.out)
    digest = file_sha256(plan_path)
    if digest != plan_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Frozen panel protocol changed")
    plan = json.loads(plan_path.read_text())
    if source_hashes() != plan["runner_source_sha256"]:
        raise ValueError("Frozen runner source changed")
    if not plan["frozen"] or args.dataset not in plan["datasets"] or (out.exists() and any(out.iterdir())):
        raise ValueError("Frozen dataset required; preserve previous output")
    source = Path(args.data_root) / args.dataset
    for filename, expected_hash in plan["data_sha256"][args.dataset].items():
        if file_sha256(source / filename) != expected_hash:
            raise ValueError("Frozen panel input changed")
    panel = load_panel(source)
    split_definition = json.loads((source / "splits.json").read_text())
    validate_splits(panel, split_definition)
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("data", "predictions", "provenance"):
        (out / directory).mkdir()
    for filename in ("dataset.npz", "splits.json", "manifest.json"):
        shutil.copyfile(source / filename, out / "provenance" / filename)
    with np.load(source / "dataset.npz", allow_pickle=False) as loaded:
        data = {key: loaded[key].copy() for key in loaded.files}
    splits = split_definition["folds"]
    expected = len(splits)*len(plan["rates"])*len(plan["gammas"])*len(plan["models"])*len(plan["modes"])
    manifest = {"status": "running", "dataset": args.dataset, "expected_cells": expected,
        "frozen_protocol_sha256": digest, "protocol": plan,
        "source_sha256": source_hashes(), "dataset_sha256": file_sha256(source / "dataset.npz"),
        "split_sha256": file_sha256(source / "splits.json"),
        "environment": capture_environment(out), "completed_cells": 0}
    atomic_json(out / "manifest.json", manifest)
    rows, errors = [], []
    dataset_id = plan["datasets"].index(args.dataset)
    settings = ModelSettings(args.device, plan["ensembles"])
    try:
        for split in splits:
            fold = split["fold"]
            ci, qi = np.asarray(split["support_indices"]), np.asarray(split["query_indices"])
            if (set(data["group_ids"][ci]) & set(data["group_ids"][qi]) or len(ci) != plan["context"]
                    or not set(data["y"][ci]) == {0, 1}):
                raise ValueError("Invalid support/query split")
            seed = plan["mask_seed_base"] + 100*dataset_id + fold
            active = int(np.random.default_rng(seed).integers(data["X"].shape[1]))
            for rate in plan["rates"]:
                for gamma in plan["gammas"]:
                    task = f"fold{fold}_r{rate:g}_g{gamma:g}"
                    xc, ic, mc = impose(data["X"][ci], data["native_mask"][ci], data["y"][ci], rate, gamma, active, seed+10000)
                    xq, iq, mq = impose(data["X"][qi], data["native_mask"][qi], data["y"][qi], rate, gamma, active, seed+20000)
                    path = out / "data" / (task + ".npz")
                    atomic_npz(path, xc=xc, yc=data["y"][ci], xq=xq, yq=data["y"][qi], mc=mc, mq=mq,
                        native_mc=data["native_mask"][ci], native_mq=data["native_mask"][qi], imposed_mc=ic, imposed_mq=iq,
                        support_row_ids=data["row_ids"][ci], query_row_ids=data["row_ids"][qi],
                        support_group_ids=data["group_ids"][ci], query_group_ids=data["group_ids"][qi])
                    atomic_json(path.with_suffix(".json"), {"active_column": active, "mask_seed": seed,
                        "rate": rate, "gamma": gamma, "class_missing_rates_query":
                        {str(y): float(iq[data["y"][qi]==y, active].mean()) for y in (0,1)}})
                    for name in plan["models"]:
                        for mode in plan["modes"]:
                            cell = f"{task}_{name.replace(':','_')}_{mode}"
                            start = time.monotonic()
                            try:
                                inputs = prepare(xc, data["y"][ci], xq, mc, mq, mode, seed)
                                model = logistic(seed, data["y"][ci], data["group_ids"][ci]) if name == "logistic" else factory(name, settings, seed)
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    p = predict_binary(model, inputs)
                                prediction = out / "predictions" / (cell + ".npz")
                                atomic_npz(prediction, probability=p, query_row_ids=data["row_ids"][qi])
                                yq = data["y"][qi]  # Evaluator access only after predictions are fixed.
                                row = {"cell_id": cell, "dataset": args.dataset, "fold": fold, "rate": rate, "gamma": gamma,
                                    "model": name, "mode": mode, "queries": len(qi), "empirical_nll": expected_nll(p, yq),
                                    "brier": float(np.mean((p-yq)**2)), "auroc": float(roc_auc_score(yq,p)),
                                    "seconds": time.monotonic()-start, "data_file": str(path.relative_to(out)),
                                    "data_sha256": file_sha256(path), "prediction_file": str(prediction.relative_to(out)),
                                    "prediction_sha256": file_sha256(prediction), "chosen_C": model.best_params_["clf__C"] if name=="logistic" else None,
                                    "support_cv_score": float(model.best_score_) if name=="logistic" else None,
                                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in caught]}
                                append_result(out, rows, row)
                                print(json.dumps({k:row[k] for k in ("cell_id","empirical_nll","seconds")}), flush=True)
                            except Exception as error:
                                errors.append({"cell_id":cell,"error":repr(error)})
                                atomic_json(out / "errors.json", errors)
                            finally:
                                if "model" in locals():
                                    del model
                                gc.collect()
    finally:
        manifest.update(status="complete" if len(rows)==expected else "incomplete", completed_cells=len(rows),
                        errors=errors, checkpoint_sha256=checkpoint_hashes(args.checkpoint_dirs))
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "errors.json", errors)
    return manifest


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol-file", default="/opt/mira/panel_protocol.json")
    parser.add_argument("--data-root", default="/opt/mira/panel")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dirs", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if run(args)["status"] != "complete":
        raise SystemExit("Incomplete panel; preserve failures and partial artifacts")
