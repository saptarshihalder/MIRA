"""CPU-only descriptive baselines fitted after the saved confirmation run.

Queries supply observations only. Hyperparameters use support-only CV. The
known-beta pilot mixture is a separate parameter-informed reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import t
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira.data import EvaluationTargets, Observations, QueryInputs, Support  # noqa: E402
from mira.metrics import evaluate  # noqa: E402

C_GRID = (.03, .3, 3., 30.)
METHODS = ("u_logistic_cv", "interaction_l2_logistic_cv", "interaction_l1_logistic_cv",
           "histgb_native_indicators", "sparse_mixture_fitted_base", "pilot_mixture_known_beta_reference")
REFERENCE = "pilot_mixture_known_beta_reference"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_inputs(path: Path) -> tuple[Support, QueryInputs]:
    """Deliberately never reads query outcomes, oracle, base or mechanism JSON."""
    with np.load(path, allow_pickle=False) as data:
        support = Support(Observations(data["uc"], data["xc"], data["mc"]), data["yc"])
        query = QueryInputs(Observations(data["uq"], data["xq"], data["mq"]))
        if np.intersect1d(data["support_row_ids"], data["query_row_ids"]).size:
            raise ValueError("Support/query IDs overlap")
    return support, query


def interaction_features(observations: Observations) -> np.ndarray:
    z = 2 * observations.mask.astype(float) - 1
    pairs = np.column_stack([z[:, i] * z[:, j] for i in range(z.shape[1])
                             for j in range(i + 1, z.shape[1])]) if z.shape[1] > 1 else np.empty((len(z), 0))
    u = observations.u[:, None]
    return np.column_stack([u, z, pairs, u * z, u * pairs])


def logistic(c: float, sparse: bool) -> LogisticRegression:
    kwargs = dict(C=c, solver="liblinear" if sparse else "lbfgs", max_iter=2000, tol=1e-6, random_state=42)
    if tuple(int(part) for part in sklearn.__version__.split(".")[:2]) >= (1, 8):
        kwargs["l1_ratio"] = 1. if sparse else 0.
    else:
        kwargs["penalty"] = "l1" if sparse else "l2"
    return LogisticRegression(**kwargs)


def empirical_nll(probability: np.ndarray, labels: np.ndarray) -> float:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(np.mean(-labels * np.log(probability) - (1 - labels) * np.log1p(-probability)))


def fit_cv_logistic(xc: np.ndarray, labels: np.ndarray, xq: np.ndarray, sparse: bool = False):
    counts = np.bincount(labels, minlength=2)
    folds = min(3, int(counts.min()))
    losses = {}
    if folds >= 2:
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=42).split(xc, labels))
        for c in C_GRID:
            held_probability = np.empty(len(labels))
            for train, held in splits:
                fitted = logistic(c, sparse).fit(xc[train], labels[train])
                held_probability[held] = fitted.predict_proba(xc[held])[:, 1]
            losses[str(c)] = empirical_nll(held_probability, labels)
        selected = min(C_GRID, key=lambda c: (losses[str(c)], c))
    else:
        selected = .3
    fitted = logistic(selected, sparse).fit(xc, labels)
    return fitted.predict_proba(xq)[:, 1], {"selected_c": selected, "support_cv_folds": folds,
                                          "support_cv_losses": losses, "params": {k: repr(v) for k, v in fitted.get_params().items()}}


def pilot_source() -> Path:
    candidates = [ROOT / "experiments/cpu_reproduction/source/headroom.py",
                  ROOT.parent / "reference/handoff/mira_pilot/headroom.py"]
    return next(path for path in candidates if path.is_file())


def load_pilot_mixture():
    path = pilot_source()
    spec = importlib.util.spec_from_file_location("mira_preserved_headroom_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sparse_likelihood_mixture


def predict_all(support: Support, query: QueryInputs) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    if not isinstance(support, Support) or not isinstance(query, QueryInputs):
        raise TypeError("Learners accept Support and QueryInputs only")
    context, queries = support.observations, query.observations
    if len(np.unique(support.labels)) < 2:
        probability = np.full(len(queries), (support.labels.sum() + 1) / (len(support.labels) + 2))
        return {name: (probability.copy(), {"cold_start": True}) for name in METHODS}
    predictions = {}
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        base, metadata = fit_cv_logistic(context.u[:, None], support.labels, queries.u[:, None])
        predictions["u_logistic_cv"] = (base, {**metadata, "seconds": time.perf_counter() - start})
        xc, xq = interaction_features(context), interaction_features(queries)
        for name, sparse in (("interaction_l2_logistic_cv", False), ("interaction_l1_logistic_cv", True)):
            start = time.perf_counter()
            probability, metadata = fit_cv_logistic(xc, support.labels, xq, sparse)
            predictions[name] = (probability, {**metadata, "seconds": time.perf_counter() - start})
        start = time.perf_counter()
        tree = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=10,
                                              l2_regularization=1., early_stopping=False, random_state=42)
        tree.fit(np.column_stack([context.u, context.values, context.mask]), support.labels)
        probability = tree.predict_proba(np.column_stack([queries.u, queries.values, queries.mask]))[:, 1]
        predictions["histgb_native_indicators"] = (probability, {"params": {k: repr(v) for k, v in tree.get_params().items()},
                                                               "seconds": time.perf_counter() - start})
        mixture = load_pilot_mixture()
        # The preserved pilot's centered-bit arithmetic expects signed masks.
        legal_context = {"u": context.u, "m": context.mask.astype(np.int64), "y": support.labels}
        legal_query_mask = queries.mask.astype(np.int64)
        for name, p0 in (("sparse_mixture_fitted_base", base), (REFERENCE, expit(.8 * queries.u))):
            start = time.perf_counter()
            probability = mixture(legal_context, {"u": queries.u, "m": legal_query_mask, "p0": p0})
            predictions[name] = (probability, {"basis": "all mask singles/pairs, optional U interactions, both signs",
                                               "gamma_prior": [0, .4, .8], "mask_reference_rate_assumption": .5,
                                               "base": "support-fitted U-only logistic" if name != REFERENCE else "known beta=.8",
                                               "seconds": time.perf_counter() - start})
    return predictions


def interval(values: list[float]) -> dict[str, Any]:
    array = np.array(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None
    half = float(t.ppf(.975, len(array) - 1)) * se if se is not None else None
    return {"n_tasks": len(array), "mean": mean, "se": se,
            "ci_low": mean - half if half is not None else None, "ci_high": mean + half if half is not None else None}


def run(source: Path, out: Path) -> None:
    source, out = source.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    source_manifest = json.loads((source / "manifest.json").read_text())
    if source_manifest["status"] != "complete":
        raise ValueError("Saved TFM run is not complete")
    configuration = {
        "analysis_status": "post-confirmation descriptive follow-up; no new blind confirmation",
        "source_manifest_sha256": sha(source / "manifest.json"), "source_run": str(source),
        "script_sha256": sha(Path(__file__)), "pilot_source_sha256": sha(pilot_source()),
        "operational_methods": [name for name in METHODS if name != REFERENCE], "parameter_informed_reference": REFERENCE,
        "c_grid": list(C_GRID), "support_cv": "3-fold stratified shuffled, seed42; full-support refit; ties favor smaller C",
        "features": "U, all centered mask singles/pairs, U interactions; no active/family/gamma inputs",
        "python": platform.python_version(), "python_executable": sys.executable,
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "mixture_limitations": "Pilot gamma prior {0,.4,.8} omits true .9; fitted-base variant is plug-in and generator-informed. Known-beta variant excluded from operational decisions.",
    }
    if (out / "manifest.json").exists():
        raise ValueError("Output manifest already exists; preserve it and select a new --out")
    write_json(out / "manifest.json", {"status": "running", **configuration})
    tfm_rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()]
    data_rows = {row["task_id"]: row for row in tfm_rows}
    rows, metadata, data_identities = [], {}, {}
    for task_id, tfm in sorted(data_rows.items()):
        path = source / tfm["data_file"]
        if sha(path) != tfm["data_sha256"]:
            raise ValueError(f"Saved source data hash failed: {path}")
        support, query = load_inputs(path)
        started = time.perf_counter()
        predictions = predict_all(support, query)
        # Materialize every learner's output before opening evaluator targets.
        paths = {}
        for method, (probability, details) in predictions.items():
            target = out / "predictions" / f"{task_id}_{method}.npz"
            with target.open("wb") as stream:
                np.savez_compressed(stream, probability=probability)
            paths[method] = target
            metadata[f"{task_id}_{method}"] = details
        with np.load(path, allow_pickle=False) as data:
            targets = EvaluationTargets(data["yq"], data["oracle"], data["base"])
            support_ids_hash = hashlib.sha256(data["support_row_ids"].tobytes()).hexdigest()
            query_ids_hash = hashlib.sha256(data["query_row_ids"].tobytes()).hexdigest()
        for saved in tfm_rows:
            if saved["task_id"] == task_id and saved["mode"] in ("native", "native_indicators"):
                saved_path = source / saved["prediction_file"]
                if sha(saved_path) != saved["prediction_sha256"]:
                    raise ValueError(f"Saved TFM prediction hash failed: {saved_path}")
                with np.load(saved_path, allow_pickle=False) as prediction:
                    checked = evaluate(prediction["probability"], targets)
                if not np.isclose(checked["expected_nll"], saved["expected_nll"], rtol=1e-10, atol=1e-10):
                    raise ValueError("Saved TFM loss disagrees with its probability artifact")
        for method, (probability, details) in predictions.items():
            rows.append({"task_id": task_id, "seed": tfm["seed"], "family": tfm["family"], "gamma": tfm["gamma"],
                         "method": method, "category": "parameter_informed_reference" if method == REFERENCE else "operational",
                         "support_rows": len(support.labels), "query_rows": len(query.observations),
                         **evaluate(probability, targets), "prediction_file": str(paths[method].relative_to(out)),
                         "prediction_sha256": sha(paths[method]), "data_sha256": tfm["data_sha256"],
                         "seconds": details.get("seconds", 0.)})
        data_identities[task_id] = {"source_file": str(path), "source_sha256": tfm["data_sha256"],
                                    "support_ids_sha256": support_ids_hash, "query_ids_sha256": query_ids_hash,
                                    "task_seconds": time.perf_counter() - started}
        write_json(out / "results.json", rows)
        write_json(out / "model_parameters.json", metadata)
        write_json(out / "data_identities.json", data_identities)
        print(f"Completed {len(data_identities)}/{len(data_rows)} saved tasks", flush=True)
    groups = {}
    for row in rows:
        groups.setdefault((row["gamma"], row["method"]), []).append(row)
    losses = [{"gamma": gamma, "method": method, "category": values[0]["category"],
               **interval([row["expected_nll"] for row in values]),
               "empirical_nll": float(np.mean([row["empirical_nll"] for row in values])),
               "oracle_nll": float(np.mean([row["oracle_nll"] for row in values])),
               "oracle_gap": float(np.mean([row["expected_nll"] - row["oracle_nll"] for row in values])),
               "seconds_total": sum(row["seconds"] for row in values)}
              for (gamma, method), values in sorted(groups.items())]
    pairs = []
    for row in rows:
        for tfm in tfm_rows:
            if tfm["task_id"] == row["task_id"] and tfm["mode"] in ("native", "native_indicators"):
                pairs.append({"task_id": row["task_id"], "seed": row["seed"], "gamma": row["gamma"],
                              "method": row["method"], "category": row["category"], "tfm_model": tfm["model"],
                              "tfm_mode": tfm["mode"], "gain_nats": tfm["expected_nll"] - row["expected_nll"]})
    effects = {}
    for pair in pairs:
        effects.setdefault((pair["gamma"], pair["method"], pair["tfm_model"], pair["tfm_mode"]), []).append(pair)
    effect_rows = [{"gamma": gamma, "method": method, "category": values[0]["category"], "tfm_model": model,
                    "tfm_mode": mode, **interval([row["gain_nats"] for row in values])}
                   for (gamma, method, model, mode), values in sorted(effects.items())]
    write_json(out / "summary.json", {"analysis_status": configuration["analysis_status"], "losses": losses,
                                       "paired_effects": effect_rows, "intervals": "descriptive paired Student-t, no multiplicity correction"})
    write_json(out / "paired_comparisons.json", pairs)
    for name, values in (("losses.csv", losses), ("paired_effects.csv", effect_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    lines = ["# Strong CPU baselines", "", configuration["analysis_status"] + ".", "",
             "Every predictor uses the same saved support labels/query observations as the TFMs. C is selected on support-only folds; selected models refit all support rows. Query targets are opened only after outputs are saved.", "",
             "| Gamma | Method | Tasks | Expected NLL | Oracle gap |",
             "|---:|---|---:|---:|---:|"]
    for row in losses:
        lines.append(f"| {row['gamma']:g} | {row['method']} | {row['n_tasks']} | {row['mean']:.6f} | {row['oracle_gap']:.6f} |")
    lines += ["", "The known-beta mixture is a parameter-informed reference outside the operational pool. The fitted-base mixture reuses support-fitted U-logistic probabilities; both retain the preserved pilot's generator-informed basis/rate and gamma prior, which omits .9.",
              "Paired effects against each saved native/indicator model are in paired_effects.csv; positive values favor the CPU method. Intervals are descriptive 95% paired Student-t intervals across 20 tasks within gamma, without multiplicity correction.",
              "These are stationary, clean support/query synthetic tasks. They do not establish a neural-adapter advantage, online-shift performance, or a real-data benchmark."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "manifest.json", {"status": "complete", **configuration, "tasks": len(data_rows),
                                        "completed_predictions": len(rows), "total_task_seconds": sum(value["task_seconds"] for value in data_identities.values())})


if __name__ == "__main__":
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source", type=Path, default=ROOT / "artifacts/runs/confirmation_pairwise_v1")
    command.add_argument("--out", type=Path, default=ROOT / "artifacts/reports/strong_baselines")
    arguments = command.parse_args()
    run(arguments.source, arguments.out)
