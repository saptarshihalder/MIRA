"""Budget-conscious, resumable paired representation study.

Only support labels and observable query inputs cross the predictor boundary.
Confirmation seeds are reserved; this module does not run on import.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import (append_result, atomic_json, atomic_npz, capture_environment,
                        checkpoint_hashes, file_sha256, load_rows, save_episode, source_hashes)
from .artifacts import verify_episode
from .data import FAMILIES, VALUE_DISTRIBUTIONS, canonical_family, generate_episode
from .metrics import evaluate
from .models import BinaryEstimator, ModelSettings, cold_start_probability, create_model, predict_binary
from .representations import MODES, transform_pair

CONFIRMATION_SEEDS = tuple(range(60000, 60020))
DEVELOPMENT_SEED_RANGE = (40000, 50000)
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def _task_id(family: str, seed: int, gamma: float) -> str:
    return f"{family}_seed{seed}_g{gamma:.12g}"


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    protocol = "development" if args.protocol in ("dev", "development") else "confirmation"
    seed_start = args.seed_start if args.seed_start is not None else (40000 if protocol == "development" else 60000)
    seeds = args.seeds if args.seeds is not None else (3 if protocol == "development" else 20)
    selected = list(range(seed_start, seed_start + seeds))
    if seeds < 1 or args.context < 0 or args.queries < 1 or args.ensembles < 1:
        raise ValueError("Require positive seeds/queries/ensembles and nonnegative context")
    allowed = range(*DEVELOPMENT_SEED_RANGE) if protocol == "development" else CONFIRMATION_SEEDS
    if any(seed not in allowed for seed in selected):
        raise ValueError(f"{protocol} seeds outside reserved range: "
                         + ("40000–49999" if protocol == "development" else "60000–60019"))
    if any(not 0 <= gamma < 1 for gamma in args.gammas):
        raise ValueError("Require 0 <= gamma < 1")
    if args.width_control_columns is not None and args.width_control_columns < 1:
        raise ValueError("Require positive width-control-columns")
    if not 0 < args.missing_rate < 1:
        raise ValueError("Require 0 < missing-rate < 1")
    if not 0 <= args.collision_probability <= 1:
        raise ValueError("Require 0 <= collision-probability <= 1")
    if not 0 < args.quantization_step < float("inf"):
        raise ValueError("Require finite positive quantization-step")
    families = [canonical_family(family) for family in args.families]
    for key, values in (("families", families), ("gammas", args.gammas),
                        ("models", args.models), ("modes", args.modes)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    if any(mode not in MODES for mode in args.modes):
        raise ValueError("Unknown representation mode")
    if any(not (model.startswith("tabpfn:") or model in ("tabicl:v2", "xgboost")) for model in args.models):
        raise ValueError("Unsupported requested model")
    return {
        "protocol": protocol, "families": families, "models": args.models, "modes": args.modes,
        "gammas": args.gammas, "seeds": selected, "context": args.context, "queries": args.queries,
        "ensembles": args.ensembles, "device": args.device, "beta": args.beta,
        "value_distribution": args.value_distribution,
        "missing_rate": args.missing_rate, "collision_probability": args.collision_probability,
        "quantization_step": args.quantization_step,
        "missing_rate_semantics": "baseline Bernoulli rate; outcome/value modulation can change marginal rates",
        "width_control_columns": args.width_control_columns,
        "checkpoint_dirs": [str(Path(directory).expanduser().resolve()) for directory in args.checkpoint_dirs],
        "generator": "exact centered-bit mask likelihood over Bernoulli(r); label_only has 8 nuisance columns",
        "oracle_information": "evaluator only; query labels/oracle/active mechanism excluded from predictor",
        "cold_start": "Laplace (n1+1)/(n+2) when context has fewer than two classes; explicitly marked",
        "interval_unit": "independent synthetic task (seed within family/gamma), not query row",
    }


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _verify_saved(out: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for path_key, hash_key in (("prediction_file", "prediction_sha256"), ("data_file", "data_sha256")):
            path = out / row[path_key]
            if not path.is_file() or file_sha256(path) != row[hash_key]:
                raise ValueError(f"Saved artifact is missing or changed: {path}")


def run(args: argparse.Namespace,
        factory: Callable[[str, ModelSettings, int], BinaryEstimator] = create_model,
        capture_lock: bool = True) -> dict[str, Any]:
    """Run requested cells; compatible reruns reuse hashed successful predictions."""
    configuration = _configuration(args)
    sources = source_hashes()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["configuration"] != configuration or manifest["source_sha256"] != sources:
            raise ValueError("Output directory has a different configuration/source; use a new --out")
    elif any(out.iterdir()):
        raise ValueError("Nonempty output directory has no manifest; use a new --out")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION, "created_utc": _now(), "configuration": configuration,
            "configuration_sha256": _fingerprint(configuration), "source_sha256": sources,
            "reserved_confirmation_seeds": list(CONFIRMATION_SEEDS),
            "expected_cells": len(configuration["seeds"]) * len(configuration["families"])
                              * len(configuration["gammas"]) * len(configuration["models"])
                              * len(configuration["modes"]),
        }
    manifest.update(status="running", last_started_utc=_now())
    atomic_json(manifest_path, manifest)
    rows = load_rows(out)
    _verify_saved(out, rows)
    environment = capture_environment(out, capture_lock)
    environment_hash = _fingerprint(environment)
    if "environment_sha256" in manifest and manifest["environment_sha256"] != environment_hash:
        raise ValueError("Environment differs from original run; use a new --out")
    manifest["environment_sha256"] = environment_hash
    atomic_json(manifest_path, manifest)
    (out / "data").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    done = {row["cell_id"] for row in rows}
    error_path = out / "errors.json"
    errors = json.loads(error_path.read_text(encoding="utf-8")) if error_path.exists() else []
    parameter_path = out / "model_parameters.json"
    parameters = json.loads(parameter_path.read_text(encoding="utf-8")) if parameter_path.exists() else {}
    blocked: set[str] = set()
    hashes = checkpoint_hashes(configuration["checkpoint_dirs"])
    atomic_json(out / "checkpoint_sha256.json", hashes)
    hashed_models: set[str] = set()
    settings = ModelSettings(args.device, args.ensembles)
    interrupted = False
    try:
        for seed in configuration["seeds"]:
            for family in configuration["families"]:
                for gamma in configuration["gammas"]:
                    task_id = _task_id(family, seed, gamma)
                    episode = generate_episode(family, seed, gamma, args.context, args.queries, args.beta,
                                               args.value_distribution, args.missing_rate,
                                               args.collision_probability, args.quantization_step)
                    data_path = out / "data" / f"{task_id}.npz"
                    if not data_path.exists():
                        save_episode(data_path, episode)
                    else:
                        verify_episode(data_path, episode)
                    data_hash = file_sha256(data_path)
                    atomic_json(out / "data" / f"{task_id}.json", asdict(episode.mechanism))
                    for name in configuration["models"]:
                        if name in blocked:
                            continue
                        for mode in configuration["modes"]:
                            cell_id = f"{task_id}_{_safe(name)}_{mode}"
                            if cell_id in done:
                                continue
                            started = time.perf_counter()
                            model = None
                            stage = "transform"
                            try:
                                inputs = transform_pair(episode.support, episode.query, mode, seed,
                                                        args.width_control_columns)
                                probability = cold_start_probability(inputs.labels, len(inputs.query))
                                cold_start = probability is not None
                                if not cold_start:
                                    stage = "model_constructor"
                                    model = factory(name, settings, seed)
                                    parameters[cell_id] = {key: repr(value) for key, value in
                                                           model.get_params(deep=False).items()}
                                    atomic_json(parameter_path, parameters)
                                    stage = "fit_predict"
                                    probability = predict_binary(model, inputs)
                                stage = "evaluate"
                                metrics = evaluate(probability, episode.targets)
                                prediction_path = out / "predictions" / f"{cell_id}.npz"
                                atomic_npz(prediction_path, probability=probability)
                                row = {
                                    "cell_id": cell_id, "task_id": task_id, "protocol": configuration["protocol"],
                                    "seed": seed, "family": family, "gamma": gamma, "model": name,
                                    "mode": mode, "context": args.context, "queries": args.queries,
                                    "input_columns": inputs.context.shape[1], "cold_start": cold_start,
                                    **metrics, "seconds": time.perf_counter() - started,
                                    "data_file": str(data_path.relative_to(out)).replace("\\", "/"),
                                    "data_sha256": data_hash,
                                    "prediction_file": str(prediction_path.relative_to(out)).replace("\\", "/"),
                                    "prediction_sha256": file_sha256(prediction_path),
                                }
                                append_result(out, rows, row)
                                done.add(cell_id)
                                # Checkpoints may materialize lazily during the first prediction.
                                if name not in hashed_models and not cold_start:
                                    hashes = checkpoint_hashes(configuration["checkpoint_dirs"])
                                    atomic_json(out / "checkpoint_sha256.json", hashes)
                                    hashed_models.add(name)
                                manifest.update(completed_cells=len(rows), last_completed_cell=cell_id)
                                atomic_json(manifest_path, manifest)
                                print(json.dumps({key: row[key] for key in
                                                  ("cell_id", "expected_nll", "cold_start", "seconds")}), flush=True)
                            except Exception as error:
                                errors.append({"cell_id": cell_id, "seed": seed, "family": family,
                                               "gamma": gamma, "model": name, "mode": mode,
                                               "stage": stage, "error": repr(error), "utc": _now()})
                                atomic_json(error_path, errors)
                                blocked.add(name)
                                print(f"FAILED {cell_id} at {stage}: {error!r}; model blocked for this invocation",
                                      file=sys.stderr, flush=True)
                            finally:
                                del model
                                gc.collect()
                                # Do not import torch merely to release an absent model.
                                torch = sys.modules.get("torch")
                                if torch is not None and torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            if name in blocked:
                                break
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise
    finally:
        hashes = checkpoint_hashes(configuration["checkpoint_dirs"])
        atomic_json(out / "checkpoint_sha256.json", hashes)
        atomic_json(error_path, errors)
        atomic_json(parameter_path, parameters)
        complete = len(rows) == manifest["expected_cells"]
        current_error_ids = {error["cell_id"] for error in errors} - done
        manifest.update(
            status="interrupted" if interrupted else ("complete" if complete else "incomplete"),
            completed_cells=len(rows), unresolved_failed_cells=sorted(current_error_ids),
            historical_errors=len(errors), blocked_models=sorted(blocked), ended_utc=_now(),
            checkpoint_identity_status="hashes recorded" if hashes else "unverified: no checkpoint files supplied/found",
            checkpoint_sha256=hashes, dependency_lock_status=environment["dependency_lock_status"],
            cold_start_cells=sum(bool(row["cold_start"]) for row in rows),
        )
        atomic_json(manifest_path, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--models", nargs="+", default=["tabpfn:v2", "tabicl:v2", "xgboost"])
    command.add_argument("--modes", nargs="+", choices=MODES,
                         default=["native", "native_indicators", "native_shuffled"])
    command.add_argument("--families", nargs="+", choices=[*FAMILIES, "joint_pair"], default=["label_only"])
    command.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
    command.add_argument("--protocol", choices=["dev", "development", "confirmation"], default="development")
    command.add_argument("--seed-start", type=int)
    command.add_argument("--seeds", type=int)
    command.add_argument("--context", type=int, default=256)
    command.add_argument("--queries", type=int, default=1024)
    command.add_argument("--ensembles", type=int, default=4)
    command.add_argument("--device", default="cuda")
    command.add_argument("--beta", type=float, default=0.8)
    command.add_argument("--value-distribution", choices=VALUE_DISTRIBUTIONS, default="gaussian")
    command.add_argument("--missing-rate", type=float, default=0.5,
                         help="Baseline independent Bernoulli mask rate, e.g. .1 or .5")
    command.add_argument("--collision-probability", type=float, default=0.5)
    command.add_argument("--quantization-step", type=float, default=1.0)
    command.add_argument("--width-control-columns", type=int)
    command.add_argument("--checkpoint-dirs", nargs="*", default=[])
    command.add_argument("--out", default="experiments/mechanism_development")
    return command


def cli() -> None:
    args = parser().parse_args()
    manifest = run(args)
    if manifest["status"] != "complete":
        raise SystemExit("Incomplete experiment; inspect manifest.json and errors.json. No model was substituted.")


if __name__ == "__main__":
    cli()
