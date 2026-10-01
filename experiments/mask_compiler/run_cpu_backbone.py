"""CPU feasibility benchmark, or explicitly requested committed-v2 validation.

This is a development device/environment change, not the frozen GPU experiment.
The JSON protocol supplies episode/weight identities, but source freeze checks
must be renewed before any complete CPU validation study. No paid jobs,
automatic resume, or retries are implemented.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
EXPECTED = {"tabpfn": "9.0.0", "torch": "2.8.0", "numpy": "2.3.5", "scikit-learn": "1.8.0", "scipy": "1.17.0"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def environment():
    packages = {}
    for distribution, module in (("tabpfn", "tabpfn"), ("torch", "torch"), ("numpy", "numpy"),
                                 ("scikit-learn", "sklearn"), ("scipy", "scipy")):
        try:
            loaded = importlib.import_module(module)
            packages[distribution] = {"metadata": importlib.metadata.version(distribution),
                "actual": getattr(loaded, "__version__", None), "path": str(loaded.__file__)}
        except Exception as error:
            packages[distribution] = {"error": repr(error)}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages}


def verify_environment(record):
    for package, version in EXPECTED.items():
        info = record["packages"][package]
        # Official CPU wheels add +cpu; retain that distinction in provenance.
        actual = str(info.get("actual", "")).split("+")[0]
        metadata = str(info.get("metadata", "")).split("+")[0]
        if actual != version or metadata != version:
            raise ValueError(f"Exact package version mismatch: {package}: {info}")


def configure_cache(cache):
    for key, value in {"TABPFN_MODEL_CACHE_DIR": str(cache / "tabpfn"), "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TABPFN_DISABLE_TELEMETRY": "1"}.items():
        os.environ[key] = value


def full_validation(args):
    """Caller must explicitly authorize this option after the v2 Git freeze."""
    if args.threads != 2 or not 1 <= args.max_seconds <= 2400:
        raise ValueError("Full CPU validation requires two threads and at most 2400 seconds")
    cache, out = Path(args.cache_root).resolve(), Path(args.out).resolve()
    configure_cache(cache)
    from run_backbone import load_protocol, verify_checkpoints, run
    from mira.models import create_model, ModelSettings
    from mira.artifacts import atomic_json
    from threadpoolctl import threadpool_limits
    import torch
    plan, protocol_sha = load_protocol(args.protocol_file)
    if "v2" not in plan["protocol_version"].lower():
        raise ValueError("Full CPU validation requires a frozen successor v2 protocol")
    own_relative = str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/")
    if plan["file_sha256"].get(own_relative) != sha(__file__):
        raise ValueError("CPU runner is not included in the frozen source map")
    relative_plan = str(Path(args.protocol_file).resolve().relative_to(ROOT)).replace("\\", "/")
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative_plan], cwd=ROOT)
    working = Path(args.protocol_file).read_bytes()
    # Git may normalize CRLF on Windows; every other committed byte must agree.
    if committed.replace(b"\r\n", b"\n") != working.replace(b"\r\n", b"\n"):
        raise ValueError("The frozen successor protocol must be committed before full validation")
    override = {"model": "tabpfn:v2", "device": "cpu", "threads": 2, "max_seconds": args.max_seconds}
    if plan.get("cpu_execution") is not None and plan["cpu_execution"] != override:
        raise ValueError("CPU execution differs from explicit frozen settings")
    imported = environment()
    verify_environment(imported)
    verify_checkpoints(cache, plan["checkpoint_sha256"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fresh output required; no resume or retry")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    bootstrap = {"status": "running", "execution": override, "protocol_sha256": protocol_sha,
        "scope": "development successor: Windows/Python3.11 CPU versus original Linux/Python3.12 GPU",
        "environment_actual": imported, "runner_sha256": sha(__file__), "checkpoint_sha256": plan["checkpoint_sha256"],
        "paid_jobs_dispatched": 0, "automatic_retries": 0, "resume": False,
        "checkpoint_transfer": {"method": "Modal volume get; no inference/app dispatch", "bytes": 139377577,
            "cli_exit": "Windows CP1252 checkmark error after completed transfer; file SHAs independently match",
            "service_bandwidth_charges": "Not independently measured; no additional paid compute reservation"},
        "numerical_equivalence_to_GPU": "not established", "bound": "2400-second maximum configured between-cell check; no automatic restart"}

    def cpu_factory(name, supplied_settings, seed):
        if name != "tabpfn:v2" or supplied_settings.ensembles != 4:
            raise ValueError("Exact TabPFN v2/four ensembles only")
        return create_model(name, ModelSettings("cpu", 4), seed)

    try:
        with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = run(plan, protocol_sha, "tabpfn:v2", out, root=ROOT, cache_root=cache,
                factory=cpu_factory, capture_lock=True, check_cache=True, max_seconds=args.max_seconds)
        bootstrap.update(status=result["status"], completed_cells=result.get("completed_cells"),
            warnings=[{"category": w.category.__name__, "message": str(w.message)} for w in captured])
    except BaseException as error:
        bootstrap.update(status="failed", error=repr(error))
        raise
    finally:
        if out.exists():
            atomic_json(out / "cpu_execution.json", bootstrap)
    if result["status"] != "complete":
        raise SystemExit("Incomplete CPU validation; preserve every partial artifact and failure")
    return result


def benchmark(args):
    cache, out = Path(args.cache_root).resolve(), Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preserve previous CPU benchmark output")
    out.mkdir(parents=True, exist_ok=True)
    configure_cache(cache)
    record = {"status": "running", "scope": "one synthetic native-view development CPU episode; no complete validation",
              "device": "cpu", "paid_jobs_dispatched": 0, "time_utc": datetime.now(timezone.utc).isoformat(),
              "python_platform_change": "Windows Python3.11 CPU; original GPU image Linux Python3.12",
              "protocol_file": str(Path(args.protocol_file).resolve()), "protocol_sha256": sha(args.protocol_file),
              "runner_sha256": sha(__file__), "threads": args.threads, "ensembles": 4}
    try:
        record["environment"] = environment()
        verify_environment(record["environment"])
        (out / "requirements-lock.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        import numpy as np
        import torch
        from mira.models import ModelSettings, create_model
        from threadpoolctl import threadpool_limits
        from mira.artifacts import atomic_npz
        from mira.metrics import expected_nll
        from run_backbone import verify_checkpoints, synthetic_episodes, prepare_inputs
        plan = json.loads(Path(args.protocol_file).read_text())
        record["source_sha256"] = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in [Path(__file__), ROOT / "experiments/mask_compiler/run_backbone.py",
                         ROOT / "experiments/mask_compiler/matched_training.py", ROOT / "src/mira/models.py"]}
        if plan.get("ensembles") != 4 or "tabpfn:v2" not in plan.get("models", []):
            raise ValueError("Require exact TabPFN v2/four ensembles")
        expected = plan["checkpoint_sha256"]
        verify_checkpoints(cache, expected)
        record["checkpoint_sha256"] = expected
        record["checkpoint_bytes"] = {name: (cache / name.removeprefix("/cache/")).stat().st_size for name in expected}
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        episodes = (e for e in synthetic_episodes(plan) if e["episode_id"] == args.episode_id)
        episode = next(episodes, None)
        if episode is None:
            raise ValueError("Episode must be in the supplied development matrix")
        inputs = prepare_inputs(episode["support"], episode["query"], episode["selected_columns"],
            np.eye(4, dtype=np.uint8), "native", episode["seed"], episode["native_context"], episode["native_query"])
        # No query targets, active/gamma metadata, or oracle enters these inputs.
        assert not hasattr(inputs, "query_labels") and not hasattr(inputs, "oracle")
        record.update(episode_id=episode["episode_id"], support_shape=list(inputs.context.shape), query_shape=list(inputs.query.shape))
        atomic_npz(out / "inputs.npz", context=inputs.context, support_labels=inputs.labels, query=inputs.query,
                   support_ids=episode["support_ids"], query_ids=episode["query_ids"])
        start = time.monotonic()
        with threadpool_limits(limits=args.threads), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = create_model("tabpfn:v2", ModelSettings("cpu", 4), episode["seed"])
            created = time.monotonic()
            model.fit(inputs.context, inputs.labels)
            fitted = time.monotonic()
            probabilities = np.asarray(model.predict_proba(inputs.query), dtype=float)
            finished = time.monotonic()
        classes = np.asarray(model.classes_)
        if probabilities.shape != (len(inputs.query), 2) or set(classes) != {0, 1} or not np.isfinite(probabilities).all():
            raise ValueError("Invalid TabPFN CPU probabilities/classes")
        if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Unnormalized probabilities")
        p = probabilities[:, np.flatnonzero(classes == 1)[0]]
        atomic_npz(out / "predictions.npz", probability=p, query_ids=episode["query_ids"])
        # Evaluator targets are opened only after fixed predictions are persisted.
        record.update(status="complete", create_seconds=created-start, fit_seconds=fitted-created,
            predict_seconds=finished-fitted, total_inference_seconds=finished-start,
            empirical_nll=expected_nll(p, episode["targets"]), expected_nll=expected_nll(p, episode["oracle"]),
            warnings=[{"category": w.category.__name__, "message": str(w.message)} for w in captured],
            inputs_sha256=sha(out / "inputs.npz"), predictions_sha256=sha(out / "predictions.npz"),
            numerical_equivalence_to_GPU="not established; this timing is one small synthetic view only")
        verify_checkpoints(cache, expected)
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        (out / "benchmark.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
    return record


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Print actual imported versions and metadata only")
    parser.add_argument("--full-validation", action="store_true", help="Explicitly authorized committed-v2 full validation; never selected by default")
    parser.add_argument("--protocol-file", type=Path, default=ROOT / "configs/trained_compiler_validation_v1.json")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts/checkpoints/compiler_cpu")
    parser.add_argument("--episode-id", default="synthetic_k2_g0.9_s74000")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=2400, help="Full-validation between-cell bound, maximum 2400")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/reports/compiler_cpu_feasibility/benchmark1")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(environment(), indent=2))
    elif args.full_validation:
        full_validation(args)
    elif not 1 <= args.threads <= 16:
        parser.error("Require bounded CPU threads")
    else:
        benchmark(args)


if __name__ == "__main__":
    cli()
