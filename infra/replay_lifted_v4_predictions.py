"""Bounded local CPU reproduction of fixed v4 score cells, after full verification.

python infra/replay_lifted_v4_predictions.py --root COMPLETE_RESULTS --panels FROZEN_PANELS

Each panel uses exactly tasks {0, n//2, n-1}. All 24 learned checkpoints,
synthetic k=0..3 and real e=0,3,6 are required. Fixed atol=2e-4, rtol=2e-5;
these tolerances must not be tuned after viewing results. The independent
verifier guards pretrained TabPFN bytes; TabPFN inference is outside this replay.
Only a new JSON audit report is written, outside the immutable input folders.
The parent subprocess watchdog is fixed at 15 minutes with no automatic retries.
"""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "e1b1d9d09e98667dc4590e6f54a6c6641c3e24a9"
DEFAULT_SOURCE = REPO / "artifacts/inputs/lifted_v4_source"
DEFAULT_IDENTITY = REPO / "artifacts/manifests/bound_source_identity_l40s.json"
DEFAULT_OUT = REPO / "artifacts/reports/lifted_v4_cpu_replay.json"
ATOL, RTOL = 2e-4, 2e-5
WALL_SECONDS = 900
CPU_THREADS = 2
METRICS = ("nll", "se", "cov", "crps")
VERIFIER_FILE = Path(__file__).with_name("verify_lifted_v4_results.py")
REPLAY_SCHEMA = 1
BUNDLE_HASH_FILES = (
    "run_identity.json", "runs/confirm_v4.json", "pretrained_weights.json",
    "modal_main_extended_runtime.json", "main_extended_execution_amendment.json",
    "main_extended_execution_gate.json", "modal_identity.json", "modal_smoke_gate.json",
    "modal_smoke_runtime.json", "source_profile.json",
)


class ReplayError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def selected_indices(n):
    require(type(n) is int and n >= 3, "Replay requires at least three tasks")
    return (0, n // 2, n - 1)


def slice_first_dimension(value, n, indices):
    """Slice task-bearing tensors/arrays/lists, including nested anchors."""
    if isinstance(value, dict):
        return {key: slice_first_dimension(item, n, indices) for key, item in value.items()}
    shape = getattr(value, "shape", ())
    if len(shape) and shape[0] == n:
        return value[list(indices)]
    if isinstance(value, (list, tuple)) and len(value) == n:
        sliced = [value[index] for index in indices]
        return tuple(sliced) if isinstance(value, tuple) else sliced
    return value


def slice_panel(panel):
    n = panel["pool"]["n"]
    indices = selected_indices(n)
    pool = slice_first_dimension(panel["pool"], n, indices)
    pool["n"] = len(indices)
    sliced = dict(pool=pool)
    if "banks" in panel:
        # Banks index masks, never tasks, even if a bank happens to have n rows.
        sliced["banks"] = panel["banks"]
    if "conds" in panel:
        sliced["conds"] = slice_first_dimension(panel["conds"], n, indices)
    return sliced, indices


def import_verifier():
    spec = importlib.util.spec_from_file_location("lifted_v4_independent_verifier", VERIFIER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source(source, expected, digest):
    require(expected.get("source_commit") == SOURCE_COMMIT, "Unknown scientific source commit")
    require((source / "MIRA_SOURCE_COMMIT.txt").read_text().strip() == SOURCE_COMMIT,
            "Frozen source archive commit differs")
    hashes = expected["source"]
    science = source / "experiments/lifted_cavity"
    expected_python = {name for name in hashes if name.startswith("experiments/lifted_cavity/") and name.endswith(".py")}
    actual_python = {path.relative_to(source).as_posix() for path in science.glob("*.py")}
    require(actual_python == expected_python, "Frozen scientific module inventory differs")
    for name, sha in hashes.items():
        path = source / name
        require(path.is_file() and digest(path) == sha, f"Frozen scientific source hash differs: {name}")
    for name in ("train.py", "evaluate2.py", "realdata.py", "pfn.py", "lct.py"):
        require(f"experiments/lifted_cavity/{name}" in expected_python, f"Required frozen source absent: {name}")
    return science


def import_scientific_source(science, expected, digest):
    module_names = {Path(name).stem for name in expected["source"]
                    if name.startswith("experiments/lifted_cavity/") and name.endswith(".py")}
    for name in module_names:
        if name in sys.modules:
            origin = getattr(sys.modules[name], "__file__", None)
            require(origin is not None and Path(origin).resolve() == (science / f"{name}.py").resolve(),
                    f"Scientific module already imported from another source: {name}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(science))
    modules = [importlib.import_module(name) for name in ("train", "evaluate2", "realdata")]
    # Check origins and bytes again after imports; never silently reuse other code.
    for name in module_names & set(sys.modules):
        origin = Path(sys.modules[name].__file__).resolve()
        require(origin == (science / f"{name}.py").resolve(), f"Unexpected scientific module origin: {name}")
        require(digest(origin) == expected["source"][f"experiments/lifted_cavity/{name}.py"],
                f"Scientific module changed during import: {name}")
    return modules


def compare_array(replayed, saved):
    replayed, saved = np.asarray(replayed), np.asarray(saved)
    require(replayed.shape == saved.shape and replayed.size > 0, "Replay/saved score shape differs")
    require(replayed.dtype.kind in "fiu" and saved.dtype.kind in "fiu"
            and np.isfinite(replayed).all() and np.isfinite(saved).all(), "Replay/saved scores are nonfinite/nonreal")
    errors = np.abs(replayed.astype(np.float64) - saved.astype(np.float64))
    allowed = ATOL + RTOL * np.abs(saved)
    mismatches = int(np.count_nonzero(errors > allowed))
    return dict(values=int(saved.size), mismatches=mismatches, matched=mismatches == 0,
                max_absolute_error=float(errors.max()), max_tolerance_ratio=float((errors / allowed).max()))


def capture_bindings(root, panels, identity_path, verifier):
    """Hash bytes without parsing outcomes; freeze before verification/replay."""
    checkpoints, scores = verifier.layout(root)
    require(all((root / name).is_file() for name in BUNDLE_HASH_FILES), "Replay provenance sidecars incomplete")
    checkpoint_hashes = {run.relative_to(root).as_posix():
                         {name: verifier.digest(run / name) for name in ("model.pt", "train.json")}
                         for run, *_ in checkpoints}
    panel_hashes = {f"{tag}.pt": verifier.digest(panels / f"{tag}.pt") for tag in verifier.TAGS}
    metadata = verifier.read_json(root / "pretrained_weights.json")
    path = root / "cache" / verifier.PRETRAINED_WEIGHT_PATH
    pretrained = {key: metadata[key] for key in ("package", "version", "n_estimators", "random_state")}
    pretrained["files"] = {verifier.PRETRAINED_WEIGHT_PATH:
                           dict(sha256=verifier.digest(path), bytes=path.stat().st_size)}
    return dict(checkpoint_hashes=checkpoint_hashes, panel_hashes=panel_hashes, pretrained_weights=pretrained,
                score_hashes={path.relative_to(root).as_posix(): verifier.digest(path) for path, *_ in scores},
                bundle_hashes={name: verifier.digest(root / name) for name in BUNDLE_HASH_FILES +
                               (("modal_main_resume_runtime.json", "main_resume_execution_amendment.json")
                                if (root / "modal_main_resume_runtime.json").exists() else ())},
                identity_sha256=verifier.digest(identity_path),
                panel_manifest_sha256=verifier.digest(panels / "SHA256SUMS"))


def require_verified_bindings(bindings, verified):
    for key in ("checkpoint_hashes", "panel_hashes", "pretrained_weights"):
        value = verified.get(key)
        if key == "checkpoint_hashes" and isinstance(value, dict):
            # The verifier's host-native run paths become portable schema paths.
            value = {name.replace("\\", "/"): hashes for name, hashes in value.items()}
        require(bindings[key] == value, f"Frozen replay/verifier {key} differ")


def require_unchanged_bindings(before, after):
    require(before == after, "Replay input bytes changed during verification/replay; no bound success")


def run_replay(root, panels, source):
    started = time.monotonic()
    verifier = import_verifier()
    # Presence preflight and raw hashes precede any outcome parsing. Full
    # independent verification precedes all scientific imports/model replay.
    verifier.preflight(root, panels, DEFAULT_IDENTITY)
    bindings = capture_bindings(root, panels, DEFAULT_IDENTITY, verifier)
    # Its endpoint values are deliberately excluded from this reproduction report.
    verified = verifier.verify(root, panels)
    require(verified.get("verified") is True, "Independent full verification failed")
    require_verified_bindings(bindings, verified)
    expected = verifier.read_json(DEFAULT_IDENTITY)
    science = check_source(source, expected, verifier.digest)
    train, evaluate2, realdata = import_scientific_source(science, expected, verifier.digest)
    import torch
    torch.set_num_threads(CPU_THREADS)
    panel_data = {tag: torch.load(panels / f"{tag}.pt", weights_only=False, map_location="cpu") for tag in verifier.TAGS}
    subsets = {tag: slice_panel(panel) for tag, panel in panel_data.items()}
    checkpoints, _ = verifier.layout(root)
    coverage, rows = [], []
    for run, model_name, seed, ds in checkpoints:
        meta = verifier.read_json(run / "train.json")
        model = train.build(meta["model"], meta.get("repo"), meta).to("cpu")
        model.load_state_dict(torch.load(run / "model.pt", weights_only=True, map_location="cpu"), strict=True)
        model.eval()
        tags = (verifier.H1, verifier.H3) if ds is None else (f"real_{ds}_s4041",)
        for tag in tags:
            subset, indices = subsets[tag]
            with torch.no_grad():
                if ds is None:
                    replayed = evaluate2.score(model, subset["pool"], subset["banks"])
                    prefix, conditions = "k", (0, 1, 2, 3)
                else:
                    replayed = {e: realdata.score_model(model, subset["pool"], subset["conds"][e]) for e in (0, 3, 6)}
                    prefix, conditions = "e", (0, 3, 6)
            require(set(replayed) == set(conditions), f"Replay conditions differ: {tag}")
            saved_path = run / f"cells_{tag}.npz"
            with np.load(saved_path, allow_pickle=False) as saved:
                for condition in conditions:
                    require(set(replayed[condition]) == set(METRICS), f"Replay metrics differ: {tag}/{condition}")
                    for metric in METRICS:
                        key = f"{prefix}{condition}_{metric}"
                        comparison = compare_array(replayed[condition][metric], saved[key][list(indices)])
                        rows.append(dict(checkpoint=run.relative_to(root).as_posix(), panel=tag,
                                         array=key, **comparison))
            coverage.append(dict(checkpoint=run.relative_to(root).as_posix(), model=model_name,
                                 seed=seed, dataset=ds, panel=tag, indices=list(indices),
                                 conditions=list(conditions), metrics=list(METRICS)))
        del model
    require(len(checkpoints) == 24 and len(coverage) == 30 and len(rows) == 408,
            "Required replay coverage incomplete")
    check_source(source, expected, verifier.digest)
    require_unchanged_bindings(bindings, capture_bindings(root, panels, DEFAULT_IDENTITY, verifier))
    mismatches = sum(row["mismatches"] for row in rows)
    return dict(replay_schema=REPLAY_SCHEMA, bindings=bindings,
                status="matched" if mismatches == 0 else "mismatched", coverage_complete=True,
                independent_full_verification=True, device="cpu", torch_threads=CPU_THREADS,
                torch=torch.__version__, source_commit=SOURCE_COMMIT,
                tolerances=dict(atol=ATOL, rtol=RTOL), selection="0,n//2,n-1", checkpoints=24,
                panel_checkpoint_pairs=30, arrays_checked=len(rows), values_checked=sum(row["values"] for row in rows),
                mismatched_values=mismatches, mismatched_arrays=sum(not row["matched"] for row in rows),
                coverage=coverage, comparisons=rows, seconds=time.monotonic() - started,
                scope="Selected learned-model score-cell reproduction; no new efficacy estimate; TabPFN inference excluded",
                automatic_retries=0, wall_seconds=WALL_SECONDS)


def validate_output(output, root, panels, source):
    output = output.resolve()
    require(not any(output.is_relative_to(path.resolve()) for path in (root, panels, source)),
            "Replay report must be outside immutable input folders")
    require(not output.exists(), "Replay report already exists; preserve it")
    return output


def bounded_replay(root, panels, source):
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker", "--root", str(root),
               "--panels", str(panels), "--source", str(source)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(CPU_THREADS),
                       MKL_NUM_THREADS=str(CPU_THREADS), PYTHONDONTWRITEBYTECODE="1",
                       MIRA_V4_REPLAY_WATCHDOG_CHILD="1")
    flags = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == "nt" else {}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=environment, **flags)
    try:
        stdout, stderr = process.communicate(timeout=WALL_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return dict(status="timed_out", coverage_complete=False, wall_seconds=WALL_SECONDS,
                    automatic_retries=0, reason="Fixed 15-minute watchdog expired; no replay pass established")
    try:
        report = json.loads(stdout)
    except ValueError:
        return dict(status="failed", coverage_complete=False, automatic_retries=0,
                    reason="Replay subprocess did not produce a valid JSON report", stderr=stderr[-4000:])
    require(process.returncode == 0 or report.get("status") in ("failed", "mismatched"),
            "Replay subprocess exit/report disagree")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args._worker:
            require(os.environ.get("MIRA_V4_REPLAY_WATCHDOG_CHILD") == "1",
                    "Replay worker must be started by the fixed watchdog")
            report = run_replay(args.root.resolve(), args.panels.resolve(), args.source.resolve())
        else:
            output = validate_output(args.out, args.root, args.panels, args.source)
            report = bounded_replay(args.root.resolve(), args.panels.resolve(), args.source.resolve())
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
    except Exception as error:
        report = dict(status="failed", coverage_complete=False, automatic_retries=0,
                      reason=f"{type(error).__name__}: {error}")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("status") == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
