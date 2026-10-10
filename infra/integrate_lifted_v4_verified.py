"""Stage complete, independently verified/replayed v4 evidence; merge explicitly.

No training, cloud access, checkpoint copying, manuscript edits or overwrites.
Hashes bind a local audit to its inputs; they are not independent attestation.
"""
import argparse
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "artifacts/reports/lifted_cavity_paper"
DEFAULT_STAGE = REPO / "artifacts/runs/lifted_v4_integration_stage"


def load_replay_module():
    spec = importlib.util.spec_from_file_location("v4_replay_integration", Path(__file__).with_name("replay_lifted_v4_predictions.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_replay(report, bindings, replay):
    require(report.get("replay_schema") == 1 and report.get("status") == "matched",
            "Successful fixed replay required")
    require(report.get("coverage_complete") is True and report.get("independent_full_verification") is True,
            "Replay coverage/verification incomplete")
    require(report.get("checkpoints") == 24 and report.get("panel_checkpoint_pairs") == 30
            and report.get("arrays_checked") == 408, "Replay counts differ")
    require(report.get("mismatched_values") == 0 and report.get("mismatched_arrays") == 0,
            "Replay has mismatches")
    require(report.get("device") == "cpu" and report.get("source_commit") == replay.SOURCE_COMMIT
            and report.get("selection") == "0,n//2,n-1"
            and report.get("tolerances") == dict(atol=replay.ATOL, rtol=replay.RTOL)
            and report.get("wall_seconds") == 900 and report.get("automatic_retries") == 0,
            "Replay contract differs")
    require(report.get("bindings") == bindings, "Replay is bound to different input bytes")
    rows, coverage = report.get("comparisons", []), report.get("coverage", [])
    require(len(rows) == 408 and len(coverage) == 30, "Replay detail incomplete")
    require(all(row.get("matched") is True and row.get("mismatches") == 0 and row.get("values", 0) > 0
                for row in rows), "Replay detail has failures")
    require(len({(row.get("checkpoint"), row.get("panel"), row.get("array")) for row in rows}) == 408,
            "Replay detail has duplicate cells")
    require(sum(row["values"] for row in rows) == report.get("values_checked"), "Replay value count differs")


def validate_inputs(root, panels, replay_path):
    replay = load_replay_module()
    verifier = replay.import_verifier()
    verifier.preflight(root, panels, replay.DEFAULT_IDENTITY)
    before = replay.capture_bindings(root, panels, replay.DEFAULT_IDENTITY, verifier)
    verified = verifier.verify(root, panels)
    require(verified.get("verified") is True, "Complete independent verification required")
    replay.require_verified_bindings(before, verified)
    report = verifier.read_json(replay_path)
    validate_replay(report, before, replay)
    replay.require_unchanged_bindings(before, replay.capture_bindings(root, panels, replay.DEFAULT_IDENTITY, verifier))
    return replay, verifier, verified, report, before


def relative_path(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name
            and str(path) == name, "Unsafe integration path")
    require(path.parts[0] in ("runs", "runs_real", "panels", "v4_audit")
            and path.suffix in (".json", ".npz"), "Unexpected integration file")
    return path


def contained(base, name):
    relative_path(name)
    destination = base / name
    require(destination.resolve().is_relative_to(base.resolve()), "Integration path escapes destination")
    require(not destination.is_symlink(), "Integration refuses symlinks")
    return destination


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def stage_results(root, panels, replay_path, stage):
    require(not stage.exists(), "Stage already exists; preserve it")
    require(not any(stage.resolve().is_relative_to(p.resolve()) for p in (root, panels, DEFAULT_TARGET)),
            "Stage must be outside immutable inputs and paper reports")
    replay, verifier, verified, report, bindings = validate_inputs(root, panels, replay_path)
    stage.mkdir(parents=True)
    checkpoints, scores = verifier.layout(root)
    files = [path for path, *_ in scores]
    files += [run / name for run, *_ in checkpoints for name in ("train.json", "checkpoint_identity.json")]
    files += [root / "runs/confirm_v4.json", root / "runs/tabpfn_v2/train.json"]
    for source in files:
        name = source.relative_to(root).as_posix()
        if name == "runs/tabpfn_v2/train.json":
            name = "runs/tabpfn_v2/train_v4.json"
        destination = contained(stage, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    # Deserialization occurs only after all frozen panel hashes were verified.
    import torch
    for tag in verifier.TAGS:
        panel = torch.load(panels / f"{tag}.pt", weights_only=False, map_location="cpu")
        arrays = {f"{c}|{model}|{metric}": np.asarray(value)
                  for c, models in panel["refs"].items() for model, metrics in models.items()
                  for metric, value in metrics.items()}
        require(arrays and all(value.dtype.kind in "fiu" and np.isfinite(value).all() for value in arrays.values()),
                "Panel reference export is empty/nonfinite")
        destination = contained(stage, f"panels/{tag}_refs.npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        info = dict(meta={key: value for key, value in panel["meta"].items()
                          if isinstance(value, (int, float, str, list, dict, bool, type(None)))},
                    n=int(panel["pool"]["n"]), keys=list(panel["pool"].get("keys", [])),
                    banks={str(key): int(len(bank)) for key, bank in panel.get("banks", {}).items()})
        write_json(contained(stage, f"panels/{tag}.json"), info)
    write_json(stage / "v4_audit/independent_verification.json", verified)
    write_json(stage / "v4_audit/selected_cpu_replay.json", report)
    replay.require_unchanged_bindings(bindings, replay.capture_bindings(root, panels, replay.DEFAULT_IDENTITY, verifier))
    inventory = {path.relative_to(stage).as_posix(): dict(sha256=verifier.digest(path), bytes=path.stat().st_size)
                 for path in sorted(stage.rglob("*")) if path.is_file()}
    manifest = dict(schema=1, status="staged", bindings=bindings, files=inventory,
                    scope="Complete v4 score/metadata integration; no model weights, raw data or scientific changes")
    write_json(stage / "STAGE_MANIFEST.json", manifest)
    return dict(status="staged", files=len(inventory), stage=str(stage), merge_required=True)


def merge_stage(root, panels, replay_path, stage, target):
    require(stage.is_dir() and not stage.is_symlink(), "Stage absent or linked")
    replay, verifier, _, _, bindings = validate_inputs(root, panels, replay_path)
    manifest = verifier.read_json(stage / "STAGE_MANIFEST.json")
    require(manifest.get("schema") == 1 and manifest.get("status") == "staged"
            and manifest.get("bindings") == bindings, "Stage provenance differs")
    inventory = manifest["files"]
    actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    require(actual == set(inventory) | {"STAGE_MANIFEST.json"}, "Stage file inventory differs")
    require(inventory and not target.is_symlink() and target.resolve() != stage.resolve(), "Invalid merge destination")
    pending, conflicts = [], []
    for name, meta in inventory.items():
        source, destination = contained(stage, name), contained(target, name)
        require(source.is_file() and not source.is_symlink() and source.stat().st_size == meta["bytes"]
                and verifier.digest(source) == meta["sha256"], f"Stage bytes differ: {name}")
        if destination.exists():
            if not destination.is_file() or verifier.digest(destination) != meta["sha256"]:
                conflicts.append(name)
        else:
            pending.append((source, destination))
    require(not conflicts, "Existing history conflicts; no files written: " + ", ".join(conflicts))
    # All conflicts and input identities are checked before any target write.
    for source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    for name, meta in inventory.items():
        require(verifier.digest(contained(target, name)) == meta["sha256"], f"Merged bytes differ: {name}")
    replay.require_unchanged_bindings(bindings, replay.capture_bindings(root, panels, replay.DEFAULT_IDENTITY, verifier))
    return dict(status="merged", files=len(inventory), new_files=len(pending), target=str(target),
                historical_files_overwritten=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=REPO / "artifacts/reports/lifted_v4_cpu_replay.json")
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    try:
        result = (merge_stage(args.root, args.panels, args.replay, args.stage, args.target) if args.merge else
                  stage_results(args.root, args.panels, args.replay, args.stage))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="failed", reason=f"{type(error).__name__}: {error}")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
