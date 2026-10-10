"""Build a review archive only from complete verified/replayed v4 and preserved history.

Contains source, numeric scores/reference exports and regeneration tools. Excludes
raw observations, all weights, private provider provenance and manuscript/style.
Does not train, use cloud compute or grant a redistribution license.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import stat
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STAMP = (1980, 1, 1, 0, 0, 0)
MAX_TOTAL = 100 * 1024**2


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def cleaned(value):
    """Remove host-specific path values from exported metadata, never numeric arrays."""
    if isinstance(value, dict):
        return {key: cleaned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cleaned(item) for item in value]
    if isinstance(value, str) and (value.startswith(("/home/", "/Users/", "/tmp/", "/content/", "/results/", "/work/"))
                                   or re.match(r"^[a-zA-Z]:[\\/]", value)):
        return "<local-path>"
    return value


def audit_text(name, data, base):
    text = data.decode("utf-8")
    if "\x00" in text or any(pattern.search(text) for pattern in base.GUARDS.values()):
        raise ValueError(f"Identity/credential/binary scan failed: {name}")
    if re.search(r"saptarshihalder|github\.com/saptarshi", text, re.I):
        raise ValueError(f"Author handle in export: {name}")


def build(root, panels, replay_path, output):
    integration = module("v4_review_integration", "integrate_lifted_v4_verified.py")
    replay, verifier, _, _, bindings = integration.validate_inputs(root, panels, replay_path)
    base = module("v4_review_base", "build_lifted_supplement.py")
    integration.require(not output.exists() and output.parent.is_dir(), "New archive path with existing parent required")
    files = base.payload(REPO)
    files["README.md"] = (REPO / "docs/LIFTED_REVIEW_PACKAGE.md").read_bytes()
    files["experiments/lifted_cavity/make_paper.py"] = (REPO / "experiments/lifted_cavity/make_paper.py").read_bytes()
    files["reproduce_paper.py"] = (REPO / "infra/reproduce_lifted_paper_exports.py").read_bytes()
    files["infra/verify_lifted_v4_results.py"] = (REPO / "infra/verify_lifted_v4_results.py").read_bytes()
    environment = REPO / "artifacts/reports/lifted_v4_launch/v4_resume_environment_20261011.json"
    files["provenance/v4_environment.json"] = environment.read_bytes()
    installed = json.loads(files["provenance/v4_environment.json"])["packages"]
    files["requirements-v4-observed.txt"] = ("# Observed installed distributions, not wheel hashes or an independently rebuilt environment.\n" +
                                            "\n".join(f"{row['name']}=={row['version']}" for row in installed) + "\n").encode()
    seed_audit = REPO / "artifacts/reports/lifted_v4_seed_sensitivity.json"
    sensitivity = json.loads(seed_audit.read_text(encoding="utf-8"))
    integration.require(sensitivity.get("bindings") == bindings, "Seed sensitivity belongs to other inputs")
    files["results/v4_audit/seed_sensitivity.json"] = (json.dumps(cleaned(sensitivity), sort_keys=True,
                                                               indent=2, allow_nan=False) + "\n").encode()
    files["docs/LIFTED_V4_SEED_AUDIT_PLAN.md"] = (REPO / "docs/LIFTED_V4_SEED_AUDIT_PLAN.md").read_bytes()
    report_root = integration.DEFAULT_TARGET
    stage = integration.DEFAULT_STAGE
    staged = verifier.read_json(stage / "STAGE_MANIFEST.json")
    integration.require(staged.get("bindings") == bindings, "Integrated result provenance differs")
    for name, meta in staged["files"].items():
        path = integration.contained(report_root, name)
        integration.require(path.is_file() and verifier.digest(path) == meta["sha256"], f"Integration incomplete: {name}")
    transformed = {}
    v1 = REPO / "artifacts/reports/lifted_cavity_v1/confirmation/conf_results.json"
    files["results/conf_results.json"] = (json.dumps(cleaned(json.loads(v1.read_text(encoding="utf-8"))),
                                                    sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    # Only numeric archives and JSON from the established paper-report tree.
    for path in sorted(report_root.rglob("*")):
        if not path.is_file() or path.suffix not in (".npz", ".json"):
            continue
        integration.require(not path.is_symlink() and path.resolve().is_relative_to(report_root.resolve()), "Linked report input")
        name = "results/" + path.relative_to(report_root).as_posix()
        data = path.read_bytes()
        original_hash = base.digest(data)
        if path.suffix == ".json":
            data = (json.dumps(cleaned(json.loads(data)), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        else:
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    value = archive[key]
                    integration.require(value.dtype.kind in "fiu" and np.isfinite(value).all(), f"Nonnumeric/nonfinite archive: {name}")
        if base.digest(data) != original_hash:
            transformed[name] = dict(original_sha256=original_hash,
                                     reason="Canonical JSON formatting and host-path removal only")
        files[name] = data
    for name, data in files.items():
        if not name.endswith(".npz"):
            audit_text(name, data, base)
    integration.require(sum(map(len, files.values())) <= MAX_TOTAL, "Review archive exceeds fixed size bound")
    manifest = dict(schema=1, scope="code-and-score-review-package", source_revision=base.SOURCE_REVISION,
                    panel_revision=base.PANEL_REVISION,
                    files={name: dict(bytes=len(data), sha256=base.digest(data)) for name, data in sorted(files.items())},
                    metadata_exports=transformed, weights_included=False, raw_observations_included=False,
                    training_reproduced=False, independent_execution_timestamps=False,
                    identity_scan="Pattern scan plus explicit export exclusions; contextual anonymity requires human review")
    files["MANIFEST.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    replay.require_unchanged_bindings(bindings, replay.capture_bindings(root, panels, replay.DEFAULT_IDENTITY, verifier))
    with zipfile.ZipFile(output, "x", zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, STAMP)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        integration.require(len(archive.namelist()) == len(set(archive.namelist())) == len(files), "Review archive inventory differs")
        for name, data in files.items():
            integration.require(archive.read(name) == data, f"Review archive round-trip differs: {name}")
    return dict(status="built", files=len(files), bytes=output.stat().st_size, sha256=verifier.digest(output),
                output=str(output), scope=manifest["scope"], training_reproduced=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=REPO / "artifacts/reports/lifted_v4_cpu_replay.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.root, args.panels, args.replay, args.out), indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps(dict(status="failed", reason=f"{type(error).__name__}: {error}")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
