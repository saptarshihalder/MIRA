"""Freeze Wine Quality development inputs; no model fitting or evaluation.

Run from any directory with Python and NumPy. This intentionally refuses to
replace a completed audit. Exact feature duplicates share a global partition,
including duplicates across colors and duplicates with conflicting labels.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/wine_acquisition"
LOCAL = ROOT / "artifacts/local/wine_acquisition"
AUDIT = ROOT / "artifacts/manifests/wine_acquisition_data_audit.json"
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
PARTITIONS = {0: "fit", 1: "validation", 2: "development"}
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def canonical_features(row: np.ndarray) -> bytes:
    values = np.asarray(row, dtype="<f8").copy()
    values[values == 0] = 0.0  # Canonical positive zero; NaN/inf rejected earlier.
    return values.tobytes(order="C")


def partition_for(group: str) -> int:
    # Integer comparisons avoid boundary changes from floating-point rounding.
    value = int(group, 16) * 10
    modulus = 1 << 256
    return 0 if value < 6 * modulus else (1 if value < 8 * modulus else 2)


def verify_ignored(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative(path)],
        cwd=ROOT, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Local output is not git-ignored: {relative(path)}")


def download(color: str) -> tuple[dict, bytes]:
    name = f"winequality-{color}.csv"
    url = BASE + name
    request = Request(url, headers={"User-Agent": "MIRA-Wine-Development-Preparation/1"})
    with urlopen(request, timeout=60) as response:
        payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("Unexpectedly large Wine Quality source")
        metadata = {
            "url": url, "resolved_url": response.geturl(),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_last_modified": response.headers.get("Last-Modified"),
            "http_etag": response.headers.get("ETag"),
            "bytes": len(payload), "sha256": sha256(payload),
            "path": relative(RAW / name),
        }
    return metadata, payload


def parse(payload: bytes, color: str) -> dict:
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig")), delimiter=";")
    if next(reader) != FEATURES + ["quality"]:
        raise ValueError(f"Unexpected {color} source header")
    rows = list(reader)
    if not rows or any(len(row) != 12 for row in rows):
        raise ValueError(f"Malformed {color} source rows")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source value")
    quality = values[:, 11]
    if not ((quality == np.floor(quality)) & (quality >= 0) & (quality <= 10)).all():
        raise ValueError("Invalid documented quality scale")
    return {
        "raw_X": values[:, :11], "quality": quality,
        "y": (quality >= 6).astype(np.uint8),
        "row_ids": np.asarray([f"winequality-{color}.csv:{i + 2}" for i in range(len(rows))]),
    }


def main() -> None:
    if AUDIT.exists():
        raise RuntimeError("Completed audit already exists; do not silently replace frozen data")
    for folder, probe in [(RAW, "probe.csv"), (LOCAL, "probe.npz")]:
        folder.mkdir(parents=True, exist_ok=True)
        verify_ignored(folder / probe)

    sources, datasets = {}, {}
    groups: dict[str, dict] = {}
    for color in ["red", "white"]:
        sources[color], payload = download(color)
        datasets[color] = data = parse(payload, color)
        (RAW / f"winequality-{color}.csv").write_bytes(payload)
        keys = []
        for row, quality, y in zip(data["raw_X"], data["quality"], data["y"]):
            canonical = canonical_features(row)
            key = sha256(canonical)
            entry = groups.setdefault(key, {
                "canonical": canonical, "colors": set(), "quality": set(),
                "labels": set(), "partitions": set(), "rows": 0,
            })
            if entry["canonical"] != canonical:
                raise RuntimeError("SHA256 collision between distinct vectors")
            entry["colors"].add(color)
            entry["quality"].add(int(quality))
            entry["labels"].add(int(y))
            entry["partitions"].add(partition_for(key))
            entry["rows"] += 1
            keys.append(key)
        data["groups"] = np.asarray(keys, dtype="U64")
        data["partition"] = np.asarray([partition_for(key) for key in keys], dtype=np.uint8)

    overlap = sum(len(entry["partitions"]) > 1 for entry in groups.values())
    if overlap:
        raise RuntimeError("Feature duplicate group crossed a partition boundary")
    for entry in groups.values():
        if len(entry["quality"]) > 1 or len(entry["labels"]) > 1:
            assert len(entry["partitions"]) == 1

    outputs = {}
    for color, data in datasets.items():
        fit = data["partition"] == 0
        if not fit.any():
            raise ValueError("No fit rows available for bin thresholds")
        thresholds = np.quantile(data["raw_X"][fit], [1 / 3, 2 / 3], axis=0, method="linear").T
        # Right insertion: values equal to a threshold enter the higher bin.
        binned = (data["raw_X"][:, :, None] >= thresholds[None, :, :]).sum(axis=2).astype(np.uint8)
        path = LOCAL / f"wine_{color}.npz"
        verify_ignored(path)
        np.savez_compressed(
            path, X=binned, y=data["y"], row_ids=data["row_ids"],
            groups=data["groups"], partition=data["partition"],
            thresholds=thresholds, feature_names=np.asarray(FEATURES),
        )
        with np.load(path, allow_pickle=False) as loaded:
            assert loaded["X"].shape == (len(data["y"]), 11)
            assert np.isin(loaded["X"], [0, 1, 2]).all()
            assert np.array_equal(loaded["groups"], data["groups"])
            assert np.array_equal(loaded["partition"], data["partition"])
            assert np.array_equal(loaded["y"], data["y"])
        outputs[color] = {
            "path": relative(path), "sha256": sha256(path.read_bytes()),
            "rows": len(data["y"]), "unique_feature_groups": len(set(data["groups"])),
            "partition_rows": {name: int((data["partition"] == i).sum()) for i, name in PARTITIONS.items()},
        }

    audit = {
        "status": "development data prepared; no model fitting, scoring or outcome summaries",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": "Wine Quality", "uci_id": 186, "doi": "10.24432/C56S3T",
                    "page": "https://archive.ics.uci.edu/dataset/186/wine+quality",
                    "license": "CC BY 4.0", "attribution": "Cortez, Cerdeira, Almeida, Matos and Reis (2009)",
                    "metadata_last_updated": "2023-11-15"},
        "versions": {"raw_version": "SHA256 of downloaded bytes; UCI does not supply an immutable release identifier",
                     "python": platform.python_version(), "numpy": np.__version__,
                     "preparation_script_sha256": sha256(Path(__file__).read_bytes())},
        "sources": sources, "outputs": outputs, "feature_names": FEATURES,
        "label_rule": "quality >= 6; no label-stratified split or outcome statistics",
        "group_rule": "SHA256 of 11 source-ordered finite IEEE754 little-endian float64 values, C order, positive zero normalization; excludes color and target",
        "split_rule": "Global full-digest integer / 2**256: [0,.6) fit, [.6,.8) validation, [.8,1) development; approximate 60/20/20 group allocation",
        "bin_rule": "Per-color, per-feature fit-row terciles; numpy linear quantiles at 1/3 and 2/3; equality enters higher bin; ties may leave bins empty",
        "npz_schema": {"X": "uint8 [n,11], bins 0..2", "y": "uint8 [n]", "row_ids": "unicode [n], source filename:CSV line",
                       "groups": "U64 [n]", "partition": "uint8 [n], 0 fit / 1 validation / 2 development",
                       "thresholds": "float64 [11,2]", "feature_names": "unicode [11]"},
        "audit": {"total_rows": sum(len(data["y"]) for data in datasets.values()),
                  "global_unique_feature_groups": len(groups),
                  "duplicate_groups": sum(entry["rows"] > 1 for entry in groups.values()),
                  "duplicate_rows_beyond_first": sum(entry["rows"] - 1 for entry in groups.values()),
                  "cross_color_feature_groups": sum(len(entry["colors"]) > 1 for entry in groups.values()),
                  "cross_partition_group_overlap": overlap,
                  "conflicting_quality_and_binary_label_groups_checked": True,
                  "conflicting_label_policy": "Retain every source row; identical features remain in one global group regardless of conflicting labels",
                  "sha256_collision_check_passed": True, "npz_roundtrip_checked": True,
                  "local_paths_git_ignored": True},
        "limitations": ["Development only; not fresh external confirmation.",
                        "Exact feature grouping does not establish patient, producer, batch, bottle or temporal independence; these identities are unavailable.",
                        "Near duplicates are not detected by exact-vector grouping; arbitrary target proxies cannot be excluded.",
                        "No empirical assay price, latency or joint measurement grouping; acquisition costs must be explicitly simulated.",
                        "Red and white are domains within one dataset, not independent dataset families.",
                        "All repeated rows retained, so duplicates affect fit quantiles and subsequent row-weighted results."],
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Saved development arrays and provenance audit: {relative(AUDIT)}")


if __name__ == "__main__":
    main()
