"""Atomic materialized artifacts and an append-only successful-result ledger."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .data import Episode


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_episode(path: Path, episode: Episode) -> None:
    support, query = episode.support.observations, episode.query.observations
    atomic_npz(path, xc=support.values, uc=support.u, mc=support.mask, yc=episode.support.labels,
               xq=query.values, uq=query.u, mq=query.mask, yq=episode.targets.labels,
               oracle=episode.targets.oracle, base=episode.targets.analytic_base,
               support_row_ids=episode.support_row_ids, query_row_ids=episode.query_row_ids)


def verify_episode(path: Path, episode: Episode) -> None:
    support, query = episode.support.observations, episode.query.observations
    expected = {"xc": support.values, "uc": support.u, "mc": support.mask, "yc": episode.support.labels,
                "xq": query.values, "uq": query.u, "mq": query.mask, "yq": episode.targets.labels,
                "oracle": episode.targets.oracle, "base": episode.targets.analytic_base,
                "support_row_ids": episode.support_row_ids, "query_row_ids": episode.query_row_ids}
    with np.load(path, allow_pickle=False) as data:
        def equal(key: str, value: Any) -> bool:
            return np.array_equal(data[key], value, equal_nan=True) if value.dtype.kind == "f" else np.array_equal(data[key], value)
        if set(data.files) != set(expected) or any(not equal(key, value) for key, value in expected.items()):
            raise ValueError(f"Stored task disagrees with reproducible generator: {path}")


def source_hashes() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    root = source.parent.parent
    paths = list(source.glob("*.py")) + list((root / "scripts").glob("*mechanism.py"))
    paths += [root / "pyproject.toml"]
    return {str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
            for path in sorted(paths) if path.is_file()}


def checkpoint_hashes(directories: list[str]) -> dict[str, str]:
    paths = set()
    for directory in directories:
        root = Path(directory).expanduser()
        if root.is_file():
            paths.add(root.resolve())
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("*") if path.is_file()
                         and path.suffix.lower() in {".ckpt", ".pt", ".pth", ".safetensors"})
    return {str(path): file_sha256(path) for path in sorted(paths)}


def capture_environment(out: Path, lock: bool = True) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "tabpfn", "tabicl", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    environment: dict[str, Any] = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "dependency_lock_status": "not requested",
    }
    if lock:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                                    capture_output=True, text=True, check=True, timeout=30)
            temporary_lock = out / "requirements-lock.txt.tmp"
            temporary_lock.write_text(result.stdout, encoding="utf-8")
            environment["dependency_lock_status"] = "captured"
            environment["dependency_lock_sha256"] = file_sha256(temporary_lock)
        except (OSError, subprocess.SubprocessError) as error:
            environment["dependency_lock_status"] = "failed"
            environment["dependency_lock_error"] = repr(error)
    existing_path = out / "environment.json"
    if existing_path.exists():
        previous = json.loads(existing_path.read_text(encoding="utf-8"))
        if previous != environment:
            raise ValueError("Environment differs from original run; use a new --out (original environment preserved)")
    if lock and environment["dependency_lock_status"] == "captured":
        (out / "requirements-lock.txt.tmp").replace(out / "requirements-lock.txt")
    atomic_json(out / "environment.json", environment)
    return environment


def load_rows(out: Path) -> list[dict[str, Any]]:
    ledger = out / "results.jsonl"
    if not ledger.exists():
        return []
    rows = []
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Interrupted/corrupt ledger at line {number}; preserve and repair explicitly") from error
        if any(existing["cell_id"] == row["cell_id"] for existing in rows):
            raise ValueError(f"Duplicate ledger cell: {row['cell_id']}")
        rows.append(row)
    return rows


def append_result(out: Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    rows.append(row)
    # CSV is a convenient view; the JSONL ledger is authoritative on resume.
    temporary = out / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(out / "results.csv")
