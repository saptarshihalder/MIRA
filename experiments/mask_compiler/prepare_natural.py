"""Freeze two official UCI naturally missing tasks; no models or imposed masks.

Run from any directory: python experiments/mask_compiler/prepare_natural.py
Cached source bytes and deterministic outputs are verified on every replay.
The legacy panel validator requires five folds; this module validates three.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mira.panel_data import canonicalize, load_panel, stratified_subset

CATALOG = {
    "hepatitis": {
        "uci_id": 46, "rows": 155, "source_columns": 20, "support_size": 64,
        "page": "https://archive.ics.uci.edu/dataset/46/hepatitis",
        "archive": "https://archive.ics.uci.edu/static/public/46/hepatitis.zip",
        "csv": "https://archive.ics.uci.edu/static/public/46/data.csv",
        "archive_sha256": "8573c2898714fb32ef8162291dcabd483a1ba9701123cab262f5acace3112d87",
        "csv_sha256": "79377934000c89e8ec579cb22486f236d96e37b57e179ec849ee790816cb8fe6",
        "data_members": ["hepatitis.data"], "documentation": ["hepatitis.names"],
        "retained_1based": list(range(2, 21)), "target_1based": 1,
        "target_map": {"1": 1, "2": 0}, "subject_1based": None,
        "excluded_1based": {"1": "target: die1/live2"},
        "citation": "Hepatitis [Dataset]. (1983). UCI Machine Learning Repository. https://doi.org/10.24432/C5Q59J",
        "doi": "10.24432/C5Q59J",
        "limitations": ["No subject identifiers or observation timestamps supplied; duplicate grouping cannot establish patient independence.",
                        "Treatment and histology variables are retained as historical benchmark covariates; no prospective clinical interpretation."],
    },
    "horse_colic": {
        "uci_id": 47, "rows": 368, "source_columns": 28, "support_size": 128,
        "page": "https://archive.ics.uci.edu/dataset/47/horse+colic",
        "archive": "https://archive.ics.uci.edu/static/public/47/horse+colic.zip",
        "csv": "https://archive.ics.uci.edu/static/public/47/data.csv",
        "archive_sha256": "e4a2d4e77573e521057b75ce00992060f2f002a1658d5a8e430138b60c3c8053",
        "csv_sha256": "7cad77c51181c532babb5c08db40dff35d3edb39797ea2f5291468a3e35d1efd",
        "data_members": ["horse-colic.data", "horse-colic.test"],
        "documentation": ["horse-colic.names", "horse-colic.names.original"],
        "retained_1based": [2, *range(4, 23)], "target_1based": 23,
        "target_map": {"1": 0, "2": 1, "3": 1}, "subject_1based": 3,
        "excluded_1based": {"1": "completed surgery/treatment decision", "3": "hospital case identifier: grouping only",
                            "23": "outcome target", "24": "retrospective surgical lesion", "25": "lesion field",
                            "26": "lesion field", "27": "lesion field", "28": "pathology-data availability flag"},
        "citation": "McLeish, M. & Cecile, M. (1989). Horse Colic [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58W23",
        "doi": "10.24432/C58W23",
        "limitations": ["Original300 train/68 test records are pooled before new grouped CV; this is not the original benchmark split.",
                        "Hospital number may recur across treatments; supplied IDs and identical-X groups are isolated, but unknown dependencies remain.",
                        "Death and euthanasia are combined; these outcomes have different causes and treatment decisions.",
                        "Observation timing is unavailable; some measurements follow interventions. No prospective, causal, or clinical claim.",
                        "Original age values include9 despite documentation describing young code2; preserve raw codes without recoding."],
    },
}
LICENSE = {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
           "verification": "Official UCI dataset page, license section, verified 2026-10-02"}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def fixed_file(path, content):
    """No silent replacement of cached source or frozen outputs."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return digest(content)


def fetch(path, url, expected=None, offline=False):
    path = Path(path)
    if path.exists():
        content = path.read_bytes()
    elif offline:
        raise FileNotFoundError(f"Offline source missing: {path}")
    else:
        content = urlopen(Request(url, headers={"User-Agent": "MIRA reproducible UCI ingestion"}), timeout=30).read()
        if expected and digest(content) != expected:
            raise ValueError(f"Official source changed: {url}")
        fixed_file(path, content)
    if expected and digest(content) != expected:
        raise ValueError(f"Cached source changed: {path}")
    return content


def number(value):
    return np.nan if value.strip() in {"?", "", "NA", "nan", "NaN"} else float(value)


def parse_sources(archive, csv_content, config):
    """Cross-check every official CSV field against the archived original records."""
    header, *csv_rows = list(csv.reader(io.StringIO(csv_content.decode("utf-8-sig"))))
    if len(header) != config["source_columns"] or len(set(header)) != len(header):
        raise ValueError("Unexpected or duplicate official CSV columns")
    archived, origins = [], []
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for filename in config["data_members"]:
            lines = source.read(filename).decode("ascii").splitlines()
            for line_number, line in enumerate(lines, 1):
                if line.strip():
                    tokens = line.split(",") if config["uci_id"] == 46 else line.split()
                    archived.append([number(token) for token in tokens])
                    origins.append({"source_file": filename, "source_line": line_number})
    expected = (config["rows"], config["source_columns"])
    original = np.asarray(archived, dtype=float)
    csv_values = np.asarray([[number(value) for value in row] for row in csv_rows], dtype=float)
    if original.shape != expected or csv_values.shape != expected or not np.array_equal(original, csv_values, equal_nan=True):
        raise ValueError("Official archive/CSV rows differ")
    labels = original[:, config["target_1based"] - 1]
    accepted = np.flatnonzero(np.isfinite(labels))
    allowed = {float(key) for key in config["target_map"]}
    if not set(labels[accepted]) <= allowed:
        raise ValueError("Unexpected observed outcome")
    selected = np.array(config["retained_1based"]) - 1
    if config["target_1based"] - 1 in selected or len(set(selected)) != len(selected):
        raise ValueError("Target or duplicate column retained")
    X = original[accepted][:, selected]
    y = np.array([config["target_map"][str(int(labels[i]))] for i in accepted], dtype=np.int64)
    subjects = None
    if config["subject_1based"]:
        identifier = original[accepted, config["subject_1based"] - 1]
        if not np.isfinite(identifier).all() or not np.equal(identifier, np.floor(identifier)).all():
            raise ValueError("Hospital identifiers must be observed integers")
        subjects = [str(int(value)) for value in identifier]
    return X, y, subjects, [header[i] for i in selected], origins, accepted


def validate_natural_splits(panel, splits):
    """Panel-compatible identities with explicit three-fold validation."""
    universe, heldout_all = set(range(len(panel.y))), []
    if splits["n_splits"] != 3 or len(splits["folds"]) != 3:
        raise ValueError("Require exactly three frozen folds")
    for fold, record in enumerate(splits["folds"]):
        if record["fold"] != fold:
            raise ValueError("Invalid fold order")
        support, query, train, heldout = [record[key] for key in
            ("support_indices", "query_indices", "training_indices", "heldout_indices")]
        for indices in (support, query, train, heldout):
            if len(indices) != len(set(indices)) or not set(indices) <= universe:
                raise ValueError("Duplicate or invalid row indices")
        if set(train) & set(heldout) or set(train) | set(heldout) != universe:
            raise ValueError("Invalid outer partition")
        if not set(support) <= set(train) or not set(query) <= set(heldout):
            raise ValueError("Support/query escaped outer partition")
        if set(panel.group_ids[train]) & set(panel.group_ids[heldout]):
            raise ValueError("Identical-X/hospital group leakage")
        if len(support) != splits["support_size"] or not 2 <= len(query) <= splits["query_cap"]:
            raise ValueError("Incorrect support/query budget")
        if len(np.unique(panel.y[support])) != 2 or len(np.unique(panel.y[query])) != 2:
            raise ValueError("Both classes required")
        for part, indices in (("support", support), ("query", query)):
            if record[part + "_row_ids"] != panel.row_ids[indices].tolist() or record[part + "_group_ids"] != panel.group_ids[indices].tolist():
                raise ValueError("Split IDs differ from canonical rows")
        rates = panel.native_mask[support].mean(axis=0)
        chosen = np.lexsort((np.arange(panel.X.shape[1]), -rates))[:4].tolist()
        if record["selected_columns"] != chosen or not np.array_equal(record["support_column_missing_rates"], rates):
            raise ValueError("Compiler columns not selected from support-only missingness")
        heldout_all.extend(heldout)
    if len(heldout_all) != len(universe) or set(heldout_all) != universe:
        raise ValueError("Each row must occur in exactly one heldout partition")


def make_natural_splits(panel, identifier, support_size):
    seed = 91000 + identifier
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    folds = []
    for fold, (train, heldout) in enumerate(splitter.split(panel.X, panel.y, panel.group_ids)):
        support_seed, query_seed = 92000 + 100 * identifier + fold, 93000 + 100 * identifier + fold
        support = stratified_subset(train, panel.y, support_size, support_seed)
        query = stratified_subset(heldout, panel.y, min(256, len(heldout)), query_seed)
        rates = panel.native_mask[support].mean(axis=0)
        chosen = np.lexsort((np.arange(panel.X.shape[1]), -rates))[:4]
        folds.append({"fold": fold, "training_indices": train.tolist(), "heldout_indices": heldout.tolist(),
            "support_indices": support.tolist(), "query_indices": query.tolist(),
            "support_row_ids": panel.row_ids[support].tolist(), "query_row_ids": panel.row_ids[query].tolist(),
            "support_group_ids": panel.group_ids[support].tolist(), "query_group_ids": panel.group_ids[query].tolist(),
            "support_seed": support_seed, "query_seed": query_seed,
            "support_class_counts": np.bincount(panel.y[support], minlength=2).tolist(),
            "query_class_counts": np.bincount(panel.y[query], minlength=2).tolist(),
            "selected_columns": chosen.tolist(), "support_column_missing_rates": rates.tolist()})
    result = {"splitter": "StratifiedGroupKFold", "splitter_seed": seed, "n_splits": 3,
              "support_size": support_size, "query_cap": 256, "folds": folds,
              "column_selection": "support native missingness descending; ties by original retained column index"}
    validate_natural_splits(panel, result)
    return result


def prepare(data_root, manifest_root, offline=False):
    data_root, manifest_root = Path(data_root), Path(manifest_root)
    selection = {"version": 1, "datasets": CATALOG, "license": LICENSE,
                 "frozen_before_scores": True, "replacement_policy": "No silent replacement; record acquisition/validation failure",
                 "predictor_encoding": "Continuous values and original binary/ordinal/nominal codes kept numeric; no one-hot, scaling or imputation. Numeric nominal codes imply artificial geometry; no clinical claim.",
                 "missingness": "Native question marks/empty CSV entries become NaN; no imposed masks or label-based column choice."}
    fixed_file(manifest_root / "selection.json", json_bytes(selection))
    summary = []
    for name, config in CATALOG.items():
        directory = data_root / name
        source = directory / "source"
        try:
            urls = {"metadata.json": f"https://archive.ics.uci.edu/api/dataset?id={config['uci_id']}",
                    "archive.zip": config["archive"], "official.csv": config["csv"], "official_page.html": config["page"]}
            content = {filename: fetch(source / filename, url,
                       config.get("archive_sha256") if filename == "archive.zip" else config.get("csv_sha256") if filename == "official.csv" else None,
                       offline=offline) for filename, url in urls.items()}
            metadata = json.loads(content["metadata.json"])["data"]
            if metadata["uci_id"] != config["uci_id"] or metadata["num_instances"] != config["rows"] or metadata["dataset_doi"] != config["doi"]:
                raise ValueError("Official metadata identity differs")
            if b"Creative Commons Attribution 4.0" not in content["official_page.html"] and b"CC BY 4.0" not in content["official_page.html"]:
                raise ValueError("Official page license not verifiable")
            with zipfile.ZipFile(io.BytesIO(content["archive.zip"])) as archive:
                for member in config["data_members"] + config["documentation"]:
                    fixed_file(source / member, archive.read(member))
            X, y, subjects, features, origins, accepted = parse_sources(content["archive.zip"], content["official.csv"], config)
            panel, maps = canonicalize(X, y, config["uci_id"], subjects)
            maps["canonical_order"] = "first labeled occurrence in official archive order (train then test for Horse)"
            maps["accepted_original_indices"] = accepted.tolist()
            maps["canonical_source_rows"] = [[{**origins[int(accepted[i])], "original_index": int(accepted[i]),
                "hospital_number": subjects[i] if subjects else None} for i in members] for members in maps["canonical_raw_row_indices"]]
            maps["excluded_missing_outcome_rows"] = [{**origins[i], "original_index": i} for i in range(config["rows"]) if i not in set(accepted)]
            splits = make_natural_splits(panel, config["uci_id"], config["support_size"])
            buffer = io.BytesIO()
            np.savez(buffer, **{key: getattr(panel, key) for key in ("X", "y", "row_ids", "group_ids", "native_mask", "multiplicity")})
            outputs = {"dataset.npz": buffer.getvalue(), "splits.json": json_bytes(splits), "row_maps.json": json_bytes(maps)}
            hashes = {filename: fixed_file(directory / filename, value) for filename, value in outputs.items()}
            source_files = {filename: {"url": urls.get(filename, config["archive"]), "sha256": digest((source / filename).read_bytes()),
                            "bytes": (source / filename).stat().st_size} for filename in [*urls, *config["data_members"], *config["documentation"]]}
            manifest = {"status": "complete", "dataset": name, "uci_id": config["uci_id"], "configuration": config,
                "features": features, "license": LICENSE, "citation": config["citation"], "source_files": source_files,
                "source_csv_archive_agreement": "Every numeric field and native NaN equals original archive, in source order",
                "original_rows": config["rows"], "labeled_rows": len(X), "canonical_rows": len(panel.y),
                "excluded_missing_outcomes": len(maps["excluded_missing_outcome_rows"]), "features_count": panel.X.shape[1],
                "class_counts": np.bincount(panel.y, minlength=2).tolist(), "native_missing_cells": int(panel.native_mask.sum()),
                "native_missing_fraction": float(panel.native_mask.mean()), "duplicates_removed": len(X)-len(panel.y),
                "groups": len(set(panel.group_ids)), "conflicting_label_groups": maps["conflicting_label_groups"],
                "predictor_encoding": selection["predictor_encoding"], "imposed_masks": False,
                "artifact_sha256": hashes, "selection_sha256": digest(json_bytes(selection)),
                "split_validation": "three-fold validator in prepare_natural.py; legacy validate_splits requires five folds",
                "compiler_U_proxy": "zero only inside restricted selector observations; full native X remains backbone input"}
            fixed_file(directory / "manifest.json", json_bytes(manifest))
            loaded = load_panel(directory)
            validate_natural_splits(loaded, splits)
            summary.append({"dataset": name, "shape": list(panel.X.shape), "class_counts": manifest["class_counts"],
                "native_missing_fraction": manifest["native_missing_fraction"], "support_size": splits["support_size"],
                "query_sizes": [len(f["query_indices"]) for f in splits["folds"]],
                "selected_columns": [f["selected_columns"] for f in splits["folds"]],
                "manifest_sha256": digest(json_bytes(manifest)), "artifact_sha256": hashes})
        except Exception as error:
            manifest_root.mkdir(parents=True, exist_ok=True)
            (manifest_root / f"{name}_failure.json").write_bytes(json_bytes({"dataset": name, "error": repr(error),
                "time_utc": datetime.now(timezone.utc).isoformat(), "replacement": None}))
            raise
    fixed_file(manifest_root / "summary.json", json_bytes({"datasets": summary, "no_model_scores": True}))
    return summary


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts/data/natural_compiler")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "artifacts/manifests/natural_compiler")
    parser.add_argument("--offline", action="store_true", help="Recompute and verify using cached official bytes only")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.manifest_root, args.offline), indent=2))


if __name__ == "__main__":
    cli()
