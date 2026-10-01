"""Canonical real-covariate data and fixed group-isolated splits; no model inference."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def _freeze(value, dtype):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PanelDataset:
    X: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray
    group_ids: np.ndarray
    native_mask: np.ndarray
    multiplicity: np.ndarray

    def __post_init__(self):
        X = _freeze(self.X, np.float64)
        y = _freeze(self.y, np.int64)
        mask = _freeze(self.native_mask, np.uint8)
        row_ids, group_ids = _freeze(self.row_ids, str), _freeze(self.group_ids, str)
        multiplicity = _freeze(self.multiplicity, np.int64)
        n = len(X)
        if X.ndim != 2 or X.shape[1] < 1 or np.isinf(X).any():
            raise ValueError("Require a numeric feature matrix without infinities")
        if any(a.shape != (n,) for a in (y, row_ids, group_ids, multiplicity)):
            raise ValueError("Canonical row arrays must align with X")
        if not np.isin(self.y, [0, 1]).all() or not np.array_equal(np.asarray(self.y), y):
            raise ValueError("Labels must be binary integers")
        if mask.shape != X.shape or not np.isin(self.native_mask, [0, 1]).all() or not np.array_equal(mask.astype(bool), np.isnan(X)):
            raise ValueError("Native masks must exactly preserve NaN locations")
        if len(set(row_ids)) != n or any(not value for value in row_ids) or any(not value for value in group_ids):
            raise ValueError("Row IDs must be unique and group IDs nonempty")
        if not (multiplicity >= 1).all() or not np.array_equal(self.multiplicity, multiplicity):
            raise ValueError("Multiplicities must be positive integers")
        for name, value in (("X", X), ("y", y), ("row_ids", row_ids), ("group_ids", group_ids),
                            ("native_mask", mask), ("multiplicity", multiplicity)):
            object.__setattr__(self, name, value)


def _feature_key(row: np.ndarray) -> bytes:
    # Equality treats signed zeros and all NaN payloads as the same value.
    normalized = np.array(row, dtype="<f8", copy=True)
    normalized[normalized == 0] = 0.
    normalized[np.isnan(normalized)] = np.nan
    return normalized.tobytes()


def canonicalize(X, y, dataset_id: int, subject_ids=None) -> tuple[PanelDataset, dict[str, Any]]:
    """Keep first source occurrence, remove exact (selected X,y) duplicates, union X/ID groups."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isin(y, [0, 1]).all() or np.isinf(X).any():
        raise ValueError("Invalid input matrix or binary labels")
    subjects = [""] * len(X) if subject_ids is None else [str(value) for value in subject_ids]
    if len(subjects) != len(X):
        raise ValueError("Subject identifiers must align with raw rows")
    prefix = f"uci:{dataset_id}:".encode()
    seen, representatives, members, mapping, keys = {}, [], [], [], []
    for raw_index, (row, label) in enumerate(zip(X, y)):
        feature = _feature_key(row)
        key = feature + bytes([int(label)])
        if key not in seen:
            seen[key] = len(representatives)
            representatives.append(raw_index)
            members.append([])
            keys.append(feature)
        index = seen[key]
        members[index].append(raw_index)
        mapping.append(index)
    row_ids = ["uci:" + str(dataset_id) + ":r:" + hashlib.sha256(prefix + key).hexdigest()
               for key in seen]
    parent = list(range(len(representatives)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(a, b):
        parent[find(b)] = find(a)
    x_seen, subject_seen = {}, {}
    for index, feature in enumerate(keys):
        if feature in x_seen:
            union(index, x_seen[feature])
        x_seen[feature] = index
    for raw_index, subject in enumerate(subjects):
        if subject and subject not in {"None", "nan"}:
            index = mapping[raw_index]
            if subject in subject_seen:
                union(index, subject_seen[subject])
            subject_seen[subject] = index
    components = {}
    for index, feature in enumerate(keys):
        components.setdefault(find(index), []).append(feature)
    component_ids = {root: "uci:" + str(dataset_id) + ":g:" + hashlib.sha256(prefix + min(features)).hexdigest()
                     for root, features in components.items()}
    groups = [component_ids[find(index)] for index in range(len(representatives))]
    selected = np.asarray(representatives, dtype=np.int64)
    panel = PanelDataset(X[selected], y[selected], row_ids, groups, np.isnan(X[selected]), [len(m) for m in members])
    return panel, {"canonical_order": "first occurrence in official source CSV",
        "raw_row_to_canonical": mapping, "canonical_raw_row_indices": members,
        "canonical_source_identifiers": [sorted(set(subjects[i] for i in member if subjects[i])) for member in members],
        "raw_rows": len(X), "canonical_rows": len(panel.y), "exact_xy_duplicates_removed": len(X)-len(panel.y),
        "identical_x_vectors": len(x_seen), "groups": len(set(groups)),
        "conflicting_label_groups": sum(len(set(panel.y[np.array(groups) == group])) > 1 for group in set(groups)),
        "group_rule": "connected components of identical selected X or supplied source identifier"}


def stratified_subset(indices, y, size: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if size < 2 or len(indices) < size or len(np.unique(indices)) != len(indices):
        raise ValueError("Require a unique pool with at least the requested support/query size")
    if len(indices) == size:
        return np.sort(indices)
    sampler = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
    chosen, _ = next(sampler.split(np.zeros((len(indices), 1)), np.asarray(y)[indices]))
    return np.sort(indices[chosen])


def make_splits(panel: PanelDataset, dataset_id: int, support_size: int = 128, query_cap: int = 1024) -> dict:
    if support_size < 2 or query_cap < 2:
        raise ValueError("Support/query sizes must retain both classes")
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=61000+dataset_id)
    folds = []
    for fold, (training, heldout) in enumerate(splitter.split(panel.X, panel.y, panel.group_ids)):
        support_seed, query_seed = 62000+100*dataset_id+fold, 63000+100*dataset_id+fold
        support = stratified_subset(training, panel.y, support_size, support_seed)
        query = stratified_subset(heldout, panel.y, min(len(heldout), query_cap), query_seed)
        folds.append({"fold": fold, "support_indices": support.tolist(), "query_indices": query.tolist(),
            "training_indices": training.tolist(), "heldout_indices": heldout.tolist(),
            "support_row_ids": panel.row_ids[support].tolist(), "query_row_ids": panel.row_ids[query].tolist(),
            "support_group_ids": panel.group_ids[support].tolist(), "query_group_ids": panel.group_ids[query].tolist(),
            "support_seed": support_seed, "query_seed": query_seed,
            "support_class_counts": np.bincount(panel.y[support], minlength=2).tolist(),
            "query_class_counts": np.bincount(panel.y[query], minlength=2).tolist()})
    result = {"splitter": "StratifiedGroupKFold", "splitter_seed": 61000+dataset_id,
              "n_splits": 5, "support_size": support_size, "query_cap": query_cap, "folds": folds}
    validate_splits(panel, result)
    return result


def validate_splits(panel: PanelDataset, splits: dict) -> None:
    heldout_all = []
    universe = set(range(len(panel.y)))
    if len(splits["folds"]) != 5:
        raise ValueError("Require five fixed folds")
    for record in splits["folds"]:
        support, query, training, heldout = [record[key] for key in
            ("support_indices", "query_indices", "training_indices", "heldout_indices")]
        for indices in (support, query, training, heldout):
            if len(indices) != len(set(indices)) or not set(indices) <= universe:
                raise ValueError("Invalid or duplicate canonical split indices")
        if set(training) & set(heldout) or set(training) | set(heldout) != universe:
            raise ValueError("Training and heldout partitions must cover disjoint canonical rows")
        if not set(support) <= set(training) or not set(query) <= set(heldout):
            raise ValueError("Support/query rows escaped their frozen partitions")
        if set(panel.group_ids[training]) & set(panel.group_ids[heldout]):
            raise ValueError("Identical-X/source groups leaked across partitions")
        if len(support) != splits["support_size"] or len(query) > splits["query_cap"]:
            raise ValueError("Incorrect support size or query cap")
        if len(set(panel.y[support])) != 2 or len(set(panel.y[query])) != 2:
            raise ValueError("Support and query partitions require both binary classes")
        if record["support_row_ids"] != panel.row_ids[support].tolist() or record["query_row_ids"] != panel.row_ids[query].tolist():
            raise ValueError("Canonical row IDs do not match saved split indices")
        if record["support_group_ids"] != panel.group_ids[support].tolist() or record["query_group_ids"] != panel.group_ids[query].tolist():
            raise ValueError("Group IDs do not match saved split indices")
        heldout_all.extend(heldout)
    if len(heldout_all) != len(universe) or set(heldout_all) != universe:
        raise ValueError("Each canonical row must be held out exactly once")


def load_panel(directory: Path | str) -> PanelDataset:
    directory = Path(directory)
    path = directory if directory.suffix == ".npz" else directory / "dataset.npz"
    with np.load(path, allow_pickle=False) as arrays:
        return PanelDataset(**{name: arrays[name] for name in
            ("X", "y", "row_ids", "group_ids", "native_mask", "multiplicity")})


def parse_uci_csv(content: bytes, metadata: dict, source: dict):
    variables = metadata["variables"]
    features = source.get("selected_features", [v["name"] for v in variables if v["role"] == "Feature"])
    targets = [v["name"] for v in variables if v["role"] == "Target"]
    identifiers = [v["name"] for v in variables if v["role"] == "ID"]
    if len(targets) != 1 or len(features) != source["features"]:
        raise ValueError("Official schema does not match predeclared features/target")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Missing or duplicate CSV headers")
    if not set([*features, *targets, *identifiers]) <= set(reader.fieldnames):
        raise ValueError("CSV header differs from official metadata")
    rows = list(reader)
    if len(rows) != source["rows"]:
        raise ValueError("Official row count differs from fixed panel")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed official CSV row")
    def number(value):
        return np.nan if value is None or value.strip() in {"", "?", "NA", "NaN", "nan"} else float(value)
    def label(value):
        value = value.strip()
        if value not in {source["positive"], source["negative"]}:
            # Official integer labels may be serialized as decimal integers.
            try:
                numeric = float(value)
                value = str(int(numeric)) if np.isfinite(numeric) and numeric.is_integer() else value
            except ValueError:
                pass
        if value not in {source["positive"], source["negative"]}:
            raise ValueError("Unexpected or missing target label")
        return int(value == source["positive"])
    X = np.array([[number(row[feature]) for feature in features] for row in rows], dtype=np.float64)
    y = np.array([label(row[targets[0]]) for row in rows], dtype=np.int64)
    subjects = ["|".join(f"{name}={row[name]}" for name in identifiers) for row in rows] if identifiers else None
    return X, y, features, subjects
