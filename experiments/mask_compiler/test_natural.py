"""Leakage/integrity tests for native-missingness ingestion; no inference."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prepare_natural", HERE / "prepare_natural.py")
natural = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = natural
spec.loader.exec_module(natural)


def test_official_sources_and_target_exclusions():
    for name, config in natural.CATALOG.items():
        source = natural.ROOT / "artifacts/data/natural_compiler" / name / "source"
        if not (source / "archive.zip").exists():
            pytest.skip("Official cached sources unavailable")
        X, y, subjects, features, _, accepted = natural.parse_sources((source / "archive.zip").read_bytes(), (source / "official.csv").read_bytes(), config)
        assert np.isnan(X).any() and set(y) == {0, 1}
        assert len(features) == (19 if name == "hepatitis" else 20)
        assert len(accepted) == (155 if name == "hepatitis" else 366)
        if name == "horse_colic":
            assert not set(features) & {"hospital_number", "outcome", "surgery", "surgical_lesion", "lesion_site", "lesion_type", "lesion_subtype", "cp_data"}
            assert subjects and len(set(subjects)) < len(subjects)
            assert np.bincount(y).tolist() == [225, 141]
        else:
            assert np.bincount(y).tolist() == [123, 32]


def test_archive_csv_drift_rejected():
    config = natural.CATALOG["hepatitis"]
    source = natural.ROOT / "artifacts/data/natural_compiler/hepatitis/source"
    text = (source / "official.csv").read_text()
    lines = text.splitlines(); values = lines[1].split(","); values[1] = str(float(values[1])+1); lines[1] = ",".join(values)
    with pytest.raises(ValueError, match="differ"):
        natural.parse_sources((source / "archive.zip").read_bytes(), "\n".join(lines).encode(), config)


def test_group_union_and_stable_support_column_selection():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(96, 6)); y = np.arange(96) % 2
    X[:, 0] = np.nan; X[:48, 1:3] = np.nan
    X[2] = X[0]; y[2] = y[0]  # exact XY duplicate
    X[3] = X[0]; y[3] = 1-y[0]  # same X, conflicting outcome
    subjects = [f"case{i}" for i in range(96)]; subjects[5] = subjects[4]
    panel, maps = natural.canonicalize(X, y, 47, subjects)
    assert maps["exact_xy_duplicates_removed"] == 1
    assert panel.group_ids[maps["raw_row_to_canonical"][0]] == panel.group_ids[maps["raw_row_to_canonical"][3]]
    assert panel.group_ids[maps["raw_row_to_canonical"][4]] == panel.group_ids[maps["raw_row_to_canonical"][5]]
    splits = natural.make_natural_splits(panel, 47, 24)
    for fold in splits["folds"]:
        assert fold["selected_columns"][0] == 0
        rates = np.asarray(fold["support_column_missing_rates"])
        assert fold["selected_columns"] == np.lexsort((np.arange(6), -rates))[:4].tolist()
        assert not set(fold["support_group_ids"]) & set(fold["query_group_ids"])
    bad = copy.deepcopy(splits); bad["folds"][0]["selected_columns"] = list(reversed(bad["folds"][0]["selected_columns"]))
    with pytest.raises(ValueError, match="support-only"):
        natural.validate_natural_splits(panel, bad)


def test_frozen_files_and_offline_hash_drift(tmp_path):
    path = tmp_path / "source.csv"
    natural.fixed_file(path, b"native?")
    assert natural.fetch(path, "https://unused.invalid", natural.digest(b"native?"), offline=True) == b"native?"
    with pytest.raises(ValueError, match="Frozen"):
        natural.fixed_file(path, b"changed")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached source"):
        natural.fetch(path, "https://unused.invalid", natural.digest(b"native?"), offline=True)


def test_saved_native_masks_and_leakage_rejected():
    root = natural.ROOT / "artifacts/data/natural_compiler"
    for name in natural.CATALOG:
        directory = root / name
        if not (directory / "dataset.npz").exists():
            pytest.skip("Prepared dataset unavailable")
        panel = natural.load_panel(directory)
        import json
        splits = json.loads((directory / "splits.json").read_text())
        natural.validate_natural_splits(panel, splits)
        assert np.array_equal(np.isnan(panel.X), panel.native_mask.astype(bool))
        bad = copy.deepcopy(splits); fold = bad["folds"][0]
        fold["query_indices"][0] = fold["support_indices"][0]
        with pytest.raises(ValueError):
            natural.validate_natural_splits(panel, bad)
