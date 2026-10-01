from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mira.panel_data import (PanelDataset, canonicalize, load_panel, make_splits,
                             parse_uci_csv, stratified_subset, validate_splits)


def source_fixture():
    rng = np.random.default_rng(721)
    X = rng.normal(size=(200, 3))
    y = np.arange(200) % 2
    X[1] = X[0]  # Identical X, conflicting labels: retain both in the same group.
    X = np.vstack([X, X[0], X[2], X[2]])
    y = np.r_[y, y[0], y[2], y[2]]
    identifiers = [f"subject{i}" for i in range(200)] + ["subject0", "subject2", "subject2"]
    identifiers[3] = identifiers[2]  # Distinct X, same supplied source identity.
    return X, y, identifiers


def test_exact_xy_deduplication_preserves_conflicting_labels_and_subject_groups():
    X, y, subjects = source_fixture()
    panel, maps = canonicalize(X, y, 17, subjects)
    assert len(panel.y) == 200 and maps["exact_xy_duplicates_removed"] == 3
    assert panel.multiplicity[0] == 2 and panel.multiplicity[2] == 3
    assert panel.multiplicity.sum() == len(X)
    assert panel.y[0] != panel.y[1] and panel.group_ids[0] == panel.group_ids[1]
    assert panel.group_ids[2] == panel.group_ids[3]
    assert len(set(panel.group_ids)) == 198 and maps["conflicting_label_groups"] == 2
    assert maps["raw_row_to_canonical"][-3:] == [0, 2, 2]
    assert maps["canonical_raw_row_indices"][2] == [2, 201, 202]


def test_content_ids_and_group_ids_are_stable_under_source_permutation():
    X, y, subjects = source_fixture()
    first, _ = canonicalize(X, y, 17, subjects)
    order = np.random.default_rng(123).permutation(len(X))
    second, _ = canonicalize(X[order], y[order], 17, np.array(subjects)[order])
    lookup = lambda panel: {r: (g, int(m)) for r, g, m in zip(panel.row_ids, panel.group_ids, panel.multiplicity)}
    assert lookup(first) == lookup(second)
    other, _ = canonicalize(X, y, 52, subjects)
    assert not set(first.row_ids) & set(other.row_ids)


def test_signed_zero_and_nan_duplicates_and_native_masks_are_preserved_immutably():
    X = np.array([[-0., np.nan], [0., np.nan], [0., np.nan]])
    panel, _ = canonicalize(X, [0, 0, 1], 94)
    assert panel.multiplicity.tolist() == [2, 1] and panel.group_ids[0] == panel.group_ids[1]
    assert panel.native_mask.tolist() == [[0, 1], [0, 1]]
    assert panel.X[0, 0] == 0 and np.isnan(panel.X[0, 1])
    X[0, 0] = 999
    assert panel.X[0, 0] == 0
    for field in ("X", "y", "row_ids", "group_ids", "native_mask", "multiplicity"):
        with pytest.raises(ValueError):
            getattr(panel, field).flat[0] = getattr(panel, field).flat[0]
    with pytest.raises(FrozenInstanceError):
        panel.X = X


def test_five_frozen_splits_are_stratified_reproducible_and_group_isolated():
    X, y, subjects = source_fixture()
    panel, _ = canonicalize(X, y, 17, subjects)
    splits = make_splits(panel, 17, support_size=128, query_cap=13)
    assert splits == make_splits(panel, 17, support_size=128, query_cap=13)
    assert splits["splitter_seed"] == 61017
    heldout = []
    for record in splits["folds"]:
        support, query = record["support_indices"], record["query_indices"]
        assert len(support) == 128 and len(query) == 13
        assert record["support_seed"] == 63700+record["fold"]
        assert not set(panel.group_ids[support]) & set(panel.group_ids[query])
        assert abs(record["support_class_counts"][0]-64) <= 2
        heldout += record["heldout_indices"]
    assert sorted(heldout) == list(range(len(panel.y)))
    group_changed = panel.group_ids.copy()
    record = splits["folds"][0]
    group_changed[record["heldout_indices"][0]] = panel.group_ids[record["training_indices"][0]]
    changed = PanelDataset(panel.X, panel.y, panel.row_ids, group_changed, panel.native_mask, panel.multiplicity)
    with pytest.raises(ValueError, match="leaked"):
        validate_splits(changed, splits)
    tampered = json.loads(json.dumps(splits))
    tampered["folds"][0]["query_row_ids"][0] = "wrong-id"
    with pytest.raises(ValueError, match="row IDs"):
        validate_splits(panel, tampered)


def test_fixed_heart_features_and_declared_positive_are_selected_before_deduplication():
    feature_names = ["age", "sex", "chest-pain", "rest-bp", "serum-chol", "fasting-blood-sugar",
                     "electrocardiographic", "max-heart-rate", "angina", "oldpeak", "slope", "major-vessels", "thal"]
    selected = ["age", "rest-bp", "serum-chol", "max-heart-rate", "oldpeak"]
    metadata = {"variables": [{"name": name, "role": "Feature"} for name in feature_names]
                             + [{"name": "heart-disease", "role": "Target"}]}
    values = [40, 1, 4, 120, 230, 0, 1, 150, 0, 0, 2, 3, 7]
    content = (",".join([*feature_names, "heart-disease"])+"\n"+
               ",".join(map(str, [*values, 2]))+"\n"+
               ",".join(map(str, [*values, 1]))+"\n").encode()
    source = {"rows": 2, "features": 5, "selected_features": selected, "positive": "2", "negative": "1"}
    X, y, names, subjects = parse_uci_csv(content, metadata, source)
    assert X.tolist() == [[40, 120, 230, 150, 0], [40, 120, 230, 150, 0]]
    assert names == selected and y.tolist() == [1, 0] and subjects is None
    panel, _ = canonicalize(X, y, 145)
    assert len(panel.y) == 2 and len(set(panel.group_ids)) == 1


def test_identifier_and_target_columns_never_enter_numeric_features():
    content = b"ID,value,label\npatient1,0,M\npatient1,2,B\n"
    metadata = {"variables": [{"name": "ID", "role": "ID"}, {"name": "value", "role": "Feature"},
                              {"name": "label", "role": "Target"}]}
    source = {"rows": 2, "features": 1, "positive": "M", "negative": "B"}
    X, y, names, subjects = parse_uci_csv(content, metadata, source)
    assert X.tolist() == [[0], [2]] and y.tolist() == [1, 0] and names == ["value"]
    panel, maps = canonicalize(X, y, 17, subjects)
    assert panel.group_ids[0] == panel.group_ids[1] and maps["canonical_source_identifiers"] == [["ID=patient1"], ["ID=patient1"]]
    with pytest.raises(ValueError, match="Unexpected"):
        parse_uci_csv(content.replace(b",M", b",unknown"), metadata, source)


def test_saved_dataset_load_is_pickle_free_and_arrays_remain_readonly(tmp_path):
    X, y, subjects = source_fixture()
    panel, _ = canonicalize(X, y, 17, subjects)
    np.savez_compressed(tmp_path / "dataset.npz", **{name: getattr(panel, name) for name in panel.__dataclass_fields__})
    loaded = load_panel(tmp_path)
    for name in panel.__dataclass_fields__:
        assert np.array_equal(getattr(loaded, name), getattr(panel, name))
        assert not getattr(loaded, name).flags.writeable


def test_insufficient_support_pool_and_nonbinary_labels_fail():
    with pytest.raises(ValueError, match="requested"):
        stratified_subset(np.arange(10), np.arange(10) % 2, 128, 1)
    with pytest.raises(ValueError, match="Invalid"):
        canonicalize(np.ones((3, 2)), [0, 1, 2], 17)


def test_catalog_has_exact_fixed_datasets_and_heart_selection():
    sources = json.loads((Path(__file__).resolve().parents[1] / "configs" / "panel_sources.json").read_text())
    assert [item["id"] for item in sources["datasets"]] == [17, 52, 267, 94, 151, 145]
    assert [item["positive"] for item in sources["datasets"]] == ["M", "g", "1", "1", "M", "2"]
    assert sources["datasets"][-1]["selected_features"] == ["age", "rest-bp", "serum-chol", "max-heart-rate", "oldpeak"]


def test_prepared_panel_artifact_hashes_ids_and_all_group_splits():
    root = Path(__file__).resolve().parents[1]
    source_path = root / "configs" / "panel_sources.json"
    sources = json.loads(source_path.read_text())
    if not (root / "artifacts" / "data" / "real_panel").exists():
        pytest.skip("Prepared public panel artifacts are absent from this checkout")
    expected_rows = [569, 350, 1348, 4210, 208, 270]
    for source, rows in zip(sources["datasets"], expected_rows):
        folder = root / "artifacts" / "data" / "real_panel" / source["slug"]
        manifest = json.loads((folder / "manifest.json").read_text())
        for filename, expected in manifest["artifact_sha256"].items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected
        assert manifest["source_sha256"]["data.csv"] == source["data_sha256"]
        assert not manifest["model_scores_inspected"] and manifest["native_missing_values_raw"] == 0
        panel = load_panel(folder)
        assert panel.X.shape == (rows, source["features"])
        assert panel.multiplicity.sum() == source["rows"]
        assert np.count_nonzero(panel.X == 0) == manifest["observed_zeros_canonical"]
        assert len({_row_key(row, label) for row, label in zip(panel.X, panel.y)}) == rows
        splits = json.loads((folder / "splits.json").read_text())
        validate_splits(panel, splits)
        assert len(splits["folds"]) == 5 and all(len(f["support_indices"]) == 128 for f in splits["folds"])
        maps = json.loads((folder / "row_maps.json").read_text())
        assert len(maps["raw_row_to_canonical"]) == source["rows"]


def _row_key(row, label):
    return tuple(row.tolist()), int(label)
