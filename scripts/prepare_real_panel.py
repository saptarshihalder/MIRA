"""Fetch the fixed six official UCI sources and freeze canonical arrays/splits, without scoring."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import platform
import sys
import urllib.parse
import urllib.request

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mira.panel_data import canonicalize, make_splits, parse_uci_csv  # noqa: E402


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_once(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen artifact differs; preserve it and investigate: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")


def fetch(url: str, path: Path, expected_sha: str | None = None) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "archive.ics.uci.edu":
        raise ValueError("Only official HTTPS UCI sources are allowed")
    if path.exists():
        content = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "MIRA-public-research-data/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if urllib.parse.urlparse(response.url).hostname != "archive.ics.uci.edu":
                raise ValueError("Official download redirected outside UCI")
            content = response.read(16*1024*1024+1)
        if len(content) > 16*1024*1024:
            raise ValueError("Unexpected source size")
    if expected_sha and digest(content) != expected_sha:
        raise ValueError("Official CSV SHA256 differs from the frozen source")
    write_once(path, content)
    return content


def prepare(source: dict, config: dict, data_root: Path, manifest_root: Path) -> dict:
    slug, dataset_id = source["slug"], source["id"]
    raw = data_root / "raw_uci" / slug
    target = data_root / "real_panel" / slug
    metadata_url = f"https://archive.ics.uci.edu/api/dataset?id={dataset_id}"
    csv_url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv"
    metadata_content = fetch(metadata_url, raw / "metadata.json")
    response = json.loads(metadata_content)
    if response["status"] != 200:
        raise ValueError("Official UCI API did not return this dataset")
    metadata = response["data"]
    if metadata["data_url"] != csv_url or metadata["dataset_doi"] != source["doi"] or metadata["num_instances"] != source["rows"]:
        raise ValueError("Official dataset identity differs from fixed catalog")
    page = fetch(source["page"], raw / "source_page.html")
    if b"CC BY 4.0" not in page and b"creativecommons.org/licenses/by/4.0" not in page:
        raise ValueError("Official page did not verify the declared license")
    csv_content = fetch(csv_url, raw / "data.csv", source.get("data_sha256"))
    X, y, features, subjects = parse_uci_csv(csv_content, metadata, source)
    panel, maps = canonicalize(X, y, dataset_id, subjects)
    splits = make_splits(panel, dataset_id)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **asdict(panel))
    artifacts = {"dataset.npz": buffer.getvalue(), "splits.json": json_bytes(splits), "row_maps.json": json_bytes(maps)}
    for name, content in artifacts.items():
        write_once(target / name, content)
    manifest = {
        "schema_version": 1, "dataset_id": dataset_id, "slug": slug,
        "official_name": metadata["name"], "prepared_date": config["verified_date"],
        "source_page": source["page"], "metadata_url": metadata_url, "data_url": csv_url,
        "doi": source["doi"], "citation": source["citation"], "license": config["license"], "license_url": config["license_url"],
        "source_sha256": {"data.csv": digest(csv_content), "metadata.json": digest(metadata_content), "source_page.html": digest(page)},
        "artifact_sha256": {name: digest(content) for name, content in artifacts.items()},
        "raw_rows": len(X), "canonical_rows": len(panel.y), "features": len(features), "feature_names": features,
        "positive_label": source["positive"], "negative_label": source["negative"],
        "class_counts_raw": np.bincount(y, minlength=2).tolist(), "class_counts_canonical": np.bincount(panel.y, minlength=2).tolist(),
        "exact_xy_duplicates_removed": maps["exact_xy_duplicates_removed"], "groups": maps["groups"],
        "conflicting_label_groups": maps["conflicting_label_groups"],
        "native_missing_values_raw": int(np.isnan(X).sum()), "native_missing_values_canonical": int(panel.native_mask.sum()),
        "observed_zeros_raw": int((X == 0).sum()), "observed_zeros_canonical": int((panel.X == 0).sum()),
        "declared_native_missing": metadata["has_missing_values"],
        "native_missing_deviation": bool(np.isnan(X).any() and metadata["has_missing_values"] == "no"),
        "splitter_seed": splits["splitter_seed"], "support_size": 128, "query_cap": 1024,
        "fold_sizes": [{"fold": f["fold"], "support": len(f["support_indices"]), "query": len(f["query_indices"]),
                        "support_class_counts": f["support_class_counts"], "query_class_counts": f["query_class_counts"]} for f in splits["folds"]],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "changes": "Retain declared numeric features, encode declared positive as 1, remove exact selected-X/label duplicates, group identical X and supplied source IDs; no scaling or zero imputation.",
        "limitations": "Unavailable object/acquisition/source groups cannot be reconstructed. Sonar repeated-object views and Spambase personalized collection may remain correlated. Dataset is the aggregation unit.",
        "model_scores_inspected": False,
    }
    write_once(target / "manifest.json", json_bytes(manifest))
    write_once(manifest_root / f"{slug}.json", json_bytes(manifest))
    print(json.dumps({k: manifest[k] for k in ("slug", "raw_rows", "canonical_rows", "features", "exact_xy_duplicates_removed", "groups", "conflicting_label_groups", "native_missing_values_raw")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "configs" / "panel_sources.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts" / "data")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "artifacts" / "manifests" / "panel_data")
    args = parser.parse_args()
    config = json.loads(args.sources.read_text(encoding="utf-8"))
    manifests = [prepare(source, config, args.data_root, args.manifest_root) for source in config["datasets"]]
    summary = {"schema_version": 1, "prepared_date": config["verified_date"], "dataset_count": len(manifests),
        "datasets": [{k: m[k] for k in ("slug", "dataset_id", "raw_rows", "canonical_rows", "features", "exact_xy_duplicates_removed", "groups", "conflicting_label_groups", "fold_sizes")} for m in manifests],
        "model_scores_inspected": False, "data_root": "artifacts/data/real_panel",
        "source_catalog_sha256": digest(args.sources.read_bytes())}
    write_once(args.manifest_root / "summary.json", json_bytes(summary))


if __name__ == "__main__":
    main()
