"""Fail-closed, local-only audit of complete frozen v4 results.

Usage: python infra/verify_lifted_v4_results.py --root LOCAL_RESULTS --panels LOCAL_PANELS

Requires the 24 final model.pt files (the small result zip omits them). No network
access, scorer imports, training, or output writes. Nothing is computed or printed
about endpoints until every required input passes validation. Copied file mtimes
are deliberately ignored: the frozen runner binds hashes before scoring, but its
identity JSON has no independent cryptographic timestamp.
"""
import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
import zlib

import numpy as np

SEEDS = (1, 2, 3)
MODELS = ("pfn_L", "lct_L")
H1 = "v2_s20261401_n256_p5_nl0.4_sr48"
H3 = "v2_s20261403_n128_p16_nl0.4_sr48"
TARGETS = ("beijing_pm10", "beijing_so2", "beijing_o3")
REAL_TAGS = tuple(f"real_{ds}_s4041" for ds in TARGETS)
TAGS = (H1, H3) + REAL_TAGS
GAUSSIAN = ("nll", "se", "cov", "crps")
TABPFN = ("nll", "se")
ENDPOINTS = (
    "E8_H1_k2_lctL_vs_pfnL", "E9_H3_k2_lctL_vs_pfnL",
    "E10_bjnew_nat_lctLft_noninf_pfnLft", "E11_bjnew_plus6_lctLft_vs_pfnLft",
    "E12_bjnew_nat_lctLft_vs_tabpfn",
)
DEFAULT_IDENTITY = Path(__file__).resolve().parents[1] / "artifacts/manifests/bound_source_identity_l40s.json"


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def layout(root):
    """Return every checkpoint and score required by the frozen main stage."""
    checkpoints, scores = [], []
    for model in MODELS:
        for seed in SEEDS:
            run = root / "runs" / f"{model}_s{seed}"
            checkpoints.append((run, model, seed, None))
            scores.extend((run / f"cells_{tag}.npz", tag, GAUSSIAN) for tag in (H1, H3))
            for ds, tag in zip(TARGETS, REAL_TAGS):
                ft = root / "runs_real" / ds / f"{model}_s{seed}_ft"
                checkpoints.append((ft, model, seed, ds))
                scores.append((ft / f"cells_{tag}.npz", tag, GAUSSIAN))
    scores.extend((root / "runs/tabpfn_v2" / f"cells_{tag}.npz", tag, TABPFN) for tag in TAGS)
    return checkpoints, scores


def preflight(root, panels, identity_path):
    """Check *all* presence before reading panels, scores, or confirm_v4.json."""
    checkpoints, scores = layout(root)
    needed = [identity_path, root / "run_identity.json", root / "runs/confirm_v4.json",
              root / "runs/tabpfn_v2/train.json", panels / "SHA256SUMS"]
    needed.extend(panels / f"{tag}.pt" for tag in TAGS)
    needed.extend(path for path, _, _ in scores)
    needed.extend(run / filename for run, *_ in checkpoints
                  for filename in ("model.pt", "train.json", "checkpoint_identity.json"))
    missing = [str(path) for path in needed if not path.is_file() or path.stat().st_size == 0]
    require(not missing, "Required inputs incomplete; no endpoint checks: " + ", ".join(missing))
    require(not any((root / name).exists() for name in ("RUNNING.lock", "MODAL_RUNNING.lock")),
            "Results root is still locked; no endpoint checks")
    return checkpoints, scores


def check_identity(root, panels, expected):
    saved = read_json(root / "run_identity.json")
    bound = saved["identity"]
    for key in ("source", "panels", "panel_manifest"):
        require(bound.get(key) == expected.get(key) and key in expected,
                f"Frozen {key} identity differs")
    package_names = ("torch", "numpy", "scipy", "tabpfn")
    require(bound.get("packages") == {name: expected["packages"][name] for name in package_names},
            "Frozen package identity differs")
    require(saved.get("metadata", {}).get("commit") == expected["source_commit"],
            "Frozen source commit differs")
    require(digest(panels / "SHA256SUMS") == expected["panel_manifest"], "Panel manifest hash differs")
    manifest = {}
    for line in (panels / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split()
            name = name.lstrip("*")
            require(name not in manifest, "Duplicate saved panel ID in SHA256SUMS")
            manifest[name] = sha
    for tag in TAGS:
        name = f"{tag}.pt"
        sha = expected["panels"].get(name)
        require(isinstance(sha, str) and manifest.get(name) == sha and digest(panels / name) == sha,
                f"Frozen panel ID/hash mismatch: {name}")


def check_checkpoint(run, model, seed, dataset):
    require(not any((run / name).exists() for name in ("ckpt.pt", "ckpt.prev.pt")),
            f"Unfinished checkpoint: {run}")
    meta = read_json(run / "train.json")
    recipe = dict(model=model.split("_")[0], seed=seed, steps=40000, steps_done=40000,
                  batch_tasks=32, queries=16, lr=5e-4, warmup=2000,
                  d=128, layers=8, heads=8, ff=512)
    require(all(meta.get(key) == value for key, value in recipe.items()),
            f"Training recipe/completion/seed differs: {run}")
    arch = dict(d=128, layers=8, heads=8, ff=512)
    if model == "lct_L":
        arch["K"] = 1
    require(meta.get("arch") == arch, f"Architecture differs: {run}")
    require(meta.get("parameters") == (2119810 if model == "pfn_L" else 2120326),
            f"Parameter count differs: {run}")
    require(meta.get("device") == "cuda", f"Training device differs: {run}")
    require(meta.get("all_masks", False) is False and meta.get("nonlin", .4) == .4,
            f"Source training masks/prior differ: {run}")
    if dataset is None:
        require("finetune" not in meta, f"Pretraining metadata contains fine-tuning: {run}")
    else:
        ft = meta.get("finetune", {})
        recipe_ft = dict(dataset=dataset, steps=2000, seed=11, lr=3e-4, episodes=4000,
                         period=None, device="cuda")
        require(all(ft.get(key) == value for key, value in recipe_ft.items()),
                f"Fine-tuning recipe/completion differs: {run}")
        init_name = str(ft.get("init", "")).replace("\\", "/").rstrip("/").split("/")[-1]
        require(init_name == f"{model}_s{seed}", f"Fine-tuning initialization differs: {run}")
        pollutant = dataset.removeprefix("beijing_").upper()
        require(ft.get("dequant_seed") == [4041, zlib.crc32(pollutant.encode())],
                f"Fine-tuning dequantization differs: {run}")
    identity = read_json(run / "checkpoint_identity.json")
    actual = {name: digest(run / name) for name in ("model.pt", "train.json")}
    require(identity == actual, f"Final checkpoint/metadata hash mismatch: {run}")
    return actual


def load_panel(path):
    # Called only after global presence and checkpoint/identity checks.
    import torch
    return torch.load(path, weights_only=False, map_location="cpu")


def panel_schema(tag, panel):
    """Extract structure and saved episode IDs; never use target/reference values."""
    pool, meta = panel["pool"], panel["meta"]
    n = pool["n"]
    require(isinstance(n, int) and not isinstance(n, bool) and n > 1, f"Invalid panel count: {tag}")
    if tag in (H1, H3):
        tasks, sensors, seed = (256, 5, 20261401) if tag == H1 else (128, 16, 20261403)
        require(n == tasks and all(meta.get(k) == v for k, v in
                dict(tasks=tasks, sensors=sensors, seed=seed, nonlin=.4, support=48).items()),
                f"Synthetic panel metadata differs: {tag}")
        expected_banks = {k: (min(math.comb(sensors, k), 20), sensors) for k in range(4)}
        require(set(panel["banks"]) == set(expected_banks) and all(
                tuple(np.shape(panel["banks"][k])) == shape for k, shape in expected_banks.items()),
                f"Synthetic mask-bank dimensions differ: {tag}")
        return {f"k{k}_{metric}": (n, shape[0]) for k, shape in expected_banks.items()
                for metric in GAUSSIAN}, None
    ds = TARGETS[REAL_TAGS.index(tag)]
    expected_n = 713 if ds == "beijing_o3" else 719
    pollutant = ds.removeprefix("beijing_").upper()
    require(n == expected_n and meta.get("episodes") == n and meta.get("dataset") == ds
            and meta.get("seed") == 4041
            and meta.get("dequant_seed") == [4041, zlib.crc32(pollutant.encode())],
            f"Real panel metadata differs: {tag}")
    require(set(panel["conds"]) == {0, 3, 6}, f"Real panel conditions differ: {tag}")
    require(all(tuple(np.shape(mask)) == (n, 48, 11) for mask in panel["conds"].values()),
            f"Real condition-mask dimensions differ: {tag}")
    keys = pool.get("keys", [])
    require(len(keys) == n and len(set(keys)) == n, f"Missing/duplicate saved episode IDs: {tag}")
    weeks = []
    for key in keys:
        require(isinstance(key, str) and key.count("|") == 1, f"Malformed saved episode ID: {tag}")
        week, station = key.split("|")
        require(station and date(2016, 1, 1) <= date.fromisoformat(week) < date(2017, 3, 1),
                f"Saved episode ID outside test period: {tag}")
        weeks.append(week)
    require(len(set(weeks)) > 1, f"Insufficient week clusters: {tag}")
    return {f"e{e}_{metric}": (n,) for e in (0, 3, 6) for metric in GAUSSIAN}, weeks


def read_arrays(path, expected, metrics):
    expected = {key: shape for key, shape in expected.items() if key.rsplit("_", 1)[1] in metrics}
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)) and set(archive.files) == set(expected),
                f"Score arrays missing/unexpected: {path}")
        arrays = {key: archive[key] for key in expected}
    for key, value in arrays.items():
        require(value.shape == expected[key], f"Wrong score shape: {path.name}/{key}")
        require(value.dtype.kind in "fiu" and np.isfinite(value).all(),
                f"Nonfinite/nonreal scores: {path.name}/{key}")
    return arrays


def seed_average(scores):
    require(set(scores) == set(SEEDS), "Exactly seeds 1, 2, 3 are required")
    arrays = [np.asarray(scores[s], dtype=np.float64) for s in SEEDS]
    require(all(a.shape == arrays[0].shape and np.isfinite(a).all() for a in arrays),
            "Seed arrays are not aligned and finite")
    averaged = np.stack(arrays, axis=0).mean(axis=0)
    require(np.isfinite(averaged).all(), "Nonfinite seed averages")
    return averaged


def paired_interval(base, candidate, weeks=None, noninferiority=False):
    base, candidate = np.asarray(base), np.asarray(candidate)
    require(base.ndim == 1 and base.shape == candidate.shape and np.isfinite(base).all()
            and np.isfinite(candidate).all(), "Paired arrays are not aligned and finite")
    gains = base - candidate
    require(np.isfinite(gains).all(), "Nonfinite paired differences")
    if weeks is not None:
        require(len(weeks) == len(gains), "Week/episode counts differ")
        # The same date across pollutants/stations is ONE cluster. Average within
        # each week, then equally weight weeks (not episodes or pollutants).
        by_week = {}
        for week, gain in zip(weeks, gains):
            require(isinstance(week, str) and week and "|" not in week, "Invalid week cluster identity")
            by_week.setdefault(week, []).append(gain)
        gains = np.array([np.mean(by_week[week]) for week in sorted(by_week)])
    require(len(gains) > 1, "At least two paired tasks/week clusters are required")
    mean = float(np.mean(gains))
    radius = 1.96 * float(np.std(gains, ddof=1)) / math.sqrt(len(gains))
    lo, hi = mean - radius, mean + radius
    require(all(math.isfinite(value) for value in (mean, lo, hi)), "Nonfinite endpoint interval")
    return dict(gain=mean, lo=lo, hi=hi, n=len(gains),
                passed=bool(lo > -.02 if noninferiority else mean >= .01 and lo > 0))


def recompute(root, arrays, weeks):
    def synthetic(model, tag):
        values = {s: arrays[root / "runs" / f"{model}_s{s}" / f"cells_{tag}.npz"]["k2_nll"] for s in SEEDS}
        return seed_average(values).mean(axis=1)

    def real(model, condition):
        parts = []
        for ds, tag in zip(TARGETS, REAL_TAGS):
            if model == "tabpfn_v2":
                value = arrays[root / "runs/tabpfn_v2" / f"cells_{tag}.npz"][f"e{condition}_nll"]
            else:
                value = seed_average({s: arrays[root / "runs_real" / ds / f"{model}_s{s}_ft" /
                                               f"cells_{tag}.npz"][f"e{condition}_nll"] for s in SEEDS})
            parts.append(value)
        return np.concatenate(parts)

    cluster_ids = [week for tag in REAL_TAGS for week in weeks[tag]]
    return {
        ENDPOINTS[0]: paired_interval(synthetic("pfn_L", H1), synthetic("lct_L", H1)),
        ENDPOINTS[1]: paired_interval(synthetic("pfn_L", H3), synthetic("lct_L", H3)),
        ENDPOINTS[2]: paired_interval(real("pfn_L", 0), real("lct_L", 0), cluster_ids, True),
        ENDPOINTS[3]: paired_interval(real("pfn_L", 6), real("lct_L", 6), cluster_ids),
        ENDPOINTS[4]: paired_interval(real("tabpfn_v2", 0), real("lct_L", 0), cluster_ids),
    }


def compare_confirmation(actual, saved):
    require(set(saved) == set(ENDPOINTS) | {"seeds"} and saved.get("seeds") == list(SEEDS),
            "Confirmation endpoint IDs/seeds differ")
    for name, calculated in actual.items():
        recorded = saved[name]
        require(isinstance(recorded, dict) and set(recorded) == set(calculated), f"Confirmation schema differs: {name}")
        for field in ("gain", "lo", "hi"):
            value = recorded[field]
            require(type(value) in (int, float) and math.isfinite(value)
                    and math.isclose(value, calculated[field], rel_tol=1e-10, abs_tol=1e-12),
                    f"Confirmation value mismatch: {name}/{field}")
        require(type(recorded["n"]) is int and recorded["n"] == calculated["n"]
                and type(recorded["passed"]) is bool and recorded["passed"] == calculated["passed"],
                f"Confirmation count/pass mismatch: {name}")


def verify(root, panels, identity_path=DEFAULT_IDENTITY, panel_loader=load_panel):
    root, panels, identity_path = Path(root), Path(panels), Path(identity_path)
    checkpoints, score_specs = preflight(root, panels, identity_path)
    expected = read_json(identity_path)
    check_identity(root, panels, expected)
    hashes = {str(run.relative_to(root)): check_checkpoint(run, model, seed, ds)
              for run, model, seed, ds in checkpoints}
    tab = read_json(root / "runs/tabpfn_v2/train.json")
    require(all(tab.get(k) == v for k, v in dict(model="tabpfn_v2", version="v2", package="9.1.0",
                                              n_estimators=8, random_state=0).items()), "TabPFN identity differs")
    schemas, weeks = {}, {}
    for tag in TAGS:
        schemas[tag], weeks[tag] = panel_schema(tag, panel_loader(panels / f"{tag}.pt"))
    arrays = {path: read_arrays(path, schemas[tag], metrics) for path, tag, metrics in score_specs}
    # Global barrier: ALL checkpoint, identity, panel, and array checks precede
    # any endpoint arithmetic or reading the saved endpoint JSON.
    endpoints = recompute(root, arrays, weeks)
    compare_confirmation(endpoints, read_json(root / "runs/confirm_v4.json"))
    return dict(verified=True, seeds=list(SEEDS), checkpoints=len(hashes), score_files=len(arrays),
                endpoints=endpoints, checkpoint_hashes=hashes,
                panel_hashes={f"{tag}.pt": expected["panels"][f"{tag}.pt"] for tag in TAGS},
                provenance=dict(pre_score_binding="Frozen runner binds final hashes before scoring; hashes match",
                                independent_cryptographic_timestamp=False,
                                limitation="Source/control-flow provenance is not independent execution-time attestation; copied mtimes ignored",
                                score_binding="Archives have no embedded panel/checkpoint hashes; binding relies on frozen paths and runner provenance"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    args = parser.parse_args(argv)
    try:
        report = verify(args.root, args.panels, args.identity)
    except (ValueError, OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        print(json.dumps(dict(verified=False, reason=str(error), endpoints_disclosed=False)), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
