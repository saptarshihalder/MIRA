"""Verify a review export inventory, then regenerate paper tables/figures locally.

This does not train models, run inference, rebuild a PDF or deserialize pickles.
The generator imports torch, but uses exported numeric references because raw
panel caches are deliberately absent. Requires numpy, matplotlib and torch.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


def recompute_endpoints(root):
    import numpy as np
    spec = importlib.util.spec_from_file_location("review_v4_statistics", root / "infra/verify_lifted_v4_results.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    results = root / "results"
    schemas, weeks = {}, {}
    for tag in verifier.TAGS:
        info = json.loads((results / "panels" / f"{tag}.json").read_text(encoding="utf-8"))
        n = info["n"]
        if tag in (verifier.H1, verifier.H3):
            expected_n, sensors = (256, 5) if tag == verifier.H1 else (128, 16)
            verifier.require(n == expected_n, "Synthetic export task count differs")
            schemas[tag] = {f"k{k}_{metric}": (n, min(math.comb(sensors, k), 20))
                            for k in range(4) for metric in verifier.GAUSSIAN}
        else:
            verifier.require(n == (713 if "o3" in tag else 719), "Real export task count differs")
            keys = info["keys"]
            verifier.require(len(keys) == n and len(set(keys)) == n
                             and all(isinstance(key, str) and key.count("|") == 1 for key in keys),
                             "Real export cluster identities differ")
            weeks[tag] = [key.split("|")[0] for key in keys]
            schemas[tag] = {f"e{e}_{metric}": (n,) for e in (0, 3, 6) for metric in verifier.GAUSSIAN}
    _, score_specs = verifier.layout(results)
    arrays = {path: verifier.read_arrays(path, schemas[tag], metrics) for path, tag, metrics in score_specs}
    endpoints = verifier.recompute(results, arrays, weeks)
    verifier.compare_confirmation(endpoints, json.loads((results / "runs/confirm_v4.json").read_text(encoding="utf-8")))
    return endpoints


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("Unsafe review file path")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Review file hash differs: {name}")
    endpoints = recompute_endpoints(root)
    results, out = root / "results", root / "regenerated"
    if out.exists():
        raise ValueError("Regeneration output already exists; preserve it")
    command = [sys.executable, str(root / "experiments/lifted_cavity/make_paper.py"),
               "--runs", str(results / "runs"), "--real", str(results / "runs_real"),
               "--panels", str(results / "panels"), "--cache", str(root / "absent_raw_cache"),
               "--posthoc", str(results / "posthoc"), "--explore", str(results / "explore"),
               "--init", str(results / "init"), "--v1", str(results / "conf_results.json"), "--out", str(out)]
    subprocess.run(command, cwd=root, check=True, timeout=300)
    (out / "v4_endpoints.json").write_text(json.dumps(endpoints, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Verified export hashes, independently recomputed E8-E12 and regenerated tables/figures; no training or inference reproduction asserted.")


if __name__ == "__main__":
    main()
