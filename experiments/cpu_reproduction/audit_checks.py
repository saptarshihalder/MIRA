"""Independent integrity, environment, posterior and leakage audit. CPU only."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
import headroom as h
from tfm_mechanism import transform_pair


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    handoff = HERE / "reference"
    manifest = json.loads((handoff / "SHA256.json").read_text())
    hashes = {}
    for name, expected in manifest.items():
        path = handoff / name
        actual = sha(path) if path.exists() else None
        hashes[name] = {"expected": expected, "actual": actual, "match": actual == expected}
    source_hashes = {}
    for path in sorted((HERE / "source").iterdir()):
        if path.is_file():
            original = handoff / "mira_pilot" / path.name
            source_hashes[path.name] = {"sha256": sha(path), "matches_preserved_original": sha(path) == sha(original)}
    (HERE / "integrity.json").write_text(json.dumps({"handoff_manifest": hashes, "source_copies": source_hashes}, indent=2))

    packages = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "threadpoolctl"]}
    with threadpool_limits(limits=1):
        environment = {"executable": sys.executable, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "packages": packages, "threadpools": threadpool_info(),
            "source_sha256": {k: v["sha256"] for k, v in source_hashes.items()}}
        (HERE / "environment.json").write_text(json.dumps(environment, indent=2))
        checks = {"families": {}}
        for index, family in enumerate(h.FAMILIES):
            rng = np.random.default_rng(915000 + index)
            task = h.make_task(family, rng)
            c = h.sample(task, 32, rng)
            q = h.sample(task, 128, rng)
            baseline = h.predict_methods(c, q, task)
            poisoned = {k: v.copy() for k, v in q.items()}
            poisoned["y"] = 1 - poisoned["y"]
            poisoned["oracle"] = np.linspace(.001, .999, len(q["y"]))
            tested = h.predict_methods(c, poisoned, task)
            diffs = {name: float(np.max(np.abs(p - tested[name])))
                for name, p in baseline.items() if name != "mechanism_oracle"}
            assert all(v == 0 for v in diffs.values()), (family, diffs)
            flipped = {k: v.copy() for k, v in c.items()}
            flipped["y"] = 1 - flipped["y"]
            response = h.predict_methods(flipped, q, task)
            sensitivity = float(np.max(np.abs(baseline["sparse_likelihood_w32"] - response["sparse_likelihood_w32"])))
            assert sensitivity > 1e-6, family
            altered_task = dict(task, gamma=.4, value=not task["value"], degree=1)
            altered = h.predict_methods(c, q, altered_task)
            structural_diffs = {name: float(np.max(np.abs(p - altered[name])))
                for name, p in baseline.items() if name not in {"mechanism_oracle", "finite_prior_reference"}}
            assert all(v == 0 for v in structural_diffs.values())
            transform_diffs = {}
            for mode in ["native", "native_indicators", "native_shuffled", "imputed", "imputed_indicators"]:
                before = transform_pair(c, q, mode, 512)
                after = transform_pair(c, poisoned, mode, 512)
                unchanged = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(before, after))
                assert unchanged
                transform_diffs[mode] = unchanged
            d = task["d"]
            masks = ((np.arange(2**d)[:, None] >> np.arange(d)) & 1)
            normalizer_errors = []
            posterior_errors = []
            min_likelihood = 1.
            for u in (-1, 1):
                f = task["sign"] * np.prod(2*masks[:, task["active"]] - 1, axis=1)
                if task["value"]:
                    f = f*u
                l1, l0 = (1 + .8*f)/(2**d), (1 - .8*f)/(2**d)
                normalizer_errors.extend([abs(float(l1.sum()) - 1), abs(float(l0.sum()) - 1)])
                p0 = expit(.8*u)
                direct = p0*l1/(p0*l1 + (1-p0)*l0)
                formula = expit(.8*u + np.log1p(.8*f) - np.log1p(-.8*f))
                posterior_errors.append(float(np.max(np.abs(direct-formula))))
                min_likelihood = min(min_likelihood, float(l1.min()), float(l0.min()))
            assert max(normalizer_errors) < 1e-12 and max(posterior_errors) < 1e-12 and min_likelihood > 0
            checks["families"][family] = {"query_label_and_oracle_perturbation_max_prediction_change": max(diffs.values()),
                "learner_methods_checked": len(diffs), "context_label_sensitivity": sensitivity,
                "generic_methods_task_metadata_change": max(structural_diffs.values()),
                "transforms_query_label_and_oracle_invariant": transform_diffs,
                "exhaustive_mask_normalization_max_error": max(normalizer_errors),
                "exhaustive_posterior_max_error": max(posterior_errors), "min_mask_likelihood": min_likelihood}
        # Imputation of each query NaN depends only on training values, including all-missing fallback.
        c = {"u": np.array([-1., 1.]), "x": np.array([[np.nan, 2.], [np.nan, 4.]]), "m": np.array([[1, 0], [1, 0]])}
        q = {"u": np.array([1., -1.]), "x": np.array([[np.nan, np.nan], [7., 999.]]), "m": np.array([[1, 1], [0, 0]])}
        xc, xq = transform_pair(c, q, "imputed", 0)
        assert xq[0, 1] == 0 and xq[0, 2] == 3
        altered_q = {k: v.copy() for k, v in q.items()}
        altered_q["x"][1] = [-1000., -2000.]
        ac, aq = transform_pair(c, altered_q, "imputed", 0)
        assert np.array_equal(xc, ac) and np.array_equal(xq[0], aq[0])
        checks["imputation_training_only_and_empty_column_fallback"] = True
        checks["scope"] = "Deterministic source-level perturbation audit; does not certify unexecuted TFM internals. mechanism_oracle intentionally reads query oracle for evaluation."
        (HERE / "leakage_and_posterior_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({"manifest_matches": sum(v["match"] for v in hashes.values()), "manifest_entries": len(hashes),
        "python": environment["python"], "packages": packages, "checked_families": list(checks["families"]),
        "query_label_leakage": False}, indent=2))


if __name__ == "__main__":
    main()
