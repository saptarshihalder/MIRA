"""Descriptive secondary metrics; requires complete verified and replayed v4.

No training, inference, new decisions or replacement of frozen NLL endpoints.
RMSE is the square root of mean squared error across fits, not ensemble error.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def point_metrics(arrays, weeks=None):
    def average(value):
        value = np.asarray(value, dtype=float)
        if value.ndim == 2:
            value = value.mean(axis=1)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("Secondary values must be finite task/episode scores")
        if weeks is None:
            return float(value.mean())
        labels = np.asarray(weeks)
        if labels.shape != value.shape:
            raise ValueError("Week identities do not align with scores")
        return float(np.mean([value[labels == week].mean() for week in sorted(set(weeks))]))

    output = {}
    for metric, value in arrays.items():
        if metric in ("se", "crps") and np.any(value < 0):
            raise ValueError("Negative squared error/CRPS")
        if metric == "cov" and (np.any(value < 0) or np.any(value > 1)):
            raise ValueError("Coverage outside [0,1]")
        mean = average(value)
        output[{"se": "rmse", "cov": "coverage90"}.get(metric, metric)] = (
            float(np.sqrt(mean)) if metric == "se" else mean)
    return output


def summarize(root, panels, replay_path):
    spec = importlib.util.spec_from_file_location("v4_secondary_integration",
            Path(__file__).with_name("integrate_lifted_v4_verified.py"))
    integration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integration)
    replay, verifier, verified, _, bindings = integration.validate_inputs(root, panels, replay_path)
    schemas, weeks = {}, {}
    for tag in verifier.TAGS:
        schemas[tag], weeks[tag] = verifier.panel_schema(tag, verifier.load_panel(panels / f"{tag}.pt"))
    _, score_specs = verifier.layout(root)
    scores = {path: verifier.read_arrays(path, schemas[tag], metrics)
              for path, tag, metrics in score_specs}
    rows = []
    for model in (*verifier.MODELS, "tabpfn_v2"):
        seeds = verifier.SEEDS if model != "tabpfn_v2" else (None,)
        metrics = verifier.GAUSSIAN if model != "tabpfn_v2" else verifier.TABPFN
        for tag in (verifier.H1, verifier.H3):
            for condition in range(4):
                paths = [root / "runs" / (model if seed is None else f"{model}_s{seed}") /
                         f"cells_{tag}.npz" for seed in seeds]
                values = {metric: np.mean([scores[p][f"k{condition}_{metric}"] for p in paths], axis=0, dtype=np.float64)
                          for metric in metrics}
                rows.append(dict(model=model, panel=tag, missing_sensors=condition,
                                 **point_metrics(values)))
        for condition in (0, 3, 6):
            targets, target_weeks = [], []
            for ds, tag in zip(verifier.TARGETS, verifier.REAL_TAGS):
                paths = [(root / "runs/tabpfn_v2" if seed is None else
                          root / "runs_real" / ds / f"{model}_s{seed}_ft") /
                         f"cells_{tag}.npz" for seed in seeds]
                values = {metric: np.mean([scores[p][f"e{condition}_{metric}"] for p in paths], axis=0, dtype=np.float64)
                          for metric in metrics}
                targets.append(values)
                target_weeks.extend(weeks[tag])
                rows.append(dict(model=model, panel=ds, extra_stations_removed=condition,
                                 **point_metrics(values, weeks[tag])))
            pooled = {metric: np.concatenate([values[metric] for values in targets]) for metric in metrics}
            rows.append(dict(model=model, panel="Beijing-new-pooled", extra_stations_removed=condition,
                             **point_metrics(pooled, target_weeks)))
    integration.require(len(rows) == 60, "Incomplete secondary metric table")
    comparisons = [(verifier.H1, "missing_sensors", 2, "pfn_L"),
                   (verifier.H3, "missing_sensors", 2, "pfn_L"),
                   ("Beijing-new-pooled", "extra_stations_removed", 0, "pfn_L"),
                   ("Beijing-new-pooled", "extra_stations_removed", 6, "pfn_L"),
                   ("Beijing-new-pooled", "extra_stations_removed", 0, "tabpfn_v2")]
    for endpoint, (panel, key, condition, baseline) in zip(verifier.ENDPOINTS, comparisons):
        values = {row["model"]: row["nll"] for row in rows if row["panel"] == panel and row.get(key) == condition}
        integration.require(np.isclose(values[baseline] - values["lct_L"], verified["endpoints"][endpoint]["gain"],
                                        rtol=1e-10, atol=1e-12), "Secondary NLL does not recover frozen endpoint")
    replay.require_unchanged_bindings(bindings, replay.capture_bindings(root, panels, replay.DEFAULT_IDENTITY, verifier))
    return dict(scope="Descriptive secondary metrics, added before complete local v4 contrasts; no new gates",
                averaging="Average individual-model score arrays over all three fits, then task or equal-week means; RMSE=sqrt(mean squared error), not ensemble or mean-seed RMSE",
                unavailable="TabPFN archives contain NLL/squared error only; no coverage/CRPS is inferred",
                rows=rows, bindings=bindings, aggregate_decisions_unchanged=verified["endpoints"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=Path("artifacts/reports/lifted_v4_cpu_replay.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/lifted_v4_secondary_metrics.json"))
    args = parser.parse_args()
    if args.out.exists() or any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.root, args.panels)):
        raise ValueError("Output must be new and outside immutable inputs")
    report = summarize(args.root, args.panels, args.replay)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="complete", rows=len(report["rows"]), output=str(args.out))))


if __name__ == "__main__":
    main()
