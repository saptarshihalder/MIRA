"""All-seed descriptive point gains, only after complete verification/replay."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def summarize(root, panels, replay_path):
    spec = importlib.util.spec_from_file_location("v4_seed_integration", Path(__file__).with_name("integrate_lifted_v4_verified.py"))
    integration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integration)
    replay, verifier, verified, _, bindings = integration.validate_inputs(root, panels, replay_path)
    _, score_specs = verifier.layout(root)
    arrays = {}
    for path, *_ in score_specs:
        with np.load(path, allow_pickle=False) as archive:
            arrays[path] = {key: archive[key].astype(np.float64) for key in archive.files}
    # Use the hash-verified original cluster identities, not a mutable export.
    weeks = [key.split("|")[0] for tag in verifier.REAL_TAGS
             for key in verifier.load_panel(panels / f"{tag}.pt")["pool"]["keys"]]

    def synthetic(model, seed, tag):
        return arrays[root / "runs" / f"{model}_s{seed}" / f"cells_{tag}.npz"]["k2_nll"].mean(axis=1)

    def real(model, seed, condition):
        return np.concatenate([arrays[(root / "runs/tabpfn_v2" if model == "tabpfn_v2" else
                                       root / "runs_real" / ds / f"{model}_s{seed}_ft") /
                                      f"cells_{tag}.npz"][f"e{condition}_nll"]
                               for ds, tag in zip(verifier.TARGETS, verifier.REAL_TAGS)])

    values = {endpoint: {} for endpoint in verifier.ENDPOINTS}
    for seed in verifier.SEEDS:
        comparisons = [(synthetic("pfn_L", seed, tag), synthetic("lct_L", seed, tag), None)
                       for tag in (verifier.H1, verifier.H3)]
        comparisons += [(real("pfn_L", seed, condition), real("lct_L", seed, condition), weeks)
                        for condition in (0, 6)]
        comparisons += [(real("tabpfn_v2", seed, 0), real("lct_L", seed, 0), weeks)]
        for endpoint, (base, candidate, clusters) in zip(verifier.ENDPOINTS, comparisons):
            values[endpoint][str(seed)] = verifier.paired_interval(base, candidate, clusters)["gain"]
    rows = {}
    for endpoint, gains in values.items():
        x = np.array(list(gains.values()))
        rows[endpoint] = dict(point_gain_by_seed=gains, minimum=float(x.min()), maximum=float(x.max()),
                              sample_sd=float(x.std(ddof=1)), mean_of_seed_gains=float(x.mean()))
        integration.require(np.isclose(x.mean(), verified["endpoints"][endpoint]["gain"], rtol=1e-10, atol=1e-12),
                            f"Seed mean does not recover fixed aggregate gain: {endpoint}")
    replay.require_unchanged_bindings(bindings, replay.capture_bindings(root, panels, replay.DEFAULT_IDENTITY, verifier))
    return dict(scope="Added descriptive audit; no new endpoint, interval, seed selection or training replication",
                seeds=list(verifier.SEEDS), source_commit=replay.SOURCE_COMMIT, bindings=bindings,
                point_gains=rows, aggregate_decisions_unchanged=verified["endpoints"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=REPO / "artifacts/reports/lifted_v4_cpu_replay.json")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/reports/lifted_v4_seed_sensitivity.json")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Seed audit output already exists; preserve it")
    if any(args.out.resolve().is_relative_to(path.resolve()) for path in (args.root, args.panels)):
        raise ValueError("Seed audit must be outside immutable result/panel inputs")
    report = summarize(args.root, args.panels, args.replay)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "bindings"}, indent=2))


if __name__ == "__main__":
    main()
