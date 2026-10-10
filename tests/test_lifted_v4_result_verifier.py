"""Fabricated inputs only; never open the frozen scientific panels/results."""
import importlib.util
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "v4_verifier", Path(__file__).resolve().parents[1] / "infra/verify_lifted_v4_results.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def test_seed_average_precedes_task_ci_and_requires_all_three():
    # Large between-seed differences must not become extra CI observations.
    base = {1: np.array([[1., 3.], [5., 7.]]),
            2: np.array([[11., 13.], [15., 17.]]),
            3: np.array([[21., 23.], [25., 27.]])}
    new = {1: base[1] - np.array([[3., 3.], [6., 6.]]),
           2: base[2], 3: base[3]}
    averaged = v.seed_average(base)
    np.testing.assert_array_equal(averaged, [[11., 13.], [15., 17.]])
    result = v.paired_interval(averaged.mean(1), v.seed_average(new).mean(1))
    assert result["n"] == 2
    assert result["gain"] == pytest.approx(1.5)
    assert result["lo"] == pytest.approx(1.5 - .98)
    with pytest.raises(v.VerificationError, match="Exactly seeds"):
        v.seed_average({1: base[1], 2: base[2]})


def test_same_week_is_one_cluster_across_pollutants_and_stations():
    # One week has three episodes across pollutants/stations; another has one.
    # Equal-week gain is 6, whereas episode-weighted gain would be 4.5.
    result = v.paired_interval(np.array([0., 3., 6., 9.]), np.zeros(4),
                               ["2016-01-01"] * 3 + ["2016-01-08"])
    assert result["n"] == 2
    assert result["gain"] == pytest.approx(6.)
    assert result["lo"] == pytest.approx(6. - 1.96 * 3.)
    with pytest.raises(v.VerificationError, match="counts"):
        v.paired_interval(np.ones(2), np.zeros(2), ["2016-01-01"])
    with pytest.raises(v.VerificationError, match="cluster identity"):
        v.paired_interval(np.ones(2), np.zeros(2), ["2016-01-01|station", "2016-01-08"])


@pytest.mark.parametrize("problem", ["missing", "shape", "nan", "object", "extra"])
def test_rejects_bad_array_archives(tmp_path, problem):
    arrays = {"e0_nll": np.ones(3), "e0_se": np.zeros(3)}
    if problem == "missing":
        del arrays["e0_se"]
    elif problem == "shape":
        arrays["e0_se"] = np.ones((3, 1))
    elif problem == "nan":
        arrays["e0_nll"][0] = np.nan
    elif problem == "object":
        arrays["e0_nll"] = np.array([1, 2, 3], dtype=object)
    else:
        arrays["episode_ids"] = np.arange(3)
    path = tmp_path / "cells_fabricated.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        v.read_arrays(path, {"e0_nll": (3,), "e0_se": (3,)}, v.TABPFN)


def test_incomplete_inputs_never_load_panels_or_recompute(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An incomplete run reached scientific inputs/arithmetic")

    monkeypatch.setattr(v, "recompute", forbidden)
    with pytest.raises(v.VerificationError, match="incomplete; no endpoint checks"):
        v.verify(tmp_path / "results", tmp_path / "panels", tmp_path / "identity.json", forbidden)


def test_recorded_confirmation_must_match_every_value_and_decision():
    row = dict(gain=.02, lo=.01, hi=.03, n=60, passed=True)
    actual = {key: dict(row) for key in v.ENDPOINTS}
    saved = dict(actual, seeds=[1, 2, 3])
    v.compare_confirmation(actual, saved)
    for field, wrong in (("gain", .021), ("lo", float("nan")), ("n", 61), ("passed", False)):
        bad = json.loads(json.dumps(saved))
        bad[v.ENDPOINTS[0]][field] = wrong
        with pytest.raises(v.VerificationError, match="mismatch"):
            v.compare_confirmation(actual, bad)


def test_checkpoint_hashes_and_ft_completion_are_required(tmp_path):
    run = tmp_path / "pfn_L_s2_ft"
    run.mkdir()
    meta = dict(model="pfn", seed=2, steps=40000, steps_done=40000,
                batch_tasks=32, queries=16, lr=5e-4, warmup=2000, d=128,
                layers=8, heads=8, ff=512, arch=dict(d=128, layers=8, heads=8, ff=512),
                parameters=2119810, device="cuda", all_masks=False, nonlin=.4,
                finetune=dict(dataset="beijing_pm10", steps=2000, seed=11, lr=3e-4,
                              episodes=4000, period=None, device="cuda", init="/runs/pfn_L_s2",
                              dequant_seed=[4041, v.zlib.crc32(b"PM10")]))
    (run / "train.json").write_text(json.dumps(meta))
    (run / "model.pt").write_bytes(b"fabricated checkpoint, not torch weights")

    def bind():
        identity = {name: v.digest(run / name) for name in ("model.pt", "train.json")}
        (run / "checkpoint_identity.json").write_text(json.dumps(identity))

    bind()
    v.check_checkpoint(run, "pfn_L", 2, "beijing_pm10")
    (run / "model.pt").write_bytes(b"replaced checkpoint")
    with pytest.raises(v.VerificationError, match="hash mismatch"):
        v.check_checkpoint(run, "pfn_L", 2, "beijing_pm10")
    bind()
    meta["finetune"]["steps"] = 1999
    (run / "train.json").write_text(json.dumps(meta))
    bind()
    with pytest.raises(v.VerificationError, match="Fine-tuning recipe/completion"):
        v.check_checkpoint(run, "pfn_L", 2, "beijing_pm10")


def test_saved_panel_identity_must_match_bytes_and_bound_manifest(tmp_path):
    root, panels = tmp_path / "results", tmp_path / "panels"
    root.mkdir(); panels.mkdir()
    hashes = {}
    for tag in v.TAGS:
        path = panels / f"{tag}.pt"
        path.write_bytes(f"fabricated panel {tag}".encode())
        hashes[path.name] = v.digest(path)
    manifest = panels / "SHA256SUMS"
    manifest.write_text("".join(f"{sha}  {name}\n" for name, sha in hashes.items()))
    expected = dict(source={"runner": "fabricated"}, source_commit="fixture_commit", panels=hashes,
                    panel_manifest=v.digest(manifest),
                    packages={name: "fixture" for name in ("torch", "numpy", "scipy", "tabpfn")})
    (root / "run_identity.json").write_text(json.dumps(dict(identity=expected, metadata=dict(commit="fixture_commit"))))
    v.check_identity(root, panels, expected)
    (panels / f"{v.H1}.pt").write_bytes(b"substituted fixture")
    with pytest.raises(v.VerificationError, match="panel ID/hash mismatch"):
        v.check_identity(root, panels, expected)


def fabricate_pretrained_provenance(root, expected, monkeypatch):
    """Patch immutable expected hashes ONLY for these fabricated test bytes."""
    path = root / "cache" / v.PRETRAINED_WEIGHT_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fabricated pretrained checkpoint")
    monkeypatch.setattr(v, "EXPECTED_WEIGHT_SHA256", v.digest(path))
    monkeypatch.setattr(v, "EXPECTED_WEIGHT_BYTES", path.stat().st_size)
    weights = dict(package="9.1.0", version="v2", n_estimators=8, random_state=0,
                   files={v.PRETRAINED_WEIGHT_PATH: dict(sha256=v.EXPECTED_WEIGHT_SHA256,
                                                       bytes=v.EXPECTED_WEIGHT_BYTES)})
    (root / "pretrained_weights.json").write_text(json.dumps(weights))
    bound = v.read_json(root / "run_identity.json")["identity"]
    gate = dict(passed=True, pretrained_weights=weights, identity=dict(inputs=expected, bounded_identity=bound))
    gate_path, smoke_path = root / "modal_smoke_gate.json", root / "modal_smoke_runtime.json"
    gate_path.write_text(json.dumps(gate))
    smoke_path.write_text(json.dumps(dict(stage="smoke", status="completed", exit_code=0)))
    monkeypatch.setattr(v, "EXPECTED_SMOKE_GATE_SHA256", v.digest(gate_path))
    monkeypatch.setattr(v, "EXPECTED_SMOKE_RUNTIME_SHA256", v.digest(smoke_path))
    amendment = dict(extension_sha256=v.EXPECTED_EXTENSION_SHA256,
                     original_wrapper_sha256=v.EXPECTED_WRAPPER_SHA256,
                     original_gate_sha256=v.EXPECTED_SMOKE_GATE_SHA256,
                     original_runtime_sha256=v.EXPECTED_SMOKE_RUNTIME_SHA256,
                     original_inputs=expected, scientific_changes=False,
                     run_id="lifted_v4_modal_main_l40s_extended", automatic_retries=0)
    (root / "main_extended_execution_amendment.json").write_text(json.dumps(amendment))
    runtime = dict(stage="main_extended", status="completed", exit_code=0, extension=amendment,
                   original_gate_sha256=v.EXPECTED_SMOKE_GATE_SHA256,
                   confirmation_sha256=v.digest(root / "runs/confirm_v4.json"))
    (root / "modal_main_extended_runtime.json").write_text(json.dumps(runtime))


@pytest.fixture
def fabricated_bundle(tmp_path, monkeypatch):
    root, panels = tmp_path / "results", tmp_path / "panels"
    root.mkdir(); panels.mkdir()
    panel_data, hashes = {}, {}
    for tag in v.TAGS:
        path = panels / f"{tag}.pt"
        path.write_bytes(f"fabricated panel {tag}".encode())
        hashes[path.name] = v.digest(path)
        if tag in (v.H1, v.H3):
            n, p, seed = (256, 5, 20261401) if tag == v.H1 else (128, 16, 20261403)
            panel_data[tag] = dict(pool=dict(n=n),
                                  banks={k: np.ones((min(v.math.comb(p, k), 20), p)) for k in range(4)},
                                  meta=dict(tasks=n, sensors=p, seed=seed, nonlin=.4, support=48))
        else:
            ds = v.TARGETS[v.REAL_TAGS.index(tag)]
            n = 713 if ds == "beijing_o3" else 719
            keys = [f"{v.date(2016, 1, 1) + timedelta(days=7 * (i // 12))}|station{i % 12}" for i in range(n)]
            panel_data[tag] = dict(pool=dict(n=n, keys=keys),
                                  conds={e: np.broadcast_to(1, (n, 48, 11)) for e in (0, 3, 6)},
                                  meta=dict(episodes=n, dataset=ds, seed=4041,
                                            dequant_seed=[4041, v.zlib.crc32(ds.removeprefix("beijing_").upper().encode())]))
    manifest = panels / "SHA256SUMS"
    manifest.write_text("".join(f"{sha}  {name}\n" for name, sha in hashes.items()))
    identity = dict(source={"runner": "fabricated"}, source_commit="fixture_commit", panels=hashes,
                    panel_manifest=v.digest(manifest),
                    wrapper_sha256=v.EXPECTED_WRAPPER_SHA256,
                    packages={name: "fixture" for name in ("torch", "numpy", "scipy", "tabpfn")})
    identity_path = tmp_path / "fixture_identity.json"
    identity_path.write_text(json.dumps(identity))
    (root / "run_identity.json").write_text(json.dumps(dict(identity=identity, metadata=dict(commit="fixture_commit"))))
    checkpoints, scores = v.layout(root)
    for run, model, seed, ds in checkpoints:
        run.mkdir(parents=True)
        arch = dict(d=128, layers=8, heads=8, ff=512)
        if model == "lct_L":
            arch["K"] = 1
        meta = dict(model=model.split("_")[0], seed=seed, steps=40000, steps_done=40000,
                    batch_tasks=32, queries=16, lr=5e-4, warmup=2000, d=128,
                    layers=8, heads=8, ff=512, arch=arch,
                    parameters=2119810 if model == "pfn_L" else 2120326, device="cuda")
        if ds:
            meta["finetune"] = dict(dataset=ds, steps=2000, seed=11, lr=3e-4,
                                    episodes=4000, period=None, device="cuda", init=f"/runs/{model}_s{seed}",
                                    dequant_seed=[4041, v.zlib.crc32(ds.removeprefix("beijing_").upper().encode())])
        (run / "train.json").write_text(json.dumps(meta))
        (run / "model.pt").write_bytes(b"fabricated model bytes")
        (run / "checkpoint_identity.json").write_text(json.dumps(
            {name: v.digest(run / name) for name in ("model.pt", "train.json")}))
    tab = root / "runs/tabpfn_v2"
    tab.mkdir()
    (tab / "train.json").write_text(json.dumps(dict(model="tabpfn_v2", version="v2", package="9.1.0",
                                                   n_estimators=8, random_state=0)))
    for path, tag, metrics in scores:
        schema, _ = v.panel_schema(tag, panel_data[tag])
        if path.parent == tab:
            value = 3.2
        else:
            model = "pfn_L" if "pfn_L" in path.parent.name else "lct_L"
            seed = int(path.parent.name.split("_s")[1][0])
            value = (1. if model == "pfn_L" else .9) + seed
        np.savez(path, **{key: np.full(shape, value) for key, shape in schema.items()
                         if key.rsplit("_", 1)[1] in metrics})
    saved = {key: dict(gain=.3 if i == 4 else .1, lo=.3 if i == 4 else .1,
                       hi=.3 if i == 4 else .1, n=256 if i == 0 else 128 if i == 1 else 60, passed=True)
             for i, key in enumerate(v.ENDPOINTS)}
    (root / "runs/confirm_v4.json").write_text(json.dumps(dict(saved, seeds=[1, 2, 3])))
    fabricate_pretrained_provenance(root, identity, monkeypatch)
    return root, panels, identity_path, lambda path: panel_data[path.stem]


def test_complete_fabricated_bundle_matches_hand_calculated_confirmation(fabricated_bundle):
    report = v.verify(*fabricated_bundle)
    assert report["verified"] and report["checkpoints"] == 24 and report["score_files"] == 35
    assert report["provenance"]["independent_cryptographic_timestamp"] is False
    assert report["pretrained_weights"]["n_estimators"] == 8


def test_bad_last_array_blocks_every_endpoint(fabricated_bundle, monkeypatch):
    root, panels, identity, loader = fabricated_bundle
    path = root / "runs/tabpfn_v2" / f"cells_{v.REAL_TAGS[-1]}.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "e6_se"}
    np.savez(path, **arrays)

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed final array reached endpoint arithmetic")

    monkeypatch.setattr(v, "recompute", forbidden)
    with pytest.raises(v.VerificationError, match="missing/unexpected"):
        v.verify(root, panels, identity, loader)


def test_production_pretrained_identity_is_fixed():
    assert v.EXPECTED_WEIGHT_SHA256 == "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
    assert v.EXPECTED_WEIGHT_BYTES == 44390977


@pytest.mark.parametrize("problem", [None, "pack", "seed1", "recipe", "code", "prior", "confirmation"])
def test_manual_continuation_requires_unchanged_seed1_and_completed_proof(fabricated_bundle, problem):
    root, panels, identity_path, loader = fabricated_bundle
    previous = root / "modal_main_extended_runtime.json"
    old = v.read_json(previous)
    old.update(status="failed", exit_code=1)
    previous.write_text(json.dumps(old))
    log = root / "logs/modal_main_extended.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text("FileNotFoundError: [Errno 2] No such file or directory: 'git'")
    checkpoints, scores = v.layout(root)
    protected = {str((run / "model.pt").relative_to(root)).replace("\\", "/"): v.digest(run / "model.pt")
                 for run, _, seed, _ in checkpoints if seed == 1}
    protected.update({path.relative_to(root).as_posix(): v.digest(path) for path, *_ in scores
                      if "_s1" in path.as_posix() or path.parent.name == "tabpfn_v2"})
    repair = v.read_json(root / "main_extended_execution_amendment.json")
    repair.update(extension_sha256=v.EXPECTED_RESUME_SHA256,
                  previous_runtime_sha256=v.digest(previous), previous_log_sha256=v.digest(log),
                  manual_continuation=True, run_id="lifted_v4_modal_main_l40s_resume",
                  child_seconds=14400, function_seconds=14700, startup_seconds=120,
                  provision_usd=9.5, preserved_seed1_files=protected)
    resumed = dict(stage="main_resume", status="completed", exit_code=0,
                   packaging_preflight_passed=True, preserved_seed1_hashes_unchanged=True,
                   extension=repair, confirmation_sha256=v.digest(root / "runs/confirm_v4.json"))
    if problem == "pack":
        resumed["packaging_preflight_passed"] = False
    elif problem == "seed1":
        repair["preserved_seed1_files"][next(iter(protected))] = "substitution"
    elif problem == "recipe":
        repair["scientific_changes"] = True
    elif problem == "code":
        repair["extension_sha256"] = "unbound repair"
    elif problem == "prior":
        old.update(timed_out=True)
        previous.write_text(json.dumps(old))
    elif problem == "confirmation":
        resumed["confirmation_sha256"] = "substituted confirmation"
    (root / "main_resume_execution_amendment.json").write_text(json.dumps(repair))
    (root / "modal_main_resume_runtime.json").write_text(json.dumps(resumed))
    if problem is None:
        assert v.verify(root, panels, identity_path, loader)["verified"] is True
    else:
        def forbidden(*args, **kwargs):
            pytest.fail("Invalid continuation reached panel loading or endpoints")
        with pytest.raises(v.VerificationError):
            v.verify(root, panels, identity_path, forbidden)


@pytest.mark.parametrize("problem", ["missing_weight", "wrong_hash", "wrong_bytes", "n_estimators",
                                     "filemap", "smoke_weights", "runtime_incomplete", "runtime_proof"])
def test_pretrained_failure_blocks_panel_load_and_endpoints(fabricated_bundle, monkeypatch, problem):
    root, panels, identity, _ = fabricated_bundle
    weight_path = root / "cache" / v.PRETRAINED_WEIGHT_PATH
    if problem == "missing_weight":
        weight_path.unlink()
    elif problem == "wrong_hash":
        weight_path.write_bytes(b"X" * v.EXPECTED_WEIGHT_BYTES)
    elif problem == "wrong_bytes":
        weight_path.write_bytes(b"short")
    elif problem in ("n_estimators", "filemap"):
        path = root / "pretrained_weights.json"
        meta = v.read_json(path)
        if problem == "n_estimators":
            meta["n_estimators"] = 7
        else:
            meta["files"]["extra.ckpt"] = dict(sha256="unexpected", bytes=1)
        path.write_text(json.dumps(meta))
    elif problem == "smoke_weights":
        path = root / "modal_smoke_gate.json"
        gate = v.read_json(path)
        gate["pretrained_weights"]["random_state"] = 1
        path.write_text(json.dumps(gate))
        # Reach the semantic check using a newly bound fabricated gate only.
        monkeypatch.setattr(v, "EXPECTED_SMOKE_GATE_SHA256", v.digest(path))
    else:
        path = root / "modal_main_extended_runtime.json"
        runtime = v.read_json(path)
        if problem == "runtime_incomplete":
            runtime["status"] = "running"
        else:
            runtime["extension"]["extension_sha256"] = "substituted code"
        path.write_text(json.dumps(runtime))

    def forbidden(*args, **kwargs):
        pytest.fail("A pretrained identity/provenance failure reached panel load or endpoints")

    monkeypatch.setattr(v, "recompute", forbidden)
    with pytest.raises(v.VerificationError):
        v.verify(root, panels, identity, forbidden)
