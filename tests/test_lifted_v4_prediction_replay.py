"""Fabricated slicing and guards only; no actual panels/checkpoints/replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "lifted_v4_replay", Path(__file__).resolve().parents[1] / "infra/replay_lifted_v4_predictions.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def test_fixed_selection_and_recursive_numpy_list_slicing_preserve_banks():
    n = 5
    values = np.arange(15).reshape(n, 3)
    bank = np.ones((n, 3))  # Deliberate dimension coincidence: never slice banks.
    static = np.arange(6).reshape(2, 3)
    panel = dict(pool=dict(n=n, sx=values, anchors={1: dict(A=values)},
                           keys=[f"episode{i}" for i in range(n)], static=static, scalar=np.array(7)),
                 banks={2: bank}, conds={0: [[i, i + 1] for i in range(n)]})
    sliced, indices = r.slice_panel(panel)
    assert indices == (0, 2, 4) and sliced["pool"]["n"] == 3 and panel["pool"]["n"] == 5
    np.testing.assert_array_equal(sliced["pool"]["sx"], values[[0, 2, 4]])
    np.testing.assert_array_equal(sliced["pool"]["anchors"][1]["A"], values[[0, 2, 4]])
    assert sliced["pool"]["keys"] == ["episode0", "episode2", "episode4"]
    assert sliced["conds"][0] == [[0, 1], [2, 3], [4, 5]]
    assert sliced["pool"]["static"] is static and sliced["banks"] is panel["banks"]
    assert r.selected_indices(128) == (0, 64, 127)
    with pytest.raises(r.ReplayError):
        r.selected_indices(2)


def test_nested_torch_slicing_preserves_type_and_cpu():
    torch = pytest.importorskip("torch")
    tensor = torch.arange(24).reshape(4, 2, 3)
    sliced, indices = r.slice_panel(dict(pool=dict(n=4, qx=tensor, anchors={1: dict(B=tensor)}),
                                         conds={6: tensor}))
    assert indices == (0, 2, 3)
    assert isinstance(sliced["pool"]["qx"], torch.Tensor)
    assert sliced["pool"]["qx"].device.type == "cpu"
    assert torch.equal(sliced["pool"]["anchors"][1]["B"], tensor[[0, 2, 3]])
    assert torch.equal(sliced["conds"][6], tensor[[0, 2, 3]])


def test_fixed_tolerances_and_mismatch_accounting():
    assert (r.ATOL, r.RTOL) == (2e-4, 2e-5)
    saved = np.array([0., 100.])
    tolerance = 2e-4 + 2e-5 * np.abs(saved)
    assert r.compare_array(saved + .9 * tolerance, saved)["matched"]
    result = r.compare_array(saved + np.array([1.1, .9]) * tolerance, saved)
    assert not result["matched"] and result["mismatches"] == 1 and result["values"] == 2
    with pytest.raises(r.ReplayError, match="shape"):
        r.compare_array(np.ones((2, 1)), saved)
    with pytest.raises(r.ReplayError, match="nonfinite"):
        r.compare_array(np.array([np.nan, 100.]), saved)


def test_full_verifier_failure_precedes_source_import_or_panel_access(tmp_path, monkeypatch):
    calls = []

    def reject(*args):
        calls.append("verify")
        raise ValueError("Fabricated incomplete bundle")

    def forbidden(*args):
        pytest.fail("Incomplete bundle reached source import or replay")

    monkeypatch.setattr(r, "import_verifier", lambda: SimpleNamespace(
        preflight=lambda *args: calls.append("preflight"), verify=reject))
    monkeypatch.setattr(r, "capture_bindings", lambda *args: calls.append("freeze") or {})
    monkeypatch.setattr(r, "check_source", forbidden)
    monkeypatch.setattr(r, "import_scientific_source", forbidden)
    with pytest.raises(ValueError, match="incomplete"):
        r.run_replay(tmp_path / "results", tmp_path / "panels", tmp_path / "source")
    assert calls == ["preflight", "freeze", "verify"]


def test_frozen_source_hash_and_inventory_guards_without_import(tmp_path):
    source = tmp_path / "source"
    science = source / "experiments/lifted_cavity"
    science.mkdir(parents=True)
    (source / "MIRA_SOURCE_COMMIT.txt").write_text(r.SOURCE_COMMIT)
    hashes = {}
    for name in ("train.py", "evaluate2.py", "realdata.py", "pfn.py", "lct.py"):
        path = science / name
        path.write_bytes(b"# fabricated source, never imported\n")
        hashes[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    expected = dict(source_commit=r.SOURCE_COMMIT, source=hashes)
    assert r.check_source(source, expected, digest) == science
    (science / "extra.py").write_text("# unexpected")
    with pytest.raises(r.ReplayError, match="inventory"):
        r.check_source(source, expected, digest)
    (science / "extra.py").unlink()
    (science / "pfn.py").write_text("# changed")
    with pytest.raises(r.ReplayError, match="hash"):
        r.check_source(source, expected, digest)


def test_report_cannot_modify_input_folders_or_overwrite_existing(tmp_path):
    root, panels, source = (tmp_path / name for name in ("results", "panels", "source"))
    for protected in (root, panels, source):
        with pytest.raises(r.ReplayError, match="outside immutable"):
            r.validate_output(protected / "report.json", root, panels, source)
    output = tmp_path / "report.json"
    assert r.validate_output(output, root, panels, source) == output
    output.write_text("preserve")
    with pytest.raises(r.ReplayError, match="already exists"):
        r.validate_output(output, root, panels, source)


def test_fixed_watchdog_kills_once_without_retry(tmp_path, monkeypatch):
    calls = []

    class Process:
        def communicate(self, timeout=None):
            calls.append(("communicate", timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired("fabricated", timeout)
            return "", ""

        def kill(self):
            calls.append(("kill",))

    def popen(command, **kwargs):
        calls.append(("spawn",))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        return Process()

    monkeypatch.setattr(r.subprocess, "Popen", popen)
    report = r.bounded_replay(tmp_path / "results", tmp_path / "panels", tmp_path / "source")
    assert report["status"] == "timed_out" and not report["coverage_complete"]
    assert report["automatic_retries"] == 0 and r.WALL_SECONDS == 900
    assert calls == [("spawn",), ("communicate", 900), ("kill",), ("communicate", None)]


@pytest.fixture
def fabricated_bindings(tmp_path):
    root, panels = tmp_path / "results", tmp_path / "panels"
    root.mkdir(); panels.mkdir()
    run = root / "runs/pfn_L_s1"
    run.mkdir(parents=True)
    score = run / "cells_fixture.npz"
    for path in (run / "model.pt", run / "train.json", score, panels / "fixture.pt", panels / "SHA256SUMS"):
        path.write_bytes(f"fabricated bytes: {path.name}".encode())
    for name in r.BUNDLE_HASH_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(fabricated=name)))
    weight_name = "tabpfn/tabpfn-v2-regressor.ckpt"
    weight = root / "cache" / weight_name
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fabricated pretrained weight")
    (root / "pretrained_weights.json").write_text(json.dumps(dict(
        package="9.1.0", version="v2", n_estimators=8, random_state=0)))
    identity = tmp_path / "identity.json"
    identity.write_bytes(b"fabricated bound identity")
    verifier = SimpleNamespace(
        TAGS=("fixture",), PRETRAINED_WEIGHT_PATH=weight_name,
        layout=lambda _: ([(run, "pfn_L", 1, None)], [(score, "fixture", r.METRICS)]),
        digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        read_json=lambda path: json.loads(path.read_text()))
    return root, panels, identity, verifier


def test_binding_schema_carries_actual_bytes_without_outcomes(fabricated_bindings):
    root, panels, identity, verifier = fabricated_bindings
    bindings = r.capture_bindings(*fabricated_bindings)
    assert r.REPLAY_SCHEMA == 1
    assert set(bindings) == {"checkpoint_hashes", "panel_hashes", "pretrained_weights", "score_hashes",
                             "bundle_hashes", "identity_sha256", "panel_manifest_sha256"}
    assert bindings["checkpoint_hashes"]["runs/pfn_L_s1"]["model.pt"] == verifier.digest(root / "runs/pfn_L_s1/model.pt")
    assert bindings["panel_hashes"] == {"fixture.pt": verifier.digest(panels / "fixture.pt")}
    assert bindings["score_hashes"] == {"runs/pfn_L_s1/cells_fixture.npz": verifier.digest(root / "runs/pfn_L_s1/cells_fixture.npz")}
    assert set(bindings["bundle_hashes"]) == set(r.BUNDLE_HASH_FILES)
    assert bindings["identity_sha256"] == verifier.digest(identity)
    assert bindings["panel_manifest_sha256"] == verifier.digest(panels / "SHA256SUMS")
    info = bindings["pretrained_weights"]["files"][verifier.PRETRAINED_WEIGHT_PATH]
    path = root / "cache" / verifier.PRETRAINED_WEIGHT_PATH
    assert info == dict(sha256=verifier.digest(path), bytes=path.stat().st_size)
    verified = {key: bindings[key] for key in ("checkpoint_hashes", "panel_hashes", "pretrained_weights")}
    r.require_verified_bindings(bindings, verified)
    verified["checkpoint_hashes"] = {name.replace("/", "\\"): hashes
                                     for name, hashes in verified["checkpoint_hashes"].items()}
    r.require_verified_bindings(bindings, verified)
    verified["panel_hashes"] = {}
    with pytest.raises(r.ReplayError, match="verifier panel_hashes differ"):
        r.require_verified_bindings(bindings, verified)


@pytest.mark.parametrize("changed", ["checkpoint", "panel", "score", "bundle", "identity", "manifest", "pretrained"])
def test_any_replay_input_mutation_prevents_bound_success(fabricated_bindings, changed):
    root, panels, identity, verifier = fabricated_bindings
    before = r.capture_bindings(*fabricated_bindings)
    paths = dict(checkpoint=root / "runs/pfn_L_s1/model.pt", panel=panels / "fixture.pt",
                 score=root / "runs/pfn_L_s1/cells_fixture.npz", bundle=root / "runs/confirm_v4.json",
                 identity=identity, manifest=panels / "SHA256SUMS",
                 pretrained=root / "cache" / verifier.PRETRAINED_WEIGHT_PATH)
    paths[changed].write_bytes(b"substituted fabricated input")
    after = r.capture_bindings(*fabricated_bindings)
    with pytest.raises(r.ReplayError, match="input bytes changed"):
        r.require_unchanged_bindings(before, after)
