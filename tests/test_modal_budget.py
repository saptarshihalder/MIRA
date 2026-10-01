"""Local-only Modal wrapper safety checks; the Modal SDK and remote calls are stubbed."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import zipfile

import pytest


class Chain:
    def __getattr__(self, name):
        return lambda *args, **kwargs: self


@pytest.fixture
def wrapper(monkeypatch):
    def decorate(function):
        def forbidden(*args, **kwargs):
            raise AssertionError("A safety test attempted a remote call")
        function.remote = forbidden
        return function
    app = SimpleNamespace(function=lambda **kwargs: decorate,
                          local_entrypoint=lambda: lambda function: function)
    fake = SimpleNamespace(App=lambda *args: app,
                           Image=SimpleNamespace(debian_slim=lambda **kwargs: Chain()),
                           Volume=SimpleNamespace(from_name=lambda *args, **kwargs: SimpleNamespace(commit=lambda: None)))
    monkeypatch.setitem(sys.modules, "modal", fake)
    path = Path(__file__).resolve().parents[1] / "infra" / "modal_mechanism.py"
    spec = importlib.util.spec_from_file_location("mira_modal_safety", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reservations_include_failed_and_crashed_calls_without_reusing_id(wrapper, tmp_path):
    ledger = tmp_path / "ledger.json"
    budget = {"total_cap_usd": 4}
    first = wrapper.reserve_call(ledger, budget, "a", {})
    assert first["reserved_usd"] == .5 and first["status"] == "reserved"
    wrapper.finish_call(ledger, "a", {"status": "failed"})
    wrapper.reserve_call(ledger, budget, "b", {})  # Simulates a crash before finalization.
    with pytest.raises(ValueError, match="already reserved"):
        wrapper.reserve_call(ledger, budget, "b", {})
    with pytest.raises(ValueError, match="Compute cap"):
        wrapper.reserve_call(ledger, budget, "c", {})
    assert [entry["status"] for entry in json.loads(ledger.read_text())] == ["failed", "reserved"]


def test_atomic_write_failure_preserves_prior_ledger(wrapper, tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    ledger.write_text('[{"run_id":"saved","reserved_usd":0.5}]')
    saved = ledger.read_bytes()
    def fail(*args):
        raise OSError("simulated interrupted replacement")
    monkeypatch.setattr(wrapper.os, "replace", fail)
    with pytest.raises(OSError):
        wrapper.reserve_call(ledger, {"total_cap_usd": 26}, "new", {})
    assert ledger.read_bytes() == saved
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_reservations_cannot_overspend_or_overwrite(wrapper, tmp_path):
    ledger = tmp_path / "ledger.json"
    results = []
    barrier = threading.Barrier(2)
    def attempt(name):
        barrier.wait()
        try:
            wrapper.reserve_call(ledger, {"total_cap_usd": 3.5}, name, {})
            results.append("reserved")
        except ValueError:
            results.append("blocked")
    workers = [threading.Thread(target=attempt, args=(name,)) for name in ["a", "b"]]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
    assert sorted(results) == ["blocked", "reserved"]
    assert len(json.loads(ledger.read_text())) == 1


def test_finalize_reloads_preserving_other_reservations(wrapper, tmp_path):
    ledger = tmp_path / "ledger.json"
    budget = {"total_cap_usd": 26}
    wrapper.reserve_call(ledger, budget, "a", {})
    wrapper.reserve_call(ledger, budget, "b", {})
    wrapper.finish_call(ledger, "a", {"status": "completed"})
    assert [e["status"] for e in json.loads(ledger.read_text())] == ["completed", "reserved"]


@pytest.mark.parametrize("argv", [
    ["--protocol", "confirmation"], ["--seed-start", "50000"],
    ["--seed-start", "49999", "--seeds", "2"], ["--models", "tabpfn:unversioned"],
    ["--out", "bad"], ["--out=bad"], ["--device", "cpu"], ["--checkpoint-dirs", "bad"],
    ["--unknown-option", "bad"], ["--seed-st", "40000"], ["--gammas", "nan"],
    ["--missing-rate", "0"], ["--collision-probability", "1.1"],
    ["--value-distribution", "quantized", "--quantization-step", "inf"],
])
def test_unsafe_or_invalid_core_args_rejected_before_paid_calls(wrapper, argv):
    with pytest.raises(ValueError):
        wrapper.validate_core_argv(argv)


def test_core_args_preserved_and_seed_boundary_accepted(wrapper):
    argv = ["--models", "xgboost", "--gammas", "0", ".9", "--seed-start", "49999", "--seeds", "1"]
    saved = argv.copy()
    configuration = wrapper.validate_core_argv(argv)
    assert configuration["seeds"] == [49999] and argv == saved


def test_matrix_selection_and_phase_exclusion(wrapper, tmp_path, monkeypatch):
    config = tmp_path / "matrix.json"
    selected = ["--models", "xgboost", "--seed-start", "40012", "--seeds", "1"]
    config.write_text(json.dumps({"entries": [{"name": "one", "argv": selected},
                                             {"name": "two", "argv": ["--models", "xgboost"]}]}))
    calls = []
    monkeypatch.setattr(wrapper, "dispatch", lambda *args: calls.append(args))
    wrapper.main(run_id="development", matrix=str(config), entry="one")
    assert len(calls) == 1 and calls[0][:3] == ("development_one", selected, "core")
    with pytest.raises(ValueError, match="mutually exclusive"):
        wrapper.main(matrix=str(config), phase="smoke")
    with pytest.raises(ValueError, match="does not exist"):
        wrapper.main(matrix=str(config), entry="absent")
    with pytest.raises(ValueError, match="requires"):
        wrapper.main(entry="one")


def test_default_smoke_backwards_compatible(wrapper, monkeypatch):
    calls = []
    monkeypatch.setattr(wrapper, "dispatch", lambda *args: calls.append(args))
    wrapper.main()
    assert calls[0][0] == "smoke_tabpfn_v2" and calls[0][2] == "pilot"
    assert calls[0][1][0:2] == ["--models", "tabpfn:v2"]


def test_day3_catalog_validates_current_cli_and_preserves_new_arguments(wrapper):
    path = Path(__file__).resolve().parents[1] / "configs" / "day3_matrix.json"
    entries = wrapper.load_matrix(path)
    assert len(entries) == 7
    for entry in entries:
        saved = entry["argv"].copy()
        configuration = wrapper.validate_core_argv(entry["argv"])
        assert configuration["protocol"] == "development" and len(configuration["seeds"]) == 3
        assert entry["argv"] == saved
    controls = entries[:2]
    assert all("native" in entry["argv"] for entry in controls)
    assert all(entry["argv"][entry["argv"].index("--width-control-columns")+1] == "16" for entry in controls)
    partial = next(entry for entry in entries if entry["name"] == "partial_collision_010")
    assert wrapper.validate_core_argv(partial["argv"])["collision_probability"] == .1
    low = next(entry for entry in entries if entry["name"] == "low_missingness")
    assert wrapper.validate_core_argv(low["argv"])["missing_rate"] == .1


def test_warm_container_calls_have_unique_paths_and_preserve_timeout_artifacts(wrapper, tmp_path, monkeypatch):
    outputs = []
    real_mkdtemp = wrapper.tempfile.mkdtemp
    monkeypatch.setattr(wrapper.tempfile, "mkdtemp", lambda **kwargs: real_mkdtemp(dir=tmp_path, **kwargs))
    def timeout(command, **kwargs):
        out = Path(command[-1])
        outputs.append(out)
        (out / "partial.json").write_text(json.dumps({"invocation": len(outputs)}))
        assert kwargs["timeout"] == 780
        raise subprocess.TimeoutExpired(command, 780)
    monkeypatch.setattr(subprocess, "run", timeout)
    first = wrapper.run_pilot(["--models", "xgboost"], "core")
    second = wrapper.run_pilot(["--models", "xgboost"], "core")
    assert first["exit_code"] == second["exit_code"] == 124 and outputs[0] != outputs[1]
    for index, payload in enumerate([first, second], start=1):
        with zipfile.ZipFile(io.BytesIO(payload["archive"])) as archive:
            assert json.loads(archive.read("partial.json"))["invocation"] == index
            assert "modal_runtime.json" in archive.namelist()


def test_dispatch_reserves_before_remote_failure_and_never_retries(wrapper, tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "budget.json").write_text('{"total_cap_usd":26}')
    ledger = tmp_path / "artifacts" / "manifests" / "compute_ledger.json"
    calls = []
    def fail(argv, engine):
        calls.append(argv)
        assert json.loads(ledger.read_text())[0]["reserved_usd"] == .5
        raise RuntimeError("simulated remote failure")
    monkeypatch.setattr(wrapper.run_pilot, "remote", fail)
    with pytest.raises(RuntimeError):
        wrapper.dispatch("failed", ["--models", "xgboost"], "core", {})
    assert len(calls) == 1 and json.loads(ledger.read_text())[0]["status"] == "failed"


def test_archive_escape_rejected_and_reservation_retained(wrapper, tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "budget.json").write_text('{"total_cap_usd":26}')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../escape.txt", "unsafe")
    monkeypatch.setattr(wrapper.run_pilot, "remote", lambda *args: {"archive": buffer.getvalue(), "seconds": 1, "exit_code": 0})
    with pytest.raises(ValueError, match="Unsafe archive"):
        wrapper.dispatch("archive", ["--models", "xgboost"], "core", {})
    ledger = json.loads((tmp_path / "artifacts" / "manifests" / "compute_ledger.json").read_text())
    assert ledger[0]["status"] == "failed" and ledger[0]["reserved_usd"] == .5
    assert not (tmp_path / "artifacts" / "escape.txt").exists()


def test_invalid_ledger_fails_closed(wrapper, tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text('[{"run_id":"bad","reserved_usd":NaN}]')
    with pytest.raises(ValueError, match="Invalid compute"):
        wrapper.reserve_call(ledger, {"total_cap_usd": 26}, "new", {})
