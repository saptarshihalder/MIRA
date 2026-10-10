"""Integration boundary tests using fabricated local files, never efficacy data."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("v4_integration_tested", Path(__file__).resolve().parents[1] / "infra/integrate_lifted_v4_verified.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fabricated_report():
    return dict(replay_schema=1, status="matched", coverage_complete=True, independent_full_verification=True,
                checkpoints=24, panel_checkpoint_pairs=30, arrays_checked=408, mismatched_values=0,
                mismatched_arrays=0, values_checked=408, device="cpu", source_commit="fixture",
                selection="0,n//2,n-1", tolerances=dict(atol=2e-4, rtol=2e-5), wall_seconds=900,
                automatic_retries=0, bindings={"fixture": "hash"}, coverage=[{}] * 30,
                comparisons=[dict(checkpoint="fixture", panel="panel", array=str(i), values=1,
                                  mismatches=0, matched=True) for i in range(408)])


@pytest.mark.parametrize("change", [dict(status="mismatched"), dict(coverage_complete=False),
                                   dict(checkpoints=23), dict(arrays_checked=407), dict(mismatched_values=1),
                                   dict(bindings={}), dict(tolerances=dict(atol=1., rtol=1.)),
                                   dict(automatic_retries=1), dict(source_commit="other")])
def test_incomplete_changed_or_mismatched_replay_rejected(change):
    report = fabricated_report()
    report.update(change)
    with pytest.raises(ValueError):
        m.validate_replay(report, {"fixture": "hash"}, SimpleNamespace(SOURCE_COMMIT="fixture", ATOL=2e-4, RTOL=2e-5))


def test_replay_details_must_be_complete_unique_and_consistent():
    report = fabricated_report()
    replay = SimpleNamespace(SOURCE_COMMIT="fixture", ATOL=2e-4, RTOL=2e-5)
    m.validate_replay(report, report["bindings"], replay)
    for field in ("duplicate", "count", "failure"):
        altered = copy.deepcopy(report)
        if field == "duplicate":
            altered["comparisons"][-1] = altered["comparisons"][0]
        elif field == "count":
            altered["values_checked"] += 1
        else:
            altered["comparisons"][0]["matched"] = False
        with pytest.raises(ValueError):
            m.validate_replay(altered, report["bindings"], replay)


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "runs/x/model.pt", "runs\\bad.json", "runs/C:bad.json", "unknown/data.json"])
def test_integration_allowlist_rejects_escape_weights_and_unexpected_files(name):
    with pytest.raises(ValueError):
        m.relative_path(name)


def setup_stage(tmp_path, monkeypatch):
    import hashlib
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    bindings = {"fixture": "hash"}
    replay = SimpleNamespace(DEFAULT_IDENTITY=tmp_path / "identity", capture_bindings=lambda *args: bindings,
                             require_unchanged_bindings=lambda a, b: None)
    verifier = SimpleNamespace(digest=digest, read_json=lambda p: json.loads(Path(p).read_text()))
    monkeypatch.setattr(m, "validate_inputs", lambda *args: (replay, verifier, {}, {}, bindings))
    stage, target = tmp_path / "stage", tmp_path / "target"
    source = stage / "runs/confirm_v4.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"fixture":true}')
    manifest = dict(schema=1, status="staged", bindings=bindings,
                    files={"runs/confirm_v4.json": dict(sha256=digest(source), bytes=source.stat().st_size)})
    (stage / "STAGE_MANIFEST.json").write_text(json.dumps(manifest))
    return stage, target


def test_conflict_preflight_writes_nothing_and_preserves_history(tmp_path, monkeypatch):
    stage, target = setup_stage(tmp_path, monkeypatch)
    old = target / "runs/confirm_v4.json"
    old.parent.mkdir(parents=True)
    old.write_text("historical evidence")
    with pytest.raises(ValueError, match="conflicts"):
        m.merge_stage(tmp_path / "input", tmp_path / "panels", tmp_path / "replay", stage, target)
    assert old.read_text() == "historical evidence"
    assert [p for p in target.rglob("*") if p.is_file()] == [old]


def test_identical_merge_is_idempotent_and_excludes_unlisted_history(tmp_path, monkeypatch):
    stage, target = setup_stage(tmp_path, monkeypatch)
    target.mkdir()
    historical = target / "historical.txt"
    historical.write_text("preserve")
    assert m.merge_stage(tmp_path / "input", tmp_path / "panels", tmp_path / "replay", stage, target)["new_files"] == 1
    assert m.merge_stage(tmp_path / "input", tmp_path / "panels", tmp_path / "replay", stage, target)["new_files"] == 0
    assert historical.read_text() == "preserve"


def test_tampered_stage_refused_before_target_creation(tmp_path, monkeypatch):
    stage, target = setup_stage(tmp_path, monkeypatch)
    (stage / "runs/confirm_v4.json").write_text("tampered")
    with pytest.raises(ValueError, match="bytes differ"):
        m.merge_stage(tmp_path / "input", tmp_path / "panels", tmp_path / "replay", stage, target)
    assert not target.exists()


def test_validation_failure_prevents_stage_creation(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("complete evidence missing")
    monkeypatch.setattr(m, "validate_inputs", fail)
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="missing"):
        m.stage_results(tmp_path / "input", tmp_path / "panels", tmp_path / "replay", stage)
    assert not stage.exists()
