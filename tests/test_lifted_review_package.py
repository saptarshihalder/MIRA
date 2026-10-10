"""Review export privacy and evidence barriers; no models or scientific scores."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("review_builder_tested", Path(__file__).resolve().parents[1] / "infra/build_lifted_review_package.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_metadata_cleanup_preserves_numbers_and_relative_identity():
    value = dict(source="abc123", numeric=[1, -2.5, True, None], relative="runs/model/train.json",
                 hosts=["C:\\Users\\name\\project", "/home/name/file", "/results/job/data"],
                 nested=dict(repo="/content/project"))
    result = m.cleaned(value)
    assert result["numeric"] == value["numeric"] and result["relative"] == value["relative"]
    assert result["source"] == value["source"] and result["hosts"] == ["<local-path>"] * 3
    assert result["nested"] == dict(repo="<local-path>")


def test_author_handles_fail_scan_while_literature_attribution_is_retained():
    base = SimpleNamespace(GUARDS={})
    m.audit_text("references.json", b'{"reference":"Minka, 2001"}', base)
    with pytest.raises(ValueError, match="handle"):
        m.audit_text("metadata.json", b'{"url":"https://github.com/saptarshihalder/MIRA"}', base)


def test_complete_verification_failure_prevents_archive_creation(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("complete verification/replay required")
    monkeypatch.setattr(m, "module", lambda *args: SimpleNamespace(validate_inputs=reject))
    output = tmp_path / "review.zip"
    with pytest.raises(ValueError, match="required"):
        m.build(tmp_path / "results", tmp_path / "panels", tmp_path / "replay", output)
    assert not output.exists()
