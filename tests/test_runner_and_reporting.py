import json
from pathlib import Path

import numpy as np
import pytest

from mira.artifacts import load_rows
from mira.reporting import mean_interval, paired_comparisons, report
from mira.runner import CONFIRMATION_SEEDS, _configuration, parser, run
from test_information_and_oracle import SpyEstimator


def arguments(out: Path, *additional: str):
    return parser().parse_args(["--models", "xgboost", "--modes", "native", "native_indicators",
                               "--gammas", "0", ".8", "--seeds", "2", "--context", "16",
                               "--queries", "24", "--device", "cpu", "--out", str(out), *additional])


def test_resumable_saved_predictions_generate_verified_task_level_report(tmp_path):
    calls = []

    def factory(name, settings, seed):
        calls.append((name, seed))
        return SpyEstimator()

    args = arguments(tmp_path / "run")
    manifest = run(args, factory=factory, capture_lock=False)
    assert manifest["status"] == "complete" and manifest["completed_cells"] == 8
    assert len(calls) == 8
    assert (tmp_path / "run" / "model_parameters.json").exists()
    run(args, factory=factory, capture_lock=False)
    assert len(calls) == 8, "Resume must not issue any additional model calls"
    summary = report(tmp_path / "run")
    assert summary["completed_cells"] == 8
    assert all(effect["n_tasks"] == 2 for effect in summary["effects"])
    assert all(effect["mean"] == 0 for effect in summary["effects"])
    assert all(row["fraction_analytic_gain"] is None for row in load_rows(tmp_path / "run") if row["gamma"] == 0)
    assert (tmp_path / "run" / "report" / "report.md").exists()
    row = load_rows(tmp_path / "run")[0]
    (tmp_path / "run" / row["prediction_file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        report(tmp_path / "run")
    with pytest.raises(ValueError, match="changed"):
        run(args, factory=factory, capture_lock=False)


def test_failures_are_incremental_and_never_silently_replaced(tmp_path):
    def failed_factory(name, settings, seed):
        raise RuntimeError("Requested checkpoint unavailable")

    out = tmp_path / "failure"
    manifest = run(arguments(out), factory=failed_factory, capture_lock=False)
    assert manifest["status"] == "incomplete"
    assert manifest["completed_cells"] == 0
    assert manifest["blocked_models"] == ["xgboost"]
    errors = json.loads((out / "errors.json").read_text())
    assert len(errors) == 1 and errors[0]["stage"] == "model_constructor"
    assert "checkpoint unavailable" in errors[0]["error"]
    resumed = run(arguments(out), factory=lambda *args: SpyEstimator(), capture_lock=False)
    assert resumed["status"] == "complete"
    assert resumed["historical_errors"] == 1 and resumed["unresolved_failed_cells"] == []


def test_configuration_mismatch_and_confirmation_seed_reuse_rejected(tmp_path):
    out = tmp_path / "run"
    run(arguments(out), factory=lambda *args: SpyEstimator(), capture_lock=False)
    with pytest.raises(ValueError, match="different configuration"):
        run(arguments(out, "--queries", "30"), factory=lambda *args: SpyEstimator(), capture_lock=False)
    with pytest.raises(ValueError, match="reserved range"):
        _configuration(arguments(tmp_path / "unused", "--seed-start", "60000"))
    confirmation = _configuration(parser().parse_args(["--protocol", "confirmation"]))
    assert confirmation["seeds"] == list(CONFIRMATION_SEEDS)
    assert confirmation["seeds"] == list(range(60000, 60020))
    assert not (tmp_path / "unused").exists()


def test_negative_gains_and_undefined_one_task_interval_are_retained():
    assert mean_interval([-1])["mean"] == -1
    assert mean_interval([-1])["ci_low"] is None
    interval = mean_interval([-0.1, -0.2, -0.3])
    assert interval["mean"] == pytest.approx(-0.2) and interval["ci_high"] < 0.05
    base = {"protocol": "development", "family": "label_only", "gamma": .8,
            "model": "xgboost", "seed": 40000, "data_sha256": "same", "oracle_nll": .3,
            "cold_start": False, "empirical_nll": .5}
    pairs = paired_comparisons([{**base, "mode": "native", "expected_nll": .5},
                                {**base, "mode": "native_indicators", "expected_nll": .6}])
    assert pairs[0]["gain_nats"] == pytest.approx(-.1)
    assert pairs[0]["fraction_baseline_gap"] == pytest.approx(-.5)


def test_cold_start_never_calls_model_constructor_and_remains_in_scoring(tmp_path):
    def forbidden(*args):
        raise AssertionError("Cold start must not be labeled as backbone inference")

    out = tmp_path / "empty_context"
    manifest = run(arguments(out, "--context", "0"), factory=forbidden, capture_lock=False)
    assert manifest["status"] == "complete" and manifest["cold_start_cells"] == 8
    assert all(row["cold_start"] for row in load_rows(out))
