import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from run_backbone import (ROOT, load_protocol, natural_episodes, prepare_inputs, run, select_views,
                          select_cv_basis, verify_checkpoints)
from mira.data import generate_episode


class Selector:
    def select(self, support, confidence=.8):
        return np.eye(4, dtype=np.uint8), {"threshold": confidence, "support_rows": len(support.labels)}


class Estimator:
    classes_ = np.array([1, 0])
    def fit(self, X, y):
        self.context = X.copy()
        return self
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), .25), np.full(len(X), .75)])
    def get_params(self, deep=False):
        return {"mock": True}


def fixture_episode():
    ep = generate_episode("pairwise", 74009, .9, context=32, queries=16)
    return {"episode_id": "mock", "kind": "synthetic", "dataset": None, "fold": None,
            "order": 2, "gamma": .9, "seed": 74009, "support": ep.support, "query": ep.query,
            "native_context": ep.support.observations.values, "native_query": ep.query.observations.values,
            "selected_columns": np.arange(4), "targets": ep.targets.labels.copy(), "oracle": ep.targets.oracle.copy(), "groups": None}


def minimal_plan(views):
    return {"protocol_version": "mock_v1", "views": views, "file_sha256": {}, "checkpoint_sha256": {}}


def test_native_values_preserved_for_synthetic_and_full_width_natural_inputs():
    episode = fixture_episode()
    support, query = episode["support"], episode["query"]
    matrix = np.eye(4, dtype=np.uint8)
    matrix[0] = [1,1,0,0]
    for real in (False, True):
        xc, xq = episode["native_context"], episode["native_query"]
        columns = np.arange(4)
        if real:
            xc, xq = np.column_stack([np.zeros((32,2)),xc]), np.column_stack([np.zeros((16,2)),xq])
            columns = np.arange(2,6)
        prepared = prepare_inputs(support, query, columns, matrix, "trained", 74009, xc, xq, real)
        offset = 0 if real else 1
        assert np.array_equal(prepared.context[:,offset:offset+xc.shape[1]],xc.astype(np.float32),equal_nan=True)
        assert np.array_equal(prepared.query[:,offset:offset+xq.shape[1]],xq.astype(np.float32),equal_nan=True)
        assert prepared.context.shape[1] == 2*xc.shape[1]+offset
        assert np.array_equal(prepared.context[:,-xc.shape[1]:][:,columns],np.column_stack([
            support.observations.mask[:,0]^support.observations.mask[:,1],support.observations.mask[:,1:]]))


def test_selection_and_fitting_ignore_query_evaluator_targets(tmp_path):
    episode = fixture_episode()
    views = ["native", "identity", "trained", "heuristic", "matched_linear", "cv_basis", "random_basis", "deepsets", "shuffled"]
    selectors = {"trained": Selector(), "matched_linear": Selector(), "deepsets": Selector(),
                 "heuristic": lambda support, confidence: (np.eye(4,dtype=np.uint8),{"threshold":.3})}
    first = select_views(episode["support"],episode["seed"],views,selectors)
    calls = []
    def factory(*args):
        calls.append(args)
        return Estimator()
    plan = minimal_plan(views)
    result = run(plan,"frozen","tabpfn:v2",tmp_path/"one",factory=factory,selectors=selectors,
                 episodes=[episode],capture_lock=False,check_cache=False)
    assert result["status"] == "complete" and len(calls) == len(views)
    episode["targets"] = 1-episode["targets"]
    episode["oracle"] = np.full(16,.01)
    second = select_views(episode["support"],episode["seed"],views,selectors)
    assert first == second
    run(plan,"frozen","tabpfn:v2",tmp_path/"two",factory=factory,selectors=selectors,
        episodes=[episode],capture_lock=False,check_cache=False)
    for view in views:
        with np.load(tmp_path/"one/predictions"/f"mock_{view}.npz") as a, np.load(tmp_path/"two/predictions"/f"mock_{view}.npz") as b:
            assert np.array_equal(a["probability"],b["probability"])
            assert np.all(a["probability"] == .25)  # Correct positive column even with reversed classes.


def test_checkpoint_mutation_preserves_failed_prediction_without_retry(tmp_path):
    cache = tmp_path/"cache"; cache.mkdir()
    path = cache/"checkpoint.ckpt"; path.write_bytes(b"pinned")
    plan = minimal_plan(["identity", "random_basis"])
    plan["checkpoint_sha256"] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()}
    calls = []
    def mutate(*args):
        calls.append(args)
        path.write_bytes(b"changed checkpoint")
        return Estimator()
    result = run(plan,"frozen","tabpfn:v2",tmp_path/"run",cache_root=cache,factory=mutate,selectors={},
                 episodes=[fixture_episode()],capture_lock=False)
    assert result["failures"] == 1 and len(calls) == 1 and result["completed_cells"] == 0
    assert (tmp_path/"run/predictions/mock_identity.npz").exists()
    assert "checkpoint changed" in (tmp_path/"run/errors.json").read_text()


def test_first_backbone_failure_preserves_episode_and_blocks_remaining_calls(tmp_path):
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError("requested checkpoint unavailable")
    result = run(minimal_plan(["identity","random_basis"]),"frozen","tabpfn:v2",tmp_path/"run",
                 factory=fail,selectors={},episodes=[fixture_episode()],capture_lock=False,check_cache=False)
    assert result["completed_cells"] == 0 and result["failures"] == 1 and len(calls) == 1
    assert (tmp_path/"run/episodes/mock.npz").exists() and (tmp_path/"run/episodes/mock.json").exists()
    assert "unavailable" in (tmp_path/"run/errors.json").read_text()
    with pytest.raises(ValueError,match="Preserve"):
        run(minimal_plan(["identity"]),"frozen","tabpfn:v2",tmp_path/"run",capture_lock=False,check_cache=False)


def test_group_cv_and_real_ranking_use_support_only():
    episode = fixture_episode()
    matrix,info = select_cv_basis(episode["support"],np.arange(32))
    assert info["group_cv"] and info["cv_fits"] == 108 and matrix.shape == (4,4)
    real = [{"directory":f"artifacts/data/natural_compiler/{name}","folds":[0,1,2]} for name in ("hepatitis","horse_colic")]
    if not (ROOT/real[0]["directory"]).exists():
        pytest.skip("Natural panel not present")
    episodes = list(natural_episodes({"real":real}))
    assert len(episodes) == 6
    for item in episodes:
        assert np.all(item["support"].observations.u == 0)
        assert len(item["selected_columns"]) == 4 and item["oracle"] is None
        assert item["native_context"].shape[1] > 4


def frozen_fixture(tmp_path):
    required = ["experiments/mask_compiler/run_backbone.py","infra/modal_compiler.py","infra/modal_mechanism.py",
                "pyproject.toml","experiments/mask_compiler/feasibility.py","experiments/mask_compiler/train_selector.py",
                "experiments/mask_compiler/matched_training.py","compiler.npz","linear.npz","src/mira/data.py"]
    hashes = {}
    for relative in required:
        p = tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"frozen")
        hashes[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    plan = {"frozen":True,"protocol_version":"compiler_backbone_v1","models":["tabpfn:v2","tabicl:v2"],"ensembles":4,
            "synthetic":{"orders":[2,4],"gammas":[0,.9],"seeds":[74000,74001,74002],"context":128,"queries":256},
            "real":[],"views":["native","identity","trained","heuristic","matched_linear","cv_basis","random_basis"],
            "compiler_checkpoint":"compiler.npz","linear_checkpoint":"linear.npz","file_sha256":hashes,
            "checkpoint_sha256":{"a.ckpt":"unused"}}
    path=tmp_path/"protocol.json";path.write_text(json.dumps(plan));path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    return path,plan


def test_protocol_requires_frozen_source_and_data_hashes(tmp_path):
    path,plan=frozen_fixture(tmp_path)
    assert load_protocol(path,tmp_path)[0]==plan
    (tmp_path/"compiler.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="changed"):
        load_protocol(path,tmp_path)
    path.write_text('{}')
    with pytest.raises(ValueError,match="protocol SHA"):
        load_protocol(path,tmp_path)


def test_cache_hashes_and_path_escape_are_rejected(tmp_path):
    cache=tmp_path/"cache";cache.mkdir();path=cache/"a.ckpt";path.write_bytes(b"checkpoint")
    verify_checkpoints(cache,{"/cache/a.ckpt":hashlib.sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match="changed"):
        verify_checkpoints(cache,{"a.ckpt":"wrong"})
    with pytest.raises(ValueError,match="inside"):
        verify_checkpoints(cache,{"../outside":"wrong"})


def test_wrapper_cap_and_hash_preflight_precede_remote_calls(tmp_path,monkeypatch):
    # Reuse the existing local-only fake Modal fixture; no SDK/network invocation.
    spec=importlib.util.spec_from_file_location("modal_safety_fixture",ROOT/"tests/test_modal_budget.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    base=fixture.wrapper.__wrapped__(monkeypatch)
    monkeypatch.setitem(sys.modules,"infra.modal_mechanism",base)
    spec=importlib.util.spec_from_file_location("compiler_wrapper_test",ROOT/"infra/modal_compiler.py")
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    ledger=tmp_path/"ledger.json";budget={"total_cap_usd":26}
    wrapper.reserve_compiler(ledger,budget,"one","tabpfn:v2","same","v1")
    with pytest.raises(ValueError,match="same frozen"):
        wrapper.reserve_compiler(ledger,budget,"two","tabicl:v2","different","v1")
    wrapper.reserve_compiler(ledger,budget,"two","tabicl:v2","same","v1")
    with pytest.raises(ValueError,match="two calls"):
        wrapper.reserve_compiler(ledger,budget,"three","tabpfn:v2","same","v1")
    assert sum(e["reserved_usd"] for e in json.loads(ledger.read_text()))==1
    called=[];monkeypatch.setattr(wrapper.run_compiler,"remote",lambda *args:called.append(args))
    monkeypatch.setattr(wrapper,"ROOT",tmp_path)
    path=tmp_path/"unfrozen.json";path.write_text('{}');path.with_suffix('.sha256').write_text('wrong')
    with pytest.raises(ValueError,match="protocol SHA"):
        wrapper.main("bad","tabpfn:v2",str(path))
    assert not called and not (tmp_path/"artifacts/manifests/compute_ledger.json").exists()
