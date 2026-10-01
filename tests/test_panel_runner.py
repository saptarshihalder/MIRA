import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
from mira.artifacts import file_sha256, source_hashes
from mira.panel_runner import impose, prepare, run, logistic
from mira.panel_data import canonicalize, make_splits


def test_masks_preserve_native_and_match_class_rates():
    n = 30000
    x = np.ones((n, 3)); native = np.zeros(x.shape, bool); native[:10, 2] = True
    original = x.copy(); y = np.arange(n) % 2
    masked, imposed, union = impose(x, native, y, .5, .8, 1, 41)
    assert np.array_equal(x, original)
    assert np.array_equal(union, native | imposed)
    assert np.array_equal(np.isnan(masked), union)
    assert abs(imposed[y==0,1].mean()-.1) < .015
    assert abs(imposed[y==1,1].mean()-.9) < .015


def test_query_boundary_and_support_only_preprocessing():
    x = np.array([[1., np.nan], [4., 3.], [5., 2.]])
    m = np.isnan(x); y = np.array([0,1,0])
    a = prepare(x, y, x, m, m, "native_indicators", 8)
    assert not hasattr(a, "query_labels") and not hasattr(a, "oracle")
    assert np.array_equal(a.query[:,2:], m)
    b = prepare(x, 1-y, x, m, m, "native_shuffled", 8)
    c = prepare(x, y, x, m, m, "native_shuffled", 8)
    assert np.array_equal(b.query, c.query, equal_nan=True)
    assert np.array_equal(np.sort(b.query[:,2:], axis=0), np.sort(m, axis=0))


def test_inner_cv_keeps_repeated_groups_together():
    y = np.tile([0,1], 24); groups = np.repeat(np.arange(24), 2)
    model = logistic(51, y, groups)
    for a,b in model.cv:
        assert not set(groups[a]) & set(groups[b])
    assert set(model.estimator.named_steps) == {"impute", "scale", "clf"}


def test_panel_partial_failure_is_saved_and_group_leak_rejected(tmp_path):
    source = tmp_path / "input" / "toy"; source.mkdir(parents=True)
    x = np.random.default_rng(2).normal(size=(80, 3)); y = np.arange(80)%2
    panel,_ = canonicalize(x,y,1)
    np.savez(source/"dataset.npz",**{k:getattr(panel,k) for k in ('X','y','row_ids','group_ids','native_mask','multiplicity')})
    split=make_splits(panel,1,support_size=12,query_cap=16)
    (source/"splits.json").write_text(json.dumps(split)); (source/"manifest.json").write_text("{}")
    plan={"frozen":True,"datasets":["toy"],"context":12,"rates":[.5],"gammas":[.8],
          "models":["broken"],"modes":["native"],"ensembles":1,"mask_seed_base":73000,"runner_source_sha256":source_hashes(),
          "data_sha256":{"toy":{name:file_sha256(source/name) for name in ('dataset.npz','splits.json','manifest.json')}}}
    path=tmp_path/"plan.json"; path.write_text(json.dumps(plan)); path.with_suffix('.sha256').write_text(file_sha256(path))
    args=Namespace(protocol_file=str(path),out=str(tmp_path/"out"),dataset="toy",data_root=str(source.parent),device="cpu",checkpoint_dirs=[])
    def broken(*_): raise RuntimeError("deliberate inference failure")
    result=run(args,factory=broken)
    assert result["status"]=="incomplete" and len(result["errors"])==5
    assert (Path(args.out)/"data/fold0_r0.5_g0.8.npz").exists()
    with pytest.raises(ValueError,match="preserve"):
        run(args,factory=broken)
    # Corrupt a split group while keeping row indices disjoint.
    with np.load(source/"dataset.npz") as f: data={k:f[k].copy() for k in f.files}
    a,b=split['folds'][0]['support_indices'][0],split['folds'][0]['query_indices'][0]
    data["group_ids"][b]=data["group_ids"][a]; np.savez(source/"dataset.npz",**data)
    plan['data_sha256']['toy']['dataset.npz']=file_sha256(source/'dataset.npz')
    path.write_text(json.dumps(plan)); path.with_suffix('.sha256').write_text(file_sha256(path))
    args.out=str(tmp_path/"leak")
    with pytest.raises(ValueError):
        run(args,factory=broken)
