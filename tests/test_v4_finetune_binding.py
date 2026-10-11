"""Future-run orchestration regression; never consumes scientific scores."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def driver(tmp_path, monkeypatch):
    code = Path(__file__).resolve().parents[1] / 'experiments/lifted_cavity'
    monkeypatch.syspath_prepend(str(code))
    spec = importlib.util.spec_from_file_location('binding_regression_driver', code / 'colab_v4.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in ('RUNS', 'REAL', 'DATA', 'POOLS', 'PANELS_DIR'):
        setattr(module, key, tmp_path / key)
    module.DEV = 'cpu'
    module.NEW, module.V4_REAL = ('fixture',), ('fixture_panel',)
    module.train_done = lambda _: True
    module.FAILED = []
    run = module.REAL / 'fixture/pfn_L_s2_ft'
    return module, run


def outputs(run):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'model.pt').write_bytes(b'constructed non-model fixture')
    (run / 'train.json').write_text(json.dumps({'finetune': {'fixture': True}}))


def test_fresh_finetune_binds_checkpoint_before_score(driver):
    module, run = driver
    calls = []
    def train(*args):
        outputs(run)
        return True
    def score(*args):
        identity = json.loads((run / 'checkpoint_identity.json').read_text())
        assert identity == {name: module.sha256(run / name) for name in ('model.pt', 'train.json')}
        calls.append('score')
        return True
    module.sh, module.score_step = train, score
    assert module.finetune('pfn_L', 2) is True
    assert calls == ['score'] and module.FAILED == []


def test_successful_subprocess_with_incomplete_outputs_cannot_score(driver):
    module, _ = driver
    module.sh = lambda *args: True
    module.score_step = lambda *args: pytest.fail('Incomplete checkpoint was scored')
    assert module.finetune('pfn_L', 2) is False
    assert module.FAILED == ['ft_pfn_L_s2_fixture_incomplete_checkpoint']


def test_existing_unbound_scores_cannot_be_backfilled(driver):
    module, run = driver
    outputs(run)
    (run / 'cells_fixture_panel.npz').write_bytes(b'constructed fixture, never parsed')
    module.sh = module.score_step = lambda *args: pytest.fail('Unbound results must be preserved')
    with pytest.raises(ValueError, match='no pre-score checkpoint binding'):
        module.finetune('pfn_L', 2)
    assert not (run / 'checkpoint_identity.json').exists()


def test_changed_bound_checkpoint_cannot_score(driver):
    module, run = driver
    outputs(run); module.bind_checkpoint(run)
    (run / 'model.pt').write_bytes(b'changed fixture')
    module.sh = module.score_step = lambda *args: pytest.fail('Changed checkpoint was scored')
    with pytest.raises(ValueError, match='Completed checkpoint changed'):
        module.finetune('pfn_L', 2)
