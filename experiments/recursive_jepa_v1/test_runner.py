"""Verify protocol isolation and exact restoration of all engineering controls."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import run
from test_model import fixture


class RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.plan = json.loads((run.ROOT / 'configs/recursive_jepa_v1.json').read_text())

    def test_seed_and_budget_contract(self):
        run.validate_plan(self.plan)
        for modification in ({'train_seeds': [450001]}, {'updates': 65}, {'validation_used': True}):
            with self.assertRaises(ValueError):
                run.validate_plan({**self.plan, **modification})
        self.assertFalse(set(self.plan['train_seeds'] + self.plan['development_seeds']) & set(range(98000, 98020)))

    def test_control_capacity_and_checkpoint_replay(self):
        inputs = fixture(6)
        counts = []
        with tempfile.TemporaryDirectory() as directory:
            for name in run.NAMES:
                model = run.make_model(name, self.plan).eval()
                counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
                with torch.no_grad():
                    expected = model(*inputs)
                checkpoint = Path(directory) / (name + '.npz')
                np.savez_compressed(checkpoint, **{key: value.detach().numpy() for key, value in model.state_dict().items()})
                restored = run.restore_model(name, self.plan, checkpoint)
                with torch.no_grad():
                    torch.testing.assert_close(expected, restored(*inputs), atol=0, rtol=0)
                if name == 'supervised_only':
                    self.assertEqual(sum(run.loss_weights(name, self.plan)[key] for key in ('jepa_weight', 'variance_weight', 'covariance_weight')), 0)
        self.assertEqual(len(set(counts)), 1)


if __name__ == '__main__':
    unittest.main()
