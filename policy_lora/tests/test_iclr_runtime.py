from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.balanced_sampling import build_balanced_weights
from src.train.train_lora_fic_hardtask import _aligned_backward
from src.utils.experiment_runtime import capture_rng_state, restore_rng_state


class _PromptDataset:
    def __init__(self):
        self.prompts = [
            "next to the ramekin",
            "easy task one",
            "on the stove",
            "easy task two",
        ]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


class ICLRRuntimeTests(unittest.TestCase):
    def test_balanced_sampling_assigns_equal_group_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights, stats = build_balanced_weights(
                _PromptDataset(),
                keywords=["ramekin", "stove"],
                hard_fraction=0.5,
                manifest_path=Path(tmp) / "manifest.json",
            )
            self.assertEqual(stats["hard_count"], 2)
            self.assertAlmostEqual(float(weights[[0, 2]].sum()), 0.5)
            self.assertAlmostEqual(float(weights[[1, 3]].sum()), 0.5)
            json.loads((Path(tmp) / "manifest.json").read_text())

    def test_conflicting_world_gradient_is_projected(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        action = (parameter - 1.0).square().sum()
        world = (parameter + 1.0).square().sum()
        metrics = _aligned_backward(
            optimizer=optimizer,
            backbone_params=[parameter],
            action_objective=action,
            world_objective=world,
        )
        self.assertTrue(metrics["gradient_conflict"])
        self.assertAlmostEqual(float(parameter.grad.item()), -2.0, places=5)

    def test_rng_state_round_trip(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(()).item())
        restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(()).item())
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
