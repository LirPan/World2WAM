"""Unit tests for Version C residual action targets."""

from __future__ import annotations

import torch

from minimal_world2wam.train.training_utils import resolve_action_flow_target


def test_resolve_action_absolute_by_default():
    batch = {
        "action_chunk": torch.tensor([[1.0, 2.0]]),
        "fastwam_action": torch.tensor([[0.5, 0.5]]),
    }
    cfg = {"loss": {"residual_delta": False}}
    out = resolve_action_flow_target(batch, cfg)
    assert torch.allclose(out, batch["action_chunk"])


def test_resolve_action_residual_delta():
    batch = {
        "action_chunk": torch.tensor([[1.0, 2.0]]),
        "fastwam_action": torch.tensor([[0.25, 0.5]]),
    }
    cfg = {"loss": {"residual_delta": True}}
    out = resolve_action_flow_target(batch, cfg)
    assert torch.allclose(out, torch.tensor([[0.75, 1.5]]))


def test_resolve_action_residual_requires_teacher():
    batch = {"action_chunk": torch.tensor([[1.0, 2.0]])}
    cfg = {"loss": {"residual_delta": True}}
    try:
        resolve_action_flow_target(batch, cfg)
    except KeyError as e:
        assert "fastwam_action" in str(e)
        return
    raise AssertionError("expected KeyError for missing fastwam_action")


if __name__ == "__main__":
    test_resolve_action_absolute_by_default()
    test_resolve_action_residual_delta()
    test_resolve_action_residual_requires_teacher()
    print("ALL residual tests OK")
