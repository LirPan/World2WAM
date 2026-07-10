#!/usr/bin/env python3
"""Unit tests for Version B modules (no FastWAM required)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

VB_ROOT = Path(__file__).resolve().parents[1]
if str(VB_ROOT) not in sys.path:
    sys.path.insert(0, str(VB_ROOT))

from world2wam_vb.models.mot_flow_action import MotFlowActionExpert
from world2wam_vb.models.mot_heads import ForwardWorldHead
from world2wam_vb.models.mot_physics_router import TokenPhysicsAttentionRouter, TokenStudentPhysicsRouter
from world2wam_vb.physics.losses import compute_physics_mot_losses, latent_delta_direction_loss


def _batch(b: int = 4):
    h = 128
    t_len, t_dim = 16, 256
    horizon, a_dim = 10, 7
    return {
        "h_t": torch.randn(b, h),
        "context": torch.randn(b, t_len, t_dim),
        "context_mask": torch.ones(b, t_len, dtype=torch.bool),
        "proprio": torch.randn(b, 9),
        "action": torch.randn(b, horizon, a_dim),
        "future_latent": torch.randn(b, 48),
        "current_latent": torch.randn(b, 48),
    }


def test_student_router_no_leak():
    router = TokenStudentPhysicsRouter(hidden_dim=128, text_dim=256, proprio_dim=9)
    batch = _batch()
    out = router(
        batch["h_t"],
        context=batch["context"],
        context_mask=batch["context_mask"],
        proprio=batch["proprio"],
    )
    assert out["physics_code"].shape == (4, 128)
    try:
        router(batch["h_t"], context=batch["context"], future_latent=batch["future_latent"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_mot_flow_inverse_future_latent():
    flow = MotFlowActionExpert(
        hidden_dim=128,
        horizon=10,
        action_dim=7,
        text_dim=256,
        flow_hidden_dim=64,
        depth=2,
        num_heads=4,
        future_latent_dim=48,
    )
    batch = _batch()
    main = flow.compute_flow_loss(
        batch["h_t"], batch["context"], batch["action"], future_latent=None
    )
    inv = flow.compute_flow_loss(
        batch["h_t"],
        batch["context"],
        batch["action"],
        future_latent=batch["future_latent"],
    )
    assert main["loss"].ndim == 0
    assert inv["loss"].ndim == 0
    main["loss"].backward()


def test_forward_head_with_physics():
    head = ForwardWorldHead(128, 7, 48, physics_dim=64)
    batch = _batch()
    phy = torch.randn(4, 64)
    z = head(batch["h_t"], batch["action"][:, 0], phy)
    assert z.shape == (4, 48)


def test_token_attention_router():
    router = TokenPhysicsAttentionRouter(hidden_dim=128, physics_dim=64)
    tokens = torch.randn(4, 10, 128)
    out = router(tokens)
    assert out["physics_code"].shape == (4, 64)


def test_physics_losses():
    batch = _batch()
    outputs = {
        "z_future_pred": batch["future_latent"] + 0.1,
        "phase_logits": torch.randn(4, 8),
    }
    losses = compute_physics_mot_losses(batch, outputs, {})
    assert "loss_phy" in losses
    assert "loss_phase" in losses


def test_latent_delta_loss():
    z_pred = torch.randn(4, 48)
    z_tgt = torch.randn(4, 48)
    z_cur = torch.randn(4, 48)
    loss = latent_delta_direction_loss(z_pred, z_tgt, z_cur)
    assert loss.ndim == 0


def _run_all():
    test_student_router_no_leak()
    test_mot_flow_inverse_future_latent()
    test_forward_head_with_physics()
    test_token_attention_router()
    test_physics_losses()
    test_latent_delta_loss()
    print("test_version_b_shapes: all passed")


if __name__ == "__main__":
    _run_all()
