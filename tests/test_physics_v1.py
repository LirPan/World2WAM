"""Unit tests for Physics-Aligned World2WAM Version A."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE.parent))

from minimal_world2wam.models.action_dit import FlowActionDiT
from minimal_world2wam.models.physics_router import StudentPhysicsRouter
from minimal_world2wam.models.physics_world2wam import PhysicsAlignedWorld2WAM
from minimal_world2wam.models.world2wam_heads import (
    ActionAdapter,
    ForwardHead,
    LightActionDiT,
    build_action_adapter,
    compute_cycle_flow_loss,
)
from minimal_world2wam.physics.phase_labeler import TeacherPhysicsLabeler
from minimal_world2wam.physics.physics_labels import batch_infer_physics_labels_v1
from minimal_world2wam.train.physics_losses import compute_physics_losses
from minimal_world2wam.train.training_utils import (
    compute_total_loss,
    compute_world2wam_losses,
    load_checkpoint,
    save_checkpoint,
)


def _batch(b: int = 4):
    return {
        "z_t": torch.randn(b, 48),
        "z_tH": torch.randn(b, 48),
        "text_embed": torch.randn(b, 128, 4096),
        "action_chunk": torch.randn(b, 10, 7),
        "state_t": torch.randn(b, 9),
        "metadata": [{"task": "pick up the bowl"} for _ in range(b)],
    }


def test_teacher_labeler_shapes():
    labeler = TeacherPhysicsLabeler(cfg={"auto_threshold": True})
    out = labeler.label_batch(_batch())
    assert out["phase_id"].shape == (4,)
    assert out["confidence"].shape == (4,)
    assert float(out["confidence"].min()) >= 0.05
    assert float(out["confidence"].max()) <= 1.0


def test_student_router_no_future_leak():
    router = StudentPhysicsRouter(
        latent_dim=48,
        text_input_dim=4096,
        text_dim=512,
        physics_dim=64,
        state_dim=9,
    )
    batch = _batch()
    out = router(
        batch["z_t"],
        text_embed=batch["text_embed"],
        state_t=batch["state_t"],
    )
    assert out["phase_logits"].shape == (4, 8)
    assert out["physics_code"].shape == (4, 64)

    try:
        router(
            batch["z_t"],
            text_embed=batch["text_embed"],
            state_t=batch["state_t"],
            z_tH=batch["z_tH"],
        )
        raise AssertionError("Expected ValueError for z_tH leakage")
    except ValueError as exc:
        assert "must not receive" in str(exc)


def test_flow_dit_future_latent_inverse():
    flow = FlowActionDiT(48, 10, 7, 4096, 512, hidden_dim=128, depth=2, num_heads=4, physics_dim=64)
    b = 4
    z_t = torch.randn(b, 48)
    z_tH = torch.randn(b, 48)
    text = torch.randn(b, 128, 4096)
    action = torch.randn(b, 10, 7)
    physics = torch.randn(b, 64)

    main_loss = flow.compute_flow_loss(z_t, text, action, physics_code=physics, future_latent=None)
    inv_loss = flow.compute_flow_loss(z_t, text, action, physics_code=physics, future_latent=z_tH)
    assert main_loss["loss"].ndim == 0
    assert inv_loss["loss"].ndim == 0

    sample_main = flow.sample(z_t, text, num_steps=2, physics_code=physics)
    sample_inv = flow.sample(z_t, text, num_steps=2, physics_code=physics, future_latent=z_tH)
    assert sample_main.shape == (b, 10, 7)
    assert sample_inv.shape == (b, 10, 7)
    assert not torch.allclose(sample_main, sample_inv)


def test_cycle_flow_loss():
    flow = FlowActionDiT(48, 10, 7, 4096, 512, hidden_dim=128, depth=2, num_heads=4)
    batch = _batch()
    z_pred = batch["z_tH"]
    loss = compute_cycle_flow_loss(
        flow,
        z_t=batch["z_t"],
        z_pred_H=z_pred,
        text_embed=batch["text_embed"],
        action_chunk=batch["action_chunk"],
    )
    assert loss.ndim == 0
    loss.backward()


def test_unified_world2wam_losses_flow():
    forward = ForwardHead(48, 10, 7, 4096, 512)
    flow = FlowActionDiT(48, 10, 7, 4096, 512, hidden_dim=128, depth=2, num_heads=4)
    batch = _batch()
    cfg = {"loss": {}, "weights": {}}
    losses = compute_world2wam_losses(
        forward_head=forward,
        action_adapter=flow,
        batch=batch,
        cfg=cfg,
    )
    assert losses["loss"].ndim == 0
    for key in ("loss_fwd", "loss_inv", "loss_cycle", "loss_flow"):
        assert key in losses
    losses["loss"].backward()


def test_physics_total_loss():
    forward = ForwardHead(48, 10, 7, 4096, 512, physics_dim=64)
    flow = FlowActionDiT(48, 10, 7, 4096, 512, hidden_dim=128, depth=2, num_heads=4, physics_dim=64)
    router = StudentPhysicsRouter(48, 4096, 512, physics_dim=64, state_dim=9)
    batch = _batch()
    cfg = {
        "loss": {"use_act": True, "use_fwd": True, "use_inv": True, "use_cycle": True},
        "weights": {"lambda_phase": 0.1, "lambda_phy": 0.1},
        "physics": {"phase_label_version": "v1"},
    }
    losses = compute_total_loss(
        forward_head=forward,
        action_adapter=flow,
        physics_router=router,
        batch=batch,
        cfg=cfg,
    )
    assert losses["loss"].ndim == 0
    assert "loss_phase" in losses
    assert "loss_phy" in losses
    losses["loss"].backward()


def test_physics_wrapper_inference_no_future():
    forward = ForwardHead(48, 10, 7, 4096, 512, physics_dim=64)
    flow = FlowActionDiT(48, 10, 7, 4096, 512, hidden_dim=128, depth=2, num_heads=4, physics_dim=64)
    router = StudentPhysicsRouter(48, 4096, 512, physics_dim=64, state_dim=9)
    model = PhysicsAlignedWorld2WAM(forward, flow, router, cfg={"eval": {"flow_sample_steps": 2}})
    batch = _batch()
    out = model.forward_inference(batch["z_t"], batch["text_embed"], state_t=batch["state_t"])
    assert out["pred_action"].shape == (4, 10, 7)
    assert "physics_code" in out


def test_adapters_ablation_with_physics_code():
    b, physics_code = 4, torch.randn(4, 64)
    z_t = torch.randn(b, 48)
    text = torch.randn(b, 128, 4096)

    mlp = ActionAdapter(48, 10, 7, 4096, 512, physics_dim=64)
    assert mlp(z_t, text, physics_code=physics_code).shape == (b, 10, 7)

    dit = LightActionDiT(48, 10, 7, 4096, 512, physics_dim=64)
    assert dit(z_t, text, physics_code=physics_code).shape == (b, 10, 7)


def test_physics_losses_slim():
    router = StudentPhysicsRouter(48, 4096, 512, physics_dim=64, state_dim=0)
    batch = _batch()
    outputs = router(batch["z_t"], text_embed=batch["text_embed"])
    outputs["z_pred_H"] = batch["z_tH"]
    losses = compute_physics_losses(batch, outputs, {"use_confidence_weight": True}, physics_cfg={"phase_label_version": "v1"})
    assert "loss_phase" in losses
    assert "loss_phy" in losses
    assert "loss_phy_smooth" not in losses


def test_checkpoint_roundtrip(tmp_path: Path):
    forward = ForwardHead(48, 10, 7, 4096, 512)
    flow = build_action_adapter(
        {"adapter_type": "flow_dit", "dit_hidden_dim": 128, "dit_depth": 2, "dit_num_heads": 4},
        latent_dim=48,
        text_input_dim=4096,
        text_dim=512,
        horizon=10,
        action_dim=7,
    )
    router = StudentPhysicsRouter(48, 4096, 512, physics_dim=64)
    cfg = {"horizon": 10, "model": {"latent_dim": 48, "action_dim": 7, "action_adapter": {"adapter_type": "flow_dit"}}}
    ckpt = tmp_path / "model.pt"
    save_checkpoint(ckpt, forward_head=forward, action_adapter=flow, physics_router=router, cfg=cfg, adapter_type="flow_dit")
    load_checkpoint(ckpt, forward, flow, physics_router=router, expected_adapter_type="flow_dit")


def test_batch_infer_physics_labels_v1():
    out = batch_infer_physics_labels_v1(_batch())
    assert out["phase_id"].shape == (4,)


def _run_all():
    import tempfile

    test_teacher_labeler_shapes()
    test_student_router_no_future_leak()
    test_flow_dit_future_latent_inverse()
    test_cycle_flow_loss()
    test_unified_world2wam_losses_flow()
    test_physics_total_loss()
    test_physics_wrapper_inference_no_future()
    test_adapters_ablation_with_physics_code()
    test_physics_losses_slim()
    with tempfile.TemporaryDirectory() as tmp:
        test_checkpoint_roundtrip(Path(tmp))
    test_batch_infer_physics_labels_v1()
    print("test_physics_v1: all passed")


if __name__ == "__main__":
    _run_all()
