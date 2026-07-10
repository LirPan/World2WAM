"""Unit tests for FlowActionDiT flow-matching action expert."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE.parent))

from minimal_world2wam.models.action_dit import FlowActionDiT
from minimal_world2wam.models.world2wam_heads import build_action_adapter, resolve_adapter_type
from minimal_world2wam.train.training_utils import load_checkpoint, save_checkpoint


def _make_adapter(**kwargs) -> FlowActionDiT:
    defaults = dict(
        latent_dim=48,
        horizon=10,
        action_dim=7,
        text_input_dim=4096,
        text_dim=512,
        hidden_dim=256,
        depth=4,
        num_heads=8,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return FlowActionDiT(**defaults)


def _make_batch(b: int = 4):
    z_t = torch.randn(b, 48)
    text_emb = torch.randn(b, 128, 4096)
    clean_action = torch.randn(b, 10, 7)
    noisy_action = torch.randn(b, 10, 7)
    tau = torch.rand(b)
    return z_t, text_emb, clean_action, noisy_action, tau


def test_forward_shape():
    adapter = _make_adapter()
    z_t, text_emb, _, noisy_action, tau = _make_batch()
    out = adapter(z_t, text_emb, noisy_action, tau)
    assert out.shape == (4, 10, 7)


def test_forward_pooled_text_shape():
    adapter = _make_adapter()
    z_t = torch.randn(4, 48)
    text_emb = torch.randn(4, 4096)
    noisy_action = torch.randn(4, 10, 7)
    tau = torch.rand(4)
    out = adapter(z_t, text_emb, noisy_action, tau)
    assert out.shape == (4, 10, 7)


def test_compute_flow_loss():
    adapter = _make_adapter()
    z_t, text_emb, clean_action, _, _ = _make_batch()
    loss_dict = adapter.compute_flow_loss(z_t, text_emb, clean_action)
    assert loss_dict["loss"].ndim == 0
    assert loss_dict["pred_velocity"].shape == clean_action.shape
    assert loss_dict["target_velocity"].shape == clean_action.shape
    assert loss_dict["noisy_action"].shape == clean_action.shape
    assert loss_dict["tau"].shape == (4,)


def test_sample():
    adapter = _make_adapter()
    z_t, text_emb, _, _, _ = _make_batch()
    sample = adapter.sample(z_t, text_emb, num_steps=2)
    assert sample.shape == (4, 10, 7)
    assert torch.isfinite(sample).all()


def test_build_action_adapter_flow_dit():
    act_cfg = {
        "adapter_type": "flow_dit",
        "dit_hidden_dim": 256,
        "dit_depth": 4,
        "dit_num_heads": 8,
        "dit_dropout": 0.1,
    }
    adapter = build_action_adapter(
        act_cfg,
        latent_dim=48,
        text_input_dim=4096,
        text_dim=512,
        horizon=10,
        action_dim=7,
    )
    assert isinstance(adapter, FlowActionDiT)
    assert adapter.adapter_type == "flow_dit"


def test_resolve_adapter_type_aliases():
    assert resolve_adapter_type({}) == "flow_dit"
    assert resolve_adapter_type({"model": {"action_adapter": {"adapter_type": "mlp"}}}) == "mlp"
    assert resolve_adapter_type({"model": {"action_adapter": {"adapter_type": "flow_dit"}}}) == "flow_dit"
    assert resolve_adapter_type({"model": {"action_adapter": {"adapter_type": "action_dit_flow"}}}) == "flow_dit"
    assert resolve_adapter_type({"model": {"action_adapter": {"adapter_type": "light_dit"}}}) == "light_dit"


def test_flow_inverse_with_future_latent():
    adapter = _make_adapter(hidden_dim=128, depth=2, num_heads=4)
    z_t, text_emb, clean_action, _, _ = _make_batch()
    z_tH = torch.randn(4, 48)
    loss_main = adapter.compute_flow_loss(z_t, text_emb, clean_action)
    loss_inv = adapter.compute_flow_loss(z_t, text_emb, clean_action, future_latent=z_tH)
    assert loss_main["loss"].ndim == 0
    assert loss_inv["loss"].ndim == 0


def test_checkpoint_roundtrip(tmp_path: Path):
    from minimal_world2wam.models.world2wam_heads import ForwardHead

    adapter = _make_adapter()
    forward_head = ForwardHead(48, 10, 7, 4096, 512)
    cfg = {
        "horizon": 10,
        "model": {
            "latent_dim": 48,
            "action_dim": 7,
            "text_dim": 512,
            "action_adapter": {
                "adapter_type": "flow_dit",
                "dit_hidden_dim": 256,
                "dit_depth": 4,
                "dit_num_heads": 8,
                "dit_dropout": 0.1,
            },
        },
    }
    ckpt_path = tmp_path / "adapter.pt"
    save_checkpoint(
        ckpt_path,
        forward_head=forward_head,
        action_adapter=adapter,
        cfg=cfg,
        adapter_type="flow_dit",
    )

    loaded = build_action_adapter(
        cfg["model"]["action_adapter"],
        latent_dim=48,
        text_input_dim=4096,
        text_dim=512,
        horizon=10,
        action_dim=7,
    )
    load_checkpoint(ckpt_path, forward_head, loaded, expected_adapter_type="flow_dit")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload["adapter_type"] == "flow_dit"

    z_t, text_emb, _, _, _ = _make_batch(b=2)
    sample = loaded.sample(z_t, text_emb, num_steps=2)
    assert sample.shape == (2, 10, 7)


def _run_all():
    import tempfile

    test_forward_shape()
    test_forward_pooled_text_shape()
    test_compute_flow_loss()
    test_sample()
    test_build_action_adapter_flow_dit()
    test_resolve_adapter_type_aliases()
    test_flow_inverse_with_future_latent()
    with tempfile.TemporaryDirectory() as tmp:
        test_checkpoint_roundtrip(Path(tmp))
    print("test_flow_action_dit: all passed")


if __name__ == "__main__":
    _run_all()
