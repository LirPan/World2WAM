"""Shared helpers for loading physics-aligned World2WAM checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from minimal_world2wam.data.latent_cache_dataset import detect_state_dim
from minimal_world2wam.models.physics_router import StudentPhysicsRouter
from minimal_world2wam.models.physics_world2wam import PhysicsAlignedWorld2WAM
from minimal_world2wam.models.world2wam_heads import build_heads_from_config, resolve_adapter_type
from minimal_world2wam.train.training_utils import load_checkpoint


def build_student_router(cfg: dict, meta: dict | None, cache_dir: Path | None) -> StudentPhysicsRouter:
    physics_cfg = cfg.get("physics", {})
    model_cfg = cfg.get("model", {})
    state_dim = detect_state_dim(cache_dir) if cache_dir else 0
    text_input_dim = int((meta or {}).get("text_embed_shape", [128, 4096])[-1])

    return StudentPhysicsRouter(
        latent_dim=int(model_cfg.get("latent_dim", 48)),
        text_input_dim=text_input_dim,
        text_dim=int(model_cfg.get("text_dim", 512)),
        num_phases=int(physics_cfg.get("num_phases", 8)),
        physics_dim=int(physics_cfg.get("physics_dim", 128)),
        router_hidden_dim=int(physics_cfg.get("router_hidden_dim", 512)),
        use_text_in_router=bool(physics_cfg.get("use_text_in_router", True)),
        state_dim=state_dim if bool(physics_cfg.get("use_state_in_router", True)) else 0,
    )


def build_physics_model(
    cfg: dict,
    meta: dict | None,
    device: str,
    *,
    cache_dir: Path | None = None,
) -> tuple[PhysicsAlignedWorld2WAM, str]:
    heads = build_heads_from_config(cfg, meta, include_inverse=False)
    adapter_type = resolve_adapter_type(cfg)
    physics_router = build_student_router(cfg, meta, cache_dir)

    model = PhysicsAlignedWorld2WAM(
        forward_head=heads["forward"],
        action_adapter=heads["adapter"],
        physics_router=physics_router,
        cfg=cfg,
    )
    model = model.to(device).eval()
    return model, adapter_type


def load_physics_checkpoint(
    model: PhysicsAlignedWorld2WAM,
    ckpt_path: Path,
    *,
    expected_adapter_type: str | None = None,
) -> dict[str, Any]:
    return load_checkpoint(
        ckpt_path,
        model.forward_head,
        model.action_adapter,
        expected_adapter_type=expected_adapter_type,
        physics_router=model.physics_router,
        inverse_head=model.inverse_head,
    )
