from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from world2wam_vb.adapters.fastwam_mot_adapter import FastWAMMotAdapter
from world2wam_vb.adapters.inference_guard import inference_guard
from world2wam_vb.losses.mot_losses import compute_bidirectional_losses, compute_total_mot_loss
from world2wam_vb.models.mot_flow_action import MotFlowActionExpert
from world2wam_vb.models.mot_heads import ForwardWorldHead
from world2wam_vb.models.mot_physics_router import TokenStudentPhysicsRouter
from world2wam_vb.physics.losses import compute_physics_mot_losses
from world2wam_vb.utils.training import anchor_from_batch, gt_action_from_batch


class BidirectionalMotWorld2WAM(nn.Module):
    """Version B idea2 baseline: frozen MoT + bidirectional heads (no physics)."""

    def __init__(
        self,
        adapter: FastWAMMotAdapter,
        forward_head: ForwardWorldHead,
        flow_expert: MotFlowActionExpert,
        cfg: dict[str, Any],
    ):
        super().__init__()
        self.adapter = adapter
        self.forward_head = forward_head
        self.flow_expert = flow_expert
        self.cfg = cfg

    def forward_train(self, batch: dict[str, Any]) -> dict[str, Any]:
        anchor = anchor_from_batch(batch, int(self.cfg.get("anchor_action_idx", 0)))
        gt_action_anchor = gt_action_from_batch(batch, anchor)
        gt_action_seq = batch["action"].float()

        mot_out = self.adapter.training_forward(batch)
        h_t = mot_out["h_t"]

        z_target = batch["future_latent"].float()
        if z_target.dim() == 1:
            z_target = z_target.unsqueeze(0)

        z_pred = self.forward_head(h_t, gt_action_anchor, physics_code=None)
        loss_fwd = F.mse_loss(z_pred, z_target)

        flow_main = self.flow_expert.compute_flow_loss(
            h_t,
            batch["context"],
            gt_action_seq,
            context_mask=batch.get("context_mask"),
            future_latent=None,
        )
        flow_inv = self.flow_expert.compute_flow_loss(
            h_t,
            batch["context"],
            gt_action_seq,
            context_mask=batch.get("context_mask"),
            future_latent=z_target,
        )
        flow_cycle = self.flow_expert.compute_flow_loss(
            h_t,
            batch["context"],
            gt_action_seq,
            context_mask=batch.get("context_mask"),
            future_latent=z_pred.detach() if self.cfg.get("cycle_detach_forward") else z_pred,
        )

        weights = self.cfg
        losses = compute_bidirectional_losses(
            loss_fastwam_action=mot_out["loss_fastwam_action"],
            loss_fastwam_video=mot_out.get("loss_fastwam_video"),
            loss_fwd=loss_fwd,
            loss_flow=flow_main["loss"],
            loss_inverse=flow_inv["loss"],
            loss_cycle=flow_cycle["loss"],
            lambda_fastwam_action=float(weights.get("lambda_fastwam_action", 0.0)),
            lambda_fastwam_video=float(weights.get("lambda_fastwam_video", 0.0)),
            lambda_fwd=float(weights.get("lambda_fwd", 1.0)),
            lambda_flow=float(weights.get("lambda_flow", 1.0)),
            lambda_inverse=float(weights.get("lambda_inverse", 0.1)),
            lambda_cycle=float(weights.get("lambda_cycle", 0.1)),
        )

        return {
            "h_t": h_t,
            "z_future_pred": z_pred,
            "losses": losses,
            "mot_out": mot_out,
        }

    @torch.no_grad()
    def forward_infer(self, batch: dict[str, Any]) -> torch.Tensor:
        with inference_guard():
            return self.adapter.infer_action_only(batch)


class PhysicsAlignedMotWorld2WAM(nn.Module):
    """
    Version B main model: FastWAM MoT + physics router + flow expert + world head.
    """

    def __init__(
        self,
        adapter: FastWAMMotAdapter,
        forward_head: ForwardWorldHead,
        flow_expert: MotFlowActionExpert,
        physics_router: TokenStudentPhysicsRouter,
        cfg: dict[str, Any],
    ):
        super().__init__()
        self.adapter = adapter
        self.forward_head = forward_head
        self.flow_expert = flow_expert
        self.physics_router = physics_router
        self.cfg = cfg
        self.enable_physics = bool(cfg.get("enable_physics", True))

    def _student_router(self, h_t: torch.Tensor, batch: dict[str, Any]) -> dict[str, Any]:
        proprio = batch.get("proprio")
        if proprio is not None and proprio.dim() == 3:
            proprio = proprio[:, 0]
        return self.physics_router(
            h_t,
            context=batch["context"],
            context_mask=batch.get("context_mask"),
            proprio=proprio,
        )

    def forward_train(self, batch: dict[str, Any]) -> dict[str, Any]:
        anchor = anchor_from_batch(batch, int(self.cfg.get("anchor_action_idx", 0)))
        gt_action_anchor = gt_action_from_batch(batch, anchor)
        gt_action_seq = batch["action"].float()

        if batch.get("future_latent") is None:
            raise ValueError("future_latent missing — precompute future VAE latents first.")

        mot_out = self.adapter.training_forward(batch)
        h_t = mot_out["h_t"]

        router_out = self._student_router(h_t, batch) if self.enable_physics else {}
        physics_code = router_out.get("physics_code")

        z_target = batch["future_latent"].float()
        if z_target.dim() == 1:
            z_target = z_target.unsqueeze(0)

        z_pred = self.forward_head(h_t, gt_action_anchor, physics_code=physics_code)
        loss_fwd = F.mse_loss(z_pred, z_target)

        flow_kwargs = dict(
            context_mask=batch.get("context_mask"),
            physics_code=physics_code,
        )
        flow_main = self.flow_expert.compute_flow_loss(
            h_t, batch["context"], gt_action_seq, future_latent=None, **flow_kwargs
        )
        flow_inv = self.flow_expert.compute_flow_loss(
            h_t, batch["context"], gt_action_seq, future_latent=z_target, **flow_kwargs
        )
        z_for_cycle = z_pred.detach() if self.cfg.get("cycle_detach_forward") else z_pred
        flow_cycle = self.flow_expert.compute_flow_loss(
            h_t, batch["context"], gt_action_seq, future_latent=z_for_cycle, **flow_kwargs
        )

        weights = self.cfg
        core = compute_bidirectional_losses(
            loss_fastwam_action=mot_out["loss_fastwam_action"],
            loss_fastwam_video=mot_out.get("loss_fastwam_video"),
            loss_fwd=loss_fwd,
            loss_flow=flow_main["loss"],
            loss_inverse=flow_inv["loss"],
            loss_cycle=flow_cycle["loss"],
            lambda_fastwam_action=float(weights.get("lambda_fastwam_action", 0.0)),
            lambda_fastwam_video=float(weights.get("lambda_fastwam_video", 0.0)),
            lambda_fwd=float(weights.get("lambda_fwd", 1.0)),
            lambda_flow=float(weights.get("lambda_flow", 1.0)),
            lambda_inverse=float(weights.get("lambda_inverse", 0.1)),
            lambda_cycle=float(weights.get("lambda_cycle", 0.1)),
        )

        outputs = {
            "h_t": h_t,
            "z_future_pred": z_pred,
            **router_out,
        }

        if self.enable_physics:
            phy = compute_physics_mot_losses(
                batch={**batch, "h_t": h_t},
                outputs=outputs,
                weights=weights,
                physics_cfg=self.cfg.get("physics", {}),
            )
            losses = compute_total_mot_loss(
                core,
                phy,
                lambda_phase=float(weights.get("lambda_phase", 0.1)),
                lambda_phy=float(weights.get("lambda_phy", 0.1)),
            )
        else:
            losses = core

        return {
            "h_t": h_t,
            "z_future_pred": z_pred,
            "losses": losses,
            "mot_out": mot_out,
            "router_out": router_out,
        }

    @torch.no_grad()
    def forward_infer(self, batch: dict[str, Any]) -> torch.Tensor:
        with inference_guard():
            return self.adapter.infer_action_only(batch)


def build_physics_mot_model(cfg: dict[str, Any]) -> PhysicsAlignedMotWorld2WAM:
    adapter = FastWAMMotAdapter.from_config(cfg)
    hidden_dim = adapter.hidden_dim
    action_dim = adapter.action_dim
    horizon = int(batch_horizon_from_cfg(cfg, adapter))

    text_dim = int(cfg.get("text_dim", 4096))
    future_latent_dim = int(cfg.get("future_latent_dim", 48))
    physics_dim = int(cfg.get("physics", {}).get("physics_dim", 128))

    forward_head = ForwardWorldHead(hidden_dim, action_dim, future_latent_dim, physics_dim)
    flow_expert = MotFlowActionExpert(
        hidden_dim=hidden_dim,
        horizon=horizon,
        action_dim=action_dim,
        text_dim=text_dim,
        flow_hidden_dim=int(cfg.get("flow_hidden_dim", 256)),
        depth=int(cfg.get("flow_depth", 4)),
        num_heads=int(cfg.get("flow_num_heads", 8)),
        physics_dim=physics_dim,
        future_latent_dim=future_latent_dim,
    )
    proprio_dim = int(cfg.get("proprio_dim", 0))
    router = TokenStudentPhysicsRouter(
        hidden_dim=hidden_dim,
        text_dim=text_dim,
        proprio_dim=proprio_dim,
        num_phases=int(cfg.get("physics", {}).get("num_phases", 8)),
        physics_dim=physics_dim,
    )
    return PhysicsAlignedMotWorld2WAM(adapter, forward_head, flow_expert, router, cfg)


def build_bidirectional_mot_model(cfg: dict[str, Any]) -> BidirectionalMotWorld2WAM:
    adapter = FastWAMMotAdapter.from_config(cfg)
    hidden_dim = adapter.hidden_dim
    action_dim = adapter.action_dim
    horizon = int(batch_horizon_from_cfg(cfg, adapter))
    text_dim = int(cfg.get("text_dim", 4096))
    future_latent_dim = int(cfg.get("future_latent_dim", 48))
    physics_dim = int(cfg.get("physics", {}).get("physics_dim", 128))

    forward_head = ForwardWorldHead(hidden_dim, action_dim, future_latent_dim, physics_dim)
    flow_expert = MotFlowActionExpert(
        hidden_dim=hidden_dim,
        horizon=horizon,
        action_dim=action_dim,
        text_dim=text_dim,
        flow_hidden_dim=int(cfg.get("flow_hidden_dim", 256)),
        depth=int(cfg.get("flow_depth", 4)),
        num_heads=int(cfg.get("flow_num_heads", 8)),
        physics_dim=physics_dim,
        future_latent_dim=future_latent_dim,
    )
    return BidirectionalMotWorld2WAM(adapter, forward_head, flow_expert, cfg)


def batch_horizon_from_cfg(cfg: dict[str, Any], adapter: FastWAMMotAdapter) -> int:
    return int(cfg.get("action_horizon", adapter.horizon))
