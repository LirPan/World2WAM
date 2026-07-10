from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from minimal_world2wam.models.action_dit import FlowActionDiT
from minimal_world2wam.models.physics_router import StudentPhysicsRouter
from minimal_world2wam.train.training_utils import call_action_adapter, is_flow_adapter, sample_action_adapter


class PhysicsAlignedWorld2WAM(nn.Module):
    """
    Physics-aligned World2WAM wrapper (Version A).

    StudentPhysicsRouter -> physics_code -> ForwardHead + FlowActionDiT only.
    No FiLM on z_t; no future information in student or main action path.
    """

    def __init__(
        self,
        forward_head: nn.Module,
        action_adapter: nn.Module | None = None,
        physics_router: StudentPhysicsRouter | None = None,
        cfg: dict | None = None,
        inverse_head: nn.Module | None = None,
    ):
        super().__init__()
        self.forward_head = forward_head
        self.action_adapter = action_adapter
        self.physics_router = physics_router
        self.inverse_head = inverse_head  # legacy checkpoint compat only
        self.cfg = cfg or {}

        model_cfg = self.cfg.get("model", {})
        self.horizon = int(self.cfg.get("horizon", 10))
        self.action_dim = int(model_cfg.get("action_dim", 7))

    def _router_forward(
        self,
        z_t: torch.Tensor,
        *,
        text_embed: torch.Tensor | None = None,
        state_t: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.physics_router is None:
            return {}
        return self.physics_router(
            z_t,
            text_embed=text_embed,
            state_t=state_t,
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_tH: torch.Tensor | None = None,
        text_embed: torch.Tensor | None = None,
        action_chunk: torch.Tensor | None = None,
        state_t: torch.Tensor | None = None,
        state_tH: torch.Tensor | None = None,
        inference: bool = False,
        predict_action: bool = True,
    ) -> dict[str, Any]:
        del inference, state_tH  # student never uses future state
        outputs: dict[str, Any] = {}

        router_out = self._router_forward(
            z_t,
            text_embed=text_embed,
            state_t=state_t,
        )
        outputs.update(router_out)
        physics_code = router_out.get("physics_code")

        head_physics = physics_code

        if action_chunk is not None and text_embed is not None:
            z_pred_H = self.forward_head(
                z_t, action_chunk, text_embed, physics_code=head_physics
            )
            outputs["z_pred_H"] = z_pred_H

        if (
            z_tH is not None
            and text_embed is not None
            and self.action_adapter is not None
            and is_flow_adapter(self.action_adapter)
        ):
            outputs["pred_action_inv"] = sample_action_adapter(
                self.action_adapter,
                z_t,
                text_embed,
                num_steps=int(self.cfg.get("eval", {}).get("flow_sample_steps", 10)),
                physics_code=physics_code,
                future_latent=z_tH,
            )
        elif z_tH is not None and text_embed is not None and self.inverse_head is not None:
            outputs["pred_action_inv"] = self.inverse_head(
                z_t, z_tH, text_embed, physics_code=head_physics
            )

        if self.action_adapter is not None and text_embed is not None and predict_action:
            adapter_physics = physics_code
            if is_flow_adapter(self.action_adapter):
                pred_act = sample_action_adapter(
                    self.action_adapter,
                    z_t,
                    text_embed,
                    num_steps=int(self.cfg.get("eval", {}).get("flow_sample_steps", 10)),
                    physics_code=adapter_physics,
                    future_latent=None,
                )
            else:
                pred_act = call_action_adapter(
                    self.action_adapter,
                    z_t,
                    text_embed,
                    physics_code=adapter_physics,
                )
            outputs["pred_action"] = pred_act
            outputs["pred_act"] = pred_act

        return outputs

    def forward_inference(
        self,
        z_t: torch.Tensor,
        text_embed: torch.Tensor,
        state_t: torch.Tensor | None = None,
        num_flow_steps: int | None = None,
    ) -> dict[str, Any]:
        """Inference: z_t + text + student physics -> action (no future latent)."""
        cfg_eval = dict(self.cfg.get("eval", {}))
        if num_flow_steps is not None:
            cfg_eval["flow_sample_steps"] = num_flow_steps
        old_eval = self.cfg.get("eval", {})
        self.cfg["eval"] = cfg_eval
        out = self.forward(
            z_t,
            text_embed=text_embed,
            state_t=state_t,
            predict_action=True,
        )
        self.cfg["eval"] = old_eval
        return out
