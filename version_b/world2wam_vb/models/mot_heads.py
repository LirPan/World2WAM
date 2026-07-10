from __future__ import annotations

import torch
import torch.nn as nn

from world2wam_vb.adapters.inference_guard import record_auxiliary_head_call


class ForwardWorldHead(nn.Module):
    """Predict future pooled VAE latent from MoT h_t + action + physics."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        future_latent_dim: int,
        physics_dim: int = 128,
    ):
        super().__init__()
        in_dim = hidden_dim + action_dim + physics_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, future_latent_dim),
        )
        self.physics_dim = physics_dim

    def forward(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
        physics_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        record_auxiliary_head_call("ForwardWorldHead")
        if physics_code is None:
            physics_code = torch.zeros(
                hidden.shape[0], self.physics_dim, device=hidden.device, dtype=hidden.dtype
            )
        return self.mlp(torch.cat([hidden, action, physics_code], dim=-1))


class InverseActionHead(nn.Module):
    """Legacy MLP inverse on MoT h_t (ablation; prefer MotFlowActionExpert)."""

    def __init__(
        self,
        hidden_dim: int,
        future_latent_dim: int,
        action_dim: int,
        physics_dim: int = 128,
        hidden_size: int = 1024,
    ):
        super().__init__()
        in_dim = hidden_dim + future_latent_dim + physics_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.physics_dim = physics_dim

    def forward(
        self,
        hidden: torch.Tensor,
        future_latent: torch.Tensor,
        physics_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        record_auxiliary_head_call("InverseActionHead")
        if physics_code is None:
            physics_code = torch.zeros(
                hidden.shape[0], self.physics_dim, device=hidden.device, dtype=hidden.dtype
            )
        return self.mlp(torch.cat([hidden, future_latent, physics_code], dim=-1))
