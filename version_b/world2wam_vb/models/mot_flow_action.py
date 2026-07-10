"""Flow-matching action expert on MoT hidden state (Version B auxiliary decoder)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _timestep_embedding(tau: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if tau.dim() > 1:
        tau = tau.reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=tau.device, dtype=tau.dtype) / half
    )
    args = tau.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class MotFlowActionExpert(nn.Module):
    """
    Flow DiT on MoT pooled hidden h_t.

    Mirrors Version A FlowActionDiT but state token comes from MoT action representation.
    Supports optional future_latent (pooled future VAE latent) for inverse/cycle paths.
    """

    adapter_type = "mot_flow"

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        action_dim: int,
        text_dim: int,
        flow_hidden_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        physics_dim: int = 128,
        future_latent_dim: int = 48,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.flow_hidden_dim = flow_hidden_dim
        self.hidden_dim = hidden_dim

        self.action_embed = nn.Linear(action_dim, flow_hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, flow_hidden_dim)
        self.text_proj = nn.Linear(text_dim, flow_hidden_dim)
        self.physics_proj = nn.Linear(physics_dim, flow_hidden_dim)
        self.future_proj = nn.Linear(future_latent_dim, flow_hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(flow_hidden_dim, flow_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(flow_hidden_dim * 4, flow_hidden_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=flow_hidden_dim,
            nhead=num_heads,
            dim_feedforward=flow_hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(flow_hidden_dim, action_dim)

    def _embed_text(self, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if context.dim() == 2:
            return self.text_proj(context).unsqueeze(1)
        if mask is not None:
            m = mask.to(dtype=context.dtype, device=context.device)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            context = context * m.unsqueeze(-1)
        pooled = context.mean(dim=1) if mask is None else (
            (context * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        )
        return self.text_proj(pooled).unsqueeze(1)

    def _cond_tokens(
        self,
        h_t: torch.Tensor,
        context: torch.Tensor,
        tau: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = [
            self.h_proj(h_t).unsqueeze(1),
            self._embed_text(context, context_mask),
            self.time_mlp(_timestep_embedding(tau, self.flow_hidden_dim)).unsqueeze(1),
        ]
        if physics_code is not None:
            tokens.append(self.physics_proj(physics_code).unsqueeze(1))
        if future_latent is not None:
            tokens.append(self.future_proj(future_latent).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        h_t: torch.Tensor,
        context: torch.Tensor,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self._cond_tokens(
            h_t, context, tau, context_mask, physics_code, future_latent
        )
        action_tokens = self.action_embed(noisy_action)
        tokens = torch.cat([cond, action_tokens], dim=1)
        out = self.transformer(tokens)[:, -self.horizon :, :]
        return self.out_proj(out)

    def compute_flow_loss(
        self,
        h_t: torch.Tensor,
        context: torch.Tensor,
        clean_action: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        a0 = clean_action
        b = a0.shape[0]
        device, dtype = a0.device, a0.dtype
        if noise is None:
            noise = torch.randn_like(a0)
        if tau is None:
            tau = torch.rand(b, device=device, dtype=dtype)

        tau_a = tau.view(b, 1, 1)
        a_tau = (1.0 - tau_a) * a0 + tau_a * noise
        v_target = noise - a0
        v_pred = self.forward(
            h_t, context, a_tau, tau, context_mask, physics_code, future_latent
        )
        return {
            "loss": F.mse_loss(v_pred, v_target),
            "pred_velocity": v_pred,
            "target_velocity": v_target,
            "tau": tau,
        }

    @torch.no_grad()
    def sample(
        self,
        h_t: torch.Tensor,
        context: torch.Tensor,
        *,
        num_steps: int = 10,
        context_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = h_t.shape[0]
        device, dtype = h_t.device, h_t.dtype
        if noise is None:
            x = torch.randn(b, self.horizon, self.action_dim, device=device, dtype=dtype)
        else:
            x = noise.to(device=device, dtype=dtype)

        steps = max(int(num_steps), 1)
        dt = 1.0 / steps
        for i in range(steps, 0, -1):
            tau = torch.full((b,), i / steps, device=device, dtype=dtype)
            v = self.forward(
                h_t, context, x, tau, context_mask, physics_code, future_latent
            )
            x = x - dt * v
        return x
