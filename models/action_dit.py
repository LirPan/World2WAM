"""Flow-matching action expert aligned with FastWAM-style velocity prediction."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal_world2wam.models.world2wam_heads import TextProjector, _pool_latent


def timestep_embedding(
    tau: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Sinusoidal embedding for flow time tau in [0, 1].

    Args:
        tau: [B] or [B, 1] float tensor
        dim: embedding dimension (must be even)
    """
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


class FlowActionDiT(nn.Module):
    """
    Flow-matching action DiT: predicts velocity field v_theta(z_t, text, a_tau, tau).

    Main path: condition on (z_t, text, physics_code[, proprio]) — no future latent.
    Inverse / cycle: optional future_latent token (z_tH or predicted z_pred_H).

    Flow convention (fixed):
        a_tau = (1 - tau) * a0 + tau * noise
        v_target = noise - a0
        sample: start x ~ N(0,I) at tau=1, integrate x = x - dt * v from tau=1 to 0
    """

    adapter_type = "flow_dit"

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        action_dim: int,
        text_input_dim: int,
        text_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_time_embed_period: int = 10000,
        physics_dim: int = 128,
        proprio_dim: int = 0,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.proprio_dim = int(proprio_dim)

        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.future_proj = nn.Linear(latent_dim, hidden_dim)
        self.text_proj = nn.Linear(text_input_dim, hidden_dim)
        self.text_pool_proj = TextProjector(text_input_dim, hidden_dim)
        self.physics_proj = nn.Linear(physics_dim, hidden_dim)
        self.proprio_proj = nn.Linear(self.proprio_dim, hidden_dim) if self.proprio_dim > 0 else None

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_proj = nn.Linear(hidden_dim, action_dim)
        self.max_time_embed_period = max_time_embed_period

    def _embed_text_tokens(
        self,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return text tokens [B, L, D] or [B, 1, D] for pooled input."""
        if text_emb.dim() == 2:
            return self.text_pool_proj(text_emb).unsqueeze(1)
        if text_mask is not None:
            m = text_mask.to(dtype=text_emb.dtype, device=text_emb.device)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            text_emb = text_emb * m.unsqueeze(-1)
        return self.text_proj(text_emb)

    def _build_cond_tokens(
        self,
        z_t: torch.Tensor,
        text_emb: torch.Tensor,
        tau: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = _pool_latent(z_t)
        state_token = self.z_proj(z).unsqueeze(1)
        text_tokens = self._embed_text_tokens(text_emb, text_mask)
        time_emb = timestep_embedding(tau, self.hidden_dim, self.max_time_embed_period)
        time_token = self.time_mlp(time_emb).unsqueeze(1)
        tokens = [state_token, text_tokens, time_token]
        if physics_code is not None:
            tokens.append(self.physics_proj(physics_code).unsqueeze(1))
        if self.proprio_proj is not None and proprio is not None:
            p = proprio.float()
            if p.dim() == 3:
                p = p[:, 0]
            if p.dim() != 2:
                raise ValueError(f"proprio must be [B,D] or [B,T,D], got {tuple(proprio.shape)}")
            if p.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"proprio dim {p.shape[-1]} != model proprio_dim {self.proprio_dim}"
                )
            tokens.append(self.proprio_proj(p).unsqueeze(1))
        if future_latent is not None:
            future = _pool_latent(future_latent)
            tokens.append(self.future_proj(future).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        z_t: torch.Tensor,
        text_emb: torch.Tensor,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict velocity v_theta.

        Args:
            z_t: [B, latent_dim]
            text_emb: [B, text_dim] or [B, L, text_dim]
            noisy_action: [B, H, action_dim]
            tau: [B] or [B, 1] in [0, 1]
            future_latent: optional [B, latent_dim] for inverse/cycle only

        Returns:
            velocity [B, H, action_dim]
        """
        if noisy_action.dim() != 3:
            raise ValueError(f"noisy_action must be [B,H,A], got {tuple(noisy_action.shape)}")

        b, h, _ = noisy_action.shape
        if h != self.horizon:
            raise ValueError(f"horizon mismatch: {h} vs {self.horizon}")

        if tau.dim() == 2:
            tau = tau.squeeze(-1)
        if tau.shape[0] != b:
            raise ValueError(f"tau batch {tau.shape[0]} != action batch {b}")

        cond_tokens = self._build_cond_tokens(
            z_t, text_emb, tau, text_mask, physics_code, future_latent, proprio
        )
        action_tokens = self.action_embed(noisy_action)
        tokens = torch.cat([cond_tokens, action_tokens], dim=1)
        tokens = self.transformer(tokens)
        action_out = tokens[:, -self.horizon :, :]
        return self.out_proj(action_out)

    def compute_flow_loss(
        self,
        z_t: torch.Tensor,
        text_emb: torch.Tensor,
        clean_action: torch.Tensor,
        tau: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample tau/noise if not provided and compute flow-matching MSE loss.

        Convention:
            a_tau = (1 - tau) * a0 + tau * noise
            v_target = noise - a0
        """
        a0 = clean_action
        b = a0.shape[0]
        device = a0.device
        dtype = a0.dtype

        if noise is None:
            noise = torch.randn_like(a0)
        if tau is None:
            tau = torch.rand(b, device=device, dtype=dtype)

        tau_action = tau.view(b, 1, 1)
        a_tau = (1.0 - tau_action) * a0 + tau_action * noise
        v_target = noise - a0

        v_pred = self.forward(
            z_t,
            text_emb,
            a_tau,
            tau,
            text_mask=text_mask,
            physics_code=physics_code,
            future_latent=future_latent,
            proprio=proprio,
        )
        loss = F.mse_loss(v_pred, v_target)

        return {
            "loss": loss,
            "pred_velocity": v_pred,
            "target_velocity": v_target,
            "noisy_action": a_tau,
            "tau": tau,
        }

    @torch.no_grad()
    def sample(
        self,
        z_t: torch.Tensor,
        text_emb: torch.Tensor,
        num_steps: int = 10,
        noise: Optional[torch.Tensor] = None,
        clamp: Optional[float] = None,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Start from Gaussian action noise at tau=1 and integrate to tau=0.

        Main inference: future_latent=None.
        Inverse inference: future_latent=z_tH or z_pred_H.
        """
        z = _pool_latent(z_t)
        b = z.shape[0]
        device = z.device
        dtype = z.dtype

        if noise is None:
            x = torch.randn(b, self.horizon, self.action_dim, device=device, dtype=dtype)
        else:
            x = noise.to(device=device, dtype=dtype)

        steps = max(int(num_steps), 1)
        dt = 1.0 / steps
        for i in range(steps, 0, -1):
            tau_val = i / steps
            tau = torch.full((b,), tau_val, device=device, dtype=dtype)
            v = self.forward(
                z_t,
                text_emb,
                x,
                tau,
                text_mask=text_mask,
                physics_code=physics_code,
                future_latent=future_latent,
                proprio=proprio,
            )
            x = x - dt * v
            if clamp is not None:
                x = x.clamp(-clamp, clamp)
        return x
