from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PHYSICS_PHASES = [
    "free_motion",
    "approach",
    "contact",
    "grasp",
    "transport",
    "place",
    "push_slide",
    "uncertain",
]


class TokenStudentPhysicsRouter(nn.Module):
    """
    Student router on MoT representation.

    Inputs: pooled h_t [B, hidden_dim], text context, proprio (current state only).
    Must NOT receive future video tokens or future proprio.
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int,
        proprio_dim: int = 0,
        num_phases: int = 8,
        physics_dim: int = 128,
        router_hidden_dim: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.proprio_dim = proprio_dim
        self.num_phases = num_phases
        self.physics_dim = physics_dim

        self.h_proj = nn.Linear(hidden_dim, router_hidden_dim)
        self.text_proj = nn.Linear(text_dim, router_hidden_dim)
        self.proprio_proj = (
            nn.Linear(proprio_dim, router_hidden_dim) if proprio_dim > 0 else None
        )

        in_dim = router_hidden_dim * (2 + int(proprio_dim > 0))
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_phases),
        )
        self.code_proj = nn.Linear(num_phases, physics_dim)

    def _pool_text(self, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if context.dim() == 2:
            return context
        if mask is None:
            return context.mean(dim=1)
        m = mask.to(dtype=context.dtype, device=context.device)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (context * m.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        h_t: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        action_tokens: torch.Tensor | None = None,
        video_tokens: torch.Tensor | None = None,
        future_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if action_tokens is not None or video_tokens is not None or future_latent is not None:
            raise ValueError(
                "TokenStudentPhysicsRouter must not receive action_tokens, "
                "video_tokens, or future_latent"
            )

        text_p = self._pool_text(context, context_mask)
        if text_p.shape[-1] != self.text_dim:
            text_p = text_p[..., : self.text_dim]

        feats = [self.h_proj(h_t), self.text_proj(text_p)]
        if self.proprio_proj is not None:
            if proprio is None:
                b = h_t.shape[0]
                proprio = torch.zeros(b, self.proprio_dim, device=h_t.device, dtype=h_t.dtype)
            elif proprio.dim() > 2:
                proprio = proprio.reshape(proprio.shape[0], -1)
            feats.append(self.proprio_proj(proprio))

        x = torch.cat(feats, dim=-1)
        logits = self.router_mlp(x)
        probs = F.softmax(logits, dim=-1)
        physics_code = self.code_proj(probs)
        return {
            "phase_logits": logits,
            "phase_prob": probs,
            "physics_code": physics_code,
            "confidence": probs.max(dim=-1).values,
        }


class TokenPhysicsAttentionRouter(nn.Module):
    """
    Optional token-level router: attend over MoT action tokens before pooling.

    Produces per-sequence physics_code from [B, T, hidden_dim].
    """

    def __init__(
        self,
        hidden_dim: int,
        num_phases: int = 8,
        physics_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.logits_proj = nn.Linear(hidden_dim, num_phases)
        self.code_proj = nn.Linear(num_phases, physics_dim)

    def forward(
        self,
        action_tokens: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = action_tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        key_padding = pad_mask if pad_mask is not None else None
        pooled, _ = self.attn(q, action_tokens, action_tokens, key_padding_mask=key_padding)
        pooled = pooled.squeeze(1)
        logits = self.logits_proj(pooled)
        probs = F.softmax(logits, dim=-1)
        return {
            "phase_logits": logits,
            "phase_prob": probs,
            "physics_code": self.code_proj(probs),
            "confidence": probs.max(dim=-1).values,
        }
