from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal_world2wam.models.world2wam_heads import TextProjector


def _pool_latent(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 1:
        return z.unsqueeze(0)
    if z.dim() == 2:
        return z
    return z.reshape(z.shape[0], -1)


def _pool_state(state: torch.Tensor) -> torch.Tensor:
    if state.dim() == 1:
        return state.unsqueeze(0)
    if state.dim() == 2:
        return state
    return state.reshape(state.shape[0], -1)


class StudentPhysicsRouter(nn.Module):
    """
    Student physics router: z_t + text + current state_t -> physics embedding.

    Must NOT receive z_tH, action_chunk, or state_tH (no future leakage).
    """

    def __init__(
        self,
        latent_dim: int,
        text_input_dim: int,
        text_dim: int,
        num_phases: int = 8,
        physics_dim: int = 128,
        router_hidden_dim: int = 512,
        use_text_in_router: bool = True,
        state_dim: int = 0,
    ):
        super().__init__()
        self.num_phases = num_phases
        self.physics_dim = physics_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.text_dim = text_dim
        self.router_hidden_dim = router_hidden_dim
        self.use_text_in_router = use_text_in_router

        self.text_proj = TextProjector(text_input_dim, text_dim) if use_text_in_router else None

        in_parts = [latent_dim]
        if use_text_in_router:
            in_parts.append(text_dim)
        if state_dim > 0:
            in_parts.append(state_dim)

        in_dim = sum(in_parts)
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_phases),
        )
        self.code_proj = nn.Linear(num_phases, physics_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        text_embed: torch.Tensor | None = None,
        state_t: torch.Tensor | None = None,
        *,
        z_tH: torch.Tensor | None = None,
        action_chunk: torch.Tensor | None = None,
        state_tH: torch.Tensor | None = None,
        inference: bool = False,
    ) -> dict[str, torch.Tensor]:
        del inference  # student always uses the same inputs
        if z_tH is not None or action_chunk is not None or state_tH is not None:
            raise ValueError(
                "StudentPhysicsRouter must not receive z_tH, action_chunk, or state_tH"
            )

        z = _pool_latent(z_t)
        b, device, dtype = z.shape[0], z.device, z.dtype
        feats = [z]

        if self.use_text_in_router:
            assert self.text_proj is not None
            if text_embed is not None:
                feats.append(self.text_proj(text_embed))
            else:
                feats.append(torch.zeros(b, self.text_dim, device=device, dtype=dtype))

        if self.state_dim > 0:
            if state_t is not None:
                st = _pool_state(state_t)
                if st.shape[-1] != self.state_dim:
                    raise ValueError(
                        f"state_t dim {st.shape[-1]} != expected {self.state_dim}"
                    )
                feats.append(st)
            else:
                feats.append(torch.zeros(b, self.state_dim, device=device, dtype=dtype))

        x = torch.cat(feats, dim=-1)
        logits = self.router_mlp(x)
        probs = F.softmax(logits, dim=-1)
        physics_code = self.code_proj(probs)
        confidence = probs.max(dim=-1).values
        return {
            "phase_logits": logits,
            "phase_prob": probs,
            "physics_logits": logits,
            "physics_probs": probs,
            "physics_code": physics_code,
            "confidence": confidence,
        }


# Backward-compatible alias (deprecated)
PhysicsPhaseRouter = StudentPhysicsRouter
