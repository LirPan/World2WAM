from __future__ import annotations

import torch
import torch.nn as nn


class FutureLatentHead(nn.Module):
    """Predict future VAE-pooled latent from FastWAM hidden + action.

    ``action`` may be GT (default, stable) or predicted action for cycle experiments.
    """

    def __init__(self, hidden_dim: int, action_dim: int, future_latent_dim: int):
        super().__init__()
        in_dim = hidden_dim + action_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, future_latent_dim),
        )

    def forward(self, hidden: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, hidden_dim]
            action: [B, action_dim]
        Returns:
            pred_future_latent: [B, future_latent_dim]
        """
        if hidden.dim() != 2 or action.dim() != 2:
            raise ValueError(
                f"Expected hidden/action rank-2, got {hidden.shape} and {action.shape}"
            )
        if hidden.shape[0] != action.shape[0]:
            raise ValueError(f"Batch mismatch: {hidden.shape[0]} vs {action.shape[0]}")
        x = torch.cat([hidden, action], dim=-1)
        return self.mlp(x)
