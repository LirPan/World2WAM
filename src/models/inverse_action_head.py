from __future__ import annotations

import torch
import torch.nn as nn


class InverseActionHead(nn.Module):
    """Predict action from FastWAM hidden + future latent (auxiliary, train-only)."""

    def __init__(
        self,
        hidden_dim: int,
        future_latent_dim: int,
        action_dim: int,
        hidden_size: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = hidden_dim + future_latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, hidden: torch.Tensor, future_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, hidden_dim]
            future_latent: [B, future_latent_dim] (target or predicted)
        Returns:
            pred_action_from_future: [B, action_dim]
        """
        if hidden.dim() != 2 or future_latent.dim() != 2:
            raise ValueError(
                f"Expected rank-2 hidden/future_latent, got {hidden.shape} and {future_latent.shape}"
            )
        if hidden.shape[0] != future_latent.shape[0]:
            raise ValueError(f"Batch mismatch: {hidden.shape[0]} vs {future_latent.shape[0]}")
        x = torch.cat([hidden, future_latent], dim=-1)
        return self.mlp(x)
