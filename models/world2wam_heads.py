from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minimal_world2wam.wrappers.inference_guard import record_auxiliary_head_call


class TextProjector(nn.Module):
    """Masked mean pool [B,L,D] -> [B, text_dim]."""

    def __init__(self, input_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, text_dim)

    def forward(self, text_embed: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # text_embed: [B,L,D] or [B,D] already pooled
        if text_embed.dim() == 2:
            return self.proj(text_embed)
        if mask is None:
            pooled = text_embed.mean(dim=1)
        else:
            m = mask.to(dtype=text_embed.dtype, device=text_embed.device)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (text_embed * m.unsqueeze(-1)).sum(dim=1) / denom
        return self.proj(pooled)


class ActionChunkProjector(nn.Module):
    """Flatten [B,H,A] -> [B, H*A] -> hidden."""

    def __init__(self, horizon: int, action_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(horizon * action_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        # action_chunk: [B, H, action_dim]
        if action_chunk.dim() != 3:
            raise ValueError(f"Expected [B,H,A], got {tuple(action_chunk.shape)}")
        b = action_chunk.shape[0]
        return self.mlp(action_chunk.reshape(b, -1))


def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for i in range(num_layers - 1):
        layers.extend([nn.Linear(d, hidden_dim), nn.GELU()])
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ForwardHead(nn.Module):
    """
    z_pred_H = ForwardHead(z_t, action_chunk, text_embed)

    Shapes:
        z_t: [B, latent_dim]
        action_chunk: [B, H, action_dim]
        text_embed: [B, L, D] or [B, text_dim]
        out: [B, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        action_dim: int,
        text_input_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        physics_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.physics_dim = physics_dim
        self.text_proj = TextProjector(text_input_dim, text_dim)
        self.action_proj = ActionChunkProjector(horizon, action_dim, hidden_dim)
        self.physics_proj = nn.Linear(physics_dim, hidden_dim)
        in_dim = latent_dim + hidden_dim + text_dim
        self.mlp = _build_mlp(in_dim, latent_dim, hidden_dim, num_layers)

    def forward(
        self,
        z_t: torch.Tensor,
        action_chunk: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        record_auxiliary_head_call("ForwardHead")
        if z_t.dim() != 2:
            raise ValueError(f"z_t must be [B,D], got {tuple(z_t.shape)}")
        text_p = self.text_proj(text_embed, text_mask)
        act_p = self.action_proj(action_chunk)
        if physics_code is not None:
            act_p = act_p + self.physics_proj(physics_code)
        x = torch.cat([z_t, act_p, text_p], dim=-1)
        return self.mlp(x)


class InverseHead(nn.Module):
    """
    Legacy MLP inverse head (ablation / checkpoint compat only).

    Main training uses FlowActionDiT with future_latent for inverse/cycle.
    """

    adapter_type = "legacy_mlp_inverse"

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        action_dim: int,
        text_input_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        physics_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.text_proj = TextProjector(text_input_dim, text_dim)
        self.physics_proj = nn.Linear(physics_dim, text_dim)
        in_dim = latent_dim * 2 + text_dim
        self.mlp = _build_mlp(in_dim, horizon * action_dim, hidden_dim, num_layers)

    def forward(
        self,
        z_t: torch.Tensor,
        z_tH: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        record_auxiliary_head_call("InverseHead")
        if z_t.shape != z_tH.shape:
            raise ValueError(f"z_t/z_tH shape mismatch: {z_t.shape} vs {z_tH.shape}")
        text_p = self.text_proj(text_embed, text_mask)
        if physics_code is not None:
            text_p = text_p + self.physics_proj(physics_code)
        x = torch.cat([z_t, z_tH, text_p], dim=-1)
        out = self.mlp(x)
        b = z_t.shape[0]
        return out.view(b, self.horizon, self.action_dim)


class ActionAdapter(nn.Module):
    """
    MLP action adapter (ablation baseline).

    Main path uses FlowActionDiT.
    """

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        action_dim: int,
        text_input_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        physics_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.text_proj = TextProjector(text_input_dim, text_dim)
        self.physics_proj = nn.Linear(physics_dim, text_dim)
        in_dim = latent_dim + text_dim
        self.mlp = _build_mlp(in_dim, horizon * action_dim, hidden_dim, num_layers)

    def forward(
        self,
        z_t: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_p = self.text_proj(text_embed, text_mask)
        if physics_code is not None:
            text_p = text_p + self.physics_proj(physics_code)
        x = torch.cat([z_t, text_p], dim=-1)
        out = self.mlp(x)
        b = z_t.shape[0]
        return out.view(b, self.horizon, self.action_dim)


def _pool_latent(z_t: torch.Tensor) -> torch.Tensor:
    if z_t.dim() == 1:
        return z_t.unsqueeze(0)
    if z_t.dim() == 2:
        return z_t
    return z_t.reshape(z_t.shape[0], -1)


class LightActionDiT(nn.Module):
    """
    Lightweight DiT-style latent action token mixer (Hybrid DiT C方案).

    One-step MSE regression over learnable action query tokens; not full diffusion.
    TODO: timestep embedding, action noising, velocity/noise prediction for flow-matching.
    """

    adapter_type = "light_dit"

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        action_dim: int,
        text_input_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        physics_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.text_proj = TextProjector(text_input_dim, hidden_dim)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.physics_proj = nn.Linear(physics_dim, hidden_dim)
        self.action_queries = nn.Parameter(torch.zeros(horizon, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        physics_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = _pool_latent(z_t)
        if z.shape[-1] != self.z_proj.in_features:
            raise ValueError(
                f"z_t last dim {z.shape[-1]} != expected {self.z_proj.in_features}"
            )
        b = z.shape[0]
        cond = self.z_proj(z) + self.text_proj(text_embed, text_mask)
        if physics_code is not None:
            cond = cond + self.physics_proj(physics_code)
        tokens = self.action_queries.unsqueeze(0).expand(b, -1, -1) + cond.unsqueeze(1)
        # TODO: inject timestep embedding for diffusion / flow-matching
        tokens = self.transformer(tokens)
        return self.out_proj(tokens)


def compute_cycle_flow_loss(
    action_adapter: nn.Module,
    *,
    z_t: torch.Tensor,
    z_pred_H: torch.Tensor,
    text_embed: torch.Tensor,
    action_chunk: torch.Tensor,
    physics_code: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cycle consistency via shared FlowActionDiT inverse path.

    Forward(z_t, action) -> z_pred_H -> flow_loss with future_latent=z_pred_H.
    """
    if not hasattr(action_adapter, "compute_flow_loss"):
        raise TypeError("compute_cycle_flow_loss requires a FlowActionDiT adapter")
    flow_out = action_adapter.compute_flow_loss(
        z_t=z_t,
        text_emb=text_embed,
        clean_action=action_chunk,
        text_mask=text_mask,
        physics_code=physics_code,
        future_latent=z_pred_H,
    )
    return flow_out["loss"]


def resolve_adapter_type(
    cfg: dict,
    payload: dict | None = None,
    cli_override: str | None = None,
) -> str:
    """Priority: CLI > checkpoint payload > config > default flow_dit."""

    def _norm(t: str) -> str:
        t = str(t).lower().strip()
        if t in ("dit", "light_dit"):
            return "light_dit"
        if t in ("flow_dit", "action_dit_flow"):
            return "flow_dit"
        if t == "mlp":
            return "mlp"
        raise ValueError(f"Unknown adapter_type: {t}")

    if cli_override is not None:
        return _norm(cli_override)

    if payload is not None:
        if "adapter_type" in payload:
            return _norm(payload["adapter_type"])
        meta = payload.get("meta") or {}
        if "adapter_type" in meta:
            return _norm(meta["adapter_type"])
        ckpt_cfg = payload.get("cfg") or {}
        act_cfg = ckpt_cfg.get("model", {}).get("action_adapter", {})
        if act_cfg.get("adapter_type"):
            return _norm(act_cfg["adapter_type"])

    act_cfg = cfg.get("model", {}).get("action_adapter", {})
    return _norm(act_cfg.get("adapter_type", "flow_dit"))


def build_action_adapter(
    act_cfg: dict,
    *,
    latent_dim: int,
    text_input_dim: int,
    text_dim: int,
    horizon: int,
    action_dim: int,
    physics_dim: int = 128,
) -> nn.Module:
    adapter_type = resolve_adapter_type({"model": {"action_adapter": act_cfg}})

    if adapter_type == "mlp":
        return ActionAdapter(
            latent_dim=latent_dim,
            horizon=horizon,
            action_dim=action_dim,
            text_input_dim=text_input_dim,
            text_dim=text_dim,
            hidden_dim=int(act_cfg.get("hidden_dim", 1024)),
            num_layers=int(act_cfg.get("num_layers", 3)),
            physics_dim=physics_dim,
        )

    if adapter_type == "light_dit":
        return LightActionDiT(
            latent_dim=latent_dim,
            horizon=horizon,
            action_dim=action_dim,
            text_input_dim=text_input_dim,
            text_dim=text_dim,
            hidden_dim=int(act_cfg.get("hidden_dim", 512)),
            num_layers=int(act_cfg.get("num_layers", 4)),
            num_heads=int(act_cfg.get("num_heads", 8)),
            dropout=float(act_cfg.get("dropout", 0.1)),
            physics_dim=physics_dim,
        )

    if adapter_type == "flow_dit":
        from minimal_world2wam.models.action_dit import FlowActionDiT

        hidden_dim = int(act_cfg.get("dit_hidden_dim", act_cfg.get("hidden_dim", 256)))
        depth = int(act_cfg.get("dit_depth", act_cfg.get("num_layers", 4)))
        num_heads = int(act_cfg.get("dit_num_heads", act_cfg.get("num_heads", 8)))
        dropout = float(act_cfg.get("dit_dropout", act_cfg.get("dropout", 0.1)))
        mlp_ratio = float(act_cfg.get("mlp_ratio", 4.0))
        return FlowActionDiT(
            latent_dim=latent_dim,
            horizon=horizon,
            action_dim=action_dim,
            text_input_dim=text_input_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            physics_dim=physics_dim,
        )

    raise ValueError(f"Unknown adapter_type: {adapter_type}")


def compute_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    """MSE action loss for ablation adapters (MLP / LightDiT)."""
    if mode == "mse":
        return F.mse_loss(pred.float(), target.float())
    raise ValueError(f"Unknown action loss mode: {mode}")


def build_heads_from_config(
    cfg: dict,
    meta: dict | None = None,
    *,
    include_inverse: bool = False,
    include_adapter: bool = True,
) -> dict[str, nn.Module]:
    model_cfg = cfg.get("model", {})
    horizon = int(cfg.get("horizon", 10))
    latent_dim = int(model_cfg.get("latent_dim", cfg.get("latent_dim", 48)))
    action_dim = int(model_cfg.get("action_dim", cfg.get("action_dim", 7)))
    text_dim = int(model_cfg.get("text_dim", cfg.get("text_dim", 512)))

    if meta and meta.get("text_embed_shape"):
        text_input_dim = int(meta["text_embed_shape"][-1])
    else:
        text_input_dim = 4096

    fwd_cfg = model_cfg.get("forward_head", {})
    inv_cfg = model_cfg.get("inverse_head", {})
    act_cfg = model_cfg.get("action_adapter", {})

    physics_dim = int(cfg.get("physics", {}).get("physics_dim", 128))

    heads: dict[str, nn.Module] = {
        "forward": ForwardHead(
            latent_dim=latent_dim,
            horizon=horizon,
            action_dim=action_dim,
            text_input_dim=text_input_dim,
            text_dim=text_dim,
            hidden_dim=int(fwd_cfg.get("hidden_dim", 1024)),
            num_layers=int(fwd_cfg.get("num_layers", 3)),
            physics_dim=physics_dim,
        ),
    }
    if include_inverse:
        heads["inverse"] = InverseHead(
            latent_dim=latent_dim,
            horizon=horizon,
            action_dim=action_dim,
            text_input_dim=text_input_dim,
            text_dim=text_dim,
            hidden_dim=int(inv_cfg.get("hidden_dim", 1024)),
            num_layers=int(inv_cfg.get("num_layers", 3)),
            physics_dim=physics_dim,
        )
    if include_adapter:
        heads["adapter"] = build_action_adapter(
            act_cfg,
            latent_dim=latent_dim,
            text_input_dim=text_input_dim,
            text_dim=text_dim,
            horizon=horizon,
            action_dim=action_dim,
            physics_dim=physics_dim,
        )
    return heads
