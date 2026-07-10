from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from minimal_world2wam.models.world2wam_heads import compute_cycle_flow_loss


def is_flow_adapter(adapter) -> bool:
    return hasattr(adapter, "compute_flow_loss") and hasattr(adapter, "sample")


def call_action_adapter(
    adapter,
    z_t: torch.Tensor,
    text_embed: torch.Tensor,
    physics_code: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    future_latent: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call action adapter; pass physics_code when supported."""
    kwargs: dict[str, Any] = {}
    if physics_code is not None:
        kwargs["physics_code"] = physics_code
    if text_mask is not None:
        kwargs["text_mask"] = text_mask
    if future_latent is not None and is_flow_adapter(adapter):
        kwargs["future_latent"] = future_latent

    if kwargs:
        try:
            return adapter(z_t, text_embed, **kwargs)
        except TypeError:
            kwargs.pop("future_latent", None)
            if kwargs:
                try:
                    return adapter(z_t, text_embed, **kwargs)
                except TypeError:
                    pass
    if text_mask is not None:
        try:
            return adapter(z_t, text_embed, text_mask=text_mask)
        except TypeError:
            pass
    return adapter(z_t, text_embed)


def sample_action_adapter(
    adapter,
    z_t: torch.Tensor,
    text_embed: torch.Tensor,
    *,
    num_steps: int = 10,
    text_mask: torch.Tensor | None = None,
    physics_code: torch.Tensor | None = None,
    future_latent: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample or predict action chunk; flow adapters use ODE integration."""
    if is_flow_adapter(adapter):
        kwargs: dict[str, Any] = {"num_steps": num_steps}
        if text_mask is not None:
            kwargs["text_mask"] = text_mask
        if physics_code is not None:
            kwargs["physics_code"] = physics_code
        if future_latent is not None:
            kwargs["future_latent"] = future_latent
        return adapter.sample(z_t, text_embed, **kwargs)
    return call_action_adapter(
        adapter,
        z_t,
        text_embed,
        physics_code=physics_code,
        text_mask=text_mask,
    )


def _legacy_inverse_loss(
    inverse_head,
    z_t: torch.Tensor,
    z_future: torch.Tensor,
    text_embed: torch.Tensor,
    action_chunk: torch.Tensor,
    physics_code: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if physics_code is not None:
        kwargs["physics_code"] = physics_code
    try:
        pred = inverse_head(z_t, z_future, text_embed, **kwargs)
    except TypeError:
        pred = inverse_head(z_t, z_future, text_embed)
    return F.mse_loss(pred, action_chunk)


def compute_world2wam_losses(
    *,
    forward_head,
    action_adapter,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    inverse_head=None,
    physics_code: torch.Tensor | None = None,
    use_act: bool = True,
    use_fwd: bool = True,
    use_inv: bool = True,
    use_cycle: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Core World2WAM losses.

    FlowActionDiT path (default):
      L_flow: main decoder, no future_latent
      L_inverse / L_cycle: flow loss with future_latent=z_tH / z_pred_H

    Legacy MLP inverse_head used only when action_adapter is not a flow adapter.
    """
    loss_cfg = cfg.get("loss", {})
    weights = cfg.get("weights", {})

    z_t = batch["z_t"]
    z_tH = batch["z_tH"]
    text_embed = batch["text_embed"]
    action_chunk = batch["action_chunk"]
    text_mask = batch.get("text_mask")

    lambda_fwd = float(weights.get("lambda_fwd", weights.get("lambda_future", 1.0)))
    lambda_inv = float(weights.get("lambda_inv", weights.get("lambda_inverse", 1.0)))
    lambda_cycle = float(weights.get("lambda_cycle", 0.1))
    lambda_act = float(weights.get("lambda_act", weights.get("lambda_flow", 1.0)))
    cycle_detach = bool(loss_cfg.get("cycle_detach_forward", False))

    device = z_t.device
    zero = torch.zeros((), device=device)

    loss_fwd = zero
    loss_inv = zero
    loss_cycle = zero
    loss_act = zero
    act_weight = lambda_act

    fwd_kwargs: dict[str, Any] = {}
    if physics_code is not None:
        fwd_kwargs["physics_code"] = physics_code

    z_pred_H = None
    if use_fwd or use_cycle:
        try:
            z_pred_H = forward_head(z_t, action_chunk, text_embed, **fwd_kwargs)
        except TypeError:
            z_pred_H = forward_head(z_t, action_chunk, text_embed)
        if use_fwd:
            loss_fwd = F.mse_loss(z_pred_H, z_tH)

    flow_metrics: dict[str, torch.Tensor] = {}
    use_flow = action_adapter is not None and is_flow_adapter(action_adapter)

    if use_flow:
        flow_kwargs: dict[str, Any] = {}
        if physics_code is not None:
            flow_kwargs["physics_code"] = physics_code
        if text_mask is not None:
            flow_kwargs["text_mask"] = text_mask

        if use_act:
            flow_out = action_adapter.compute_flow_loss(
                z_t=z_t,
                text_emb=text_embed,
                clean_action=action_chunk,
                **flow_kwargs,
            )
            loss_act = flow_out["loss"]
            act_weight = float(weights.get("flow_loss_weight", lambda_act))
            flow_metrics = {
                "loss_action_flow": flow_out["loss"].detach(),
                "loss_flow": flow_out["loss"].detach(),
                "flow_tau_mean": flow_out["tau"].mean().detach(),
            }

        if use_inv:
            inv_out = action_adapter.compute_flow_loss(
                z_t=z_t,
                text_emb=text_embed,
                clean_action=action_chunk,
                future_latent=z_tH,
                **flow_kwargs,
            )
            loss_inv = inv_out["loss"]

        if use_cycle:
            if z_pred_H is None:
                try:
                    z_pred_H = forward_head(z_t, action_chunk, text_embed, **fwd_kwargs)
                except TypeError:
                    z_pred_H = forward_head(z_t, action_chunk, text_embed)
            z_for_cycle = z_pred_H.detach() if cycle_detach else z_pred_H
            loss_cycle = compute_cycle_flow_loss(
                action_adapter,
                z_t=z_t,
                z_pred_H=z_for_cycle,
                text_embed=text_embed,
                action_chunk=action_chunk,
                physics_code=physics_code,
                text_mask=text_mask,
            )

    elif inverse_head is not None:
        if use_inv:
            loss_inv = _legacy_inverse_loss(
                inverse_head, z_t, z_tH, text_embed, action_chunk, physics_code
            )
        if use_cycle:
            if z_pred_H is None:
                try:
                    z_pred_H = forward_head(z_t, action_chunk, text_embed, **fwd_kwargs)
                except TypeError:
                    z_pred_H = forward_head(z_t, action_chunk, text_embed)
            z_for_cycle = z_pred_H.detach() if cycle_detach else z_pred_H
            loss_cycle = _legacy_inverse_loss(
                inverse_head, z_t, z_for_cycle, text_embed, action_chunk, physics_code
            )

        if use_act and action_adapter is not None:
            from minimal_world2wam.models.world2wam_heads import compute_action_loss

            pred_act = call_action_adapter(
                action_adapter, z_t, text_embed, physics_code=physics_code, text_mask=text_mask
            )
            loss_act = compute_action_loss(
                pred_act,
                action_chunk,
                mode=str(loss_cfg.get("action_loss_mode", "mse")),
            )
            act_weight = lambda_act

    total = (
        lambda_fwd * loss_fwd
        + lambda_inv * loss_inv
        + lambda_cycle * loss_cycle
        + act_weight * loss_act
    )

    result = {
        "loss": total,
        "loss_fwd": loss_fwd.detach(),
        "loss_future": loss_fwd.detach(),
        "loss_inv": loss_inv.detach(),
        "loss_inverse": loss_inv.detach(),
        "loss_cycle": loss_cycle.detach(),
        "loss_act": loss_act.detach(),
        "loss_flow": loss_act.detach(),
    }
    result.update(flow_metrics)
    return result


def compute_total_loss(
    *,
    forward_head,
    action_adapter,
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    physics_router=None,
    inverse_head=None,
    use_act: bool = True,
    use_fwd: bool = True,
    use_inv: bool = True,
    use_cycle: bool = True,
    use_physics: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Unified L_total = L_flow + λ_future·L_future + λ_inverse·L_inverse
                    + λ_cycle·L_cycle + λ_phase·L_phase + λ_phy·L_phy
    """
    from minimal_world2wam.train.physics_losses import compute_physics_losses

    physics_code = None
    router_out: dict[str, Any] = {}
    state_t = batch.get("state_t")

    if physics_router is not None and use_physics:
        router_out = physics_router(
            batch["z_t"],
            text_embed=batch.get("text_embed"),
            state_t=state_t,
        )
        physics_code = router_out.get("physics_code")

    core = compute_world2wam_losses(
        forward_head=forward_head,
        action_adapter=action_adapter,
        inverse_head=inverse_head,
        batch=batch,
        cfg=cfg,
        physics_code=physics_code,
        use_act=use_act,
        use_fwd=use_fwd,
        use_inv=use_inv,
        use_cycle=use_cycle,
    )

    if not use_physics or physics_router is None:
        return core

    weights = cfg.get("weights", {})
    outputs = dict(router_out)
    if core.get("loss_fwd") is not None:
        z_t = batch["z_t"]
        fwd_kwargs: dict[str, Any] = {}
        if physics_code is not None:
            fwd_kwargs["physics_code"] = physics_code
        try:
            outputs["z_pred_H"] = forward_head(
                z_t, batch["action_chunk"], batch["text_embed"], **fwd_kwargs
            )
        except TypeError:
            outputs["z_pred_H"] = forward_head(
                z_t, batch["action_chunk"], batch["text_embed"]
            )

    phy = compute_physics_losses(
        batch, outputs, weights, physics_cfg=cfg.get("physics", {})
    )

    lambda_phase = float(weights.get("lambda_phase", weights.get("lambda_phy_router", 0.1)))
    lambda_phy = float(weights.get("lambda_phy", weights.get("lambda_phy_delta", 0.1)))

    total = (
        core["loss"]
        + lambda_phase * phy["loss_phase"]
        + lambda_phy * phy["loss_phy"]
    )

    result = dict(core)
    result["loss"] = total
    result["loss_phase"] = phy["loss_phase"].detach()
    result["loss_phy"] = phy["loss_phy"].detach()
    for k, v in phy.items():
        if k not in result and hasattr(v, "detach"):
            result[k] = v.detach() if torch.is_tensor(v) else v
    return result


def count_trainable_params(modules: list) -> int:
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    *,
    forward_head,
    action_adapter=None,
    cfg: dict,
    meta: dict | None = None,
    adapter_type: str | None = None,
    physics_router=None,
    inverse_head=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from minimal_world2wam.models.world2wam_heads import resolve_adapter_type

    meta = dict(meta or {})
    if adapter_type is None:
        adapter_type = resolve_adapter_type(cfg)
    meta.setdefault("adapter_type", adapter_type)
    if action_adapter is not None:
        act_cfg = cfg.get("model", {}).get("action_adapter", {})
        meta.setdefault(
            "model_config",
            {
                "latent_dim": int(cfg.get("model", {}).get("latent_dim", 48)),
                "action_dim": int(cfg.get("model", {}).get("action_dim", 7)),
                "horizon": int(cfg.get("horizon", 10)),
                "hidden_dim": int(act_cfg.get("dit_hidden_dim", act_cfg.get("hidden_dim", 256))),
                "depth": int(act_cfg.get("dit_depth", act_cfg.get("num_layers", 4))),
                "num_heads": int(act_cfg.get("dit_num_heads", act_cfg.get("num_heads", 8))),
                "dropout": float(act_cfg.get("dit_dropout", act_cfg.get("dropout", 0.1))),
            },
        )

    payload = {
        "forward_head": forward_head.state_dict(),
        "adapter_type": adapter_type,
        "cfg": cfg,
        "meta": meta,
    }
    if inverse_head is not None:
        payload["inverse_head"] = inverse_head.state_dict()
    else:
        payload["inverse_head"] = {}
    if action_adapter is not None:
        payload["action_adapter"] = action_adapter.state_dict()
    if physics_router is not None:
        payload["physics_router"] = physics_router.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    forward_head,
    action_adapter=None,
    expected_adapter_type: str | None = None,
    physics_router=None,
    inverse_head=None,
) -> dict:
    from minimal_world2wam.models.world2wam_heads import resolve_adapter_type

    payload = torch.load(path, map_location="cpu", weights_only=False)
    forward_head.load_state_dict(payload["forward_head"])
    if inverse_head is not None and payload.get("inverse_head"):
        inv_result = inverse_head.load_state_dict(payload["inverse_head"], strict=False)
        if inv_result.missing_keys:
            print(f"  inverse missing keys (legacy compat): {inv_result.missing_keys}")
    if action_adapter is not None and "action_adapter" in payload:
        ckpt_type = resolve_adapter_type({}, payload)
        if expected_adapter_type is not None and ckpt_type != expected_adapter_type:
            print(
                f"Warning: skip adapter weights from {path} "
                f"(checkpoint type={ckpt_type}, expected={expected_adapter_type})"
            )
        else:
            action_adapter.load_state_dict(payload["action_adapter"])
    if physics_router is not None and "physics_router" in payload:
        physics_router.load_state_dict(payload["physics_router"], strict=False)
    return payload


def load_heads_warm_start(
    path: Path,
    forward_head,
    inverse_head=None,
) -> dict:
    """Load ForwardHead (+ optional legacy InverseHead) from a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    fwd_result = forward_head.load_state_dict(payload["forward_head"], strict=False)
    print(f"Warm-start from {path}")
    print(f"  forward missing: {fwd_result.missing_keys}")
    if inverse_head is not None and payload.get("inverse_head"):
        inv_result = inverse_head.load_state_dict(payload["inverse_head"], strict=False)
        print(f"  inverse missing: {inv_result.missing_keys}")
    if "action_adapter" in payload:
        print("  note: action_adapter weights present in checkpoint (load separately)")
    return payload


def append_json_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
    history.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
