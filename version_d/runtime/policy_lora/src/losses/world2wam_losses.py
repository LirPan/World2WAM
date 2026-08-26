from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

FUTURE_LATENT_MISSING_MSG = (
    "future_latent is missing. Please run precompute_future_latents first "
    "or enable online future_obs encoding."
)


def compute_action_loss(
    pred_action: torch.Tensor | None,
    batch: dict[str, Any],
    action_loss_from_fastwam: torch.Tensor | None = None,
    loss_dict: dict[str, float] | None = None,
) -> torch.Tensor:
    """Prefer FastWAM training_loss action component; else MSE on pred vs GT."""
    if action_loss_from_fastwam is not None:
        return action_loss_from_fastwam
    if loss_dict is not None and "loss_action" in loss_dict:
        return torch.tensor(float(loss_dict["loss_action"]), device=_infer_device(batch))
    if pred_action is None:
        raise ValueError(
            "compute_action_loss: no FastWAM action loss and pred_action is None."
        )
    target = batch["action"]
    if target.dim() == 2:
        target = target.unsqueeze(0)
    mask = batch.get("action_is_pad")
    per_token = F.mse_loss(pred_action.float(), target.float(), reduction="none").mean(dim=-1)
    if mask is not None:
        valid = (~mask).to(dtype=per_token.dtype, device=per_token.device)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return ((per_token * valid).sum(dim=1) / valid_sum).mean()
    return per_token.mean()


def compute_future_latent_loss(
    pred_future_latent: torch.Tensor,
    target_future_latent: torch.Tensor,
) -> torch.Tensor:
    if target_future_latent is None:
        raise ValueError(FUTURE_LATENT_MISSING_MSG)
    return F.mse_loss(pred_future_latent.float(), target_future_latent.float())


def compute_forward_loss(
    pred_future_latent: torch.Tensor,
    target_future_latent: torch.Tensor,
) -> torch.Tensor:
    """L_fwd = MSE(pred_future_latent, target_future_latent)."""
    return compute_future_latent_loss(pred_future_latent, target_future_latent)


def compute_inverse_loss(
    pred_action_from_target_future: torch.Tensor,
    target_action: torch.Tensor,
) -> torch.Tensor:
    """L_inv = MSE(pred_action_from_target_future, target_action)."""
    return F.mse_loss(
        pred_action_from_target_future.float(),
        target_action.float(),
    )


def compute_cycle_loss(
    reconstructed_action: torch.Tensor,
    action_source: torch.Tensor,
) -> torch.Tensor:
    """L_cycle = MSE(reconstructed_action, action_source)."""
    return F.mse_loss(reconstructed_action.float(), action_source.float())


def compute_bidirectional_world2wam_loss(
    *,
    action_loss_monitor: torch.Tensor,
    pred_future_latent: torch.Tensor,
    target_future_latent: torch.Tensor,
    pred_action_from_target_future: torch.Tensor | None = None,
    target_action: torch.Tensor | None = None,
    reconstructed_action: torch.Tensor | None = None,
    action_source: torch.Tensor | None = None,
    lambda_fwd: float = 1.0,
    lambda_inv: float = 1.0,
    lambda_cycle: float = 0.0,
    enable_inverse: bool = False,
    enable_cycle: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Returns loss dict with separate logging vs backward tensors.

    loss_train_backward = lambda_fwd * loss_fwd + lambda_inv * loss_inv + lambda_cycle * loss_cycle
    loss_action_monitor is logged only (no grad into FastWAM when frozen).
    """
    loss_fwd = compute_forward_loss(pred_future_latent, target_future_latent)
    loss_inv = torch.zeros((), device=loss_fwd.device, dtype=loss_fwd.dtype)
    loss_cycle = torch.zeros((), device=loss_fwd.device, dtype=loss_fwd.dtype)

    if enable_inverse:
        if pred_action_from_target_future is None or target_action is None:
            raise ValueError("inverse loss requires pred_action_from_target_future and target_action")
        loss_inv = compute_inverse_loss(pred_action_from_target_future, target_action)

    if enable_cycle:
        if reconstructed_action is None or action_source is None:
            raise ValueError("cycle loss requires reconstructed_action and action_source")
        loss_cycle = compute_cycle_loss(reconstructed_action, action_source)

    loss_train_backward = (
        float(lambda_fwd) * loss_fwd
        + float(lambda_inv) * loss_inv
        + float(lambda_cycle) * loss_cycle
    )

    loss_action_monitor = action_loss_monitor.detach()
    loss_total_logged = (
        loss_action_monitor
        + float(lambda_fwd) * loss_fwd.detach()
        + float(lambda_inv) * loss_inv.detach()
        + float(lambda_cycle) * loss_cycle.detach()
    )

    return {
        "loss_fwd": loss_fwd,
        "loss_inv": loss_inv,
        "loss_cycle": loss_cycle,
        "loss_action_monitor": loss_action_monitor,
        "loss_total_logged": loss_total_logged,
        "loss_train_backward": loss_train_backward,
    }


def compute_total_loss(
    action_loss: torch.Tensor,
    future_loss: torch.Tensor | None,
    *,
    use_future_latent_distill: bool,
    lambda_fwd: float,
    batch: dict[str, Any],
    current_lambda_fwd: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    lam = float(lambda_fwd if current_lambda_fwd is None else current_lambda_fwd)
    metrics: dict[str, float] = {"loss_action": float(action_loss.detach().item())}
    if not use_future_latent_distill:
        metrics["loss_total"] = metrics["loss_action"]
        metrics["current_lambda_fwd"] = lam
        return action_loss, metrics

    fl = batch.get("future_latent")
    if fl is None:
        raise ValueError(FUTURE_LATENT_MISSING_MSG)
    if future_loss is None:
        raise ValueError("future_loss is None but use_future_latent_distill=True")

    total = action_loss + lam * future_loss
    metrics["loss_future"] = float(future_loss.detach().item())
    metrics["loss_total"] = float(total.detach().item())
    metrics["current_lambda_fwd"] = lam
    return total, metrics


def build_policy_distill_metrics(
    *,
    backbone_mode: str,
    trainable_param_count: int,
    hidden_detached: bool,
    loss_action: float,
    loss_future: float,
    current_lambda_fwd: float,
    loss_total: float,
    experiment_role: str,
    global_step: int,
    epoch: int,
) -> dict[str, float | int | str | bool]:
    return {
        "backbone_mode": backbone_mode,
        "trainable_param_count": int(trainable_param_count),
        "hidden_detached": bool(hidden_detached),
        "loss_action": float(loss_action),
        "loss_future": float(loss_future),
        "current_lambda_fwd": float(current_lambda_fwd),
        "loss_total": float(loss_total),
        "experiment_role": experiment_role,
        "global_step": int(global_step),
        "epoch": int(epoch),
    }


def _infer_device(batch: dict[str, Any]) -> torch.device:
    for key in ("video", "action", "future_latent"):
        v = batch.get(key)
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")
