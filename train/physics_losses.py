from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from minimal_world2wam.physics.physics_labels import PHYSICS_PHASES, batch_infer_physics_labels_v1


def _zero(ref: torch.Tensor | None) -> torch.Tensor:
    if ref is not None:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return torch.zeros(())


def latent_delta_direction_loss(
    z_t: torch.Tensor,
    z_pred_H: torch.Tensor,
    z_tH: torch.Tensor,
) -> torch.Tensor:
    """L_phy: 1 - cosine(delta_pred, delta_real)."""
    if z_t is None or z_pred_H is None or z_tH is None:
        return _zero(z_t)
    delta_pred = z_pred_H - z_t
    delta_gt = z_tH - z_t
    cos = F.cosine_similarity(delta_pred, delta_gt, dim=-1)
    return (1.0 - cos).mean()


def physics_router_loss(
    logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    confidence: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    use_confidence_weight: bool = True,
) -> torch.Tensor:
    """L_phase: confidence-weighted CE(student, teacher)."""
    if logits is None or labels is None:
        return _zero(logits)
    ce = F.cross_entropy(logits, labels, reduction="none", weight=class_weights)
    if use_confidence_weight and confidence is not None:
        ce = ce * confidence.to(ce.dtype)
    return ce.mean()


def _class_weights_from_freq(freq: dict[str, float], num_phases: int, eps: float = 1e-6) -> torch.Tensor:
    weights = []
    for i in range(num_phases):
        name = PHYSICS_PHASES[i]
        f = float(freq.get(name, freq.get(str(i), 1.0)))
        weights.append(1.0 / (f + eps) ** 0.5)
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.mean()


def compute_physics_metrics(
    logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    probs: torch.Tensor | None,
    confidence: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    metrics: dict[str, torch.Tensor | float] = {}
    if logits is None or labels is None:
        return metrics
    pred = logits.argmax(dim=-1)
    metrics["phase_acc_pseudo"] = (pred == labels).float().mean().detach()
    if probs is not None:
        p_mean = probs.mean(dim=0).clamp(min=1e-8)
        metrics["phase_entropy"] = (-(p_mean * p_mean.log()).sum()).detach()
    if confidence is not None:
        metrics["confidence_mean"] = confidence.mean().detach()
    return metrics


def compute_physics_losses(
    batch: dict[str, Any],
    outputs: dict[str, Any],
    weights: dict[str, float],
    physics_cfg: dict | None = None,
) -> dict[str, torch.Tensor]:
    """
    Physics losses: L_phase + L_phy only (Version A).
    """
    physics_cfg = physics_cfg or {}
    z_t = batch.get("z_t")
    z_tH = batch.get("z_tH")

    z_pred_H = outputs.get("z_pred_H")

    physics_logits = outputs.get("physics_logits")
    if physics_logits is None:
        physics_logits = outputs.get("phase_logits")
    physics_probs = outputs.get("physics_probs")
    if physics_probs is None:
        physics_probs = outputs.get("phase_prob")

    ref = z_t if isinstance(z_t, torch.Tensor) else None

    loss_phy = latent_delta_direction_loss(z_t, z_pred_H, z_tH)

    labels = None
    label_confidence = None
    if batch.get("action_chunk") is not None:
        try:
            label_out = batch_infer_physics_labels_v1(batch, cfg=physics_cfg)
            labels = label_out["phase_id"]
            label_confidence = label_out["confidence"]
            if ref is not None:
                labels = labels.to(ref.device)
                if label_confidence is not None:
                    label_confidence = label_confidence.to(ref.device)
        except Exception:
            labels = None
            label_confidence = None

    class_weights = None
    if weights.get("use_class_balance", physics_cfg.get("use_class_balance", False)):
        freq = physics_cfg.get("phase_freq") or weights.get("phase_freq")
        if freq and physics_logits is not None:
            class_weights = _class_weights_from_freq(freq, len(PHYSICS_PHASES)).to(
                physics_logits.device
            )

    use_conf = bool(weights.get("use_confidence_weight", physics_cfg.get("use_confidence_weight", True)))

    loss_phase = physics_router_loss(
        physics_logits,
        labels,
        confidence=label_confidence,
        class_weights=class_weights,
        use_confidence_weight=use_conf,
    )

    result = {
        "loss_phy": loss_phy,
        "loss_phy_delta": loss_phy,
        "loss_phase": loss_phase,
        "loss_phy_router": loss_phase,
    }
    result.update(
        compute_physics_metrics(physics_logits, labels, physics_probs, label_confidence)
    )
    return result
