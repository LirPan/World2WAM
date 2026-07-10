from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from world2wam_vb.models.mot_physics_router import PHYSICS_PHASES


def _zero(ref: torch.Tensor | None) -> torch.Tensor:
    if ref is not None:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return torch.zeros(())


def latent_delta_direction_loss(
    z_pred: torch.Tensor | None,
    z_target: torch.Tensor | None,
    z_current: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_phy: cosine alignment of predicted vs real future latent transition."""
    if z_pred is None or z_target is None:
        return _zero(z_pred)
    if z_current is not None and z_current.shape == z_target.shape:
        delta_pred = z_pred - z_current
        delta_gt = z_target - z_current
    else:
        delta_pred = z_pred
        delta_gt = z_target
    cos = F.cosine_similarity(delta_pred, delta_gt, dim=-1)
    return (1.0 - cos).mean()


def physics_router_loss(
    logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits is None or labels is None:
        return _zero(logits)
    ce = F.cross_entropy(logits, labels, reduction="none")
    if confidence is not None:
        ce = ce * confidence.to(ce.dtype)
    return ce.mean()


def infer_teacher_labels_from_batch(batch: dict[str, Any], cfg: dict | None = None) -> dict[str, torch.Tensor]:
    """
    Teacher pseudo-labels using privileged future information.

    Reuses Version A TeacherPhysicsLabeler when minimal_world2wam is importable;
    otherwise falls back to simple action-norm heuristic.
    """
    cfg = cfg or {}
    try:
        import sys
        from pathlib import Path

        vb_root = Path(__file__).resolve().parents[2]
        workspace = vb_root.parent
        for candidate in (workspace, workspace.parent):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            from physics.phase_labeler import TeacherPhysicsLabeler
        except ImportError:
            from minimal_world2wam.physics.phase_labeler import TeacherPhysicsLabeler

        action = batch["action"]
        b, t, a = action.shape
        pseudo_batch = {
            "action_chunk": action,
            "z_t": batch.get("h_t", torch.zeros(b, 48, device=action.device)),
            "z_tH": batch.get("future_latent", torch.zeros(b, 48, device=action.device)),
        }
        if batch.get("proprio") is not None:
            pseudo_batch["state_t"] = batch["proprio"][:, 0]
        labeler = TeacherPhysicsLabeler(cfg=cfg)
        out = labeler.label_batch(pseudo_batch)
        return {"phase_id": out["phase_id"], "confidence": out["confidence"]}
    except Exception:
        act = batch["action"]
        norm = act.float().norm(dim=-1).mean(dim=-1)
        phase_id = torch.zeros(act.shape[0], dtype=torch.long, device=act.device)
        phase_id[norm > 0.2] = PHYSICS_PHASES.index("transport")
        phase_id[norm < 0.05] = PHYSICS_PHASES.index("free_motion")
        conf = torch.full((act.shape[0],), 0.5, device=act.device)
        return {"phase_id": phase_id, "confidence": conf}


def compute_physics_mot_losses(
    batch: dict[str, Any],
    outputs: dict[str, Any],
    weights: dict[str, float],
    *,
    physics_cfg: dict | None = None,
) -> dict[str, torch.Tensor]:
    physics_cfg = physics_cfg or {}
    z_pred = outputs.get("z_future_pred")
    z_target = batch.get("future_latent")
    if z_target is not None and z_target.dim() == 1:
        z_target = z_target.unsqueeze(0)

    loss_phy = latent_delta_direction_loss(
        z_pred,
        z_target.float() if z_target is not None else None,
        batch.get("current_latent"),
    )

    labels = infer_teacher_labels_from_batch(batch, physics_cfg)
    loss_phase = physics_router_loss(
        outputs.get("phase_logits"),
        labels["phase_id"],
        labels.get("confidence"),
    )

    result = {
        "loss_phy": loss_phy,
        "loss_phase": loss_phase,
    }
    if outputs.get("phase_logits") is not None:
        pred = outputs["phase_logits"].argmax(dim=-1)
        result["phase_acc_pseudo"] = (pred == labels["phase_id"]).float().mean().detach()
    return result
