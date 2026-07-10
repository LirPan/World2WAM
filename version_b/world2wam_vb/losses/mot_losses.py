from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_bidirectional_losses(
    *,
    loss_fastwam_action: torch.Tensor,
    loss_fastwam_video: torch.Tensor | None = None,
    loss_fwd: torch.Tensor,
    loss_flow: torch.Tensor,
    loss_inverse: torch.Tensor,
    loss_cycle: torch.Tensor,
    lambda_fastwam_action: float = 0.0,
    lambda_fastwam_video: float = 0.0,
    lambda_fwd: float = 1.0,
    lambda_flow: float = 1.0,
    lambda_inverse: float = 0.1,
    lambda_cycle: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Core Version B losses before physics terms."""
    device = loss_fwd.device
    zero = torch.zeros((), device=device)

    l_vid = loss_fastwam_video if loss_fastwam_video is not None else zero
    l_act = loss_fastwam_action.detach() if lambda_fastwam_action == 0 else loss_fastwam_action

    total = (
        float(lambda_fastwam_action) * l_act
        + float(lambda_fastwam_video) * l_vid
        + float(lambda_fwd) * loss_fwd
        + float(lambda_flow) * loss_flow
        + float(lambda_inverse) * loss_inverse
        + float(lambda_cycle) * loss_cycle
    )

    return {
        "loss": total,
        "loss_fastwam_action": l_act.detach() if hasattr(l_act, "detach") else l_act,
        "loss_fastwam_video": l_vid.detach() if hasattr(l_vid, "detach") else l_vid,
        "loss_fwd": loss_fwd.detach(),
        "loss_future": loss_fwd.detach(),
        "loss_flow": loss_flow.detach(),
        "loss_inverse": loss_inverse.detach(),
        "loss_cycle": loss_cycle.detach(),
    }


def compute_total_mot_loss(
    core: dict[str, torch.Tensor],
    physics: dict[str, torch.Tensor],
    *,
    lambda_phase: float = 0.1,
    lambda_phy: float = 0.1,
) -> dict[str, torch.Tensor]:
    total = (
        core["loss"]
        + float(lambda_phase) * physics["loss_phase"]
        + float(lambda_phy) * physics["loss_phy"]
    )
    out = dict(core)
    out["loss"] = total
    out["loss_phase"] = physics["loss_phase"].detach()
    out["loss_phy"] = physics["loss_phy"].detach()
    for k, v in physics.items():
        if k.startswith("phase_") and k not in out:
            out[k] = v.detach() if torch.is_tensor(v) else v
    return out
