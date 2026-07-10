from __future__ import annotations

from typing import Any

import torch

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

PHASE_TO_ID = {name: i for i, name in enumerate(PHYSICS_PHASES)}


def batch_infer_physics_labels_v1(
    batch: dict[str, Any],
    cfg: dict | None = None,
) -> dict[str, Any]:
    from minimal_world2wam.physics.phase_labeler import get_labeler

    labeler = get_labeler("v1", cfg=cfg or {})
    return labeler.label_batch(batch)


def batch_infer_physics_labels(
    batch: dict[str, Any],
    version: str = "v1",
    cfg: dict | None = None,
) -> torch.LongTensor:
    version = str(version).lower().strip()
    if version not in ("v1", "1"):
        raise ValueError(f"Only v1 phase labels are supported; got {version}")
    out = batch_infer_physics_labels_v1(batch, cfg=cfg)
    return out["phase_id"]
