from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def anchor_from_batch(batch: dict[str, Any], default: int = 0) -> int:
    anchor_t = batch.get("anchor_action_idx", default)
    if isinstance(anchor_t, torch.Tensor):
        return int(anchor_t[0].item())
    if isinstance(anchor_t, list):
        return int(anchor_t[0])
    return int(anchor_t)


def gt_action_from_batch(batch: dict[str, Any], anchor: int) -> torch.Tensor:
    act = batch["action"]
    if act.dim() == 3:
        return act[:, anchor].float()
    return act.float()
