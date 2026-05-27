from __future__ import annotations

import time
from typing import Any

import torch

from .fastwam_wrapper import FastWAMWrapper


def run_action_only_batch(
    wrapper: FastWAMWrapper,
    batch: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """
    Run action-only forward and return outputs + latency_ms.
    future_head_called is always False at this layer.
    """
    start = time.perf_counter()
    with torch.no_grad():
        out = wrapper.forward_action_only(batch)
    latency_ms = (time.perf_counter() - start) * 1000.0
    out["future_head_called"] = False
    return out, latency_ms
