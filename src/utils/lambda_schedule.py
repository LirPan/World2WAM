from __future__ import annotations


def current_lambda_fwd(step: int, lambda_fwd: float, warmup_steps: int) -> float:
    """Linear warmup from 0 to lambda_fwd over warmup_steps (step is 0-based after increment)."""
    if warmup_steps <= 0:
        return float(lambda_fwd)
    if step <= 0:
        return 0.0
    return float(lambda_fwd) * min(1.0, step / float(warmup_steps))
