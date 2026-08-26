from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def config_sha256(cfg: dict[str, Any]) -> str:
    raw = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_resume_checkpoint(value: str | None, ckpt_dir: Path) -> Path | None:
    if value in (None, "", "none"):
        return None
    if value != "auto":
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    candidates = sorted(
        ckpt_dir.glob("world2wam_step*.pt"),
        key=lambda path: int(path.stem.removeprefix("world2wam_step") or "0"),
    )
    return candidates[-1] if candidates else None


def make_trainer_state(
    *,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "seed": int(seed),
        "rng": capture_rng_state(),
        "config_sha256": config_sha256(cfg),
    }
