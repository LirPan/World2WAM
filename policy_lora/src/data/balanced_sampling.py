from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _prompt_from_sample(sample: dict[str, Any]) -> str:
    return str(sample.get("prompt") or sample.get("language") or "")


def build_balanced_weights(
    dataset,
    *,
    keywords: Sequence[str],
    hard_fraction: float,
    manifest_path: Path,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    if not 0.0 < hard_fraction < 1.0:
        raise ValueError(f"hard_fraction must be in (0,1), got {hard_fraction}")
    lowered = [word.lower() for word in keywords]
    labels: list[bool]
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("length") == len(dataset) and payload.get("keywords") == lowered:
            labels = [bool(value) for value in payload["is_hard"]]
        else:
            labels = []
    else:
        labels = []

    if not labels:
        labels = []
        prompts: list[str] = []
        for idx in range(len(dataset)):
            prompt = _prompt_from_sample(dataset[idx])
            prompts.append(prompt)
            labels.append(any(word in prompt.lower() for word in lowered))
        _atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "length": len(dataset),
                "keywords": lowered,
                "is_hard": labels,
                "prompts": prompts,
            },
        )

    hard_count = sum(labels)
    easy_count = len(labels) - hard_count
    if hard_count == 0 or easy_count == 0:
        raise RuntimeError(
            f"Balanced sampling needs both groups; hard={hard_count} easy={easy_count}"
        )
    hard_weight = hard_fraction / hard_count
    easy_weight = (1.0 - hard_fraction) / easy_count
    weights = torch.tensor(
        [hard_weight if is_hard else easy_weight for is_hard in labels],
        dtype=torch.double,
    )
    return weights, {
        "hard_count": hard_count,
        "easy_count": easy_count,
        "hard_fraction": hard_fraction,
    }


def cached_indices(
    *,
    cache,
    max_samples: int,
    anchor_action_idx: int,
    future_horizon: int,
) -> list[int]:
    return [
        idx
        for idx in range(max_samples)
        if cache.has_future_latent(idx, anchor_action_idx, future_horizon)
    ]
