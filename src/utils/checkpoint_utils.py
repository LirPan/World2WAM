from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from .path_utils import minimal_project_root, resolve_path

OFFICIAL_CKPT_HINT = (
    "Download the official FastWAM LIBERO checkpoint to:\n"
    "  {fastwam_root}/checkpoints/fastwam_release/libero_uncond_2cam224.pt\n"
    "See FastWAM README / https://github.com/yuantianyuan01/FastWAM"
)

FUTURE_CACHE_HINT = (
    "Precompute future latents first:\n"
    "  bash scripts/02_precompute_future_latents.sh\n"
    "Expected cache under: {cache_dir}/{project_name}/*.pt"
)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve paths and align checkpoint_path with official_fastwam_checkpoint."""
    out = copy.deepcopy(cfg)
    root = minimal_project_root()
    for key in (
        "fastwam_root",
        "libero_root",
        "cache_dir",
        "output_dir",
        "official_fastwam_checkpoint",
        "checkpoint_path",
        "dataset_stats_path",
        "future_head_checkpoint",
        "inverse_head_checkpoint",
    ):
        if key in out and out[key] is not None:
            out[key] = str(resolve_path(out[key], root))

    official = out.get("official_fastwam_checkpoint") or out.get("checkpoint_path")
    if official:
        out["official_fastwam_checkpoint"] = str(official)
        if not out.get("checkpoint_path"):
            out["checkpoint_path"] = str(official)

    dirs = out.get("lerobot_dataset_dirs")
    if dirs:
        out["lerobot_dataset_dirs"] = [str(resolve_path(d, root)) for d in dirs]

    return out


def resolve_official_checkpoint(cfg: dict[str, Any]) -> Path:
    cfg = normalize_config(cfg)
    path_str = cfg.get("official_fastwam_checkpoint") or cfg.get("checkpoint_path")
    if not path_str:
        fastwam_root = cfg.get("fastwam_root", "../code/FastWAM")
        path_str = str(
            resolve_path(
                f"{fastwam_root}/checkpoints/fastwam_release/libero_uncond_2cam224.pt",
                minimal_project_root(),
            )
        )
    path = Path(path_str)
    if not path.is_file():
        hint = OFFICIAL_CKPT_HINT.format(
            fastwam_root=cfg.get("fastwam_root", "<fastwam_root>")
        )
        raise FileNotFoundError(f"Official FastWAM checkpoint not found: {path}\n{hint}")
    return path


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_resolved_config(cfg: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(normalize_config(cfg), f, sort_keys=False, allow_unicode=True)
    return path


def verify_future_latent_cache(cfg: dict[str, Any], sample_indices: list[int] | None = None) -> None:
    from src.data.future_latent_cache import FutureLatentCache

    cfg = normalize_config(cfg)
    cache = FutureLatentCache(cfg["cache_dir"], dataset_name=cfg.get("project_name", "world2wam_minimal"))
    anchor = int(cfg.get("anchor_action_idx", 0))
    horizon = int(cfg.get("future_horizon", 1))
    indices = sample_indices if sample_indices is not None else [0, 1, 2, 3, 4]
    found = [i for i in indices if cache.has_future_latent(i, anchor, horizon)]
    if not found:
        missing = indices
        hint = FUTURE_CACHE_HINT.format(
            cache_dir=cfg["cache_dir"],
            project_name=cfg.get("project_name", "world2wam_minimal"),
        )
        raise FileNotFoundError(
            f"future_latent cache missing for sample indices {missing}.\n{hint}"
        )


def save_world2wam_checkpoint(
    path: Path,
    *,
    backbone_mode: str,
    official_checkpoint: str | Path,
    future_head_state: dict[str, Any] | None = None,
    backbone_extra: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "backbone_mode": backbone_mode,
        "official_checkpoint": str(official_checkpoint),
        "meta": meta or {},
    }
    if future_head_state is not None:
        payload["future_head"] = future_head_state
    if backbone_extra:
        payload["backbone_extra"] = backbone_extra
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_world2wam_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"World2WAM checkpoint bundle not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
