from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..utils.import_utils import add_fastwam_path
from .future_latent_cache import FutureLatentCache


def future_video_index(
    t_anchor: int,
    future_horizon: int,
    t_vid: int,
    actions_per_vid: int,
) -> int | None:
    """Map action-step horizon to video frame index within clip."""
    vid_idx = (t_anchor + future_horizon) // actions_per_vid + 1
    if vid_idx >= t_vid:
        return None
    return vid_idx


def collate_world2wam_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors; keep prompt/language as lists."""
    if not samples:
        raise ValueError("Empty batch")
    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        vals = [s[key] for s in samples]
        if vals[0] is None:
            batch[key] = None
            continue
        if isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            batch[key] = vals
        else:
            batch[key] = vals
    return batch


def build_fastwam_dataset(cfg: dict[str, Any]):
    """
    Instantiate FastWAM RobotVideoDataset via Hydra (read-only import).
    """
    fastwam_root = Path(cfg["fastwam_root"])
    add_fastwam_path(fastwam_root)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    task = cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4")
    config_dir = fastwam_root / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(config_name="train", overrides=[f"task={task}"])

    data_cfg = hydra_cfg.data
    if cfg.get("lerobot_dataset_dirs"):
        OmegaConf.update(data_cfg.train, "dataset_dirs", cfg["lerobot_dataset_dirs"])

    import os

    os.chdir(fastwam_root)
    runs_dir = fastwam_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    stats_path = cfg.get("dataset_stats_path")
    if not stats_path:
        default_stats = fastwam_root / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
        if default_stats.is_file():
            stats_path = str(default_stats.resolve())
    if stats_path and not Path(stats_path).is_file():
        raise FileNotFoundError(
            f"dataset_stats_path not found: {stats_path}. "
            "Download libero_uncond_2cam224_dataset_stats.json from yuanty/fastwam."
        )

    if stats_path:
        dataset = instantiate(data_cfg.train, pretrained_norm_stats=stats_path)
    else:
        dataset = instantiate(data_cfg.train)
    return dataset, hydra_cfg


class LiberoDatasetAdapter(Dataset):
    """
    Wraps FastWAM RobotVideoDataset (LeRobot LIBERO).
    Adds future_obs, future_latent (from cache), language alias.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        future_horizon: int = 1,
        anchor_action_idx: int = 0,
        cache: FutureLatentCache | None = None,
        dataset_name: str = "libero_lerobot",
    ):
        self.base = base_dataset
        self.future_horizon = int(future_horizon)
        self.anchor_action_idx = int(anchor_action_idx)
        self.cache = cache
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        video = sample["video"]  # [3, T_vid, H, W]
        action = sample["action"]  # [T_a, 7]
        t_vid = int(video.shape[1])
        t_a = int(action.shape[0])
        actions_per_vid = t_a // max(t_vid - 1, 1)

        vid_idx = future_video_index(
            self.anchor_action_idx,
            self.future_horizon,
            t_vid,
            actions_per_vid,
        )

        future_obs = None
        valid_future = False
        if vid_idx is not None:
            future_obs = video[:, vid_idx : vid_idx + 1].clone()
            valid_future = True

        future_latent = None
        if self.cache is not None and valid_future:
            if self.cache.has_future_latent(idx, self.anchor_action_idx, self.future_horizon):
                future_latent = self.cache.load_future_latent(
                    idx, self.anchor_action_idx, self.future_horizon
                )

        obs = video[:, 0]  # [3, H, W]

        return {
            "obs": obs,
            "video": video,
            "language": sample.get("prompt", ""),
            "prompt": sample.get("prompt", ""),
            "action": action,
            "proprio": sample.get("proprio"),
            "context": sample.get("context"),
            "context_mask": sample.get("context_mask"),
            "image_is_pad": sample.get("image_is_pad"),
            "action_is_pad": sample.get("action_is_pad"),
            "proprio_is_pad": sample.get("proprio_is_pad"),
            "future_obs": future_obs,
            "future_latent": future_latent,
            "valid_future": valid_future,
            "anchor_action_idx": self.anchor_action_idx,
            "sample_idx": idx,
        }
