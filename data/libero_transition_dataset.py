from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from minimal_world2wam.wrappers.fastwam_encoder import add_fastwam_path


def action_to_video_frame(action_idx: int, actions_per_vid: int) -> int:
    """Map action step index to subsampled video frame index."""
    return action_idx // actions_per_vid


def future_video_frame(action_idx: int, horizon: int, actions_per_vid: int, num_video_frames: int) -> int | None:
    """Frame index for obs at t+H; None if out of clip."""
    target_action = action_idx + horizon
    vid_idx = target_action // actions_per_vid
    if vid_idx >= num_video_frames:
        return None
    return vid_idx


def build_fastwam_dataset(cfg: dict[str, Any]):
    """Instantiate FastWAM RobotVideoDataset via Hydra (read-only)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    fastwam_root = Path(cfg["fastwam_root"])
    add_fastwam_path(fastwam_root)

    task = cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4")
    config_dir = fastwam_root / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(config_name="train", overrides=[f"task={task}"])

    data_cfg = hydra_cfg.data
    if cfg.get("lerobot_dataset_dirs"):
        rel_dirs = []
        for d in cfg["lerobot_dataset_dirs"]:
            p = Path(d)
            try:
                rel_dirs.append("./" + str(p.relative_to(fastwam_root)).replace("\\", "/"))
            except ValueError:
                rel_dirs.append(str(p))
        OmegaConf.update(data_cfg.train, "dataset_dirs", rel_dirs)

    import os

    os.chdir(fastwam_root)
    runs_dir = fastwam_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    stats_path = cfg.get("dataset_stats_path")
    if stats_path and not Path(stats_path).is_file():
        raise FileNotFoundError(f"dataset_stats_path not found: {stats_path}")

    if stats_path:
        dataset = instantiate(data_cfg.train, pretrained_norm_stats=stats_path)
    else:
        dataset = instantiate(data_cfg.train)
    return dataset, hydra_cfg


class LiberoTransitionIndex:
    """Flat index over (clip_idx, action_t) transitions within clips."""

    def __init__(self, base_len: int, num_actions: int, horizon: int, max_samples: int | None = None):
        self.entries: list[tuple[int, int]] = []
        self.horizon = horizon
        for clip_idx in range(base_len):
            for t in range(num_actions - horizon):
                self.entries.append((clip_idx, t))
                if max_samples is not None and len(self.entries) >= max_samples:
                    break
            if max_samples is not None and len(self.entries) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.entries)


class LiberoTransitionDataset(Dataset):
    """
    Transition-level LIBERO samples from FastWAM RobotVideoDataset clips.

    Each sample:
        obs_t, obs_tH, instruction, action_chunk[H,7], state_t, state_tH, metadata
    Physics fields reserved for idea3 (None if unavailable).
    """

    PHYSICS_KEYS = (
        "object_state_t",
        "object_state_tH",
        "eef_state_t",
        "eef_state_tH",
        "gripper_state",
        "contact_flag",
        "object_displacement",
    )

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        horizon: int = 10,
        max_samples: int | None = None,
    ):
        self.base = base_dataset
        self.horizon = int(horizon)

        sample0 = base_dataset[0]
        self._num_actions = int(sample0["action"].shape[0])
        self._num_video = int(sample0["video"].shape[1])
        self._actions_per_vid = self._num_actions // max(self._num_video - 1, 1)

        self._index = LiberoTransitionIndex(
            len(base_dataset),
            self._num_actions,
            self.horizon,
            max_samples=max_samples,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_idx, t = self._index.entries[idx]
        sample = self.base[clip_idx]
        video = sample["video"]  # [3, T_vid, H, W]
        action = sample["action"]  # [T_a, 7]
        proprio = sample.get("proprio")

        frame_t = action_to_video_frame(t, self._actions_per_vid)
        frame_tH = future_video_frame(t, self.horizon, self._actions_per_vid, video.shape[1])
        if frame_tH is None:
            raise IndexError(f"Invalid transition idx={idx} clip={clip_idx} t={t}")

        obs_t = video[:, frame_t].clone()
        obs_tH = video[:, frame_tH].clone()
        action_chunk = action[t : t + self.horizon].clone()

        state_t = proprio[frame_t].clone() if proprio is not None else None
        state_tH = proprio[frame_tH].clone() if proprio is not None else None

        instruction = sample.get("prompt", "")

        out: dict[str, Any] = {
            "obs_t": obs_t,
            "obs_tH": obs_tH,
            "instruction": instruction,
            "action_chunk": action_chunk,
            "state_t": state_t,
            "state_tH": state_tH,
            "task_id": sample.get("task_id", clip_idx),
            "episode_id": sample.get("episode_index", clip_idx),
            "t": t,
            "clip_idx": clip_idx,
            "transition_idx": idx,
            "context": sample.get("context"),
            "context_mask": sample.get("context_mask"),
            "video": video,
            "proprio": proprio,
        }
        for k in self.PHYSICS_KEYS:
            out[k] = None
        return out


def collate_transitions(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Empty batch")
    batch: dict[str, Any] = {}
    for key in samples[0].keys():
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
