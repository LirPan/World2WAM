from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset


def list_cache_files(cache_dir: str | Path) -> list[Path]:
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")
    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt cache files in {cache_dir}")
    return files


def load_meta(cache_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found in {cache_dir}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


class LatentCacheDataset(Dataset):
    """Load precomputed World2WAM latent cache files."""

    def __init__(self, cache_dir: str | Path, load_state: bool = True):
        self.cache_dir = Path(cache_dir)
        self.files = list_cache_files(self.cache_dir)
        self.meta = load_meta(self.cache_dir)
        self.load_state = load_state

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        payload = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        sample = {
            "z_t": payload["z_t"].float(),
            "z_tH": payload["z_tH"].float(),
            "text_embed": payload["text_embed"].float(),
            "action_chunk": payload["action_chunk"].float(),
            "metadata": {
                k: payload.get(k)
                for k in ("task_id", "episode_id", "t", "transition_idx", "clip_idx")
            },
        }
        if self.load_state:
            if "state_t" in payload:
                sample["state_t"] = payload["state_t"].float()
            if "state_tH" in payload:
                sample["state_tH"] = payload["state_tH"].float()
        return sample


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def collate_latent_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "z_t": torch.stack([s["z_t"] for s in samples], dim=0),
        "z_tH": torch.stack([s["z_tH"] for s in samples], dim=0),
        "text_embed": torch.stack([s["text_embed"] for s in samples], dim=0),
        "action_chunk": torch.stack([s["action_chunk"] for s in samples], dim=0),
        "metadata": [s["metadata"] for s in samples],
    }
    if "state_t" in samples[0]:
        batch["state_t"] = torch.stack([s["state_t"] for s in samples], dim=0)
    if "state_tH" in samples[0]:
        batch["state_tH"] = torch.stack([s["state_tH"] for s in samples], dim=0)
    return batch


def detect_state_dim(cache_dir: str | Path) -> int:
    """Return state dim from first cache file with state_t, else 0."""
    files = list_cache_files(cache_dir)
    for fp in files[:32]:
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        if "state_t" in payload:
            st = payload["state_t"].float()
            return int(st.reshape(-1).shape[0])
    return 0
