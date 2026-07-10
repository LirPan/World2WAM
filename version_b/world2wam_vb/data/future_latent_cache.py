from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import torch

PathLike = Union[str, Path]


class FutureLatentCache:
    """Disk cache for precomputed VAE-pooled latents (current + future)."""

    def __init__(self, cache_dir: PathLike, dataset_name: str = "physics_world2wam_vb"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name

    def _key_parts(self, idx: int, t_anchor: int, horizon: int) -> str:
        raw = f"{self.dataset_name}:{idx}:{t_anchor}:{horizon}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def get_cache_path(self, idx: int, t_anchor: int, horizon: int) -> Path:
        key = self._key_parts(idx, t_anchor, horizon)
        return self.cache_dir / self.dataset_name / f"{key}.pt"

    def has_future_latent(self, idx: int, t_anchor: int, horizon: int) -> bool:
        return self.get_cache_path(idx, t_anchor, horizon).exists()

    def load_future_latent(self, idx: int, t_anchor: int, horizon: int) -> torch.Tensor:
        path = self.get_cache_path(idx, t_anchor, horizon)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached latent at {path}. Run scripts/01_precompute_future_latents.py"
            )
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "future_latent" in obj:
            return obj["future_latent"]
        return obj

    def save_future_latent(
        self,
        idx: int,
        t_anchor: int,
        horizon: int,
        latent: torch.Tensor,
        meta: dict | None = None,
    ) -> Path:
        path = self.get_cache_path(idx, t_anchor, horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"future_latent": latent.detach().cpu()}
        if meta:
            payload["meta"] = meta
        torch.save(payload, path)
        return path

    def has_current_latent(self, idx: int, t_anchor: int) -> bool:
        return self.has_future_latent(idx, t_anchor, horizon=0)

    def load_current_latent(self, idx: int, t_anchor: int) -> torch.Tensor:
        return self.load_future_latent(idx, t_anchor, horizon=0)

    def save_current_latent(
        self,
        idx: int,
        t_anchor: int,
        latent: torch.Tensor,
        meta: dict | None = None,
    ) -> Path:
        return self.save_future_latent(idx, t_anchor, horizon=0, latent=latent, meta=meta)
