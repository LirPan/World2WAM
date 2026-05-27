#!/usr/bin/env python3
"""Precompute future VAE-pooled latents for LIBERO LeRobot clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
if str(_MINIMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMAL_ROOT))

from src.data.future_latent_cache import FutureLatentCache
from src.data.libero_dataset_adapter import LiberoDatasetAdapter, build_fastwam_dataset, collate_world2wam_batch
from src.utils.config import load_config
from src.utils.import_utils import add_fastwam_path
from src.utils.seed import set_seed
from src.wrappers.fastwam_wrapper import FastWAMWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute future latents with FastWAM VAE")
    parser.add_argument("--config", type=str, default="configs/fastwam_future_distill.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(_MINIMAL_ROOT / args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = args.device or cfg.get("device", "cuda")

    add_fastwam_path(cfg["fastwam_root"])
    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(cfg["cache_dir"], dataset_name=cfg.get("project_name", "world2wam"))
    adapter = LiberoDatasetAdapter(
        base_ds,
        future_horizon=int(cfg.get("future_horizon", 1)),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=None,
    )

    print("Loading FastWAM for VAE encode (weights must be available)...")
    try:
        wrapper = FastWAMWrapper(
            fastwam_root=cfg["fastwam_root"],
            fastwam_task_config=cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4"),
            checkpoint_path=cfg.get("checkpoint_path"),
            freeze_backbone=True,
            device=device,
            mixed_precision=cfg.get("mixed_precision", "bf16"),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load FastWAM for VAE encoding. "
            "Confirm Wan2.2 weights and ActionDiT checkpoint paths in FastWAM configs. "
            f"Original error: {exc}"
        ) from exc

    wrapper.model.eval()
    n = len(adapter) if args.max_samples is None else min(len(adapter), args.max_samples)
    skipped = 0
    saved = 0

    for idx in tqdm(range(n), desc="precompute"):
        sample = adapter[idx]
        if not sample.get("valid_future") or sample.get("future_obs") is None:
            skipped += 1
            continue
        if cache.has_future_latent(idx, adapter.anchor_action_idx, adapter.future_horizon):
            continue

        fo = sample["future_obs"].unsqueeze(0).to(device=wrapper.model.device)
        try:
            latent = wrapper.encode_future_latent(fo)
        except Exception as exc:
            raise RuntimeError(
                "vae.encode failed. TODO: confirm FastWAM._encode_video_latents path and "
                "that future_obs shape is [B,3,1,H,W] with H,W multiples of 16. "
                f"sample_idx={idx}, shape={tuple(fo.shape)}, error={exc}"
            ) from exc

        cache.save_future_latent(
            idx,
            adapter.anchor_action_idx,
            adapter.future_horizon,
            latent.squeeze(0),
            meta={"future_latent_dim": int(latent.shape[-1])},
        )
        saved += 1

    print(f"Done. saved={saved}, skipped_invalid={skipped}, cache_dir={cache.cache_dir}")


if __name__ == "__main__":
    main()
