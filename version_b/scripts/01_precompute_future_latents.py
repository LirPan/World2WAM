#!/usr/bin/env python3
"""Precompute current + future VAE-pooled latents for Version B training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

VB_ROOT = Path(__file__).resolve().parents[1]
if str(VB_ROOT) not in sys.path:
    sys.path.insert(0, str(VB_ROOT))

from world2wam_vb.adapters.fastwam_mot_adapter import FastWAMMotAdapter
from world2wam_vb.data.future_latent_cache import FutureLatentCache
from world2wam_vb.data.libero_batch_adapter import LiberoBatchAdapter, build_fastwam_dataset
from world2wam_vb.utils.config import load_config
from world2wam_vb.utils.training import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(VB_ROOT / "configs/precompute_latents.yaml"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = args.device or cfg.get("device", "cuda")

    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(
        cfg["cache_dir"],
        dataset_name=cfg.get("project_name", "physics_world2wam_vb"),
    )
    adapter_ds = LiberoBatchAdapter(
        base_ds,
        future_horizon=int(cfg.get("future_horizon", 1)),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=None,
        dataset_name=cache.dataset_name,
    )

    print("Loading FastWAM MoT adapter for VAE encode...")
    fastwam = FastWAMMotAdapter.from_config({**cfg, "device": device})
    fastwam.model.eval()

    n_limit = args.max_samples or cfg.get("precompute_max_samples")
    n = len(adapter_ds) if n_limit is None else min(len(adapter_ds), int(n_limit))
    saved_current = saved_future = skipped = 0

    for idx in tqdm(range(n), desc="precompute_latents"):
        sample = adapter_ds[idx]
        anchor = adapter_ds.anchor_action_idx

        video = sample["video"]
        t_vid = int(video.shape[1])
        t_a = int(sample["action"].shape[0])
        actions_per_vid = t_a // max(t_vid - 1, 1)
        cur_vid = anchor // actions_per_vid
        cur_obs = video[:, cur_vid : cur_vid + 1]

        if not cache.has_current_latent(idx, anchor):
            cur_latent = fastwam.encode_obs_latent(cur_obs.unsqueeze(0))
            cache.save_current_latent(
                idx,
                anchor,
                cur_latent.squeeze(0),
                meta={"future_latent_dim": int(cur_latent.shape[-1])},
            )
            saved_current += 1

        if not sample.get("valid_future") or sample.get("future_obs") is None:
            skipped += 1
            continue
        if cache.has_future_latent(idx, anchor, adapter_ds.future_horizon):
            continue

        fo = sample["future_obs"].unsqueeze(0)
        latent = fastwam.encode_future_frames(fo)
        cache.save_future_latent(
            idx,
            anchor,
            adapter_ds.future_horizon,
            latent.squeeze(0),
            meta={"future_latent_dim": int(latent.shape[-1])},
        )
        saved_future += 1

    print(
        f"Done. saved_current={saved_current}, saved_future={saved_future}, "
        f"skipped_invalid={skipped}, cache={cache.cache_dir}"
    )


if __name__ == "__main__":
    main()
