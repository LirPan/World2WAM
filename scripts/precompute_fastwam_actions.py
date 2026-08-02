#!/usr/bin/env python3
"""Precompute FastWAM teacher actions into existing latent cache .pt files.

Writes field `fastwam_action` (normalized action chunk, same space as `action_chunk`)
so Version C can train δ = action_chunk - fastwam_action.

Usage:
  python minimal_world2wam/scripts/precompute_fastwam_actions.py \\
    --config configs/world2wam_physics_residual_flow_dit_vc.yaml \\
    --cache_dir cache/libero_spatial_h10_full_fastwam \\
    --resume --shard_id 0 --num_shards 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from minimal_world2wam.data.libero_transition_dataset import LiberoTransitionDataset, build_fastwam_dataset
from minimal_world2wam.utils.config import load_config
from minimal_world2wam.utils.seed import set_seed
from minimal_world2wam.wrappers.fastwam_encoder import load_frozen_fastwam


def _list_cache_indices(cache_dir: Path) -> list[int]:
    idxs: list[int] = []
    for p in cache_dir.glob("*.pt"):
        if p.stem.isdigit():
            idxs.append(int(p.stem))
    return sorted(idxs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/world2wam_physics_residual_flow_dit_vc.yaml")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--resume", action="store_true", help="Skip files that already have fastwam_action")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None, help="Only process first N cache indices")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument(
        "--require_existing_cache",
        action="store_true",
        default=True,
        help="Only write into existing {idx:06d}.pt files (default True)",
    )
    args = parser.parse_args()

    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id must be in [0, {args.num_shards}), got {args.shard_id}")

    cfg = load_config(WORKSPACE / args.config)
    if args.device:
        cfg.setdefault("fastwam", {})["device"] = args.device
    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(cfg["cache"]["output_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (WORKSPACE / cache_dir).resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    indices = _list_cache_indices(cache_dir)
    if args.max_samples is not None:
        indices = [i for i in indices if i < int(args.max_samples)]
    if not indices:
        raise FileNotFoundError(f"No numeric .pt cache files in {cache_dir}")

    horizon = int(cfg.get("horizon", 10))
    print(f"Cache files to consider: {len(indices)} under {cache_dir}")
    print(f"Shard {args.shard_id}/{args.num_shards}")

    print("Loading FastWAM encoder (frozen)...")
    encoder = load_frozen_fastwam(cfg)
    encoder.model.eval()
    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("Loading LIBERO transition dataset (same indexing as latent precompute)...")
    base_ds, _ = build_fastwam_dataset(cfg)
    # Index must cover the largest cache index we will touch.
    max_idx = max(indices) + 1
    ds = LiberoTransitionDataset(base_ds, horizon=horizon, max_samples=max_idx)
    print(f"Transition dataset size: {len(ds)} (need >= {max_idx})")

    written = 0
    skipped = 0
    shard_indices = [i for i in indices if i % args.num_shards == args.shard_id]

    for i in tqdm(shard_indices, desc=f"fw_action[{args.shard_id}/{args.num_shards}]"):
        out_path = cache_dir / f"{i:06d}.pt"
        if not out_path.is_file():
            continue
        payload = torch.load(out_path, map_location="cpu", weights_only=False)
        if args.resume and "fastwam_action" in payload and payload["fastwam_action"] is not None:
            skipped += 1
            continue
        if i >= len(ds):
            raise IndexError(f"Cache index {i} out of transition dataset range {len(ds)}")

        sample = ds[i]
        obs_t = sample["obs_t"].unsqueeze(0).to(device=device, dtype=torch.float32)
        # Match eval residual path: obs already in dataset scale; FastWAM expects model input.
        # Latent precompute uses obs directly from dataset (same tensor as video frame).
        proprio = sample.get("state_t")
        if proprio is not None:
            proprio = proprio.unsqueeze(0).float()

        context = sample.get("context")
        context_mask = sample.get("context_mask")
        batch = {
            "obs_t": obs_t,
            "action_horizon": horizon,
            "num_inference_steps": int(args.num_inference_steps),
        }
        if context is not None and context_mask is not None:
            if context.dim() == 2:
                context = context.unsqueeze(0)
            if context_mask.dim() == 1:
                context_mask = context_mask.unsqueeze(0)
            batch["context"] = context
            batch["context_mask"] = context_mask
        else:
            batch["prompt"] = sample.get("instruction") or "libero task"
        if proprio is not None:
            batch["proprio"] = proprio

        with torch.no_grad():
            a_fw = encoder.infer_action_only(batch)
        if a_fw.dim() == 3:
            a_fw = a_fw.squeeze(0)
        a_fw = a_fw.float().cpu()
        gt = payload["action_chunk"].float()
        if tuple(a_fw.shape) != tuple(gt.shape):
            raise ValueError(
                f"Shape mismatch at {i}: fastwam_action {tuple(a_fw.shape)} vs action_chunk {tuple(gt.shape)}"
            )

        payload["fastwam_action"] = a_fw
        # Atomic-ish write
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)
        written += 1

    print(
        f"Shard {args.shard_id}/{args.num_shards} done: written={written} skipped={skipped} "
        f"considered={len(shard_indices)}"
    )


if __name__ == "__main__":
    main()
