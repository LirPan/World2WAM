#!/usr/bin/env python3
"""Precompute future VAE-pooled latents for LIBERO LeRobot clips."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
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


def _atomic_manifest(path: Path, records: list[dict], *, complete: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "complete": bool(complete),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "counts": {
            "processed": len(records),
            "valid": sum(bool(record.get("valid")) for record in records),
            "invalid": sum(not bool(record.get("valid")) for record in records),
        },
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute future latents with FastWAM VAE")
    parser.add_argument("--config", type=str, default="configs/fastwam_future_distill.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--manifest-every", type=int, default=100)
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
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else cache.cache_dir
        / cache.dataset_name
        / f"manifest_h{adapter.future_horizon}_n{n}.json"
    )
    skipped = 0
    saved = 0
    reused = 0
    records: list[dict] = []

    for idx in tqdm(range(n), desc="precompute"):
        sample = adapter[idx]
        if not sample.get("valid_future") or sample.get("future_obs") is None:
            skipped += 1
            records.append(
                {
                    "dataset_index": idx,
                    "valid": False,
                    "reason": "no_valid_future",
                    "future_horizon": adapter.future_horizon,
                }
            )
            if len(records) % max(args.manifest_every, 1) == 0:
                _atomic_manifest(manifest_path, records, complete=False)
            continue
        cache_path = cache.get_cache_path(
            idx, adapter.anchor_action_idx, adapter.future_horizon
        )
        if cache_path.is_file():
            latent = cache.load_future_latent(
                idx, adapter.anchor_action_idx, adapter.future_horizon
            )
            reused += 1
        else:
            fo = sample["future_obs"].unsqueeze(0).to(device=wrapper.model.device)
            try:
                latent = wrapper.encode_future_latent(fo).squeeze(0)
            except Exception as exc:
                raise RuntimeError(
                    "vae.encode failed. Confirm future_obs is [B,3,1,H,W]. "
                    f"sample_idx={idx}, shape={tuple(fo.shape)}, error={exc}"
                ) from exc
            cache_path = cache.save_future_latent(
                idx,
                adapter.anchor_action_idx,
                adapter.future_horizon,
                latent,
                meta={"future_latent_dim": int(latent.shape[-1])},
            )
            saved += 1

        latent = latent.detach().cpu()
        valid = bool(torch.isfinite(latent).all().item()) and latent.numel() > 0
        records.append(
            {
                "dataset_index": idx,
                "valid": valid,
                "cache_path": str(cache_path),
                "future_horizon": adapter.future_horizon,
                "shape": list(latent.shape),
                "dtype": str(latent.dtype),
            }
        )
        if not valid:
            raise ValueError(f"Non-finite or empty future latent at index {idx}")
        if len(records) % max(args.manifest_every, 1) == 0:
            _atomic_manifest(manifest_path, records, complete=False)

    _atomic_manifest(manifest_path, records, complete=True)
    print(
        "Done. "
        f"saved={saved}, reused={reused}, skipped_invalid={skipped}, "
        f"manifest={manifest_path}, cache_dir={cache.cache_dir}"
    )


if __name__ == "__main__":
    main()
