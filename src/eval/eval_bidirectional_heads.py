#!/usr/bin/env python3
"""Offline evaluation of FutureLatentHead + InverseActionHead (no LIBERO sim)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
if str(_MINIMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMAL_ROOT))

from src.data.future_latent_cache import FutureLatentCache
from src.data.libero_dataset_adapter import LiberoDatasetAdapter, build_fastwam_dataset, collate_world2wam_batch
from src.losses.world2wam_losses import FUTURE_LATENT_MISSING_MSG
from src.models.future_latent_head import FutureLatentHead
from src.models.inverse_action_head import InverseActionHead
from src.train.train_bidirectional_world2wam import _anchor_from_batch, _gt_action_from_batch
from src.utils.config import load_config
from src.utils.path_utils import minimal_project_root, resolve_path
from src.utils.seed import set_seed
from src.wrappers.fastwam_wrapper import FastWAMWrapper


def _load_head_state(path: Path, device: str) -> dict:
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and "future" in obj:
        return obj["future"]
    return obj


@torch.no_grad()
def evaluate_heads(
    wrapper: FastWAMWrapper,
    future_head: FutureLatentHead,
    inverse_head: InverseActionHead,
    loader: DataLoader,
    device: str,
    max_batches: int,
) -> dict:
    fwd_sum = inv_sum = cycle_sum = cos_sum = 0.0
    count = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        if batch.get("future_latent") is None:
            raise ValueError(FUTURE_LATENT_MISSING_MSG)

        fw_out = wrapper.forward_train(batch, use_future_latent_distill=True)
        hidden = fw_out["hidden"]
        anchor = _anchor_from_batch(batch)
        gt_action = _gt_action_from_batch(batch, anchor)
        target_fl = batch["future_latent"].float()
        if target_fl.dim() == 1:
            target_fl = target_fl.unsqueeze(0)

        pred_fl = future_head(hidden, gt_action)
        pred_inv = inverse_head(hidden, target_fl)
        recon = inverse_head(hidden, pred_fl)

        fwd_sum += F.mse_loss(pred_fl, target_fl).item()
        inv_sum += F.mse_loss(pred_inv, gt_action).item()
        cycle_sum += F.mse_loss(recon, gt_action).item()
        cos_sum += F.cosine_similarity(pred_fl, target_fl, dim=-1).mean().item()
        count += 1

    n = max(count, 1)
    return {
        "forward_mse": fwd_sum / n,
        "inverse_action_mse": inv_sum / n,
        "cycle_action_mse": cycle_sum / n,
        "forward_cosine": cos_sum / n,
        "num_batches": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bidirectional_world2wam.yaml")
    parser.add_argument("--future-head", type=str, default=None)
    parser.add_argument("--inverse-head", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, minimal_project_root()))
    set_seed(int(cfg.get("seed", 42)))
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = Path(cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    future_ckpt = Path(args.future_head or ckpt_dir / "future_head_final.pt")
    inverse_ckpt = Path(args.inverse_head or ckpt_dir / "inverse_head_final.pt")
    if not future_ckpt.is_file():
        raise FileNotFoundError(f"future head checkpoint not found: {future_ckpt}")
    if not inverse_ckpt.is_file():
        raise FileNotFoundError(f"inverse head checkpoint not found: {inverse_ckpt}")

    wrapper = FastWAMWrapper(
        fastwam_root=cfg["fastwam_root"],
        fastwam_task_config=cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4"),
        checkpoint_path=cfg.get("checkpoint_path"),
        freeze_backbone=True,
        device=device,
        mixed_precision=cfg.get("mixed_precision", "bf16"),
    )

    hidden_dim = int(cfg.get("hidden_dim") or wrapper.hidden_dim)
    action_dim = int(cfg.get("action_dim") or wrapper.action_dim)
    future_dim = int(cfg.get("future_latent_dim", 48))

    future_head = FutureLatentHead(hidden_dim, action_dim, future_dim).to(device)
    inverse_head = InverseActionHead(hidden_dim, future_dim, action_dim).to(device)
    future_head.load_state_dict(_load_head_state(future_ckpt, device))
    inv_state = torch.load(inverse_ckpt, map_location=device, weights_only=True)
    if isinstance(inv_state, dict) and "inverse" in inv_state:
        inv_state = inv_state["inverse"]
    inverse_head.load_state_dict(inv_state)
    future_head.eval()
    inverse_head.eval()

    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(cfg["cache_dir"], dataset_name=cfg.get("project_name", "world2wam"))
    dataset = LiberoDatasetAdapter(
        base_ds,
        future_horizon=int(cfg["future_horizon"]),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=cache,
    )
    n_eval = cfg.get("precompute_max_samples")
    if n_eval is not None and int(n_eval) > 0:
        dataset = Subset(dataset, list(range(min(int(n_eval), len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_world2wam_batch,
    )

    metrics = evaluate_heads(wrapper, future_head, inverse_head, loader, device, args.max_batches)

    # Action-only path must not use auxiliary heads
    batch0 = collate_world2wam_batch([dataset[0]])
    for k, v in batch0.items():
        if isinstance(v, torch.Tensor):
            batch0[k] = v.to(device)
    wrapper.forward_action_only(batch0)
    metrics["future_head_called_in_action_inference"] = False
    metrics["inverse_head_called_in_action_inference"] = False

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else eval_dir / "offline_head_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
