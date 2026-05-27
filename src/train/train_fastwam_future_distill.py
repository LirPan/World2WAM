#!/usr/bin/env python3
"""World2WAM minimal training: baseline sanity / future latent distillation."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
if str(_MINIMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMAL_ROOT))

from src.data.future_latent_cache import FutureLatentCache
from src.data.libero_dataset_adapter import LiberoDatasetAdapter, build_fastwam_dataset, collate_world2wam_batch
from src.losses.world2wam_losses import (
    FUTURE_LATENT_MISSING_MSG,
    compute_action_loss,
    compute_future_latent_loss,
    compute_total_loss,
)
from src.models.future_latent_head import FutureLatentHead
from src.utils.config import load_config
from src.utils.path_utils import minimal_project_root, resolve_path
from src.utils.seed import set_seed
from src.wrappers.fastwam_wrapper import FastWAMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _delegate_fastwam_baseline(cfg: dict) -> int:
    """Run original FastWAM training without modifying its repo."""
    fastwam_root = Path(cfg["fastwam_root"])
    script = fastwam_root / "scripts" / "train_zero1.sh"
    task = cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4")
    if not script.exists():
        logger.error("FastWAM train_zero1.sh not found at %s", script)
        return 1
    cmd = ["bash", str(script), "1", f"task={task}"]
    logger.info("Delegating baseline to FastWAM: %s (cwd=%s)", " ".join(cmd), fastwam_root)
    return subprocess.call(cmd, cwd=str(fastwam_root))


def train_future_distill(cfg: dict, args: argparse.Namespace) -> None:
    out_dir = Path(cfg["output_dir"])
    log_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = "cpu"

    wrapper = FastWAMWrapper(
        fastwam_root=cfg["fastwam_root"],
        fastwam_task_config=cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4"),
        checkpoint_path=cfg.get("checkpoint_path"),
        freeze_backbone=bool(cfg.get("freeze_fastwam_backbone", True)),
        device=device,
        mixed_precision=cfg.get("mixed_precision", "bf16"),
    )

    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(cfg["cache_dir"], dataset_name=cfg.get("project_name", "world2wam"))
    dataset = LiberoDatasetAdapter(
        base_ds,
        future_horizon=int(cfg["future_horizon"]),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=cache,
    )
    n_cached = cfg.get("precompute_max_samples")
    if n_cached is None:
        n_cached = 128
    n_cached = int(n_cached)
    # 0 or negative => use full dataset (all precomputed indices).
    if n_cached > 0:
        n_use = min(n_cached, len(dataset))
        dataset = Subset(dataset, list(range(n_use)))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_world2wam_batch,
        drop_last=True,
    )

    hidden_dim = int(cfg.get("hidden_dim") or wrapper.hidden_dim)
    action_dim = int(cfg.get("action_dim") or wrapper.action_dim)
    future_dim = int(cfg.get("future_latent_dim", 48))

    head = FutureLatentHead(hidden_dim, action_dim, future_dim).to(device)
    if cfg.get("future_head_checkpoint"):
        head.load_state_dict(torch.load(cfg["future_head_checkpoint"], map_location=device, weights_only=True))

    optim = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=1e-4,
    )

    lambda_fwd = float(cfg.get("lambda_fwd", 0.1))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 500))
    global_step = 0
    history = []
    max_train_steps = cfg.get("max_train_steps")
    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        if max_train_steps <= 0:
            max_train_steps = None

    for epoch in range(int(cfg["num_epochs"])):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            if max_train_steps is not None and global_step >= max_train_steps:
                break
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            if batch.get("future_latent") is None:
                raise ValueError(FUTURE_LATENT_MISSING_MSG)

            fw_out = wrapper.forward_train(batch, use_future_latent_distill=True)
            hidden = fw_out["hidden"]

            if cfg.get("use_gt_action_for_future_head", True):
                act = batch["action"]
                anchor_t = batch.get("anchor_action_idx", 0)
                if isinstance(anchor_t, torch.Tensor):
                    anchor = int(anchor_t[0].item())
                elif isinstance(anchor_t, list):
                    anchor = int(anchor_t[0])
                else:
                    anchor = int(anchor_t)
                if act.dim() == 3:
                    act_in = act[:, anchor]
                else:
                    act_in = act
            else:
                raise NotImplementedError("pred_action for future head not wired yet.")

            pred_fl = head(hidden, act_in.float())
            target_fl = batch["future_latent"].float()
            if target_fl.dim() == 1:
                target_fl = target_fl.unsqueeze(0)

            future_loss = compute_future_latent_loss(pred_fl, target_fl)
            action_loss = compute_action_loss(
                fw_out.get("pred_action"),
                batch,
                action_loss_from_fastwam=fw_out.get("action_loss"),
                loss_dict=fw_out.get("loss_dict"),
            )
            total, metrics = compute_total_loss(
                action_loss,
                future_loss,
                use_future_latent_distill=True,
                lambda_fwd=lambda_fwd,
                batch=batch,
            )

            optim.zero_grad(set_to_none=True)
            future_loss.backward()
            optim.step()

            global_step += 1
            metrics["epoch"] = epoch
            history.append(metrics)
            if global_step % log_every == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

            if global_step % save_every == 0:
                torch.save(
                    head.state_dict(),
                    ckpt_dir / f"future_head_step{global_step}.pt",
                )

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    torch.save(head.state_dict(), ckpt_dir / "future_head_final.pt")
    with open(log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved future head to %s", ckpt_dir / "future_head_final.pt")


def train_baseline_sanity(cfg: dict, args: argparse.Namespace) -> None:
    """Single-batch forward through FastWAM training_loss (no future head)."""
    device = cfg.get("device", "cuda")
    wrapper = FastWAMWrapper(
        fastwam_root=cfg["fastwam_root"],
        fastwam_task_config=cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4"),
        checkpoint_path=cfg.get("checkpoint_path"),
        freeze_backbone=False,
        device=device,
    )
    base_ds, _ = build_fastwam_dataset(cfg)
    adapter = LiberoDatasetAdapter(base_ds, future_horizon=1, cache=None)
    batch = collate_world2wam_batch([adapter[0]])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    out = wrapper.forward_train(batch, use_future_latent_distill=False)
    logger.info("Baseline sanity forward OK. loss_dict=%s", out.get("loss_dict"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fastwam_future_distill.yaml")
    parser.add_argument("--mode", choices=["baseline", "future_distill"], required=True)
    parser.add_argument("--fastwam-root", type=str, default=None)
    parser.add_argument("--libero-root", type=str, default=None)
    parser.add_argument("--delegate-baseline", action="store_true", help="Call FastWAM train_zero1.sh")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, minimal_project_root()))
    if args.fastwam_root:
        cfg["fastwam_root"] = str(resolve_path(args.fastwam_root, minimal_project_root()))
    if args.libero_root:
        cfg["libero_root"] = str(resolve_path(args.libero_root, minimal_project_root()))

    set_seed(int(cfg.get("seed", 42)))

    if args.mode == "baseline":
        if args.delegate_baseline:
            sys.exit(_delegate_fastwam_baseline(cfg))
        train_baseline_sanity(cfg, args)
        return

    if args.mode == "future_distill":
        train_future_distill(cfg, args)
        return


if __name__ == "__main__":
    main()
