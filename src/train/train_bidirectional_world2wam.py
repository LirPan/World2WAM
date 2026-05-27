#!/usr/bin/env python3
"""Bidirectional World2WAM: forward / inverse / cycle distillation on frozen FastWAM."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import chain
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
    compute_bidirectional_world2wam_loss,
)
from src.models.future_latent_head import FutureLatentHead
from src.models.inverse_action_head import InverseActionHead
from src.utils.config import load_config
from src.utils.path_utils import minimal_project_root, resolve_path
from src.utils.seed import set_seed
from src.wrappers.fastwam_wrapper import FastWAMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _use_gt_action_for_forward(cfg: dict) -> bool:
    if "use_gt_action_for_forward_head" in cfg:
        return bool(cfg["use_gt_action_for_forward_head"])
    return bool(cfg.get("use_gt_action_for_future_head", True))


def _anchor_from_batch(batch: dict, default: int = 0) -> int:
    anchor_t = batch.get("anchor_action_idx", default)
    if isinstance(anchor_t, torch.Tensor):
        return int(anchor_t[0].item())
    if isinstance(anchor_t, list):
        return int(anchor_t[0])
    return int(anchor_t)


def _gt_action_from_batch(batch: dict, anchor: int) -> torch.Tensor:
    act = batch["action"]
    if act.dim() == 3:
        return act[:, anchor].float()
    return act.float()


def _infer_action_dim(batch: dict, anchor: int) -> int:
    act = batch["action"]
    if act.dim() == 3:
        return int(act.shape[-1])
    return int(act.shape[-1])


def build_dataloader(cfg: dict) -> DataLoader:
    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(cfg["cache_dir"], dataset_name=cfg.get("project_name", "world2wam"))
    dataset = LiberoDatasetAdapter(
        base_ds,
        future_horizon=int(cfg["future_horizon"]),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=cache,
    )
    n_cached = cfg.get("precompute_max_samples")
    if n_cached is not None:
        n_cached = int(n_cached)
        if n_cached > 0:
            n_use = min(n_cached, len(dataset))
            dataset = Subset(dataset, list(range(n_use)))
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_world2wam_batch,
        drop_last=True,
    )


def train_bidirectional(cfg: dict, mode: str) -> None:
    if mode not in ("forward_only", "bidirectional", "cycle"):
        raise ValueError(f"Unknown mode: {mode}")

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

    loader = build_dataloader(cfg)

    hidden_dim = int(cfg.get("hidden_dim") or wrapper.hidden_dim)
    future_dim = int(cfg.get("future_latent_dim", 48))
    action_dim_cfg = cfg.get("action_dim")
    action_dim = int(action_dim_cfg) if action_dim_cfg is not None else int(wrapper.action_dim)

    future_head = FutureLatentHead(hidden_dim, action_dim, future_dim).to(device)
    inverse_head: InverseActionHead | None = None
    if mode in ("bidirectional", "cycle"):
        inverse_head = InverseActionHead(
            hidden_dim,
            future_dim,
            action_dim,
            hidden_size=int(cfg.get("inverse_hidden_size", 1024)),
            dropout=float(cfg.get("inverse_dropout", 0.0)),
        ).to(device)

    if cfg.get("future_head_checkpoint"):
        future_head.load_state_dict(
            torch.load(cfg["future_head_checkpoint"], map_location=device, weights_only=True)
        )
    if inverse_head is not None and cfg.get("inverse_head_checkpoint"):
        inverse_head.load_state_dict(
            torch.load(cfg["inverse_head_checkpoint"], map_location=device, weights_only=True)
        )

    if mode == "forward_only":
        optim_params = future_head.parameters()
    else:
        assert inverse_head is not None
        optim_params = chain(future_head.parameters(), inverse_head.parameters())

    optim = torch.optim.AdamW(optim_params, lr=float(cfg["lr"]), weight_decay=1e-4)

    lambda_fwd = float(cfg.get("lambda_fwd", 1.0))
    lambda_inv = float(cfg.get("lambda_inv", 1.0))
    lambda_cycle = float(cfg.get("lambda_cycle", 0.0))
    log_every = int(cfg.get("log_every", 20))
    save_every = int(cfg.get("save_every", 1000))
    use_gt_action = _use_gt_action_for_forward(cfg)
    enable_inverse = mode in ("bidirectional", "cycle")
    enable_cycle = mode == "cycle"

    global_step = 0
    history: list[dict] = []
    action_dim_logged = False

    max_train_steps = cfg.get("max_train_steps")
    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        if max_train_steps <= 0:
            max_train_steps = None

    for epoch in range(int(cfg["num_epochs"])):
        pbar = tqdm(loader, desc=f"{mode} epoch {epoch}")
        for batch in pbar:
            if max_train_steps is not None and global_step >= max_train_steps:
                break

            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            if batch.get("future_latent") is None:
                raise ValueError(FUTURE_LATENT_MISSING_MSG)

            anchor = _anchor_from_batch(batch, int(cfg.get("anchor_action_idx", 0)))

            if not action_dim_logged and action_dim_cfg is None:
                action_dim = _infer_action_dim(batch, anchor)
                logger.info("Inferred action_dim=%d from first batch", action_dim)
                action_dim_logged = True

            with torch.no_grad():
                fw_out = wrapper.forward_train(batch, use_future_latent_distill=True)
            hidden = fw_out["hidden"]

            if use_gt_action:
                gt_action = _gt_action_from_batch(batch, anchor)
            else:
                raise NotImplementedError("pred_action for forward head not wired yet.")

            target_fl = batch["future_latent"].float()
            if target_fl.dim() == 1:
                target_fl = target_fl.unsqueeze(0)

            pred_fl = future_head(hidden, gt_action)

            pred_inv = None
            recon_action = None
            if inverse_head is not None:
                if bool(cfg.get("use_target_future_for_inverse_head", True)):
                    pred_inv = inverse_head(hidden, target_fl)
                else:
                    pred_inv = inverse_head(hidden, pred_fl.detach())

            if enable_cycle and inverse_head is not None:
                recon_action = inverse_head(hidden, pred_fl)

            action_loss_monitor = compute_action_loss(
                fw_out.get("pred_action"),
                batch,
                action_loss_from_fastwam=fw_out.get("action_loss"),
                loss_dict=fw_out.get("loss_dict"),
            )

            loss_dict = compute_bidirectional_world2wam_loss(
                action_loss_monitor=action_loss_monitor,
                pred_future_latent=pred_fl,
                target_future_latent=target_fl,
                pred_action_from_target_future=pred_inv,
                target_action=gt_action if enable_inverse else None,
                reconstructed_action=recon_action,
                action_source=gt_action if enable_cycle else None,
                lambda_fwd=lambda_fwd,
                lambda_inv=lambda_inv if enable_inverse else 0.0,
                lambda_cycle=lambda_cycle if enable_cycle else 0.0,
                enable_inverse=enable_inverse,
                enable_cycle=enable_cycle,
            )

            optim.zero_grad(set_to_none=True)
            loss_dict["loss_train_backward"].backward()
            optim.step()

            global_step += 1
            metrics = {
                "loss_fwd": float(loss_dict["loss_fwd"].detach().item()),
                "loss_inv": float(loss_dict["loss_inv"].detach().item()),
                "loss_cycle": float(loss_dict["loss_cycle"].detach().item()),
                "loss_train_backward": float(loss_dict["loss_train_backward"].detach().item()),
                "loss_action_monitor": float(loss_dict["loss_action_monitor"].item()),
                "loss_total_logged": float(loss_dict["loss_total_logged"].item()),
                "lr": float(optim.param_groups[0]["lr"]),
                "batch_size": int(batch["action"].shape[0]),
                "from_cache": True,
                "epoch": epoch,
                "mode": mode,
            }
            history.append(metrics)

            if global_step % log_every == 0:
                pbar.set_postfix(
                    {k: f"{v:.4f}" for k, v in metrics.items() if k.startswith("loss")}
                )
                logger.info(
                    "step=%d mode=%s loss_fwd=%.4f loss_inv=%.4f loss_cycle=%.4f "
                    "loss_train_backward=%.4f loss_action_monitor=%.4f lr=%.2e bs=%d cache=%s",
                    global_step,
                    mode,
                    metrics["loss_fwd"],
                    metrics["loss_inv"],
                    metrics["loss_cycle"],
                    metrics["loss_train_backward"],
                    metrics["loss_action_monitor"],
                    metrics["lr"],
                    metrics["batch_size"],
                    metrics["from_cache"],
                )

            if global_step % save_every == 0:
                torch.save(future_head.state_dict(), ckpt_dir / f"future_head_step{global_step}.pt")
                if inverse_head is not None:
                    torch.save(inverse_head.state_dict(), ckpt_dir / f"inverse_head_step{global_step}.pt")

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    torch.save(future_head.state_dict(), ckpt_dir / "future_head_final.pt")
    if inverse_head is not None:
        torch.save(inverse_head.state_dict(), ckpt_dir / "inverse_head_final.pt")
        torch.save(
            {"future": future_head.state_dict(), "inverse": inverse_head.state_dict()},
            ckpt_dir / "bidirectional_heads_final.pt",
        )

    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved checkpoints to %s; log %s", ckpt_dir, log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bidirectional_world2wam.yaml")
    parser.add_argument(
        "--mode",
        choices=["forward_only", "bidirectional", "cycle"],
        required=True,
    )
    parser.add_argument("--fastwam-root", type=str, default=None)
    parser.add_argument("--libero-root", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, minimal_project_root()))
    if args.fastwam_root:
        cfg["fastwam_root"] = str(resolve_path(args.fastwam_root, minimal_project_root()))
    if args.libero_root:
        cfg["libero_root"] = str(resolve_path(args.libero_root, minimal_project_root()))

    set_seed(int(cfg.get("seed", 42)))
    train_bidirectional(cfg, args.mode)


if __name__ == "__main__":
    main()
