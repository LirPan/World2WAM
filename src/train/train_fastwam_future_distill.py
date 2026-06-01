#!/usr/bin/env python3
"""World2WAM minimal training: baseline sanity / future latent distillation / policy improve."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
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
    build_policy_distill_metrics,
    compute_future_latent_loss,
    compute_total_loss,
)
from src.models.future_latent_head import FutureLatentHead
from src.utils.checkpoint_utils import (
    count_trainable_params,
    normalize_config,
    resolve_official_checkpoint,
    save_resolved_config,
    save_world2wam_checkpoint,
    verify_future_latent_cache,
)
from src.utils.config import load_config
from src.utils.lambda_schedule import current_lambda_fwd
from src.utils.path_utils import minimal_project_root, resolve_path
from src.utils.seed import set_seed
from src.wrappers.backbone_modes import resolve_backbone_mode
from src.wrappers.fastwam_wrapper import FastWAMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _delegate_fastwam_baseline(cfg: dict) -> int:
    fastwam_root = Path(cfg["fastwam_root"])
    script = fastwam_root / "scripts" / "train_zero1.sh"
    task = cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4")
    if not script.exists():
        logger.error("FastWAM train_zero1.sh not found at %s", script)
        return 1
    cmd = ["bash", str(script), "1", f"task={task}"]
    logger.info("Delegating baseline to FastWAM: %s (cwd=%s)", " ".join(cmd), fastwam_root)
    return subprocess.call(cmd, cwd=str(fastwam_root))


def _gt_action_from_batch(batch: dict, anchor: int) -> torch.Tensor:
    act = batch["action"]
    if act.dim() == 3:
        return act[:, anchor].float()
    return act.float()


def _anchor_from_batch(batch: dict, default: int = 0) -> int:
    anchor_t = batch.get("anchor_action_idx", default)
    if isinstance(anchor_t, torch.Tensor):
        return int(anchor_t[0].item())
    if isinstance(anchor_t, list):
        return int(anchor_t[0])
    return int(anchor_t)


def _prune_step_checkpoints(ckpt_dir: Path, prefix: str, keep_last: int) -> None:
    """Keep only the newest `keep_last` step checkpoints matching prefix (e.g. world2wam_step)."""
    if keep_last <= 0:
        return
    paths = sorted(
        ckpt_dir.glob(f"{prefix}*.pt"),
        key=lambda p: int(p.stem.removeprefix(prefix) or "0"),
    )
    for old in paths[:-keep_last]:
        old.unlink(missing_ok=True)
        logger.info("Pruned old checkpoint %s", old.name)


def train_future_distill(cfg: dict, args: argparse.Namespace) -> None:
    cfg = normalize_config(cfg)
    official_ckpt = resolve_official_checkpoint(cfg)
    backbone_mode = args.backbone_mode or resolve_backbone_mode(cfg)

    out_dir = Path(cfg["output_dir"])
    log_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config({**cfg, "backbone_mode": backbone_mode}, out_dir)

    verify_future_latent_cache(cfg)

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = "cpu"

    logger.info(
        "future_distill: role=%s backbone_mode=%s official_ckpt=%s",
        cfg.get("experiment_role", "probe"),
        backbone_mode,
        official_ckpt,
    )

    wrapper = FastWAMWrapper.from_config(cfg, backbone_mode=backbone_mode)
    backbone_mode = wrapper.backbone_mode

    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(cfg["cache_dir"], dataset_name=cfg.get("project_name", "world2wam_minimal"))
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
        head.load_state_dict(
            torch.load(cfg["future_head_checkpoint"], map_location=device, weights_only=True)
        )

    train_head = bool(cfg.get("train_future_head", True))
    optim_groups = []
    if train_head:
        optim_groups.append({"params": head.parameters(), "lr": float(cfg["lr"])})

    backbone_params = [p for p in wrapper.model.parameters() if p.requires_grad]
    if wrapper._adapter_bank is not None:
        backbone_params = list(chain(backbone_params, wrapper._adapter_bank.parameters()))
    if backbone_params:
        optim_groups.append(
            {"params": backbone_params, "lr": float(cfg.get("backbone_lr", cfg["lr"]) * 0.5)}
        )

    if not optim_groups:
        raise RuntimeError("No trainable parameters — check backbone_mode and train_future_head.")

    optim = torch.optim.AdamW(optim_groups, weight_decay=1e-4)

    lambda_fwd = float(cfg.get("lambda_fwd", 0.1))
    warmup_steps = int(cfg.get("future_loss_warmup_steps", 1000))
    log_every = int(cfg.get("log_every", 10))
    save_every = int(cfg.get("save_every", 500))
    keep_last = int(cfg.get("keep_last_checkpoints", 5))
    experiment_role = str(cfg.get("experiment_role", "probe"))
    global_step = 0
    history: list[dict] = []

    max_train_steps = args.max_steps or cfg.get("max_train_steps")
    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        if max_train_steps <= 0:
            max_train_steps = None

    trainable_count = count_trainable_params(head) + wrapper.trainable_param_count

    for epoch in range(int(cfg["num_epochs"])):
        pbar = tqdm(loader, desc=f"{backbone_mode} epoch {epoch}")
        for batch in pbar:
            if max_train_steps is not None and global_step >= max_train_steps:
                break
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            if batch.get("future_latent") is None:
                raise ValueError(FUTURE_LATENT_MISSING_MSG)

            fw_out = wrapper.forward_train(
                batch,
                use_future_latent_distill=True,
                policy_action_only_loss=bool(cfg.get("policy_action_only_loss", True)),
            )
            hidden = fw_out["hidden"]
            anchor = _anchor_from_batch(batch, int(cfg.get("anchor_action_idx", 0)))

            if cfg.get("use_gt_action_for_future_head", True):
                act_in = _gt_action_from_batch(batch, anchor)
            else:
                raise NotImplementedError("pred_action for future head not wired yet.")

            pred_fl = head(hidden, act_in)
            target_fl = batch["future_latent"].float()
            if target_fl.dim() == 1:
                target_fl = target_fl.unsqueeze(0)

            future_loss = compute_future_latent_loss(pred_fl, target_fl)
            action_loss = fw_out["action_loss"]
            current_lam = current_lambda_fwd(global_step + 1, lambda_fwd, warmup_steps)
            total_loss, _ = compute_total_loss(
                action_loss,
                future_loss,
                use_future_latent_distill=True,
                lambda_fwd=lambda_fwd,
                current_lambda_fwd=current_lam,
                batch=batch,
            )

            optim.zero_grad(set_to_none=True)
            total_loss.backward()
            optim.step()

            global_step += 1
            metrics = build_policy_distill_metrics(
                backbone_mode=backbone_mode,
                trainable_param_count=trainable_count,
                hidden_detached=bool(fw_out["hidden_detached"]),
                loss_action=float(action_loss.detach().item()),
                loss_future=float(future_loss.detach().item()),
                current_lambda_fwd=current_lam,
                loss_total=float(total_loss.detach().item()),
                experiment_role=experiment_role,
                global_step=global_step,
                epoch=epoch,
            )
            history.append(metrics)

            if global_step % log_every == 0:
                pbar.set_postfix(
                    {
                        "loss_action": f"{metrics['loss_action']:.4f}",
                        "loss_future": f"{metrics['loss_future']:.4f}",
                        "loss_total": f"{metrics['loss_total']:.4f}",
                        "lam": f"{metrics['current_lambda_fwd']:.4f}",
                    }
                )
                logger.info(
                    "step=%d role=%s backbone_mode=%s trainable=%d hidden_detached=%s "
                    "loss_action=%.4f loss_future=%.4f current_lambda_fwd=%.4f loss_total=%.4f",
                    global_step,
                    experiment_role,
                    backbone_mode,
                    trainable_count,
                    metrics["hidden_detached"],
                    metrics["loss_action"],
                    metrics["loss_future"],
                    metrics["current_lambda_fwd"],
                    metrics["loss_total"],
                )

            if global_step % save_every == 0:
                torch.save(head.state_dict(), ckpt_dir / f"future_head_step{global_step}.pt")
                if backbone_mode != "frozen":
                    save_world2wam_checkpoint(
                        ckpt_dir / f"world2wam_step{global_step}.pt",
                        backbone_mode=backbone_mode,
                        official_checkpoint=official_ckpt,
                        future_head_state=head.state_dict(),
                        backbone_extra=wrapper.get_backbone_state_for_save(),
                        meta={"global_step": global_step, "epoch": epoch},
                    )
                _prune_step_checkpoints(ckpt_dir, "future_head_step", keep_last)
                _prune_step_checkpoints(ckpt_dir, "world2wam_step", keep_last)

        if max_train_steps is not None and global_step >= max_train_steps:
            break

        torch.save(head.state_dict(), ckpt_dir / f"future_head_epoch{epoch}.pt")
        if backbone_mode != "frozen":
            save_world2wam_checkpoint(
                ckpt_dir / f"world2wam_epoch{epoch}.pt",
                backbone_mode=backbone_mode,
                official_checkpoint=official_ckpt,
                future_head_state=head.state_dict(),
                backbone_extra=wrapper.get_backbone_state_for_save(),
                meta={"global_step": global_step, "epoch": epoch},
            )
        logger.info("Saved epoch %d checkpoint at step %d", epoch, global_step)

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    torch.save(head.state_dict(), ckpt_dir / "future_head_final.pt")
    save_world2wam_checkpoint(
        ckpt_dir / "world2wam_final.pt",
        backbone_mode=backbone_mode,
        official_checkpoint=official_ckpt,
        future_head_state=head.state_dict(),
        backbone_extra=wrapper.get_backbone_state_for_save(),
        meta={"global_step": global_step, "experiment_role": experiment_role},
    )

    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved future_head and world2wam bundle to %s; log %s", ckpt_dir, log_path)


def train_baseline_sanity(cfg: dict, args: argparse.Namespace) -> None:
    device = cfg.get("device", "cuda")
    official_ckpt = resolve_official_checkpoint(cfg)
    wrapper = FastWAMWrapper.from_config(cfg, backbone_mode="frozen")
    base_ds, _ = build_fastwam_dataset(cfg)
    adapter = LiberoDatasetAdapter(base_ds, future_horizon=1, cache=None)
    batch = collate_world2wam_batch([adapter[0]])
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    out = wrapper.forward_train(batch, use_future_latent_distill=False)
    logger.info("Baseline sanity OK official_ckpt=%s loss_dict=%s", official_ckpt, out.get("loss_dict"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fastwam_future_distill.yaml")
    parser.add_argument("--mode", choices=["baseline", "future_distill"], required=True)
    parser.add_argument("--backbone-mode", type=str, default=None, choices=["frozen", "lora", "adapter", "full"])
    parser.add_argument("--max-steps", type=int, default=None)
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
