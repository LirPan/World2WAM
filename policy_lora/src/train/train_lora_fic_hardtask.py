#!/usr/bin/env python3
"""LoRA on FastWAM ActionDiT + Forward/Inverse/Cycle regularizers + hard-task reweight.

Trains a policy bundle that can be exported to an official FastWAM .pt for LIBERO eval.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
if str(_MINIMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMAL_ROOT))

from src.data.future_latent_cache import FutureLatentCache
from src.data.libero_dataset_adapter import LiberoDatasetAdapter, build_fastwam_dataset, collate_world2wam_batch
from src.data.balanced_sampling import build_balanced_weights, cached_indices
from src.losses.world2wam_losses import (
    FUTURE_LATENT_MISSING_MSG,
    compute_action_loss,
    compute_bidirectional_world2wam_loss,
)
from src.models.future_latent_head import FutureLatentHead
from src.models.inverse_action_head import InverseActionHead
from src.utils.checkpoint_utils import (
    count_trainable_params,
    normalize_config,
    resolve_official_checkpoint,
    save_resolved_config,
    save_world2wam_checkpoint,
    verify_future_latent_cache,
    load_world2wam_checkpoint,
)
from src.utils.config import load_config
from src.utils.lambda_schedule import current_lambda_fwd
from src.utils.path_utils import minimal_project_root, resolve_path
from src.utils.seed import set_seed
from src.utils.experiment_runtime import (
    make_trainer_state,
    resolve_resume_checkpoint,
    restore_rng_state,
)
from src.wrappers.backbone_modes import resolve_backbone_mode
from src.wrappers.fastwam_wrapper import FastWAMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def _prune_step_checkpoints(ckpt_dir: Path, prefix: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    paths = sorted(
        ckpt_dir.glob(f"{prefix}*.pt"),
        key=lambda p: int(p.stem.removeprefix(prefix) or "0"),
    )
    for old in paths[:-keep_last]:
        old.unlink(missing_ok=True)


def _aligned_backward(
    *,
    optimizer: torch.optim.Optimizer,
    backbone_params: list[torch.nn.Parameter],
    action_objective: torch.Tensor,
    world_objective: torch.Tensor,
) -> dict[str, float | bool]:
    optimizer.zero_grad(set_to_none=True)
    world_objective.backward(retain_graph=True)
    world_grads = [
        None if param.grad is None else param.grad.detach().clone()
        for param in backbone_params
    ]
    for param in backbone_params:
        param.grad = None

    action_objective.backward()
    action_grads = [
        None if param.grad is None else param.grad.detach().clone()
        for param in backbone_params
    ]
    dot = torch.zeros((), device=action_objective.device, dtype=torch.float32)
    action_norm_sq = torch.zeros_like(dot)
    world_norm_sq = torch.zeros_like(dot)
    for action_grad, world_grad in zip(action_grads, world_grads):
        if action_grad is not None:
            action_norm_sq += action_grad.float().square().sum()
        if world_grad is not None:
            world_norm_sq += world_grad.float().square().sum()
        if action_grad is not None and world_grad is not None:
            dot += (action_grad.float() * world_grad.float()).sum()

    conflict = bool(dot.item() < 0.0 and action_norm_sq.item() > 0.0)
    coefficient = dot / action_norm_sq.clamp_min(1e-12) if conflict else dot.new_zeros(())
    for param, action_grad, world_grad in zip(
        backbone_params, action_grads, world_grads
    ):
        if world_grad is not None and conflict and action_grad is not None:
            world_grad = world_grad - coefficient.to(world_grad.dtype) * action_grad
        if action_grad is None:
            param.grad = world_grad
        elif world_grad is None:
            param.grad = action_grad
        else:
            param.grad = action_grad + world_grad

    cosine = dot / (
        action_norm_sq.sqrt() * world_norm_sq.sqrt()
    ).clamp_min(1e-12)
    return {
        "gradient_cosine": float(cosine.detach().item()),
        "gradient_conflict": conflict,
        "action_grad_norm": float(action_norm_sq.sqrt().detach().item()),
        "world_grad_norm": float(world_norm_sq.sqrt().detach().item()),
    }


def train_lora_fic(cfg: dict, args: argparse.Namespace) -> None:
    cfg = normalize_config(cfg)
    official_ckpt = resolve_official_checkpoint(cfg)
    backbone_mode = args.backbone_mode or resolve_backbone_mode(cfg)
    if backbone_mode != "lora":
        logger.warning("Expected backbone_mode=lora for this recipe; got %s", backbone_mode)

    out_dir = Path(cfg["output_dir"])
    if args.run_id:
        out_dir = out_dir.parent / args.run_id
        cfg["output_dir"] = str(out_dir)
    log_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config({**cfg, "backbone_mode": backbone_mode}, out_dir)
    verify_future_latent_cache(cfg)

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    wrapper = FastWAMWrapper.from_config(cfg, backbone_mode=backbone_mode)
    backbone_mode = wrapper.backbone_mode

    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(cfg["cache_dir"], dataset_name=cfg.get("project_name", "world2wam_lora_fic_hard"))
    dataset = LiberoDatasetAdapter(
        base_ds,
        future_horizon=int(cfg["future_horizon"]),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=cache,
    )
    n_cached = min(int(cfg.get("precompute_max_samples") or len(dataset)), len(dataset))
    usable_indices = cached_indices(
        cache=cache,
        max_samples=n_cached,
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        future_horizon=int(cfg["future_horizon"]),
    )
    if not usable_indices:
        raise FileNotFoundError("No valid future-latent cache entries were found")
    dataset = Subset(dataset, usable_indices)
    logger.info(
        "Using %d cached samples from first %d dataset indices",
        len(usable_indices),
        n_cached,
    )

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    hard_keywords = list(cfg.get("hard_task_keywords") or [])
    weights, sampling_stats = build_balanced_weights(
        dataset,
        keywords=hard_keywords,
        hard_fraction=float(cfg.get("hard_sample_fraction", 0.5)),
        manifest_path=out_dir / "sampling_manifest.json",
    )
    logger.info("Balanced sampling: %s", sampling_stats)

    hidden_dim = int(cfg.get("hidden_dim") or wrapper.hidden_dim)
    action_dim = int(cfg.get("action_dim") or wrapper.action_dim)
    future_dim = int(cfg.get("future_latent_dim", 48))

    future_head = FutureLatentHead(hidden_dim, action_dim, future_dim).to(device)
    inverse_head = InverseActionHead(
        hidden_dim,
        future_dim,
        action_dim,
        hidden_size=int(cfg.get("inverse_hidden_size", 1024)),
    ).to(device)

    optim_groups = [
        {"params": future_head.parameters(), "lr": float(cfg["lr"])},
        {"params": inverse_head.parameters(), "lr": float(cfg["lr"])},
    ]
    backbone_params = [p for p in wrapper.model.parameters() if p.requires_grad]
    if backbone_params:
        optim_groups.append(
            {"params": backbone_params, "lr": float(cfg.get("backbone_lr", cfg["lr"]))}
        )
    optim = torch.optim.AdamW(optim_groups, weight_decay=1e-4)

    lambda_action = float(cfg.get("lambda_action", 1.0))
    lambda_fwd = float(cfg.get("lambda_fwd", 0.1))
    lambda_inv = float(cfg.get("lambda_inv", 0.05))
    lambda_cycle = float(cfg.get("lambda_cycle", 0.05))
    warmup_steps = int(cfg.get("future_loss_warmup_steps", 500))
    log_every = int(cfg.get("log_every", 20))
    save_every = int(cfg.get("save_every", 500))
    keep_last = int(cfg.get("keep_last_checkpoints", 3))

    max_train_steps = (
        args.max_steps if args.max_steps is not None else cfg.get("max_train_steps")
    )
    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        if max_train_steps <= 0:
            max_train_steps = None

    trainable = (
        count_trainable_params(future_head)
        + count_trainable_params(inverse_head)
        + wrapper.trainable_param_count
    )
    logger.info(
        "LoRA+F/I/C train: backbone=%s trainable=%d official=%s hard_keywords=%d",
        backbone_mode,
        trainable,
        official_ckpt,
        len(hard_keywords),
    )

    gradient_mode = args.gradient_mode or str(
        cfg.get("world_gradient_mode", "naive")
    )
    if gradient_mode not in {"naive", "project_conflicts"}:
        raise ValueError(f"Unsupported world_gradient_mode: {gradient_mode}")
    world_loss_scale = float(cfg.get("world_loss_scale", 1.0))
    gradient_log_every = int(cfg.get("gradient_log_every", 100))

    resume_path = resolve_resume_checkpoint(args.resume_from, ckpt_dir)
    resume_payload = None
    start_epoch = 0
    start_batch = 0
    global_step = 0
    if resume_path is not None:
        resume_payload = load_world2wam_checkpoint(resume_path)
        wrapper.load_world2wam_bundle(resume_path)
        if resume_payload.get("future_head"):
            future_head.load_state_dict(resume_payload["future_head"])
        inverse_state = (resume_payload.get("backbone_extra") or {}).get("inverse_head")
        if inverse_state:
            inverse_head.load_state_dict(inverse_state)
        trainer_state = resume_payload.get("trainer_state") or {}
        if trainer_state.get("optimizer"):
            optim.load_state_dict(trainer_state["optimizer"])
        global_step = int(
            trainer_state.get(
                "global_step", resume_payload.get("meta", {}).get("global_step", 0)
            )
        )
        start_epoch = int(trainer_state.get("epoch", 0))
        start_batch = int(trainer_state.get("batch_in_epoch", 0))
        logger.info(
            "Resuming from %s at step=%d epoch=%d batch=%d",
            resume_path,
            global_step,
            start_epoch,
            start_batch,
        )
    history: list[dict] = []
    last_epoch = start_epoch
    last_batch = start_batch
    if resume_payload is not None:
        restore_rng_state((resume_payload.get("trainer_state") or {}).get("rng"))

    for epoch in range(start_epoch, int(cfg["num_epochs"])):
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["batch_size"]),
            sampler=sampler,
            num_workers=int(cfg.get("num_workers", 0)),
            collate_fn=collate_world2wam_batch,
            drop_last=True,
        )
        pbar = tqdm(loader, desc=f"lora_fic epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if max_train_steps is not None and global_step >= max_train_steps:
                break
            last_epoch = epoch
            last_batch = batch_idx + 1
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
            gt_action = _gt_action_from_batch(batch, anchor)
            target_fl = batch["future_latent"].float()
            if target_fl.dim() == 1:
                target_fl = target_fl.unsqueeze(0)

            pred_fl = future_head(hidden, gt_action)
            if bool(cfg.get("use_target_future_for_inverse_head", True)):
                pred_inv = inverse_head(hidden, target_fl)
            else:
                pred_inv = inverse_head(hidden, pred_fl.detach())
            recon_action = inverse_head(hidden, pred_fl)

            action_loss = compute_action_loss(
                fw_out.get("pred_action"),
                batch,
                action_loss_from_fastwam=fw_out.get("action_loss"),
                loss_dict=fw_out.get("loss_dict"),
            )
            # Warm up world losses like future distill.
            lam_fwd = current_lambda_fwd(global_step + 1, lambda_fwd, warmup_steps)
            lam_inv = current_lambda_fwd(global_step + 1, lambda_inv, warmup_steps)
            lam_cyc = current_lambda_fwd(global_step + 1, lambda_cycle, warmup_steps)

            fic = compute_bidirectional_world2wam_loss(
                action_loss_monitor=action_loss,
                pred_future_latent=pred_fl,
                target_future_latent=target_fl,
                pred_action_from_target_future=pred_inv,
                target_action=gt_action,
                reconstructed_action=recon_action,
                action_source=gt_action,
                lambda_fwd=lam_fwd,
                lambda_inv=lam_inv,
                lambda_cycle=lam_cyc,
                enable_inverse=bool(cfg.get("enable_inverse", True)),
                enable_cycle=bool(cfg.get("enable_cycle", True)),
            )

            action_objective = lambda_action * action_loss
            world_objective = world_loss_scale * fic["loss_train_backward"]
            total = action_objective + world_objective
            gradient_metrics = {
                "gradient_cosine": float("nan"),
                "gradient_conflict": False,
                "action_grad_norm": float("nan"),
                "world_grad_norm": float("nan"),
            }
            if gradient_mode == "project_conflicts":
                gradient_metrics = _aligned_backward(
                    optimizer=optim,
                    backbone_params=backbone_params,
                    action_objective=action_objective,
                    world_objective=world_objective,
                )
            else:
                optim.zero_grad(set_to_none=True)
                total.backward()
            optim.step()

            global_step += 1
            metrics = {
                "global_step": global_step,
                "epoch": epoch,
                "loss_total": float(total.detach().item()),
                "loss_action": float(action_loss.detach().item()),
                "loss_fwd": float(fic["loss_fwd"].detach().item()),
                "loss_inv": float(fic["loss_inv"].detach().item()),
                "loss_cycle": float(fic["loss_cycle"].detach().item()),
                "hard_sample_fraction": float(sampling_stats["hard_fraction"]),
                "lam_fwd": float(lam_fwd),
                "lam_inv": float(lam_inv),
                "lam_cycle": float(lam_cyc),
                "backbone_mode": backbone_mode,
                "world_gradient_mode": gradient_mode,
                **gradient_metrics,
            }
            history.append(metrics)

            if global_step % log_every == 0:
                pbar.set_postfix(
                    {
                        "act": f"{metrics['loss_action']:.4f}",
                        "fwd": f"{metrics['loss_fwd']:.4f}",
                        "inv": f"{metrics['loss_inv']:.4f}",
                        "cyc": f"{metrics['loss_cycle']:.4f}",
                        "cos": f"{metrics['gradient_cosine']:.3f}",
                    }
                )
                logger.info("step=%d %s", global_step, metrics)

            if global_step % save_every == 0:
                save_world2wam_checkpoint(
                    ckpt_dir / f"world2wam_step{global_step}.pt",
                    backbone_mode=backbone_mode,
                    official_checkpoint=official_ckpt,
                    future_head_state=future_head.state_dict(),
                    backbone_extra={
                        **(wrapper.get_backbone_state_for_save() or {}),
                        "inverse_head": inverse_head.state_dict(),
                    },
                    meta={"global_step": global_step, "epoch": epoch},
                    trainer_state=make_trainer_state(
                        optimizer=optim,
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=batch_idx + 1,
                        seed=seed,
                        cfg=cfg,
                    ),
                )
                _prune_step_checkpoints(ckpt_dir, "world2wam_step", keep_last)

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    save_world2wam_checkpoint(
        ckpt_dir / "world2wam_final.pt",
        backbone_mode=backbone_mode,
        official_checkpoint=official_ckpt,
        future_head_state=future_head.state_dict(),
        backbone_extra={
            **(wrapper.get_backbone_state_for_save() or {}),
            "inverse_head": inverse_head.state_dict(),
        },
        meta={"global_step": global_step, "experiment_role": cfg.get("experiment_role")},
        trainer_state=make_trainer_state(
            optimizer=optim,
            global_step=global_step,
            epoch=last_epoch,
            batch_in_epoch=last_batch,
            seed=seed,
            cfg=cfg,
        ),
    )
    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(history, indent=2))
    logger.info("Saved final bundle to %s ; log %s", ckpt_dir / "world2wam_final.pt", log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/world2wam_policy_lora_fic_hard.yaml")
    parser.add_argument("--backbone-mode", type=str, default=None, choices=["frozen", "lora", "adapter", "full"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--gradient-mode",
        default=None,
        choices=["naive", "project_conflicts"],
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, minimal_project_root()))
    set_seed(int(args.seed if args.seed is not None else cfg.get("seed", 42)))
    train_lora_fic(cfg, args)


if __name__ == "__main__":
    main()
