#!/usr/bin/env python3
"""Fast path: LoRA on FastWAM ActionDiT with action loss + hard-task reweight.

No future_latent cache required. Export bundle -> official LIBERO eval.
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

from src.data.libero_dataset_adapter import LiberoDatasetAdapter, build_fastwam_dataset, collate_world2wam_batch
from src.data.balanced_sampling import build_balanced_weights
from src.utils.checkpoint_utils import (
    count_trainable_params,
    normalize_config,
    resolve_official_checkpoint,
    save_resolved_config,
    save_world2wam_checkpoint,
    load_world2wam_checkpoint,
)
from src.utils.config import load_config
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


def _prune(ckpt_dir: Path, prefix: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    paths = sorted(ckpt_dir.glob(f"{prefix}*.pt"), key=lambda p: p.stat().st_mtime)
    for old in paths[:-keep_last]:
        old.unlink(missing_ok=True)


def train(cfg: dict, args: argparse.Namespace) -> None:
    cfg = normalize_config(cfg)
    official_ckpt = resolve_official_checkpoint(cfg)
    backbone_mode = args.backbone_mode or resolve_backbone_mode(cfg)

    out_dir = Path(cfg["output_dir"])
    if args.run_id:
        out_dir = out_dir.parent / args.run_id
        cfg["output_dir"] = str(out_dir)
    log_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config({**cfg, "backbone_mode": backbone_mode}, out_dir)

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    wrapper = FastWAMWrapper.from_config(cfg, backbone_mode=backbone_mode)
    backbone_mode = wrapper.backbone_mode

    base_ds, _ = build_fastwam_dataset(cfg)
    dataset = LiberoDatasetAdapter(
        base_ds,
        future_horizon=int(cfg.get("future_horizon", 1)),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=None,
    )
    n_cap = cfg.get("precompute_max_samples")
    if n_cap is not None and int(n_cap) > 0:
        n_use = min(int(n_cap), len(dataset))
        dataset = Subset(dataset, list(range(n_use)))
        logger.info("Using first %d / %d samples", n_use, len(base_ds))

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    hard_keywords = list(cfg.get("hard_task_keywords") or [])
    weights, sampling_stats = build_balanced_weights(
        dataset,
        keywords=hard_keywords,
        hard_fraction=float(cfg.get("hard_sample_fraction", 0.5)),
        manifest_path=out_dir / "sampling_manifest.json",
    )
    logger.info("Balanced sampling: %s", sampling_stats)

    backbone_params = [p for p in wrapper.model.parameters() if p.requires_grad]
    if not backbone_params:
        raise RuntimeError("No trainable LoRA params")
    optim = torch.optim.AdamW(
        [{"params": backbone_params, "lr": float(cfg.get("backbone_lr", cfg["lr"]))}],
        weight_decay=1e-4,
    )

    log_every = int(cfg.get("log_every", 20))
    save_every = int(cfg.get("save_every", 500))
    keep_last = int(cfg.get("keep_last_checkpoints", 2))
    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_train_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            max_steps = None

    trainable = wrapper.trainable_param_count
    logger.info("LoRA-action-hard: mode=%s trainable=%d ckpt=%s", backbone_mode, trainable, official_ckpt)

    resume_path = resolve_resume_checkpoint(args.resume_from, ckpt_dir)
    resume_payload = None
    start_epoch = 0
    start_batch = 0
    global_step = 0
    if resume_path is not None:
        resume_payload = load_world2wam_checkpoint(resume_path)
        wrapper.load_world2wam_bundle(resume_path)
        trainer_state = resume_payload.get("trainer_state") or {}
        if trainer_state.get("optimizer"):
            optim.load_state_dict(trainer_state["optimizer"])
        global_step = int(trainer_state.get("global_step", resume_payload.get("meta", {}).get("global_step", 0)))
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

    for epoch in range(start_epoch, int(cfg.get("num_epochs", 1))):
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
        pbar = tqdm(loader, desc=f"lora_action epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if max_steps is not None and global_step >= max_steps:
                break
            last_epoch = epoch
            last_batch = batch_idx + 1
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            fw_out = wrapper.forward_train(
                batch,
                use_future_latent_distill=False,
                policy_action_only_loss=True,
            )
            action_loss = fw_out["action_loss"]
            total = action_loss

            optim.zero_grad(set_to_none=True)
            total.backward()
            optim.step()

            global_step += 1
            metrics = {
                "global_step": global_step,
                "epoch": epoch,
                "loss_action": float(action_loss.detach().item()),
                "loss_total": float(total.detach().item()),
                "hard_sample_fraction": float(sampling_stats["hard_fraction"]),
            }
            history.append(metrics)
            if global_step % log_every == 0:
                pbar.set_postfix(loss=f"{metrics['loss_action']:.4f}")
                logger.info("step=%d %s", global_step, metrics)
            if global_step % save_every == 0:
                save_world2wam_checkpoint(
                    ckpt_dir / f"world2wam_step{global_step}.pt",
                    backbone_mode=backbone_mode,
                    official_checkpoint=official_ckpt,
                    future_head_state=None,
                    backbone_extra=wrapper.get_backbone_state_for_save(),
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
                _prune(ckpt_dir, "world2wam_step", keep_last)

        if max_steps is not None and global_step >= max_steps:
            break

    save_world2wam_checkpoint(
        ckpt_dir / "world2wam_final.pt",
        backbone_mode=backbone_mode,
        official_checkpoint=official_ckpt,
        future_head_state=None,
        backbone_extra=wrapper.get_backbone_state_for_save(),
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
    logger.info("Saved %s ; log %s", ckpt_dir / "world2wam_final.pt", log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world2wam_policy_lora_action_hard.yaml")
    parser.add_argument("--backbone-mode", default="lora", choices=["lora", "full", "adapter", "frozen"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config, minimal_project_root()))
    set_seed(int(args.seed if args.seed is not None else cfg.get("seed", 42)))
    train(cfg, args)


if __name__ == "__main__":
    main()
