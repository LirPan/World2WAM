#!/usr/bin/env python3
"""Train ForwardHead on cached latents (Stage 1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from minimal_world2wam.data.latent_cache_dataset import (
    LatentCacheDataset,
    collate_latent_batch,
    load_meta,
    train_val_split,
)
from minimal_world2wam.models.world2wam_heads import build_heads_from_config
from minimal_world2wam.train.training_utils import (
    append_json_log,
    compute_world2wam_losses,
    count_trainable_params,
    save_checkpoint,
)
from minimal_world2wam.utils.config import load_config, save_config_copy
from minimal_world2wam.utils.seed import set_seed


def _parse_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y")


def run_epoch(model_heads, loader, device, cfg, train: bool, toggles: dict) -> dict:
    if train:
        model_heads["forward"].train()
        if model_heads.get("inverse") is not None:
            model_heads["inverse"].train()
    else:
        model_heads["forward"].eval()
        if model_heads.get("inverse") is not None:
            model_heads["inverse"].eval()

    sums = {"loss": 0.0, "loss_fwd": 0.0, "loss_inv": 0.0, "loss_cycle": 0.0}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            losses = compute_world2wam_losses(
                forward_head=model_heads["forward"],
                action_adapter=None,
                inverse_head=model_heads.get("inverse"),
                batch=batch,
                cfg=cfg,
                use_act=False,
                use_fwd=toggles["use_fwd"],
                use_inv=toggles["use_inv"],
                use_cycle=toggles["use_cycle"],
            )
            if train:
                losses["loss"].backward()
            for k in sums:
                if k in losses:
                    sums[k] += float(losses[k].item())
            n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world2wam_libero_spatial_h10_paper.yaml")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--use_fwd", default=None)
    parser.add_argument("--use_inv", default=None)
    parser.add_argument("--use_cycle", default=None)
    parser.add_argument("--legacy_inverse", action="store_true", help="Use legacy MLP InverseHead")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    cfg = load_config(WORKSPACE / args.config)
    if args.device:
        cfg.setdefault("fastwam", {})["device"] = args.device
    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    loss_cfg = dict(cfg.get("loss", {}))
    toggles = {
        "use_fwd": _parse_bool(args.use_fwd) if args.use_fwd is not None else bool(loss_cfg.get("use_fwd", True)),
        "use_inv": _parse_bool(args.use_inv) if args.use_inv is not None else bool(loss_cfg.get("use_inv", False)),
        "use_cycle": _parse_bool(args.use_cycle) if args.use_cycle is not None else bool(loss_cfg.get("use_cycle", False)),
    }

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(cfg["cache"]["output_dir"])
    meta = load_meta(cache_dir)
    dataset = LatentCacheDataset(cache_dir)
    train_cfg = cfg.get("train", {})
    train_ds, val_ds = train_val_split(dataset, float(train_cfg.get("val_ratio", 0.05)), int(train_cfg.get("seed", 42)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        collate_fn=collate_latent_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_latent_batch,
    )

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    heads = build_heads_from_config(
        cfg, meta, include_inverse=args.legacy_inverse, include_adapter=False
    )
    forward_head = heads["forward"].to(device)
    inverse_head = heads.get("inverse")
    if inverse_head is not None:
        inverse_head = inverse_head.to(device)

    params = [forward_head]
    if inverse_head is not None:
        params.append(inverse_head)
    print(f"Trainable params: {count_trainable_params(params):,}")

    optim = torch.optim.AdamW(
        [p for m in params for p in m.parameters()],
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=1e-4,
    )

    out_dir = Path(train_cfg.get("output_dir", "experiments/world2wam_heads"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy({**cfg, "toggles": toggles}, out_dir)
    log_path = out_dir / "train_log.json"
    log_every = int(train_cfg.get("log_every", 50))
    global_step = 0

    for epoch in range(int(train_cfg.get("epochs", 10))):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        forward_head.train()
        if inverse_head is not None:
            inverse_head.train()
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            losses = compute_world2wam_losses(
                forward_head=forward_head,
                action_adapter=None,
                inverse_head=inverse_head,
                batch=batch,
                cfg=cfg,
                use_act=False,
                **toggles,
            )
            losses["loss"].backward()
            optim.step()
            global_step += 1
            if global_step % log_every == 0:
                rec = {k: float(losses[k].item()) for k in ("loss", "loss_fwd", "loss_inv", "loss_cycle")}
                rec["step"] = global_step
                rec["epoch"] = epoch
                append_json_log(log_path, rec)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in rec.items() if k.startswith("loss")})

        val_metrics = run_epoch(
            {"forward": forward_head, "inverse": inverse_head},
            val_loader,
            device,
            cfg,
            train=False,
            toggles=toggles,
        )
        print(f"Epoch {epoch} val: {val_metrics}")
        append_json_log(log_path, {"epoch": epoch, "val": val_metrics, "step": global_step})

        if int(train_cfg.get("save_every", 1)) > 0 and (epoch + 1) % int(train_cfg.get("save_every", 1)) == 0:
            save_checkpoint(
                out_dir / f"heads_epoch{epoch}.pt",
                forward_head=forward_head,
                inverse_head=inverse_head,
                cfg=cfg,
                meta={"epoch": epoch, "val": val_metrics},
            )

    save_checkpoint(
        out_dir / "heads_final.pt",
        forward_head=forward_head,
        inverse_head=inverse_head,
        cfg=cfg,
        meta={"epochs": train_cfg.get("epochs"), "toggles": toggles},
    )
    print(f"Saved final checkpoint to {out_dir / 'heads_final.pt'}")


if __name__ == "__main__":
    main()
