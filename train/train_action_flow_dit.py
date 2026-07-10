#!/usr/bin/env python3
"""Train FlowActionDiT + ForwardHead with flow inverse/cycle losses (Stage 2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
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
from minimal_world2wam.models.world2wam_heads import build_heads_from_config, resolve_adapter_type
from minimal_world2wam.train.training_utils import (
    append_json_log,
    compute_world2wam_losses,
    count_trainable_params,
    is_flow_adapter,
    load_heads_warm_start,
    sample_action_adapter,
    save_checkpoint,
)
from minimal_world2wam.utils.config import load_config, save_config_copy
from minimal_world2wam.utils.seed import set_seed


def _parse_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world2wam_libero_spatial_h10_flow_dit.yaml")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--use_act", default=None)
    parser.add_argument("--use_fwd", default=None)
    parser.add_argument("--use_inv", default=None)
    parser.add_argument("--use_cycle", default=None)
    parser.add_argument("--warm_start_heads", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--adapter_type", choices=["mlp", "light_dit", "flow_dit"], default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--flow_sample_steps", type=int, default=None)
    parser.add_argument("--dit_hidden_dim", type=int, default=None)
    parser.add_argument("--dit_depth", type=int, default=None)
    parser.add_argument("--dit_num_heads", type=int, default=None)
    parser.add_argument("--dit_dropout", type=float, default=None)
    parser.add_argument("--flow_loss_weight", type=float, default=None)
    parser.add_argument("--act_sample_log_interval", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    cfg = load_config(WORKSPACE / args.config)
    if args.device:
        cfg.setdefault("fastwam", {})["device"] = args.device
    if args.adapter_type:
        cfg.setdefault("model", {}).setdefault("action_adapter", {})["adapter_type"] = args.adapter_type
    act_cfg = cfg.setdefault("model", {}).setdefault("action_adapter", {})
    if args.dit_hidden_dim is not None:
        act_cfg["dit_hidden_dim"] = args.dit_hidden_dim
        act_cfg["hidden_dim"] = args.dit_hidden_dim
    if args.dit_depth is not None:
        act_cfg["dit_depth"] = args.dit_depth
        act_cfg["num_layers"] = args.dit_depth
    if args.dit_num_heads is not None:
        act_cfg["dit_num_heads"] = args.dit_num_heads
        act_cfg["num_heads"] = args.dit_num_heads
    if args.dit_dropout is not None:
        act_cfg["dit_dropout"] = args.dit_dropout
        act_cfg["dropout"] = args.dit_dropout
    if args.flow_loss_weight is not None:
        cfg.setdefault("weights", {})["flow_loss_weight"] = args.flow_loss_weight
    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    loss_cfg = dict(cfg.get("loss", {}))
    toggles = {
        "use_act": _parse_bool(args.use_act) if args.use_act is not None else bool(loss_cfg.get("use_act", True)),
        "use_fwd": _parse_bool(args.use_fwd) if args.use_fwd is not None else bool(loss_cfg.get("use_fwd", True)),
        "use_inv": _parse_bool(args.use_inv) if args.use_inv is not None else bool(loss_cfg.get("use_inv", True)),
        "use_cycle": _parse_bool(args.use_cycle) if args.use_cycle is not None else bool(loss_cfg.get("use_cycle", True)),
    }

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(cfg["cache"]["output_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (WORKSPACE / cache_dir).resolve()
    meta = load_meta(cache_dir)
    dataset = LatentCacheDataset(cache_dir)
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))
    train_cfg = cfg.get("train", {})
    adapter_cfg = cfg.get("adapter", {})
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        adapter_cfg.get("output_dir") or train_cfg.get("output_dir", "experiments/world2wam_adapter_flow_dit")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, _val_ds = train_val_split(dataset, float(train_cfg.get("val_ratio", 0.05)), int(train_cfg.get("seed", 42)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        collate_fn=collate_latent_batch,
        drop_last=True,
    )

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    heads = build_heads_from_config(cfg, meta, include_inverse=False)
    forward_head = heads["forward"].to(device)
    action_adapter = heads["adapter"].to(device)

    if args.warm_start_heads:
        warm_path = Path(args.warm_start_heads)
        if not warm_path.is_absolute():
            warm_path = (WORKSPACE / warm_path).resolve()
        load_heads_warm_start(warm_path, forward_head)

    adapter_type = resolve_adapter_type(cfg, cli_override=args.adapter_type)
    horizon = int(cfg.get("horizon", 10))
    action_dim = int(cfg.get("model", {}).get("action_dim", 7))
    modules = [forward_head, action_adapter]
    print(f"Adapter type: {adapter_type}")
    print(f"Action output shape: [B, {horizon}, {action_dim}]")
    print(f"Trainable params: {count_trainable_params(modules):,}")

    optim = torch.optim.AdamW(
        [p for m in modules for p in m.parameters()],
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=1e-4,
    )

    save_config_copy({**cfg, "toggles": toggles}, out_dir)
    log_path = out_dir / "train_log.json"
    log_every = int(train_cfg.get("log_every", 50))
    flow_sample_steps = int(args.flow_sample_steps or cfg.get("eval", {}).get("flow_sample_steps", 10))
    act_sample_log_interval = int(args.act_sample_log_interval or log_every)
    global_step = 0
    max_steps = args.max_steps
    use_flow = is_flow_adapter(action_adapter)

    for epoch in range(int(train_cfg.get("epochs", 10))):
        pbar = tqdm(train_loader, desc=f"flow_dit epoch {epoch}")
        for m in modules:
            m.train()
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            losses = compute_world2wam_losses(
                forward_head=forward_head,
                action_adapter=action_adapter,
                batch=batch,
                cfg=cfg,
                **toggles,
            )
            losses["loss"].backward()
            optim.step()
            global_step += 1
            if global_step % log_every == 0:
                rec = {
                    k: float(losses[k].item())
                    for k in ("loss", "loss_fwd", "loss_inv", "loss_cycle", "loss_act", "loss_flow")
                    if k in losses
                }
                for k in ("loss_action_flow", "flow_tau_mean"):
                    if k in losses:
                        rec[k] = float(losses[k].item())
                rec.update({"step": global_step, "epoch": epoch, "adapter_type": adapter_type})
                if use_flow and global_step % act_sample_log_interval == 0:
                    with torch.no_grad():
                        sampled = sample_action_adapter(
                            action_adapter,
                            batch["z_t"],
                            batch["text_embed"],
                            num_steps=flow_sample_steps,
                        )
                        rec["mse_act_sample"] = float(
                            torch.nn.functional.mse_loss(sampled, batch["action_chunk"]).item()
                        )
                append_json_log(log_path, rec)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in rec.items() if k.startswith("loss") or k == "mse_act_sample"})

            if max_steps is not None and global_step >= max_steps:
                break

        save_checkpoint(
            out_dir / f"adapter_epoch{epoch}.pt",
            forward_head=forward_head,
            action_adapter=action_adapter,
            cfg=cfg,
            meta={"epoch": epoch},
            adapter_type=adapter_type,
        )
        if max_steps is not None and global_step >= max_steps:
            break

    save_checkpoint(
        out_dir / "adapter_final.pt",
        forward_head=forward_head,
        action_adapter=action_adapter,
        cfg=cfg,
        meta={"toggles": toggles, "steps": global_step},
        adapter_type=adapter_type,
    )
    print(f"Saved {out_dir / 'adapter_final.pt'}")


if __name__ == "__main__":
    main()
