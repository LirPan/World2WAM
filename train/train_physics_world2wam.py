#!/usr/bin/env python3
"""Train Physics-Aligned World2WAM with unified L_total."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    detect_state_dim,
    load_meta,
    require_fastwam_action_in_cache,
    train_val_split,
)
from minimal_world2wam.eval.physics_eval_utils import build_student_router
from minimal_world2wam.models.physics_world2wam import PhysicsAlignedWorld2WAM
from minimal_world2wam.models.world2wam_heads import build_heads_from_config, resolve_adapter_type
from minimal_world2wam.physics.physics_labels import PHYSICS_PHASES, batch_infer_physics_labels_v1
from minimal_world2wam.train.training_utils import (
    compute_total_loss,
    count_trainable_params,
    is_flow_adapter,
    load_checkpoint,
    load_heads_warm_start,
    sample_action_adapter,
    save_checkpoint,
)
from minimal_world2wam.utils.config import load_config, save_config_copy
from minimal_world2wam.utils.seed import set_seed


def _parse_bool(s: str | None, default: bool = True) -> bool:
    if s is None:
        return default
    return str(s).lower() in ("1", "true", "yes", "y")


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (WORKSPACE / p).resolve()
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world2wam_physics_flow_dit_main.yaml")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--heads_ckpt", default=None)
    parser.add_argument("--warm_start_heads", default=None)
    parser.add_argument("--adapter_ckpt", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--indices_file",
        default=None,
        help="JSON list of dataset indices (e.g. cache files with fastwam_action only)",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--adapter_type", choices=["mlp", "light_dit", "flow_dit"], default=None)
    parser.add_argument("--lambda_phase", type=float, default=None)
    parser.add_argument("--lambda_phy", type=float, default=None)
    parser.add_argument("--phase_label_version", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    cfg = load_config(WORKSPACE / args.config)
    if args.device:
        cfg.setdefault("fastwam", {})["device"] = args.device
    if args.adapter_type:
        cfg.setdefault("model", {}).setdefault("action_adapter", {})["adapter_type"] = args.adapter_type
    if args.phase_label_version:
        cfg.setdefault("physics", {})["phase_label_version"] = args.phase_label_version
    if args.lambda_phase is not None:
        cfg.setdefault("weights", {})["lambda_phase"] = args.lambda_phase
    if args.lambda_phy is not None:
        cfg.setdefault("weights", {})["lambda_phy"] = args.lambda_phy
    set_seed(int(cfg.get("train", {}).get("seed", 42)))

    train_cfg = cfg.get("train", {})
    physics_cfg = cfg.get("physics", {})
    loss_cfg = cfg.get("loss", {})

    cache_dir = _resolve(args.cache_dir) or Path(cfg["cache"]["output_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (WORKSPACE / cache_dir).resolve()

    meta = load_meta(cache_dir)
    state_dim = detect_state_dim(cache_dir)
    residual_delta = bool(loss_cfg.get("residual_delta", False))
    if residual_delta:
        if args.indices_file:
            print("Version C residual_delta=ON with --indices_file (skip global cache probe).")
        else:
            require_fastwam_action_in_cache(cache_dir)
        print("Version C residual_delta=ON: Flow/Inv/Cycle target δ = a_GT - a_FW; Forward uses absolute GT.")
    dataset = LatentCacheDataset(cache_dir, load_state=state_dim > 0)
    if args.indices_file:
        indices_path = _resolve(args.indices_file)
        if indices_path is None or not indices_path.is_file():
            raise FileNotFoundError(f"--indices_file not found: {args.indices_file}")
        with open(indices_path, encoding="utf-8") as f:
            subset_indices = json.load(f)
        if not isinstance(subset_indices, list) or not subset_indices:
            raise ValueError(f"--indices_file must contain a non-empty JSON list: {indices_path}")
        dataset = Subset(dataset, subset_indices)
        print(f"Using {len(subset_indices)} samples from {indices_path}")
    elif args.max_samples is not None and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))

    train_ds, _val_ds = train_val_split(
        dataset,
        float(train_cfg.get("val_ratio", 0.05)),
        int(train_cfg.get("seed", 42)),
    )
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 32))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_latent_batch,
        drop_last=True,
    )

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    heads = build_heads_from_config(cfg, meta, include_inverse=False)
    forward_head = heads["forward"].to(device)
    action_adapter = heads["adapter"].to(device)
    adapter_type = resolve_adapter_type(cfg, cli_override=args.adapter_type)

    heads_ckpt = _resolve(args.heads_ckpt or args.warm_start_heads)
    adapter_ckpt = _resolve(args.adapter_ckpt)

    if heads_ckpt and heads_ckpt.is_file():
        load_heads_warm_start(heads_ckpt, forward_head)
    if adapter_ckpt and adapter_ckpt.is_file():
        load_checkpoint(
            adapter_ckpt,
            forward_head,
            action_adapter,
            expected_adapter_type=adapter_type,
        )

    physics_router = build_student_router(cfg, meta, cache_dir).to(device)

    model = PhysicsAlignedWorld2WAM(
        forward_head=forward_head,
        action_adapter=action_adapter,
        physics_router=physics_router,
        cfg=cfg,
    ).to(device)

    if heads_ckpt and heads_ckpt.is_file() and "physics_router" in torch.load(
        heads_ckpt, map_location="cpu", weights_only=False
    ):
        load_checkpoint(
            heads_ckpt,
            forward_head,
            action_adapter,
            physics_router=physics_router,
        )

    modules = [forward_head, action_adapter, physics_router]
    print(f"Adapter type: {adapter_type}")
    print(f"State dim: {state_dim}")
    print(f"Trainable params: {count_trainable_params(modules):,}")

    optim = torch.optim.AdamW(
        [p for m in modules for p in m.parameters()],
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=1e-4,
    )

    out_dir = _resolve(args.output_dir) or _resolve(train_cfg.get("output_dir")) or Path(
        "experiments/world2wam_physics_flow_dit_main"
    )
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(cfg, out_dir)
    log_path = out_dir / "train_log.jsonl"
    log_every = int(train_cfg.get("log_every", 50))
    max_steps = args.max_steps or int(train_cfg.get("max_steps", 0)) or None
    flow_sample_steps = int(cfg.get("eval", {}).get("flow_sample_steps", 10))

    global_step = 0
    phase_counter: Counter[str] = Counter()

    for epoch in range(int(train_cfg.get("epochs", 3))):
        pbar = tqdm(train_loader, desc=f"physics epoch {epoch}")
        for m in modules:
            m.train()
        for batch in pbar:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optim.zero_grad(set_to_none=True)

            losses = compute_total_loss(
                forward_head=forward_head,
                action_adapter=action_adapter,
                physics_router=physics_router,
                batch=batch_dev,
                cfg=cfg,
                use_act=bool(loss_cfg.get("use_act", True)),
                use_fwd=bool(loss_cfg.get("use_fwd", True)),
                use_inv=bool(loss_cfg.get("use_inv", True)),
                use_cycle=bool(loss_cfg.get("use_cycle", True)),
                use_physics=True,
            )

            with torch.no_grad():
                label_out = batch_infer_physics_labels_v1(batch_dev, cfg=physics_cfg)
                for lid in label_out["phase_id"].tolist():
                    phase_counter[PHYSICS_PHASES[lid]] += 1

            losses["loss"].backward()
            optim.step()
            global_step += 1

            if global_step % log_every == 0:
                rec = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(losses["loss"].item()),
                }
                for k in (
                    "loss_flow", "loss_fwd", "loss_inverse", "loss_cycle",
                    "loss_phase", "loss_phy", "phase_acc_pseudo", "phase_entropy",
                    "delta_abs_mean",
                ):
                    if k in losses:
                        v = losses[k]
                        rec[k] = float(v.item()) if hasattr(v, "item") else float(v)
                rec["residual_delta"] = bool(loss_cfg.get("residual_delta", False))
                if is_flow_adapter(action_adapter) and global_step % (log_every * 2) == 0:
                    with torch.no_grad():
                        router_out = physics_router(
                            batch_dev["z_t"],
                            text_embed=batch_dev["text_embed"],
                            state_t=batch_dev.get("state_t"),
                        )
                        sampled = sample_action_adapter(
                            action_adapter,
                            batch_dev["z_t"],
                            batch_dev["text_embed"],
                            num_steps=flow_sample_steps,
                            physics_code=router_out.get("physics_code"),
                        )
                        from minimal_world2wam.train.training_utils import resolve_action_flow_target

                        target = resolve_action_flow_target(batch_dev, cfg)
                        rec["mse_act_sample"] = float(
                            torch.nn.functional.mse_loss(sampled, target).item()
                        )
                        if "fastwam_action" in batch_dev:
                            # Reconstructed absolute action vs GT (sanity for residual mode).
                            abs_pred = batch_dev["fastwam_action"] + sampled
                            rec["mse_abs_recon"] = float(
                                torch.nn.functional.mse_loss(abs_pred, batch_dev["action_chunk"]).item()
                            )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                pbar.set_postfix(loss=f"{rec['loss']:.4f}")

            if max_steps is not None and global_step >= max_steps:
                break

        save_checkpoint(
            out_dir / f"physics_world2wam_epoch{epoch}.pt",
            forward_head=forward_head,
            action_adapter=action_adapter,
            cfg=cfg,
            meta={"epoch": epoch},
            adapter_type=adapter_type,
            physics_router=physics_router,
        )
        if max_steps is not None and global_step >= max_steps:
            break

    save_checkpoint(
        out_dir / "physics_world2wam_final.pt",
        forward_head=forward_head,
        action_adapter=action_adapter,
        cfg=cfg,
        meta={"phase_distribution": dict(phase_counter), "steps": global_step},
        adapter_type=adapter_type,
        physics_router=physics_router,
    )
    print(f"Phase distribution: {dict(phase_counter)}")
    print(f"Saved {out_dir / 'physics_world2wam_final.pt'}")


if __name__ == "__main__":
    main()
