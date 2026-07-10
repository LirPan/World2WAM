#!/usr/bin/env python3
"""Train bidirectional MoT World2WAM (Version B, no physics)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

VB_ROOT = Path(__file__).resolve().parents[1]
if str(VB_ROOT) not in sys.path:
    sys.path.insert(0, str(VB_ROOT))

from world2wam_vb.data.future_latent_cache import FutureLatentCache
from world2wam_vb.data.libero_batch_adapter import LiberoBatchAdapter, build_fastwam_dataset, collate_world2wam_batch
from world2wam_vb.models.physics_mot_model import build_bidirectional_mot_model
from world2wam_vb.utils.config import load_config
from world2wam_vb.utils.training import count_trainable_params, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(VB_ROOT / "configs/bidirectional_mot_train.yaml"))
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    set_seed(int(cfg.get("seed", 42)))

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        cfg["device"] = device

    logger.info("Building BidirectionalMotWorld2WAM (Version B baseline)...")
    model = build_bidirectional_mot_model(cfg).to(device)

    trainable = [model.forward_head, model.flow_expert]
    logger.info("Trainable params: %s", f"{count_trainable_params(trainable):,}")

    optim = torch.optim.AdamW(
        [p for m in trainable for p in m.parameters()],
        lr=float(cfg.get("lr", 1e-4)),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    base_ds, _ = build_fastwam_dataset(cfg)
    cache = FutureLatentCache(
        cfg["cache_dir"],
        dataset_name=cfg.get("project_name", "physics_world2wam_vb"),
    )
    ds = LiberoBatchAdapter(
        base_ds,
        future_horizon=int(cfg.get("future_horizon", 1)),
        anchor_action_idx=int(cfg.get("anchor_action_idx", 0)),
        cache=cache,
        dataset_name=cache.dataset_name,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_world2wam_batch,
    )

    max_steps = args.max_steps or int(cfg.get("max_train_steps") or 0)
    step = 0
    model.train()

    for epoch in range(int(cfg.get("num_epochs", 1))):
        for batch in loader:
            if batch.get("future_latent") is None:
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            out = model.forward_train(batch)
            loss = out["losses"]["loss"]
            loss.backward()
            optim.step()
            step += 1
            if step % int(cfg.get("log_every", 10)) == 0:
                logger.info(
                    "step=%d loss=%.4f loss_flow=%.4f loss_fwd=%.4f",
                    step,
                    float(loss.item()),
                    float(out["losses"]["loss_flow"].item()),
                    float(out["losses"]["loss_fwd"].item()),
                )
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    ckpt = out_dir / "bidirectional_mot_vb_final.pt"
    torch.save(
        {
            "cfg": cfg,
            "forward_head": model.forward_head.state_dict(),
            "flow_expert": model.flow_expert.state_dict(),
        },
        ckpt,
    )
    logger.info("Saved %s", ckpt)


if __name__ == "__main__":
    main()
