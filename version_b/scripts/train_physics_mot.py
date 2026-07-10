#!/usr/bin/env python3
"""Train Physics-Aligned MoT World2WAM (Version B)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

VB_ROOT = Path(__file__).resolve().parents[1]
if str(VB_ROOT) not in sys.path:
    sys.path.insert(0, str(VB_ROOT))

from world2wam_vb.data.future_latent_cache import FutureLatentCache
from world2wam_vb.data.libero_batch_adapter import LiberoBatchAdapter, build_fastwam_dataset, collate_world2wam_batch
from world2wam_vb.models.physics_mot_model import build_physics_mot_model
from world2wam_vb.utils.config import load_config
from world2wam_vb.utils.training import count_trainable_params, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _PlaceholderMotDataset(Dataset):
    """Synthetic batch for dry-run smoke only."""

    def __init__(self, n: int, horizon: int, action_dim: int, text_len: int, text_dim: int):
        self.n = n
        self.horizon = horizon
        self.action_dim = action_dim
        self.text_len = text_len
        self.text_dim = text_dim

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        del idx
        return {
            "video": torch.randn(3, 2, 224, 224),
            "action": torch.randn(self.horizon, self.action_dim),
            "context": torch.randn(self.text_len, self.text_dim),
            "context_mask": torch.ones(self.text_len, dtype=torch.bool),
            "proprio": torch.randn(1, 9),
            "future_latent": torch.randn(48),
            "current_latent": torch.randn(48),
            "anchor_action_idx": 0,
        }


def _collate(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(VB_ROOT / "configs/physics_mot_train.yaml"))
    parser.add_argument("--dry_run", action="store_true", help="Use synthetic data; still needs FastWAM")
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

    logger.info("Building PhysicsAlignedMotWorld2WAM (Version B)...")
    model = build_physics_mot_model(cfg).to(device)

    trainable = [
        model.forward_head,
        model.flow_expert,
        model.physics_router,
    ]
    logger.info("Trainable params (heads+flow+router): %s", f"{count_trainable_params(trainable):,}")

    optim = torch.optim.AdamW(
        [p for m in trainable for p in m.parameters()],
        lr=float(cfg.get("lr", 1e-4)),
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if args.dry_run:
        logger.warning("dry_run uses synthetic batches — FastWAM backbone still required for forward.")
        horizon = int(cfg.get("action_horizon", 10))
        action_dim = int(cfg.get("action_dim") or model.adapter.action_dim)
        ds = _PlaceholderMotDataset(
            n=8,
            horizon=horizon,
            action_dim=action_dim,
            text_len=32,
            text_dim=int(cfg.get("text_dim", 4096)),
        )
        loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 2)), collate_fn=_collate)
    else:
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

    max_steps = args.max_steps or int(cfg.get("max_train_steps") or 2)
    step = 0
    model.train()

    for epoch in range(int(cfg.get("num_epochs", 1))):
        for batch in loader:
            if not args.dry_run and batch.get("future_latent") is None:
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            out = model.forward_train(batch)
            loss = out["losses"]["loss"]
            loss.backward()
            optim.step()
            step += 1
            logger.info(
                "step=%d loss=%.4f loss_flow=%.4f loss_phase=%.4f",
                step,
                float(loss.item()),
                float(out["losses"].get("loss_flow", loss).item()),
                float(out["losses"].get("loss_phase", torch.tensor(0.0)).item()),
            )
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    ckpt = out_dir / "physics_mot_vb_final.pt"
    torch.save(
        {
            "cfg": cfg,
            "forward_head": model.forward_head.state_dict(),
            "flow_expert": model.flow_expert.state_dict(),
            "physics_router": model.physics_router.state_dict(),
        },
        ckpt,
    )
    logger.info("Saved %s", ckpt)


if __name__ == "__main__":
    main()
