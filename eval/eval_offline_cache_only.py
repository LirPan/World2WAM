#!/usr/bin/env python3
"""Cache-only offline evaluation — does NOT load FastWAM or LIBERO."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from minimal_world2wam.data.latent_cache_dataset import (
    LatentCacheDataset,
    collate_latent_batch,
    load_meta,
)
from minimal_world2wam.eval.physics_eval_utils import build_physics_model, load_physics_checkpoint
from minimal_world2wam.models.world2wam_heads import build_heads_from_config, resolve_adapter_type
from minimal_world2wam.physics.physics_labels import PHYSICS_PHASES, batch_infer_physics_labels_v1
from minimal_world2wam.train.training_utils import (
    call_action_adapter,
    is_flow_adapter,
    load_checkpoint,
    sample_action_adapter,
)
from minimal_world2wam.utils.config import load_config


def _parse_bool(s: str | None) -> bool:
    if s is None:
        return False
    return str(s).lower() in ("1", "true", "yes", "y")


def _resolve_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (WORKSPACE / path).resolve()
    return path


def _verify_batch_shapes(batch: dict[str, torch.Tensor], meta: dict) -> None:
    z_t = batch["z_t"]
    z_tH = batch["z_tH"]
    text_embed = batch["text_embed"]
    action_chunk = batch["action_chunk"]

    latent_dim = int(meta.get("latent_dim", z_t.shape[-1]))
    horizon = int(meta.get("horizon", action_chunk.shape[1] if action_chunk.dim() == 3 else 10))
    action_dim = int(meta.get("action_dim", action_chunk.shape[-1] if action_chunk.dim() >= 2 else 7))

    for key, t, exp_tail in (
        ("z_t", z_t, (latent_dim,)),
        ("z_tH", z_tH, (latent_dim,)),
    ):
        if t.dim() != 2 or tuple(t.shape[1:]) != exp_tail:
            raise ValueError(f"Shape mismatch for {key}: got {tuple(t.shape)}, expected [B, {exp_tail}]")

    if action_chunk.dim() != 3 or tuple(action_chunk.shape[1:]) != (horizon, action_dim):
        raise ValueError(
            f"Shape mismatch for action_chunk: got {tuple(action_chunk.shape)}, "
            f"expected [B, {horizon}, {action_dim}]"
        )

    if text_embed.dim() not in (2, 3):
        raise ValueError(f"text_embed must be [B,L,D] or [B,D], got {tuple(text_embed.shape)}")


@torch.no_grad()
def evaluate(
    *,
    cfg: dict,
    cache_dir: Path,
    heads_ckpt: Path,
    adapter_ckpt: Path | None,
    max_samples: int | None,
    batch_size: int,
    device: str,
    adapter_type: str | None = None,
    flow_sample_steps: int = 10,
    use_physics: bool = False,
    phase_label_version: str = "v1",
) -> dict:
    meta = load_meta(cache_dir)
    physics_cfg = dict(cfg.get("physics", {}))
    physics_cfg["phase_label_version"] = phase_label_version

    dataset = LatentCacheDataset(cache_dir)
    if max_samples is not None and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_latent_batch,
    )

    adapter_payload = None
    ckpt_path = adapter_ckpt if adapter_ckpt and adapter_ckpt.is_file() else heads_ckpt
    if ckpt_path.is_file():
        adapter_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    resolved_type = resolve_adapter_type(cfg, adapter_payload, adapter_type)
    cfg = dict(cfg)
    cfg.setdefault("model", {}).setdefault("action_adapter", {})["adapter_type"] = resolved_type

    physics_model = None
    if use_physics:
        physics_model, resolved_type = build_physics_model(cfg, meta, device, cache_dir=cache_dir)
        load_physics_checkpoint(physics_model, ckpt_path, expected_adapter_type=resolved_type)
        forward_head = physics_model.forward_head
        action_adapter = physics_model.action_adapter
        inverse_head = None
    else:
        heads = build_heads_from_config(cfg, meta, include_inverse=False)
        forward_head = heads["forward"].to(device).eval()
        action_adapter = heads["adapter"].to(device).eval()
        inverse_head = None
        load_checkpoint(heads_ckpt, forward_head, action_adapter=None)
        has_adapter_ckpt = adapter_ckpt is not None and adapter_ckpt.is_file()
        if has_adapter_ckpt:
            load_checkpoint(
                adapter_ckpt,
                forward_head,
                action_adapter,
                expected_adapter_type=resolved_type,
            )

    has_adapter = action_adapter is not None and (
        use_physics or (adapter_ckpt is not None and adapter_ckpt.is_file())
    )

    fwd_sum = inv_sum = cycle_sum = act_sum = 0.0
    n_samples = 0
    phase_counter: Counter[str] = Counter()
    phase_acc_sum = 0.0
    phase_entropy_sum = 0.0
    n_phase_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        _verify_batch_shapes(batch, meta)

        z_t = batch["z_t"]
        z_tH = batch["z_tH"]
        text_embed = batch["text_embed"]
        action_chunk = batch["action_chunk"]
        bs = z_t.shape[0]

        if use_physics and physics_model is not None:
            out = physics_model.forward_inference(
                z_t,
                text_embed,
                state_t=batch.get("state_t"),
                num_flow_steps=flow_sample_steps,
            )
            z_pred = forward_head(z_t, action_chunk, text_embed)
            pred_a = out.get("pred_action")
            pred_act = pred_a
            logits = out.get("phase_logits") or out.get("physics_logits")
            probs = out.get("phase_prob") or out.get("physics_probs")
            if logits is not None and probs is not None:
                label_out = batch_infer_physics_labels_v1(batch, cfg=physics_cfg)
                phase_acc_sum += (logits.argmax(dim=-1) == label_out["phase_id"]).float().mean().item()
                p_mean = probs.mean(dim=0).clamp(min=1e-8)
                phase_entropy_sum += float((-(p_mean * p_mean.log()).sum()).item())
                n_phase_batches += 1
                for lid in label_out["phase_id"].tolist():
                    phase_counter[PHYSICS_PHASES[lid]] += 1
        else:
            z_pred = forward_head(z_t, action_chunk, text_embed)
            pred_a = None
            pred_act = None
            if is_flow_adapter(action_adapter):
                pred_a = sample_action_adapter(
                    action_adapter,
                    z_t,
                    text_embed,
                    num_steps=flow_sample_steps,
                    future_latent=z_tH,
                )
            elif inverse_head is not None:
                pred_a = inverse_head(z_t, z_tH, text_embed)

        if is_flow_adapter(action_adapter):
            a_cycle = sample_action_adapter(
                action_adapter,
                z_t,
                text_embed,
                num_steps=flow_sample_steps,
                future_latent=z_pred,
            )
        elif inverse_head is not None:
            a_cycle = inverse_head(z_t, z_pred, text_embed)
        else:
            a_cycle = pred_a if pred_a is not None else action_chunk

        fwd_sum += F.mse_loss(z_pred, z_tH, reduction="sum").item()
        inv_sum += F.mse_loss(pred_a, action_chunk, reduction="sum").item() if pred_a is not None else 0.0
        cycle_sum += F.mse_loss(a_cycle, action_chunk, reduction="sum").item()

        if has_adapter:
            if pred_act is None:
                if is_flow_adapter(action_adapter):
                    pred_act = sample_action_adapter(
                        action_adapter,
                        z_t,
                        text_embed,
                        num_steps=flow_sample_steps,
                    )
                else:
                    pred_act = call_action_adapter(action_adapter, z_t, text_embed)
            act_sum += F.mse_loss(pred_act, action_chunk, reduction="sum").item()

        n_samples += bs

    denom = max(n_samples, 1)
    result = {
        "mse_fwd": fwd_sum / denom,
        "mse_inv": inv_sum / denom,
        "mse_cycle": cycle_sum / denom,
        "num_evaluated_samples": n_samples,
        "cache_dir": str(cache_dir),
        "heads_ckpt": str(heads_ckpt),
        "adapter_ckpt": str(adapter_ckpt) if adapter_ckpt else str(ckpt_path),
        "has_adapter": has_adapter,
        "adapter_type": resolved_type if has_adapter else None,
        "device": device,
        "use_physics": use_physics,
        "phase_label_version": phase_label_version,
    }
    if has_adapter:
        if is_flow_adapter(action_adapter):
            result["mse_act_sample"] = act_sum / denom
            result["mse_act"] = act_sum / denom
            result["flow_sample_steps"] = flow_sample_steps
        else:
            result["mse_act"] = act_sum / denom
    if use_physics and n_phase_batches > 0:
        result["phase_counts_eval"] = dict(phase_counter)
        result["phase_acc_pseudo"] = phase_acc_sum / n_phase_batches
        result["phase_entropy"] = phase_entropy_sum / n_phase_batches
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world2wam_physics_flow_dit_main.yaml")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--heads_ckpt", default=None)
    parser.add_argument("--adapter_ckpt", default=None)
    parser.add_argument("--output", default="experiments/eval_offline_physics_v1.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--adapter_type", choices=["mlp", "light_dit", "flow_dit"], default=None)
    parser.add_argument("--flow_sample_steps", type=int, default=10)
    parser.add_argument("--use_physics", default=None)
    parser.add_argument("--phase_label_version", default="v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    cfg = load_config(WORKSPACE / args.config)
    if args.adapter_type:
        cfg.setdefault("model", {}).setdefault("action_adapter", {})["adapter_type"] = args.adapter_type
    device = args.device or cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cache_dir = _resolve_path(args.cache_dir)
    heads_ckpt = _resolve_path(args.heads_ckpt)
    adapter_ckpt = _resolve_path(args.adapter_ckpt)
    output = _resolve_path(args.output)

    assert cache_dir is not None and cache_dir.is_dir()
    ckpt = adapter_ckpt or heads_ckpt
    assert ckpt is not None and ckpt.is_file()

    use_physics = _parse_bool(args.use_physics) or bool(cfg.get("train", {}).get("use_physics", False))
    if heads_ckpt is None:
        heads_ckpt = ckpt

    results = evaluate(
        cfg=cfg,
        cache_dir=cache_dir,
        heads_ckpt=heads_ckpt,
        adapter_ckpt=adapter_ckpt or ckpt,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=device,
        adapter_type=args.adapter_type,
        flow_sample_steps=args.flow_sample_steps,
        use_physics=use_physics,
        phase_label_version=args.phase_label_version,
    )

    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
