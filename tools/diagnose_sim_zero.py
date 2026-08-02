#!/usr/bin/env python3
"""Phase A smoke diagnosis for Version A sim 0% success.

Checks:
  L1/L4 — cache GT action stats, Flow sample MSE/MAE per-dim (physics / no-physics)
  L1    — postprocess (denorm + gripper) stats for GT vs pred
  L2    — checkpoint load missing/unexpected keys, router state_dim
  L4    — flow_sample_steps sweep

Writes: experiments/diagnose_sim_zero_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from minimal_world2wam.data.latent_cache_dataset import (
    LatentCacheDataset,
    collate_latent_batch,
    detect_state_dim,
    load_meta,
)
from minimal_world2wam.eval.physics_eval_utils import build_physics_model, load_physics_checkpoint
from minimal_world2wam.models.world2wam_heads import build_heads_from_config, resolve_adapter_type
from minimal_world2wam.train.training_utils import (
    is_flow_adapter,
    load_checkpoint,
    sample_action_adapter,
)
from minimal_world2wam.utils.config import load_config
from minimal_world2wam.utils.seed import set_seed


def _tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().float().reshape(-1, x.shape[-1]) if x.ndim >= 2 else x.detach().float().unsqueeze(-1)
    # flatten to [N, D] over batch*horizon
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    return {
        "mean_per_dim": x.mean(dim=0).tolist(),
        "std_per_dim": x.std(dim=0, unbiased=False).tolist(),
        "min_per_dim": x.min(dim=0).values.tolist(),
        "max_per_dim": x.max(dim=0).values.tolist(),
        "global_mean": float(x.mean().item()),
        "global_std": float(x.std(unbiased=False).item()),
        "global_min": float(x.min().item()),
        "global_max": float(x.max().item()),
    }


def _mse_mae_per_dim(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    err = (pred.float() - gt.float()).reshape(-1, pred.shape[-1])
    return {
        "mse": float(F.mse_loss(pred.float(), gt.float()).item()),
        "mae": float((pred.float() - gt.float()).abs().mean().item()),
        "mse_per_dim": (err ** 2).mean(dim=0).tolist(),
        "mae_per_dim": err.abs().mean(dim=0).tolist(),
        "gripper_mse": float((err[:, -1] ** 2).mean().item()),
        "gripper_sign_agree": float(
            ((pred.float().reshape(-1, pred.shape[-1])[:, -1] * gt.float().reshape(-1, gt.shape[-1])[:, -1]) > 0)
            .float()
            .mean()
            .item()
        ),
    }


def _load_keys_report(module: torch.nn.Module, state: dict, name: str) -> dict:
    result = module.load_state_dict(state, strict=False)
    return {
        "module": name,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "n_missing": len(result.missing_keys),
        "n_unexpected": len(result.unexpected_keys),
        "n_loaded_params": len(state),
    }


def _try_build_processor(cfg: dict):
    """Best-effort FastWAM processor for denorm; skip if FastWAM env incomplete."""
    try:
        import sys as _sys
        from pathlib import Path as _Path

        fastwam_root = _Path(cfg["fastwam_root"]).resolve()
        for p in (fastwam_root, fastwam_root / "src"):
            s = str(p)
            if s not in _sys.path:
                _sys.path.insert(0, s)
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # type: ignore
        from hydra import compose, initialize_config_dir  # type: ignore
        from hydra.core.global_hydra import GlobalHydra  # type: ignore
        from hydra.utils import instantiate  # type: ignore

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(fastwam_root / "configs"), version_base="1.3"):
            hydra_cfg = compose(config_name="train", overrides=[f"task={cfg['fastwam_task_config']}"])
        dataset_stats = load_dataset_stats_from_json(str(cfg["dataset_stats_path"]))
        processor = instantiate(hydra_cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(dataset_stats)
        return processor
    except Exception as exc:  # noqa: BLE001
        return f"SKIP_PROCESSOR: {type(exc).__name__}: {exc}"


def _denorm_only(action: torch.Tensor, processor) -> torch.Tensor:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    return normalizer.backward(action.detach().float().cpu())


def _postprocess_numpy(action: torch.Tensor, processor, invert_gripper_action) -> "object":
    import numpy as np

    denorm = _denorm_only(action, processor).numpy()
    if denorm.ndim == 3:
        denorm = denorm[0]
    denorm = denorm.copy()
    denorm[..., -1] = denorm[..., -1] * 2 - 1
    return invert_gripper_action(denorm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world2wam_physics_flow_dit_main.yaml")
    parser.add_argument("--cache_dir", default="cache/libero_spatial_h10_full_fastwam")
    parser.add_argument(
        "--ckpt",
        default="experiments/world2wam_physics_flow_dit_main/physics_world2wam_final.pt",
    )
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--flow_sample_steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/diagnose_sim_zero_report.json")
    parser.add_argument(
        "--steps_sweep",
        default="1,5,10,20,50",
        help="Comma-separated flow steps to sweep for physics sample MSE",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cfg = load_config(WORKSPACE / args.config)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (WORKSPACE / cache_dir).resolve()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (WORKSPACE / ckpt_path).resolve()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (WORKSPACE / out_path).resolve()

    meta = load_meta(cache_dir)
    state_dim = detect_state_dim(cache_dir)
    report: dict = {
        "config": str(args.config),
        "cache_dir": str(cache_dir),
        "ckpt": str(ckpt_path),
        "device": device,
        "num_samples": args.num_samples,
        "meta_num_samples": meta.get("num_samples"),
        "state_dim": state_dim,
        "findings": [],
        "verdicts": {},
    }

    # ---- L2: load physics model + key reports ----
    physics_model, adapter_type = build_physics_model(cfg, meta, device, cache_dir=cache_dir)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_reports = []
    load_reports.append(_load_keys_report(physics_model.forward_head, payload["forward_head"], "forward_head"))
    if "action_adapter" in payload and physics_model.action_adapter is not None:
        load_reports.append(
            _load_keys_report(physics_model.action_adapter, payload["action_adapter"], "action_adapter")
        )
    if "physics_router" in payload and physics_model.physics_router is not None:
        load_reports.append(
            _load_keys_report(physics_model.physics_router, payload["physics_router"], "physics_router")
        )
    else:
        load_reports.append(
            {
                "module": "physics_router",
                "missing_keys": ["CHECKPOINT_MISSING_physics_router"],
                "unexpected_keys": [],
                "n_missing": 1,
                "n_unexpected": 0,
            }
        )
    physics_model.eval()
    report["adapter_type"] = adapter_type
    report["ckpt_keys"] = list(payload.keys())
    report["load_reports"] = load_reports
    report["router_state_dim"] = int(getattr(physics_model.physics_router, "state_dim", -1))
    report["router_in_dim_ok"] = report["router_state_dim"] == state_dim

    if not report["router_in_dim_ok"]:
        report["findings"].append(
            f"L2 FAIL: router.state_dim={report['router_state_dim']} != cache state_dim={state_dim}"
        )
        report["verdicts"]["L2_load"] = "FAIL_state_dim"
    elif any(r["n_missing"] > 0 for r in load_reports):
        report["findings"].append(
            "L2 WARN: missing keys on load: "
            + "; ".join(f"{r['module']}={r['missing_keys'][:5]}" for r in load_reports if r["n_missing"])
        )
        report["verdicts"]["L2_load"] = "WARN_missing_keys"
    else:
        report["findings"].append("L2 OK: state_dim match and no missing keys")
        report["verdicts"]["L2_load"] = "OK"

    # Also probe adapter-only load (non-physics eval path)
    heads = build_heads_from_config(cfg, meta, include_inverse=False)
    adapter_only = heads["adapter"].to(device).eval()
    fwd_only = heads["forward"].to(device).eval()
    load_checkpoint(ckpt_path, fwd_only, adapter_only, expected_adapter_type="flow_dit")
    report["non_physics_path_note"] = (
        "ours_onestep_flow_dit loads adapter WITHOUT physics_router; "
        "model was trained WITH physics_code always injected."
    )

    # ---- data ----
    dataset = LatentCacheDataset(cache_dir, load_state=state_dim > 0)
    n = min(args.num_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:n].tolist()
    batch_list = [dataset[i] for i in indices]
    # pad to batches
    loader_batches = []
    for start in range(0, n, args.batch_size):
        chunk = batch_list[start : start + args.batch_size]
        loader_batches.append(collate_latent_batch(chunk))

    gt_actions = []
    pred_physics = []
    pred_nophys = []
    with torch.no_grad():
        for batch in loader_batches:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            gt = batch["action_chunk"]
            gt_actions.append(gt.cpu())

            out = physics_model.forward_inference(
                batch["z_t"],
                batch["text_embed"],
                state_t=batch.get("state_t"),
                num_flow_steps=args.flow_sample_steps,
            )
            pred_p = out["pred_action"]
            pred_physics.append(pred_p.cpu())

            # no-physics: sample adapter without physics_code
            pred_np = sample_action_adapter(
                adapter_only,
                batch["z_t"],
                batch["text_embed"],
                num_steps=args.flow_sample_steps,
            )
            pred_nophys.append(pred_np.cpu())

    gt_cat = torch.cat(gt_actions, dim=0)
    phy_cat = torch.cat(pred_physics, dim=0)
    nop_cat = torch.cat(pred_nophys, dim=0)

    report["gt_action_stats_normalized"] = _tensor_stats(gt_cat)
    report["pred_physics_stats_normalized"] = _tensor_stats(phy_cat)
    report["pred_nophysics_stats_normalized"] = _tensor_stats(nop_cat)
    report["mse_physics_vs_gt"] = _mse_mae_per_dim(phy_cat, gt_cat)
    report["mse_nophysics_vs_gt"] = _mse_mae_per_dim(nop_cat, gt_cat)

    mse_p = report["mse_physics_vs_gt"]["mse"]
    mse_np = report["mse_nophysics_vs_gt"]["mse"]
    if mse_p <= 0.2:
        report["verdicts"]["L4_physics_mse"] = "OK"
        report["findings"].append(f"L4 OK: physics sample MSE={mse_p:.4f} (<=0.2)")
    else:
        report["verdicts"]["L4_physics_mse"] = "FAIL_high_mse"
        report["findings"].append(f"L4 FAIL: physics sample MSE={mse_p:.4f} (>0.2)")

    if mse_np > mse_p * 1.5 and mse_np > 0.3:
        report["findings"].append(
            f"L2/L5: no-physics MSE={mse_np:.4f} >> physics MSE={mse_p:.4f} "
            "(non-physics eval drops physics_code used in training)"
        )
        report["verdicts"]["L2_physics_code_mismatch"] = "LIKELY"

    # steps sweep (physics)
    sweep = []
    for s in [int(x) for x in args.steps_sweep.split(",") if x.strip()]:
        errs = []
        with torch.no_grad():
            for batch in loader_batches:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = physics_model.forward_inference(
                    batch["z_t"],
                    batch["text_embed"],
                    state_t=batch.get("state_t"),
                    num_flow_steps=s,
                )
                errs.append(F.mse_loss(out["pred_action"].float(), batch["action_chunk"].float()).item())
        sweep.append({"flow_sample_steps": s, "mse": float(sum(errs) / max(len(errs), 1))})
    report["flow_steps_sweep"] = sweep

    # ---- L1 postprocess ----
    processor = _try_build_processor(cfg)
    if isinstance(processor, str):
        report["postprocess"] = {"status": processor}
        report["findings"].append(f"L1 SKIP postprocess: {processor}")
        report["verdicts"]["L1_postprocess"] = "SKIP"
    else:
        # Match FastWAM libero_utils.invert_gripper_action without importing libero.
        def invert_gripper_action(action):
            action = action.copy() if hasattr(action, "copy") else action
            action[..., -1] = action[..., -1] * -1.0
            return action

        import numpy as np

        # use first 8 samples for postprocess dump
        n_pp = min(8, gt_cat.shape[0])
        gt_pp = []
        phy_pp = []
        for i in range(n_pp):
            gt_pp.append(_postprocess_numpy(gt_cat[i], processor, invert_gripper_action))
            phy_pp.append(_postprocess_numpy(phy_cat[i], processor, invert_gripper_action))

        gt_pp_t = torch.tensor(np.stack(gt_pp), dtype=torch.float32)
        phy_pp_t = torch.tensor(np.stack(phy_pp), dtype=torch.float32)
        report["postprocess"] = {
            "status": "OK",
            "gt_physical_stats": _tensor_stats(gt_pp_t),
            "pred_physics_physical_stats": _tensor_stats(phy_pp_t),
            "physical_mse": _mse_mae_per_dim(phy_pp_t, gt_pp_t),
            "sample0_gt_physical": gt_pp_t[0, 0].tolist(),
            "sample0_pred_physical": phy_pp_t[0, 0].tolist(),
            "denorm_only_gt_grip_before_star2": float(
                _denorm_only(gt_cat[:1], processor)[0, 0, -1].item()
            ),
            "denorm_only_pred_grip_before_star2": float(
                _denorm_only(phy_cat[:1], processor)[0, 0, -1].item()
            ),
        }
        # Heuristic: if denorm gripper already in ~[-1,1] and then *2-1, polarity may be wrong
        g = report["postprocess"]["denorm_only_gt_grip_before_star2"]
        if abs(g) <= 1.05:
            report["findings"].append(
                f"L1 WARN: after denorm, GT gripper={g:.3f} already in ~[-1,1]; "
                "then code does grip*2-1 + invert_gripper — verify this matches FastWAM official path"
            )
            report["verdicts"]["L1_gripper_transform"] = "WARN_possible_double_transform"
        else:
            report["verdicts"]["L1_gripper_transform"] = "OK_or_ambiguous"
            report["findings"].append(f"L1: denorm GT gripper={g:.3f} (outside [-1,1] before *2-1)")

    # ---- summary recommendation ----
    if report["verdicts"].get("L4_physics_mse") == "OK" and report["verdicts"].get("L2_load") == "OK":
        report["recommended_next"] = (
            "Cache-side model looks healthy. Proceed to Phase B/C: "
            "suspect L5 onestep-vs-residual or L3 sim domain gap."
        )
    elif report["verdicts"].get("L4_physics_mse") != "OK":
        report["recommended_next"] = (
            "High cache MSE — fix L2 load / L4 flow / training quality before more sim."
        )
    else:
        report["recommended_next"] = "Inspect load warnings and postprocess gripper carefully."

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("verdicts", "findings", "recommended_next", "mse_physics_vs_gt", "mse_nophysics_vs_gt", "flow_steps_sweep", "router_state_dim")}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
