#!/usr/bin/env python3
"""Action-only evaluation: FastWAM.infer_action, no FutureLatentHead / InverseActionHead."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
if str(_MINIMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMAL_ROOT))

from src.data.libero_dataset_adapter import LiberoDatasetAdapter, build_fastwam_dataset, collate_world2wam_batch
from src.utils.checkpoint_utils import load_world2wam_checkpoint, resolve_official_checkpoint
from src.utils.config import load_config
from src.utils.path_utils import minimal_project_root, resolve_path
from src.utils.seed import set_seed
from src.wrappers.action_only_inference import run_action_only_batch
from src.wrappers.fastwam_wrapper import FastWAMWrapper
from src.wrappers.inference_guard import inference_guard


def offline_action_mse(wrapper: FastWAMWrapper, loader: DataLoader, device: str, max_batches: int) -> float:
    total = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        with torch.no_grad():
            fw = wrapper.forward_train(
                batch,
                use_future_latent_distill=False,
                policy_action_only_loss=True,
            )
        pred = fw.get("pred_action")
        if pred is None:
            continue
        gt = batch["action"]
        if gt.dim() == 3 and pred.shape != gt.shape:
            pred = pred[:, : gt.shape[1]]
        mse = torch.nn.functional.mse_loss(pred.float(), gt.float()).item()
        total += mse
        count += 1
    return total / max(count, 1)


def _resolve_eval_checkpoint(cfg: dict, cli_ckpt: str | None) -> tuple[Path, str]:
    if cli_ckpt:
        path = Path(resolve_path(cli_ckpt, minimal_project_root()))
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        role = "world2wam_bundle" if path.name.startswith("world2wam") else "official"
        return path, role
    official = resolve_official_checkpoint(cfg)
    return official, "official"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fastwam_future_distill.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Official or world2wam_final.pt bundle")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--run-libero-sim", action="store_true")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, minimal_project_root()))
    set_seed(int(cfg.get("seed", 42)))
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt_path, eval_role = _resolve_eval_checkpoint(cfg, args.checkpoint)
    backbone_mode = "official"
    if ckpt_path.name.startswith("world2wam") or eval_role == "world2wam_bundle":
        payload = load_world2wam_checkpoint(ckpt_path)
        backbone_mode = str(payload.get("backbone_mode", "lora"))
        official = Path(payload.get("official_checkpoint", resolve_official_checkpoint(cfg)))
        wrapper = FastWAMWrapper.from_config(cfg, backbone_mode=backbone_mode)
        wrapper.model.load_checkpoint(str(official))
        wrapper.load_world2wam_bundle(ckpt_path)
        if payload.get("future_head"):
            pass  # must NOT load into eval — action-only only
    else:
        wrapper = FastWAMWrapper.from_config(cfg, backbone_mode="frozen")

    base_ds, _ = build_fastwam_dataset(cfg)
    adapter = LiberoDatasetAdapter(base_ds, future_horizon=int(cfg.get("future_horizon", 1)), cache=None)
    loader = DataLoader(
        adapter,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_world2wam_batch,
    )

    latencies: list[float] = []
    with inference_guard():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            _, ms = run_action_only_batch(wrapper, batch)
            latencies.append(ms)

    results = {
        "future_head_called": False,
        "inverse_head_called": False,
        "uses_future_video": False,
        "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
        "num_batches": len(latencies),
        "success_rate": None,
        "offline_action_mse": None,
        "checkpoint": str(ckpt_path.resolve()),
        "eval_role": eval_role,
        "backbone_mode": backbone_mode,
    }

    try:
        with inference_guard():
            results["offline_action_mse"] = offline_action_mse(wrapper, loader, device, args.max_batches)
    except Exception as exc:
        results["offline_action_mse_error"] = str(exc)

    if args.run_libero_sim:
        eval_script = Path(cfg["fastwam_root"]) / "experiments" / "libero" / "eval_libero_single.py"
        if eval_script.exists():
            results["libero_sim_note"] = (
                f"Run FastWAM eval: python {eval_script} ckpt={results['checkpoint']}"
            )
        else:
            results["libero_sim_note"] = f"eval script not found: {eval_script}"

    out_path = args.output or str(Path(cfg["output_dir"]) / "eval_results.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
