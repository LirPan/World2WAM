from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
if str(_MINIMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMAL_ROOT))

from src.utils.checkpoint_utils import load_world2wam_checkpoint, normalize_config, resolve_official_checkpoint
from src.utils.config import load_config
from src.utils.path_utils import minimal_project_root, resolve_path
from src.wrappers.backbone_modes import sync_action_expert_to_mot
from src.wrappers.fastwam_wrapper import FastWAMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_action_expert(model: torch.nn.Module, lora_scale: float = 1.0) -> None:
    from peft import PeftModel

    if not isinstance(model.action_expert, PeftModel):
        logger.warning("action_expert is not PeftModel; skip merge_and_unload")
        return
    if lora_scale < 0:
        raise ValueError(f"lora_scale must be non-negative, got {lora_scale}")
    scaled_adapters = 0
    for module in model.action_expert.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name, value in list(scaling.items()):
            scaling[adapter_name] = value * lora_scale
            scaled_adapters += 1
    if scaled_adapters == 0:
        raise RuntimeError("No LoRA adapter scaling entries were found before merge")
    model.action_expert = model.action_expert.merge_and_unload()
    sync_action_expert_to_mot(model)
    logger.info(
        "Merged LoRA into action_expert at scale=%s (scaled entries=%d) and synced MoT",
        lora_scale,
        scaled_adapters,
    )


def build_fastwam_payload(model: torch.nn.Module, meta: dict | None = None) -> dict:
    payload: dict = {
        "mot": model.mot.state_dict(),
        "step": int((meta or {}).get("global_step", 0)),
        "torch_dtype": str(model.torch_dtype),
    }
    if getattr(model, "proprio_encoder", None) is not None:
        payload["proprio_encoder"] = model.proprio_encoder.state_dict()
    return payload


def export_libero_checkpoint(
    *,
    bundle_path: Path,
    config_path: Path,
    out_path: Path,
    sidecar_path: Path | None = None,
    lora_scale: float = 1.0,
) -> Path:
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"World2WAM bundle not found: {bundle_path}")

    cfg = normalize_config(load_config(config_path))
    payload = load_world2wam_checkpoint(bundle_path)
    backbone_mode = str(payload.get("backbone_mode", cfg.get("backbone_mode", "lora"))).lower()
    # Resolve from the active config so bundles remain portable across servers.
    official = resolve_official_checkpoint(cfg)
    export_device = os.environ.get("EXPORT_DEVICE", cfg.get("export_device", cfg.get("device", "cuda")))

    logger.info("Export bundle=%s mode=%s official=%s device=%s", bundle_path, backbone_mode, official, export_device)

    wrapper = FastWAMWrapper.from_config(
        {**cfg, "backbone_mode": backbone_mode, "device": export_device},
    )
    wrapper.model.load_checkpoint(str(official))
    wrapper.load_world2wam_bundle(bundle_path)
    sync_action_expert_to_mot(wrapper.model)

    if backbone_mode == "lora":
        merge_lora_action_expert(wrapper.model, lora_scale=lora_scale)
    elif backbone_mode == "adapter":
        raise NotImplementedError(
            "Adapter-mode export is not supported. Train policy with backbone_mode=lora "
            "or implement adapter bake-in before LIBERO sim eval."
        )
    elif backbone_mode == "full":
        sync_action_expert_to_mot(wrapper.model)
    elif backbone_mode == "frozen":
        logger.info("Frozen bundle: exporting official MoT weights only (no policy delta).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = payload.get("meta") or {}
    torch.save(build_fastwam_payload(wrapper.model, meta), out_path)
    logger.info("Saved merged FastWAM checkpoint: %s", out_path)

    sidecar = {
        "merged_checkpoint": str(out_path),
        "bundle_source": str(bundle_path),
        "official_checkpoint": str(official),
        "backbone_mode": backbone_mode,
        "lora_scale": lora_scale,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
    }
    sidecar_out = sidecar_path or out_path.with_suffix(".json")
    with open(sidecar_out, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    logger.info("Wrote sidecar: %s", sidecar_out)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export World2WAM bundle to FastWAM LIBERO-compatible .pt")
    parser.add_argument("--bundle", type=str, required=True, help="world2wam_final.pt bundle path")
    parser.add_argument("--config", type=str, default="configs/world2wam_policy_improve.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path")
    parser.add_argument("--tag", type=str, default=None, help="Tag for default output filename")
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to every LoRA adapter delta before merge.",
    )
    args = parser.parse_args()

    root = minimal_project_root()
    bundle = resolve_path(args.bundle, root)
    config = resolve_path(args.config, root)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (
        resolve_path(args.output, root)
        if args.output
        else root / "experiments/exported_ckpts" / f"world2wam_merged_{tag}.pt"
    )
    export_libero_checkpoint(
        bundle_path=bundle,
        config_path=config,
        out_path=out,
        lora_scale=args.lora_scale,
    )
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
