from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from ..models.action_expert_adapter import ActionExpertAdapterBank
from ..utils.checkpoint_utils import count_trainable_params

logger = logging.getLogger(__name__)

BACKBONE_MODES = ("frozen", "lora", "adapter", "full")


def resolve_backbone_mode(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("backbone_mode", "frozen")).lower()
    if mode not in BACKBONE_MODES:
        if cfg.get("freeze_fastwam_backbone", True):
            mode = "frozen"
        else:
            mode = "full"
    return mode


def hidden_should_detach(mode: str) -> bool:
    return mode == "frozen"


def sync_action_expert_to_mot(model: nn.Module) -> None:
    """Keep MoT action mixture aligned with model.action_expert after LoRA / replace."""
    if not hasattr(model, "mot") or not hasattr(model, "action_expert"):
        return
    mixtures = getattr(model.mot, "mixtures", None)
    if mixtures is not None and "action" in mixtures:
        mixtures["action"] = model.action_expert


def _freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _freeze_video_and_vae(model: nn.Module) -> None:
    for name in ("video_expert", "vae"):
        if hasattr(model, name):
            _freeze_module(getattr(model, name))


def _apply_lora(action_expert: nn.Module, cfg: dict[str, Any]) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "backbone_mode=lora requires `peft`. Install with: pip install peft>=0.10.0"
        ) from exc

    target_modules = cfg.get("lora_target_modules", ["q", "k", "v", "o"])
    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_rank", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        target_modules=list(target_modules),
        bias="none",
    )
    peft_model = get_peft_model(action_expert, lora_cfg)
    matched = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info(
        "LoRA on action_expert: rank=%s alpha=%s targets=%s trainable=%d",
        cfg.get("lora_rank", 8),
        cfg.get("lora_alpha", 16),
        target_modules,
        matched,
    )
    if matched == 0:
        raise RuntimeError(
            "LoRA matched 0 trainable parameters on action_expert. "
            "Check lora_target_modules against DiTBlock layer names (q,k,v,o)."
        )
    return peft_model


def apply_backbone_mode(
    model: nn.Module,
    mode: str,
    cfg: dict[str, Any],
) -> tuple[nn.Module, ActionExpertAdapterBank | None, bool, str]:
    """
    Configure FastWAM model trainability.

    Returns:
        model (possibly with patched action_expert),
        adapter_bank or None,
        hidden_detached flag,
        effective backbone mode actually applied
    """
    mode = mode.lower()
    if mode not in BACKBONE_MODES:
        raise ValueError(f"Unknown backbone_mode: {mode}. Choose from {BACKBONE_MODES}")

    adapter_bank: ActionExpertAdapterBank | None = None
    _freeze_video_and_vae(model)

    if not hasattr(model, "action_expert"):
        raise AttributeError("FastWAM model has no action_expert")

    if mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        return model, None, True, "frozen"

    if mode == "lora":
        for p in model.parameters():
            p.requires_grad = False
        try:
            model.action_expert = _apply_lora(model.action_expert, cfg)
            sync_action_expert_to_mot(model)
        except (ImportError, RuntimeError) as exc:
            logger.warning("LoRA failed (%s); falling back to adapter mode.", exc)
            mode = "adapter"
        else:
            return model, None, False, "lora"

    if mode == "adapter":
        for p in model.parameters():
            p.requires_grad = False
        adapter_bank = ActionExpertAdapterBank(
            model.action_expert,
            bottleneck_dim=int(cfg.get("adapter_hidden_dim", 256)),
        )
        adapter_bank.attach(model.action_expert)
        sync_action_expert_to_mot(model)
        for p in adapter_bank.parameters():
            p.requires_grad = True
        logger.info(
            "Adapter bank on action_expert: %d blocks, trainable=%d",
            len(adapter_bank.adapters),
            count_trainable_params(adapter_bank),
        )
        return model, adapter_bank, False, "adapter"

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = False
        _freeze_video_and_vae(model)
        for p in model.action_expert.parameters():
            p.requires_grad = True
        if hasattr(model, "mot"):
            for name, p in model.mot.named_parameters():
                if "action" in name.lower():
                    p.requires_grad = True
        logger.info("Full action-path finetune trainable=%d", count_trainable_params(model))
        return model, None, False, "full"

    raise ValueError(f"Unhandled backbone_mode: {mode}")
