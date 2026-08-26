from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from ..models.action_expert_adapter import ActionExpertAdapterBank
from ..utils.checkpoint_utils import count_trainable_params, load_world2wam_checkpoint, resolve_official_checkpoint
from .backbone_modes import apply_backbone_mode, hidden_should_detach, resolve_backbone_mode, sync_action_expert_to_mot
from ..utils.import_utils import add_fastwam_path

logger = logging.getLogger(__name__)


class FastWAMWrapper(nn.Module):
    """
    External wrapper around FastWAM (no edits to FastWAM source).
    Captures MoT action tokens via forward hook during training_loss.
    """

    def __init__(
        self,
        fastwam_config_path: str | Path | None = None,
        fastwam_root: str | Path | None = None,
        fastwam_task_config: str = "libero_uncond_2cam224_1e-4",
        checkpoint_path: str | Path | None = None,
        official_checkpoint: str | Path | None = None,
        freeze_backbone: bool | None = None,
        backbone_mode: str | None = None,
        cfg: dict[str, Any] | None = None,
        device: str = "cuda",
        mixed_precision: str = "bf16",
    ):
        super().__init__()
        if fastwam_root is None and fastwam_config_path is None:
            raise ValueError("Provide fastwam_root or fastwam_config_path.")

        if fastwam_root is not None:
            self.fastwam_root = Path(fastwam_root).resolve()
        else:
            self.fastwam_root = Path(fastwam_config_path).resolve().parents[1]

        self._cfg = cfg or {}
        if backbone_mode is None:
            backbone_mode = resolve_backbone_mode(self._cfg)
        if freeze_backbone is not None and backbone_mode is None:
            backbone_mode = "frozen" if freeze_backbone else "full"

        self.backbone_mode = str(backbone_mode).lower()
        self.hidden_detached = hidden_should_detach(self.backbone_mode)
        self._adapter_bank: ActionExpertAdapterBank | None = None
        self._uses_future_video = False
        self._orig_loss_lambda_video: float | None = None

        add_fastwam_path(self.fastwam_root)
        os.chdir(self.fastwam_root)

        self._captured_action_tokens: torch.Tensor | None = None
        self._hook_handle = None
        self.device_str = device
        self.model, self._hydra_cfg = self._load_model(
            task_name=fastwam_task_config,
            mixed_precision=mixed_precision,
            device=device,
        )

        ckpt = official_checkpoint or checkpoint_path
        if ckpt is not None:
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            self.model.load_checkpoint(str(ckpt_path))
            self._loaded_checkpoint = str(ckpt_path.resolve())
        else:
            self._loaded_checkpoint = None

        self.model, self._adapter_bank, self.hidden_detached, effective_mode = apply_backbone_mode(
            self.model, self.backbone_mode, self._cfg
        )
        self.backbone_mode = effective_mode
        self.trainable_param_count = count_trainable_params(self.model)
        if self._adapter_bank is not None:
            self.trainable_param_count += count_trainable_params(self._adapter_bank)

        self._register_mot_hook()

    def _load_model(self, task_name: str, mixed_precision: str, device: str):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        from fastwam.runtime import _mixed_precision_to_model_dtype

        config_dir = self.fastwam_root / "configs"
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg: DictConfig = compose(config_name="train", overrides=[f"task={task_name}"])

        model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
        model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
        return model, cfg

    def _register_mot_hook(self) -> None:
        if not hasattr(self.model, "mot"):
            raise AttributeError(
                "Loaded model has no `mot` module. Expected FastWAM / FastWAMJoint / FastWAMIDM."
            )

        def _hook(_module, _inputs, output):
            if isinstance(output, dict) and "action" in output:
                self._captured_action_tokens = output["action"]

        self._hook_handle = self.model.mot.register_forward_hook(_hook)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self._adapter_bank is not None:
            self._adapter_bank.detach()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **kwargs: Any) -> "FastWAMWrapper":
        official = resolve_official_checkpoint(cfg)
        mode = kwargs.pop("backbone_mode", None) or resolve_backbone_mode(cfg)
        return cls(
            fastwam_root=cfg["fastwam_root"],
            fastwam_task_config=cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4"),
            official_checkpoint=official,
            backbone_mode=mode,
            cfg=cfg,
            device=cfg.get("device", "cuda"),
            mixed_precision=cfg.get("mixed_precision", "bf16"),
            **kwargs,
        )

    def load_world2wam_bundle(self, bundle_path: str | Path) -> None:
        """Load policy bundle: optional LoRA/adapter + metadata (official ckpt already loaded)."""
        payload = load_world2wam_checkpoint(Path(bundle_path))
        extra = payload.get("backbone_extra") or {}
        mode = str(payload.get("backbone_mode", self.backbone_mode))

        if mode == "lora" and "lora" in extra:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(self.model.action_expert, extra["lora"])
            sync_action_expert_to_mot(self.model)
        elif mode == "adapter" and "adapter" in extra and self._adapter_bank is not None:
            self._adapter_bank.load_state_dict(extra["adapter"])
        elif mode == "full" and "action_expert" in extra:
            self.model.action_expert.load_state_dict(extra["action_expert"], strict=False)
            sync_action_expert_to_mot(self.model)

        self._world2wam_bundle_path = str(Path(bundle_path).resolve())

    def get_backbone_state_for_save(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self._adapter_bank is not None:
            extra["adapter"] = self._adapter_bank.state_dict_adapter_only()
        elif self.backbone_mode == "lora":
            try:
                from peft.utils import get_peft_model_state_dict

                extra["lora"] = get_peft_model_state_dict(self.model.action_expert)
            except Exception:
                extra["lora"] = {
                    k: v.cpu()
                    for k, v in self.model.action_expert.state_dict().items()
                    if "lora" in k.lower()
                }
        elif self.backbone_mode == "full":
            extra["action_expert"] = self.model.action_expert.state_dict()
        return extra

    @property
    def action_dim(self) -> int:
        return int(self.model.action_expert.action_dim)

    @property
    def hidden_dim(self) -> int:
        return int(self.model.action_expert.hidden_dim)

    def forward_train(
        self,
        batch: dict[str, Any],
        use_future_latent_distill: bool = False,
        policy_action_only_loss: bool | None = None,
    ) -> dict[str, Any]:
        del use_future_latent_distill
        self._captured_action_tokens = None
        train_batch = self._to_fastwam_batch(batch)

        if not hasattr(self.model, "training_loss"):
            raise AttributeError("Model does not implement training_loss().")

        use_action_only = policy_action_only_loss
        if use_action_only is None:
            use_action_only = bool(self._cfg.get("policy_action_only_loss", True))

        if use_action_only:
            if self._orig_loss_lambda_video is None:
                self._orig_loss_lambda_video = float(getattr(self.model, "loss_lambda_video", 1.0))
            self.model.loss_lambda_video = 0.0
        elif self._orig_loss_lambda_video is not None:
            self.model.loss_lambda_video = self._orig_loss_lambda_video

        loss_total, loss_dict = self.model.training_loss(train_batch)

        if use_action_only:
            action_loss_tensor = loss_total
        else:
            action_loss_tensor = loss_total
            if "loss_action" in loss_dict and "loss_video" in loss_dict:
                action_loss_tensor = loss_total

        hidden = self.extract_hidden({"action_tokens": self._captured_action_tokens}, batch)
        if self.hidden_detached:
            hidden = hidden.detach()

        return {
            "pred_action": None,
            "action_loss": action_loss_tensor,
            "loss_dict": loss_dict,
            "loss_total": loss_total,
            "hidden": hidden,
            "hidden_detached": self.hidden_detached,
            "backbone_mode": self.backbone_mode,
            "raw_outputs": {
                "action_tokens": self._captured_action_tokens,
                "loss_dict": loss_dict,
            },
        }

    def forward_action_only(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Inference via FastWAM.infer_action — no future latent head."""
        if not hasattr(self.model, "infer_action"):
            raise AttributeError("Model does not implement infer_action().")

        self.model.eval()
        video = batch.get("video")
        if video is None:
            raise ValueError("forward_action_only requires batch['video'] or batch['obs'].")

        if video.dim() == 4:
            video = video.unsqueeze(0)
        if video.shape[1] != 3:
            if video.shape[-1] == 3:
                video = video.permute(0, 3, 1, 2)
            else:
                video = video.unsqueeze(1) if video.dim() == 4 else video

        frame = video[:, :, 0] if video.dim() == 5 else video[:, 0]
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        prompt = batch.get("prompt") or batch.get("language")
        if isinstance(prompt, list):
            prompt = prompt[0]
        context = batch.get("context")
        context_mask = batch.get("context_mask")
        proprio = batch.get("proprio")
        action_horizon = batch.get("action_horizon")
        if action_horizon is None:
            action = batch.get("action")
            if action is not None:
                action_horizon = int(action.shape[-2] if action.dim() >= 2 else 8)
            else:
                action_horizon = 8

        infer_kwargs: dict[str, Any] = {
            "input_image": frame.to(device=self.model.device, dtype=self.model.torch_dtype),
            "action_horizon": int(action_horizon),
            "num_inference_steps": batch.get("num_inference_steps", 20),
        }
        if context is not None and context_mask is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = context.to(device=self.model.device) if isinstance(context, torch.Tensor) else context
            infer_kwargs["context_mask"] = (
                context_mask.to(device=self.model.device)
                if isinstance(context_mask, torch.Tensor)
                else context_mask
            )
        else:
            if not prompt:
                raise ValueError("forward_action_only needs context/context_mask or prompt in batch.")
            infer_kwargs["prompt"] = str(prompt)
            infer_kwargs["context"] = None
            infer_kwargs["context_mask"] = None
        if proprio is not None:
            infer_kwargs["proprio"] = proprio[:, 0] if proprio.dim() == 3 else proprio

        out = self.model.infer_action(**infer_kwargs)
        action = out.get("action") if isinstance(out, dict) else out
        return {"pred_action": action}

    def extract_hidden(self, raw_outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
        tokens = raw_outputs.get("action_tokens")
        if tokens is None:
            tokens = self._captured_action_tokens
        if tokens is None:
            raise RuntimeError(
                "No action tokens captured. MoT forward hook did not run — "
                "confirm training_loss executes mot.forward."
            )

        mask = batch.get("action_is_pad")
        anchor_t = batch.get("anchor_action_idx", 0)
        if isinstance(anchor_t, torch.Tensor):
            anchor = int(anchor_t[0].item())
        elif isinstance(anchor_t, list):
            anchor = int(anchor_t[0])
        else:
            anchor = int(anchor_t)
        if anchor < 0 or anchor >= tokens.shape[1]:
            anchor = 0

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            valid = (~mask).to(dtype=tokens.dtype, device=tokens.device)
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (tokens * valid.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = tokens.mean(dim=1)

        if mask is None and tokens.shape[1] > anchor:
            pooled = tokens[:, anchor, :]

        return pooled

    @torch.no_grad()
    def encode_future_latent(self, future_obs: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        self._uses_future_video = True
        if not hasattr(self.model, "vae"):
            raise AttributeError(
                "FastWAM model has no `vae`. Cannot encode future_latent. "
                "Confirm Wan weights are loaded under FastWAM/checkpoints/."
            )
        x = future_obs
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.shape[1] != 3:
            raise ValueError(f"future_obs must have channel dim 3, got {x.shape}")
        x = x.to(device=self.model.device, dtype=self.model.torch_dtype)
        z = self.model._encode_video_latents(x, tiled=tiled)
        return z.float().mean(dim=(2, 3, 4))

    def _to_fastwam_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key in (
            "video",
            "action",
            "proprio",
            "prompt",
            "context",
            "context_mask",
            "image_is_pad",
            "action_is_pad",
            "proprio_is_pad",
        ):
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor) and key in ("context", "context_mask", "video", "action", "proprio"):
                    out[key] = val.to(device=self.model.device)
                else:
                    out[key] = val
        if "language" in batch and "prompt" not in out:
            out["prompt"] = batch["language"]
        missing = [k for k in ("video", "action", "context", "context_mask") if k not in out]
        if missing:
            raise KeyError(
                f"Batch missing required FastWAM keys: {missing}. "
                "Use LiberoDatasetAdapter / collate_world2wam_batch."
            )
        return out
