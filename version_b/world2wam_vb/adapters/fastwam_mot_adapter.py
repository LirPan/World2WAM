from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from world2wam_vb.utils.import_paths import add_fastwam_path
from world2wam_vb.utils.training import count_trainable_params

logger = logging.getLogger(__name__)


class FastWAMMotAdapter(nn.Module):
    """
    Frozen FastWAM backbone with MoT action-token hook.

    Captures post-MoT action tokens [B, T, hidden_dim] during training_loss.
    Supports optional λ_video > 0 for joint video-action training (Version B).
    """

    FROZEN_MODULES = ("video_expert", "action_expert", "vae", "text_encoder")

    def __init__(
        self,
        fastwam_root: str | Path,
        fastwam_task_config: str = "libero_uncond_2cam224_1e-4",
        official_checkpoint: str | Path | None = None,
        cfg: dict[str, Any] | None = None,
        device: str = "cuda",
        mixed_precision: str = "bf16",
    ):
        super().__init__()
        self._cfg = cfg or {}
        self.fastwam_root = Path(fastwam_root).resolve()
        self.device_str = device
        self.backbone_mode = str(self._cfg.get("backbone_mode", "frozen")).lower()
        self.hidden_detached = self.backbone_mode == "frozen"
        self.lambda_video = float(self._cfg.get("lambda_video", 0.0))
        self.lambda_action = float(self._cfg.get("lambda_action", 1.0))

        self._captured_action_tokens: torch.Tensor | None = None
        self._captured_video_tokens: torch.Tensor | None = None
        self._last_loss_dict: dict[str, torch.Tensor] = {}
        self._hook_handle = None

        add_fastwam_path(self.fastwam_root)
        os.chdir(self.fastwam_root)

        self.model, self._hydra_cfg = self._load_model(
            task_name=fastwam_task_config,
            mixed_precision=mixed_precision,
            device=device,
        )

        ckpt = official_checkpoint or self._cfg.get("official_fastwam_checkpoint")
        if ckpt is not None:
            ckpt_path = Path(ckpt)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"FastWAM checkpoint not found: {ckpt_path}")
            self.model.load_checkpoint(str(ckpt_path))
            self._loaded_checkpoint = str(ckpt_path.resolve())
        else:
            self._loaded_checkpoint = None

        self._apply_loss_weights()
        if self.backbone_mode == "frozen":
            self._freeze_fastwam()
        self._register_mot_hook()
        logger.info(
            "FastWAMMotAdapter ckpt=%s trainable=%d λ_video=%.3f λ_action=%.3f",
            self._loaded_checkpoint,
            count_trainable_params(self.model),
            self.lambda_video,
            self.lambda_action,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FastWAMMotAdapter":
        return cls(
            fastwam_root=cfg["fastwam_root"],
            fastwam_task_config=cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4"),
            official_checkpoint=cfg.get("official_fastwam_checkpoint"),
            cfg=cfg,
            device=cfg.get("device", "cuda"),
            mixed_precision=cfg.get("mixed_precision", "bf16"),
        )

    def _load_model(self, task_name: str, mixed_precision: str, device: str):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        from fastwam.runtime import _mixed_precision_to_model_dtype

        config_dir = self.fastwam_root / "configs"
        if not config_dir.is_dir():
            raise FileNotFoundError(f"FastWAM configs/ missing at {config_dir}")

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            hydra_cfg: DictConfig = compose(config_name="train", overrides=[f"task={task_name}"])

        model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
        model = instantiate(hydra_cfg.model, model_dtype=model_dtype, device=device)

        for attr in ("mot", "training_loss", "infer_action", "action_expert", "video_expert"):
            if not hasattr(model, attr):
                raise AttributeError(f"Loaded model missing `{attr}`. Expected FastWAM MoT build.")

        return model, hydra_cfg

    def _apply_loss_weights(self) -> None:
        self.model.loss_lambda_video = self.lambda_video
        self.model.loss_lambda_action = self.lambda_action

    def _freeze_fastwam(self) -> None:
        for name in self.FROZEN_MODULES:
            module = getattr(self.model, name)
            for param in module.parameters():
                param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _register_mot_hook(self) -> None:
        def _hook(_module, _inputs, output):
            if isinstance(output, dict):
                if "action" in output:
                    self._captured_action_tokens = output["action"]
                if "video" in output:
                    self._captured_video_tokens = output["video"]

        self._hook_handle = self.model.mot.register_forward_hook(_hook)

    @property
    def action_dim(self) -> int:
        return int(self.model.action_expert.action_dim)

    @property
    def hidden_dim(self) -> int:
        return int(self.model.action_expert.hidden_dim)

    @property
    def horizon(self) -> int:
        return int(getattr(self.model.action_expert, "horizon", 10))

    def _to_fastwam_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
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
                if isinstance(val, torch.Tensor) and key in (
                    "context",
                    "context_mask",
                    "video",
                    "action",
                    "proprio",
                ):
                    out[key] = val.to(device=self.model.device)
                else:
                    out[key] = val
        if "language" in batch and "prompt" not in out:
            out["prompt"] = batch["language"]
        missing = [k for k in ("video", "action", "context", "context_mask") if k not in out]
        if missing:
            raise KeyError(f"Batch missing FastWAM keys: {missing}")
        return out

    def pool_action_tokens(self, tokens: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
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
            return (tokens * valid.unsqueeze(-1)).sum(dim=1) / denom
        if tokens.shape[1] > anchor:
            return tokens[:, anchor, :]
        return tokens.mean(dim=1)

    def training_forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run FastWAM training_loss and capture MoT action tokens."""
        self._captured_action_tokens = None
        self._captured_video_tokens = None
        train_batch = self._to_fastwam_batch(batch)

        loss_total, loss_dict = self.model.training_loss(train_batch)
        self._last_loss_dict = loss_dict

        if self._captured_action_tokens is None:
            raise RuntimeError(
                "MoT hook did not capture action tokens. "
                "Check that model.mot.forward returns dict with 'action' key."
            )

        action_tokens = self._captured_action_tokens
        h_t = self.pool_action_tokens(action_tokens, batch)
        if self.hidden_detached:
            h_t = h_t.detach()
            action_tokens = action_tokens.detach()

        loss_action = loss_dict.get("loss_action", loss_total)
        loss_video = loss_dict.get("loss_video", torch.zeros((), device=loss_total.device))

        return {
            "h_t": h_t,
            "action_tokens": action_tokens,
            "video_tokens": self._captured_video_tokens,
            "loss_fastwam_total": loss_total,
            "loss_fastwam_action": loss_action,
            "loss_fastwam_video": loss_video,
            "loss_dict": loss_dict,
        }

    @torch.no_grad()
    def infer_action_only(self, batch: dict[str, Any]) -> torch.Tensor:
        self.model.eval()
        video = batch.get("video") or batch.get("obs")
        if video is None:
            raise ValueError("infer_action_only requires video/obs")

        if video.dim() == 4:
            video = video.unsqueeze(0)
        frame = video[:, :, 0] if video.dim() == 5 else video[:, 0]
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        prompt = batch.get("prompt") or batch.get("language")
        if isinstance(prompt, list):
            prompt = prompt[0]

        infer_kwargs: dict[str, Any] = {
            "input_image": frame.to(device=self.model.device, dtype=self.model.torch_dtype),
            "action_horizon": int(batch.get("action_horizon", 10)),
            "num_inference_steps": int(batch.get("num_inference_steps", 20)),
        }
        context = batch.get("context")
        context_mask = batch.get("context_mask")
        if context is not None and context_mask is not None:
            infer_kwargs["context"] = context.to(device=self.model.device)
            infer_kwargs["context_mask"] = context_mask.to(device=self.model.device)
            infer_kwargs["prompt"] = None
        else:
            infer_kwargs["prompt"] = str(prompt)

        proprio = batch.get("proprio")
        if proprio is not None:
            infer_kwargs["proprio"] = proprio[:, 0] if proprio.dim() == 3 else proprio

        out = self.model.infer_action(**infer_kwargs)
        return out.get("action") if isinstance(out, dict) else out

    @torch.no_grad()
    def encode_future_frames(self, future_obs: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """Encode future RGB frames to pooled VAE latent [B, latent_dim]."""
        if not hasattr(self.model, "vae"):
            raise AttributeError("FastWAM model has no `vae` for future frame encoding.")
        if not hasattr(self.model, "_encode_video_latents"):
            raise AttributeError("FastWAM model has no `_encode_video_latents`.")

        x = future_obs
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.shape[1] != 3:
            raise ValueError(f"future_obs must have channel dim 3, got {x.shape}")

        x = x.to(device=self.model.device, dtype=self.model.torch_dtype)
        z = self.model._encode_video_latents(x, tiled=tiled)
        return z.float().mean(dim=(2, 3, 4))

    @torch.no_grad()
    def encode_obs_latent(self, obs: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """Encode current observation frame(s) to pooled VAE latent."""
        return self.encode_future_frames(obs, tiled=tiled)
