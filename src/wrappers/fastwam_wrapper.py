from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from ..utils.import_utils import add_fastwam_path


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
        freeze_backbone: bool = False,
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

        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            self.model.load_checkpoint(str(ckpt))

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

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
    ) -> dict[str, Any]:
        """
        Run FastWAM training_loss and optionally expose hidden for distillation.
        """
        del use_future_latent_distill  # head applied outside wrapper
        self._captured_action_tokens = None
        train_batch = self._to_fastwam_batch(batch)

        if not hasattr(self.model, "training_loss"):
            raise AttributeError("Model does not implement training_loss().")

        loss_total, loss_dict = self.model.training_loss(train_batch)

        hidden = self.extract_hidden({"action_tokens": self._captured_action_tokens}, batch)

        action_loss_tensor = loss_total
        if "loss_action" in loss_dict and "loss_video" in loss_dict:
            # Reconstruct scalar action-only component when both logged
            action_loss_tensor = loss_total  # caller uses loss_dict for logging

        return {
            "pred_action": None,
            "action_loss": action_loss_tensor,
            "loss_dict": loss_dict,
            "loss_total": loss_total,
            "hidden": hidden,
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

        # Current frame as [1,3,H,W]
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
            infer_kwargs["context"] = context
            infer_kwargs["context_mask"] = context_mask
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
        """
        Pool MoT action tokens -> [B, hidden_dim].
        """
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

        # If anchor specified and no mask, prefer anchor token
        if mask is None and tokens.shape[1] > anchor:
            pooled = tokens[:, anchor, :]

        return pooled

    @torch.no_grad()
    def encode_future_latent(self, future_obs: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """
        Encode future RGB clip with FastWAM VAE; mean-pool to vector [B, C_lat].
        future_obs: [B,3,T,H,W] or [B,3,H,W]
        """
        if not hasattr(self.model, "vae"):
            raise AttributeError(
                "FastWAM model has no `vae`. Cannot encode future_latent. "
                "TODO: confirm Wan weights loaded."
            )
        x = future_obs
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.shape[1] != 3:
            raise ValueError(f"future_obs must have channel dim 3, got {x.shape}")
        x = x.to(device=self.model.device, dtype=self.model.torch_dtype)
        z = self.model._encode_video_latents(x, tiled=tiled)
        # z: [B, C, T', H', W']
        return z.float().mean(dim=(2, 3, 4))

    def _to_fastwam_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Ensure batch keys match FastWAM training_loss expectations."""
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
                out[key] = batch[key]
        if "language" in batch and "prompt" not in out:
            out["prompt"] = batch["language"]
        missing = [k for k in ("video", "action", "context", "context_mask") if k not in out]
        if missing:
            raise KeyError(
                f"Batch missing required FastWAM keys: {missing}. "
                "Use LiberoDatasetAdapter / collate_world2wam_batch."
            )
        return out
