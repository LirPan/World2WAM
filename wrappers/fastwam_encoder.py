from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def add_fastwam_path(fastwam_root: str | Path) -> None:
    root = Path(fastwam_root).resolve()
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"FastWAM src/ missing at {src}")
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


def pool_vae_latent(z: torch.Tensor) -> torch.Tensor:
    """Pool VAE latent [B,C,T,H,W] or [B,C,H,W] -> [B,C]."""
    if z.dim() == 5:
        return z.float().mean(dim=(2, 3, 4))
    if z.dim() == 4:
        return z.float().mean(dim=(2, 3))
    raise ValueError(f"Expected 4D or 5D VAE latent, got {tuple(z.shape)}")


class FastWAMEncoder(nn.Module):
    """
    Frozen FastWAM wrapper for World2WAM.

    - encode_obs_latent: single-frame VAE encode -> pooled z_t [B, latent_dim]
    - encode_text: return precomputed context [B,L,D] or live T5 encode
    - infer_action_only: official FastWAM action-only path (no auxiliary heads)
    """

    FROZEN_MODULES = ("video_expert", "action_expert", "vae", "text_encoder")

    def __init__(
        self,
        fastwam_root: str | Path,
        checkpoint: str | Path,
        task_config: str = "libero_uncond_2cam224_1e-4",
        freeze: bool = True,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        latent_dim: int = 48,
        load_text_encoder: bool = False,
    ):
        super().__init__()
        self.fastwam_root = Path(fastwam_root).resolve()
        self.latent_dim = int(latent_dim)
        self.freeze_backbone = bool(freeze)
        self.device_str = device

        add_fastwam_path(self.fastwam_root)
        os.chdir(self.fastwam_root)

        self.model, self._hydra_cfg = self._load_model(
            task_config, mixed_precision, device, load_text_encoder=load_text_encoder
        )
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = self.fastwam_root / ckpt_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"FastWAM checkpoint not found: {ckpt_path}")
        self.model.load_checkpoint(str(ckpt_path))
        self._loaded_checkpoint = str(ckpt_path.resolve())

        if self.freeze_backbone:
            self._freeze()

        self.model.eval()
        logger.info("FastWAMEncoder loaded ckpt=%s freeze=%s", self._loaded_checkpoint, freeze)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FastWAMEncoder":
        fw = cfg.get("fastwam", {})
        return cls(
            fastwam_root=fw.get("root") or cfg["fastwam_root"],
            checkpoint=fw.get("checkpoint") or cfg["official_fastwam_checkpoint"],
            task_config=fw.get("task_config") or cfg.get("fastwam_task_config", "libero_uncond_2cam224_1e-4"),
            freeze=bool(fw.get("freeze", True)),
            device=fw.get("device") or cfg.get("device", "cuda"),
            mixed_precision=fw.get("mixed_precision", "bf16"),
            latent_dim=int(cfg.get("latent_dim") or cfg.get("model", {}).get("latent_dim", 48)),
            load_text_encoder=bool(fw.get("load_text_encoder", False)),
        )

    def _load_model(
        self,
        task_name: str,
        mixed_precision: str,
        device: str,
        *,
        load_text_encoder: bool = False,
    ):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        from fastwam.runtime import _mixed_precision_to_model_dtype

        config_dir = self.fastwam_root / "configs"
        GlobalHydra.instance().clear()
        overrides = [f"task={task_name}"]
        if load_text_encoder:
            overrides.append("model.load_text_encoder=true")
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            hydra_cfg = compose(config_name="train", overrides=overrides)

        model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
        model = instantiate(hydra_cfg.model, model_dtype=model_dtype, device=device)

        for attr in ("vae", "infer_action", "_encode_input_image_latents_tensor"):
            if not hasattr(model, attr):
                raise AttributeError(f"Loaded model missing `{attr}`. Expected real FastWAM.")

        return model, hydra_cfg

    def _freeze(self) -> None:
        for name in self.FROZEN_MODULES:
            mod = getattr(self.model, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def action_dim(self) -> int:
        return int(self.model.action_expert.action_dim)

    @torch.no_grad()
    def encode_obs_latent(self, obs: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        """
        Encode single observation frame to pooled VAE latent.

        Args:
            obs: [B,3,H,W] or [3,H,W]
        Returns:
            z: [B, latent_dim]
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[1] != 3:
            raise ValueError(f"obs must have channel dim 3, got {tuple(obs.shape)}")

        x = obs.to(device=self.model.device, dtype=self.model.torch_dtype)
        z = self.model._encode_input_image_latents_tensor(x, tiled=tiled)
        pooled = pool_vae_latent(z)
        if pooled.shape[-1] != self.latent_dim:
            logger.warning(
                "latent_dim mismatch: pooled=%d config=%d",
                pooled.shape[-1],
                self.latent_dim,
            )
        return pooled

    @torch.no_grad()
    def encode_text_from_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return text embedding tensor as stored in cache [B,L,D] or [L,D]."""
        if context.dim() == 2:
            context = context.unsqueeze(0)
        return context.float()

    @torch.no_grad()
    def encode_text_from_prompt(self, prompt: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self.model, "encode_prompt"):
            raise AttributeError("Model has no encode_prompt for live T5 encoding.")
        emb, mask = self.model.encode_prompt(prompt)
        return emb.float(), mask

    @torch.no_grad()
    def infer_action_only(self, batch: dict[str, Any]) -> torch.Tensor:
        """Official FastWAM action-only inference — no auxiliary heads."""
        self.model.eval()
        video = batch.get("video")
        if video is None:
            # Do NOT use `or` on Tensors (Boolean value ambiguous).
            obs = batch.get("obs")
            if obs is None:
                obs = batch.get("obs_t")
            if obs is None:
                raise ValueError("infer_action_only requires video or obs/obs_t.")
            video = obs

        if video.dim() == 5:
            frame = video[:, 0]
        elif video.dim() == 4:
            frame = video
        elif video.dim() == 3:
            frame = video.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected video/obs shape: {tuple(video.shape)}")

        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        prompt = batch.get("prompt")
        if prompt is None:
            prompt = batch.get("instruction")
        if prompt is None:
            prompt = batch.get("language")
        if isinstance(prompt, list):
            prompt = prompt[0]

        context = batch.get("context")
        context_mask = batch.get("context_mask")
        proprio = batch.get("proprio")
        if proprio is None:
            proprio = batch.get("state_t")
        action_horizon = batch.get("action_horizon")
        if action_horizon is None:
            ac = batch.get("action_chunk")
            if ac is None:
                ac = batch.get("action")
            action_horizon = int(ac.shape[-2]) if ac is not None and ac.dim() >= 2 else 8

        infer_kwargs: dict[str, Any] = {
            "input_image": frame.to(device=self.model.device, dtype=self.model.torch_dtype),
            "action_horizon": int(action_horizon),
            "num_inference_steps": batch.get("num_inference_steps", 20),
        }
        if context is not None and context_mask is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = context.to(device=self.model.device)
            infer_kwargs["context_mask"] = context_mask.to(device=self.model.device)
        else:
            if not prompt:
                raise ValueError("infer_action_only needs context/context_mask or prompt.")
            infer_kwargs["prompt"] = str(prompt)
            infer_kwargs["context"] = None
            infer_kwargs["context_mask"] = None

        if proprio is not None:
            p = proprio
            if p.dim() == 3:
                p = p[:, 0]
            infer_kwargs["proprio"] = p.to(device=self.model.device)

        out = self.model.infer_action(**infer_kwargs)
        action = out.get("action") if isinstance(out, dict) else out
        return action


def load_frozen_fastwam(cfg: dict[str, Any]) -> FastWAMEncoder:
    return FastWAMEncoder.from_config(cfg)
