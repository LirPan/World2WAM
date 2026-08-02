"""Unified policy backend for sim eval (LIBERO now, Robotwin later)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PolicyBackend(ABC):
    """Load a policy once and run action-only inference in sim loops."""

    @abstractmethod
    def load(self, checkpoint_or_bundle: str | Path, **kwargs: Any) -> None:
        """Load official ckpt, merged ckpt, or World2WAM bundle."""

    @abstractmethod
    def infer_action(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return dict with at least `pred_action`. Action-only; no auxiliary heads."""


class FastWAMLiberoBackend(PolicyBackend):
    """
    LIBERO eval backend using FastWAMWrapper.

    Prefer merged FastWAM `.pt` for sim (export via src.tools.export_libero_checkpoint).
    Bundles are also supported via load_world2wam_bundle.
    """

    def __init__(self) -> None:
        self._wrapper = None

    def load(self, checkpoint_or_bundle: str | Path, **kwargs: Any) -> None:
        from ..utils.config import load_config
        from ..utils.path_utils import minimal_project_root, resolve_path
        from ..wrappers.fastwam_wrapper import FastWAMWrapper

        cfg = kwargs.get("config")
        if cfg is None:
            cfg = load_config(resolve_path("configs/world2wam_policy_improve.yaml", minimal_project_root()))
        elif isinstance(cfg, (str, Path)):
            cfg = load_config(resolve_path(str(cfg), minimal_project_root()))

        path = Path(checkpoint_or_bundle)
        backbone_mode = kwargs.get("backbone_mode", "frozen")

        if path.name.startswith("world2wam") or kwargs.get("is_bundle"):
            payload_path = path
            from ..utils.checkpoint_utils import load_world2wam_checkpoint

            payload = load_world2wam_checkpoint(payload_path)
            backbone_mode = str(payload.get("backbone_mode", backbone_mode))
            wrapper = FastWAMWrapper.from_config(cfg, backbone_mode=backbone_mode)
            official = payload.get("official_checkpoint")
            if official:
                wrapper.model.load_checkpoint(str(official))
            wrapper.load_world2wam_bundle(payload_path)
        else:
            wrapper = FastWAMWrapper.from_config(cfg, backbone_mode=backbone_mode)
            wrapper.model.load_checkpoint(str(path))

        self._wrapper = wrapper

    def infer_action(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self._wrapper is None:
            raise RuntimeError("PolicyBackend.load() must be called first.")
        return self._wrapper.forward_action_only(batch)


class FastWAMRobotwinBackend(PolicyBackend):
    """Stub for future Robotwin sim integration."""

    def load(self, checkpoint_or_bundle: str | Path, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Robotwin backend is not implemented yet. "
            "Reuse FastWAMLiberoBackend after Robotwin obs adapter is wired."
        )

    def infer_action(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Robotwin backend is not implemented yet.")
