"""Motion/contact phase pseudo-labeling v1 with confidence scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from minimal_world2wam.physics.physics_labels import PHASE_TO_ID, PHYSICS_PHASES

PHASE_NAMES = PHYSICS_PHASES


def get_optional_tensor(batch: dict[str, Any], candidate_names: list[str]) -> torch.Tensor | None:
    """Return first matching tensor from batch keys."""
    for name in candidate_names:
        val = batch.get(name)
        if isinstance(val, torch.Tensor):
            return val
    return None


def get_task_text_from_batch(batch: dict[str, Any]) -> list[str | None]:
    """Extract raw task text per sample from batch metadata if available."""
    texts: list[str | None] = []
    metadata = batch.get("metadata")
    if not metadata:
        b = batch.get("action_chunk")
        n = int(b.shape[0]) if isinstance(b, torch.Tensor) else 1
        return [None] * n

    if isinstance(metadata, dict):
        metadata = [metadata]

    for meta in metadata:
        text = None
        if isinstance(meta, dict):
            for key in ("task", "task_name", "language", "instruction", "prompt"):
                if meta.get(key):
                    text = str(meta[key])
                    break
        texts.append(text)
    return texts


def _keyword_flags(text: str | None) -> dict[str, bool]:
    if not text:
        return {
            "kw_push_slide": False,
            "kw_grasp": False,
            "kw_place": False,
            "kw_gripper": False,
        }
    t = text.lower()
    return {
        "kw_push_slide": any(k in t for k in ("push", "slide", "press")),
        "kw_grasp": any(k in t for k in ("pick", "grasp", "lift")),
        "kw_place": any(k in t for k in ("place", "put", "insert")),
        "kw_gripper": any(k in t for k in ("open", "close")),
    }


@dataclass
class PhaseThresholds:
    motion_low: float = 0.04
    motion_mid: float = 0.10
    motion_high: float = 0.16
    latent_low: float = 0.15
    latent_mid: float = 0.35
    latent_high: float = 0.55
    grip_low: float = 0.08
    grip_mid: float = 0.20
    horizontal_high: float = 0.10
    vertical_low: float = 0.04
    confidence_threshold: float = 0.3

    @classmethod
    def from_cfg(cls, cfg: dict) -> PhaseThresholds:
        t = cfg.get("thresholds", {})
        return cls(
            motion_low=float(t.get("motion_low", cfg.get("motion_low", 0.04))),
            motion_mid=float(t.get("motion_mid", cfg.get("motion_mid", 0.10))),
            motion_high=float(t.get("motion_high", cfg.get("motion_high", 0.16))),
            latent_low=float(t.get("latent_low", cfg.get("latent_low", 0.15))),
            latent_mid=float(t.get("latent_mid", cfg.get("latent_mid", 0.35))),
            latent_high=float(t.get("latent_high", cfg.get("latent_high", 0.55))),
            grip_low=float(t.get("grip_low", cfg.get("grip_low", 0.08))),
            grip_mid=float(t.get("grip_mid", cfg.get("grip_mid", 0.20))),
            horizontal_high=float(t.get("horizontal_high", cfg.get("horizontal_high", 0.10))),
            vertical_low=float(t.get("vertical_low", cfg.get("vertical_low", 0.04))),
            confidence_threshold=float(
                cfg.get("phase_confidence_threshold", t.get("confidence_threshold", 0.3))
            ),
        )


def extract_phase_features(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Compute batch motion/latent/gripper features."""
    action = batch["action_chunk"]
    z_t = batch["z_t"]
    z_tH = batch.get("z_tH")

    if action.dim() != 3:
        raise ValueError(f"action_chunk must be [B,H,A], got {tuple(action.shape)}")

    b = action.shape[0]
    device = action.device
    dtype = action.dtype

    action_xyz = action[..., :3]
    motion_mag = action_xyz.norm(dim=-1).mean(dim=-1)
    horizontal_motion = action_xyz[..., :2].norm(dim=-1).mean(dim=-1)
    vertical_motion = action_xyz[..., 2].abs().mean(dim=-1)

    gripper_signal = action[..., -1] if action.shape[-1] >= 4 else torch.zeros(b, action.shape[1], device=device, dtype=dtype)
    gripper_delta = gripper_signal[:, -1] - gripper_signal[:, 0]
    gripper_abs_change = gripper_delta.abs()
    gripper_mean = gripper_signal.mean(dim=-1)

    if z_tH is not None:
        latent_delta = (z_tH - z_t).norm(dim=-1)
    else:
        latent_delta = torch.zeros(b, device=device, dtype=dtype)

    state_t = get_optional_tensor(batch, ["state_t", "state", "robot_state_t", "obs_state"])
    state_tH = get_optional_tensor(batch, ["state_tH", "next_state", "future_state", "robot_state_tH"])
    state_delta = torch.zeros(b, device=device, dtype=dtype)
    if state_t is not None and state_tH is not None:
        state_delta = (state_tH - state_t).reshape(b, -1).norm(dim=-1)

    return {
        "motion_mag": motion_mag,
        "horizontal_motion": horizontal_motion,
        "vertical_motion": vertical_motion,
        "gripper_delta": gripper_delta,
        "gripper_abs_change": gripper_abs_change,
        "gripper_mean": gripper_mean,
        "latent_delta": latent_delta,
        "state_delta": state_delta,
    }


def _auto_calibrate_thresholds(features: dict[str, torch.Tensor], th: PhaseThresholds) -> PhaseThresholds:
    """Use batch percentiles to set thresholds when auto_threshold enabled."""
    mm = features["motion_mag"]
    ld = features["latent_delta"]
    gc = features["gripper_abs_change"]
    return PhaseThresholds(
        motion_low=float(torch.quantile(mm, 0.25).item()),
        motion_mid=float(torch.quantile(mm, 0.50).item()),
        motion_high=float(torch.quantile(mm, 0.75).item()),
        latent_low=float(torch.quantile(ld, 0.25).item()),
        latent_mid=float(torch.quantile(ld, 0.50).item()),
        latent_high=float(torch.quantile(ld, 0.75).item()),
        grip_low=float(torch.quantile(gc, 0.25).item()),
        grip_mid=float(torch.quantile(gc, 0.50).item()),
        horizontal_high=float(torch.quantile(features["horizontal_motion"], 0.70).item()),
        vertical_low=float(torch.quantile(features["vertical_motion"], 0.30).item()),
        confidence_threshold=th.confidence_threshold,
    )


@dataclass
class TeacherPhysicsLabeler:
    """Rule-based motion/contact phase labeler with confidence (train-only teacher)."""

    cfg: dict = field(default_factory=dict)
    thresholds: PhaseThresholds | None = None
    gripper_close_sign: str = "auto"
    _calibrated: bool = False

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = PhaseThresholds.from_cfg(self.cfg)
        self.gripper_close_sign = str(self.cfg.get("gripper_close_sign", "auto")).lower()

    def _gripper_closing(self, gripper_delta: float, gripper_mean: float) -> bool:
        if self.gripper_close_sign == "positive":
            return gripper_delta > self.thresholds.grip_mid
        if self.gripper_close_sign == "negative":
            return gripper_delta < -self.thresholds.grip_mid
        return gripper_delta < -self.thresholds.grip_mid or (
            gripper_mean < -0.1 and gripper_delta < -self.thresholds.grip_low
        )

    def _gripper_opening(self, gripper_delta: float, gripper_mean: float) -> bool:
        if self.gripper_close_sign == "positive":
            return gripper_delta < -self.thresholds.grip_mid
        if self.gripper_close_sign == "negative":
            return gripper_delta > self.thresholds.grip_mid
        return gripper_delta > self.thresholds.grip_mid or (
            gripper_mean > 0.1 and gripper_delta > self.thresholds.grip_low
        )

    def _score_rules(
        self,
        feats: dict[str, float],
        kw: dict[str, bool],
    ) -> list[tuple[str, float]]:
        th = self.thresholds
        scores: list[tuple[str, float]] = []

        mm = feats["motion_mag"]
        hm = feats["horizontal_motion"]
        vm = feats["vertical_motion"]
        ld = feats["latent_delta"]
        gac = feats["gripper_abs_change"]
        gd = feats["gripper_delta"]
        gm = feats["gripper_mean"]

        closing = self._gripper_closing(gd, gm)
        opening = self._gripper_opening(gd, gm)

        if kw["kw_push_slide"] and hm >= th.horizontal_high * 0.8:
            scores.append(("push_slide", 0.95))
        if kw["kw_grasp"] and (closing or gac >= th.grip_mid):
            scores.append(("grasp", 0.90))
        if kw["kw_place"] and (opening or ld >= th.latent_mid):
            scores.append(("place", 0.88))

        if mm < th.motion_low and ld < th.latent_low and gac < th.grip_low:
            scores.append(("free_motion", 0.85))
        if closing:
            scores.append(("grasp", 0.80 + min(gac, 0.2)))
        if opening and ld >= th.latent_mid * 0.8:
            scores.append(("place", 0.75 + min(gac, 0.2)))
        if (
            hm >= th.horizontal_high
            and vm <= th.vertical_low
            and ld >= th.latent_mid
            and not closing
        ):
            scores.append(("push_slide", 0.70 + min(hm, 0.2)))
        if mm >= th.motion_high and ld >= th.latent_mid and (closing or gm < -0.05):
            scores.append(("transport", 0.72))
        elif mm >= th.motion_mid and ld >= th.latent_mid and not closing:
            scores.append(("transport", 0.55))
        if th.latent_low <= ld <= th.latent_mid and (mm <= th.motion_mid or gac >= th.grip_low):
            scores.append(("contact", 0.60))
        if mm >= th.motion_mid and ld < th.latent_mid and not closing and not opening:
            scores.append(("approach", 0.58))

        if not scores:
            scores.append(("uncertain", 0.35))
        return scores

    def label_single(
        self,
        feats: dict[str, float],
        kw: dict[str, bool],
    ) -> tuple[int, float, str]:
        scores = self._score_rules(feats, kw)
        scores.sort(key=lambda x: x[1], reverse=True)
        best_phase, best_score = scores[0]
        if len(scores) > 1 and scores[1][1] >= best_score - 0.08:
            best_phase = "uncertain"
            best_score = min(best_score, 0.35)
        confidence = max(0.05, min(1.0, best_score))
        if confidence < self.thresholds.confidence_threshold:
            best_phase = "uncertain"
        return PHASE_TO_ID[best_phase], confidence, best_phase

    def label_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        features = extract_phase_features(batch)
        if self.cfg.get("auto_threshold", False) and not self._calibrated:
            self.thresholds = _auto_calibrate_thresholds(features, self.thresholds)
            self._calibrated = True

        b = features["motion_mag"].shape[0]
        device = features["motion_mag"].device
        phase_ids = []
        confidences = []
        phase_names = []
        task_texts = get_task_text_from_batch(batch)

        for i in range(b):
            feats_i = {k: float(v[i].item()) for k, v in features.items()}
            kw = _keyword_flags(task_texts[i] if i < len(task_texts) else None)
            pid, conf, pname = self.label_single(feats_i, kw)
            phase_ids.append(pid)
            confidences.append(conf)
            phase_names.append(pname)

        return {
            "phase_id": torch.tensor(phase_ids, dtype=torch.long, device=device),
            "confidence": torch.tensor(confidences, dtype=torch.float32, device=device),
            "phase_name": phase_names,
            "phase_features": features,
        }


# Backward-compatible alias (deprecated)
PhysicsPhaseLabeler = TeacherPhysicsLabeler


def get_labeler(version: str = "v1", cfg: dict | None = None) -> TeacherPhysicsLabeler:
    version = str(version).lower().strip()
    if version in ("v1", "1"):
        return TeacherPhysicsLabeler(cfg=cfg or {})
    raise ValueError(f"Unknown phase_label_version: {version}")
