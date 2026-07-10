#!/usr/bin/env python3
"""Inspect physics pseudo-label distribution, confidence, and feature stats."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from minimal_world2wam.data.latent_cache_dataset import LatentCacheDataset, collate_latent_batch, list_cache_files
from minimal_world2wam.physics.physics_labels import PHYSICS_PHASES, batch_infer_physics_labels_v1
from minimal_world2wam.physics.phase_labeler import TeacherPhysicsLabeler, extract_phase_features


def _feature_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "p25": float(torch.quantile(t, 0.25).item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p75": float(torch.quantile(t, 0.75).item()),
    }


def _check_warnings(phase_counts: dict[str, int], confidence_mean: float) -> list[str]:
    warnings: list[str] = []
    total = max(sum(phase_counts.values()), 1)
    nonzero = sum(1 for c in phase_counts.values() if c > 0)
    ratios = {k: v / total for k, v in phase_counts.items()}
    max_ratio = max(ratios.values()) if ratios else 1.0
    uncertain_ratio = ratios.get("uncertain", 0.0)

    if nonzero < 4:
        warnings.append(f"Only {nonzero} phases are non-zero (need >= 4).")
    if max_ratio >= 0.70:
        top = max(ratios, key=ratios.get)
        warnings.append(f"Max phase ratio {max_ratio:.2%} for '{top}' (need < 70%).")
    if uncertain_ratio >= 0.35:
        warnings.append(f"Uncertain ratio {uncertain_ratio:.2%} (need < 35%).")
    if confidence_mean <= 0.2:
        warnings.append(f"confidence_mean={confidence_mean:.3f} (need > 0.2).")
    return warnings


def inspect_cache(
    cache_dir: Path,
    max_samples: int,
    phase_label_version: str,
    physics_cfg: dict,
    seed: int = 42,
) -> dict:
    files = list_cache_files(cache_dir)
    rng = random.Random(seed)
    chosen = rng.sample(files, min(max_samples, len(files)))

    phase_counter: Counter[str] = Counter()
    conf_by_phase: dict[str, list[float]] = {p: [] for p in PHYSICS_PHASES}
    feature_accum: dict[str, list[float]] = {
        "motion_mag": [],
        "latent_delta": [],
        "horizontal_motion": [],
        "vertical_motion": [],
        "gripper_abs_change": [],
    }

    labeler_cfg = dict(physics_cfg)
    labeler_cfg.setdefault("auto_threshold", physics_cfg.get("auto_threshold", True))
    labeler = TeacherPhysicsLabeler(cfg=labeler_cfg)

    for fp in chosen:
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        sample = {
            "z_t": payload["z_t"].float().unsqueeze(0),
            "z_tH": payload["z_tH"].float().unsqueeze(0),
            "text_embed": payload["text_embed"].float().unsqueeze(0),
            "action_chunk": payload["action_chunk"].float().unsqueeze(0),
            "metadata": [{"task_id": payload.get("task_id")}],
        }
        if "state_t" in payload:
            sample["state_t"] = payload["state_t"].float().unsqueeze(0)
        if "state_tH" in payload:
            sample["state_tH"] = payload["state_tH"].float().unsqueeze(0)

        if phase_label_version in ("v1", "1"):
            out = labeler.label_batch(sample)
        else:
            from minimal_world2wam.physics.physics_labels import infer_physics_phase_from_action

            lid, _ = infer_physics_phase_from_action(sample["action_chunk"][0])
            out = {
                "phase_id": torch.tensor([lid]),
                "confidence": torch.tensor([0.5]),
                "phase_name": [PHYSICS_PHASES[lid]],
            }

        phase = out["phase_name"][0]
        conf = float(out["confidence"][0].item())
        phase_counter[phase] += 1
        conf_by_phase[phase].append(conf)

        feats = extract_phase_features(sample)
        for k in feature_accum:
            feature_accum[k].append(float(feats[k][0].item()))

    total = max(sum(phase_counter.values()), 1)
    phase_counts = {p: phase_counter.get(p, 0) for p in PHYSICS_PHASES}
    phase_ratios = {p: phase_counts[p] / total for p in PHYSICS_PHASES}
    confidence_all = [c for vals in conf_by_phase.values() for c in vals]
    confidence_mean = mean(confidence_all) if confidence_all else 0.0
    confidence_by_phase = {
        p: mean(conf_by_phase[p]) if conf_by_phase[p] else 0.0 for p in PHYSICS_PHASES
    }
    feature_stats = {k: _feature_stats(v) for k, v in feature_accum.items()}
    warnings = _check_warnings(phase_counts, confidence_mean)

    return {
        "num_samples": len(chosen),
        "phase_label_version": phase_label_version,
        "phase_counts": phase_counts,
        "phase_ratios": phase_ratios,
        "confidence_mean": confidence_mean,
        "confidence_by_phase": confidence_by_phase,
        "feature_stats": feature_stats,
        "warnings": warnings,
        "cache_dir": str(cache_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--phase_label_version", default="v1")
    parser.add_argument("--output_json", default="experiments/physics_label_inspect_v1.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (WORKSPACE / cache_dir).resolve()

    result = inspect_cache(
        cache_dir,
        args.max_samples,
        args.phase_label_version,
        physics_cfg={"auto_threshold": True, "phase_confidence_threshold": 0.3},
        seed=args.seed,
    )

    out = Path(args.output_json)
    if not out.is_absolute():
        out = (WORKSPACE / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    if result["warnings"]:
        print("\nWARNINGS:")
        for w in result["warnings"]:
            print(f"  - {w}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
