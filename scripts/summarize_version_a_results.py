#!/usr/bin/env python3
"""Aggregate Version A eval JSON into a readable summary with success rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt_rate(obj: dict | None) -> str:
    if not obj:
        return "N/A"
    if "success_rate" in obj:
        sr = float(obj["success_rate"])
        succ = obj.get("successes", "?")
        total = obj.get("total_episodes", "?")
        return f"{sr * 100:.2f}% ({succ}/{total})"
    return "N/A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default="/DATA/disk0/jianhua",
        help="Workspace root containing experiments/",
    )
    parser.add_argument("--output", default=None, help="Write summary JSON path")
    args = parser.parse_args()

    ws = Path(args.workspace)
    exp = ws / "experiments"

    paths = {
        "offline": exp / "eval_offline_physics_flow_dit_main.json",
        "baseline": exp / "eval_baseline_version_a_main.json",
        "ours_physics_flow_dit": exp / "eval_ours_onestep_physics_flow_dit_main.json",
        "ours_flow_dit_no_physics": exp / "eval_ours_onestep_flow_dit_main.json",
        "train_log": exp / "world2wam_physics_flow_dit_main" / "train_log.jsonl",
        "checkpoint": exp / "world2wam_physics_flow_dit_main" / "physics_world2wam_final.pt",
    }

    loaded = {k: _load(v) if k not in ("train_log", "checkpoint") else None for k, v in paths.items()}

    summary = {
        "checkpoint_exists": paths["checkpoint"].is_file(),
        "checkpoint_path": str(paths["checkpoint"]),
        "libero_success_rates": {
            "baseline_fastwam": _fmt_rate(loaded["baseline"]),
            "ours_onestep_physics_flow_dit": _fmt_rate(loaded["ours_physics_flow_dit"]),
            "ours_onestep_flow_dit_no_physics": _fmt_rate(loaded["ours_flow_dit_no_physics"]),
        },
        "offline_metrics": {},
    }

    off = loaded["offline"]
    if off:
        for k in ("mse_fwd", "mse_inv", "mse_cycle", "infer_latency_ms"):
            if k in off:
                summary["offline_metrics"][k] = off[k]
        if "phase_acc_pseudo" in off:
            summary["offline_metrics"]["phase_acc_pseudo"] = off["phase_acc_pseudo"]

    for name, obj in loaded.items():
        if obj and (name.startswith("ours") or name == "baseline"):
            summary.setdefault("raw", {})[name] = {
                    "success_rate": obj.get("success_rate"),
                    "successes": obj.get("successes"),
                    "total_episodes": obj.get("total_episodes"),
                    "average_episode_length": obj.get("average_episode_length"),
                    "inference_latency_ms": obj.get("inference_latency_ms"),
                }

    out_path = Path(args.output) if args.output else exp / "VERSION_A_SUMMARY.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("Version A — LIBERO Success Rates")
    print("=" * 60)
    for label, key in (
        ("Baseline (FastWAM)", "baseline_fastwam"),
        ("Ours onestep + physics + FlowDiT", "ours_onestep_physics_flow_dit"),
        ("Ours onestep FlowDiT (no physics)", "ours_onestep_flow_dit_no_physics"),
    ):
        print(f"  {label}: {summary['libero_success_rates'][key]}")
    if summary["offline_metrics"]:
        print("\nOffline cache metrics:")
        for k, v in summary["offline_metrics"].items():
            print(f"  {k}: {v}")
    print(f"\nCheckpoint: {paths['checkpoint']} ({'OK' if paths['checkpoint'].is_file() else 'MISSING'})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
