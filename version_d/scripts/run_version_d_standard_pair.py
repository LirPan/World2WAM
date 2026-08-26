#!/usr/bin/env python3
"""Run a fair standard RoboTwin pair for Version D.

Both R0 and Version D use the same official evaluator and explicitly disable
the optional GraphLite wrapper.  This script is intended for the FiveAges
server, where the corrected 14D B5 checkpoint is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--policy-root", required=True)
    parser.add_argument("--r0-ckpt", required=True)
    parser.add_argument("--version-d-ckpt", required=True)
    args = parser.parse_args()

    policy_root = Path(args.policy_root)
    sys.path.insert(0, str(policy_root))
    from src.eval.run_paired_fixed5 import (  # type: ignore
        detect_host,
        evaluate_method,
        load_tasks,
    )

    tasks = load_tasks(Path(args.tasks_json))
    host = detect_host()
    host["policy_root"] = str(policy_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    common = {
        "host": host["alias"],
        "protocol": "official_robotwin_standard_no_graph_lite",
        "episodes": args.episodes,
        "gpu_id": args.gpu_id,
        "tasks": tasks,
        "r0_checkpoint": args.r0_ckpt,
        "version_d_checkpoint": args.version_d_ckpt,
        "graph_lite": False,
    }
    (output_root / "pair_meta.json").write_text(
        json.dumps(common, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    specs = [
        (
            "R0_standard",
            Path(args.r0_ckpt),
            output_root / f"R0_standard_fixed5_n{args.episodes}",
        ),
        (
            "VersionD_B5_standard",
            Path(args.version_d_ckpt),
            output_root / f"VersionD_B5_standard_fixed5_n{args.episodes}",
        ),
    ]
    summaries = []
    for method, checkpoint, out_dir in specs:
        if (out_dir / "summary.json").exists():
            summaries.append(json.loads((out_dir / "summary.json").read_text()))
            continue
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        summaries.append(
            evaluate_method(
                host,
                method=method,
                ckpt=checkpoint,
                tasks=tasks,
                episodes=args.episodes,
                gpu_id=args.gpu_id,
                out_dir=out_dir,
                graph_lite=False,
                dry_run=False,
            )
        )

    result = {**common, "summaries": summaries}
    (output_root / f"pair_summary_n{args.episodes}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
