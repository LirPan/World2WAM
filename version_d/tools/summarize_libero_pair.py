#!/usr/bin/env python3
"""Summarize a fixed, paired LIBERO evaluation without selecting favorable rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(root: Path, task_id: int) -> dict:
    path = root / "libero_spatial" / f"gpu0_task{task_id}_results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text())
    expected = int(result.get("total_episodes", 0))
    successes = int(result.get("successes", 0))
    if expected <= 0 or successes < 0 or successes > expected:
        raise ValueError(f"Invalid result in {path}: successes={successes}, total={expected}")
    return {
        "task_id": task_id,
        "successes": successes,
        "total_episodes": expected,
        "success_rate": successes / expected,
        "failure_episodes": result.get("failure_episodes", []),
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--version-d-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = [load_result(args.official_dir, task_id) for task_id in args.task_ids]
    version_d = [load_result(args.version_d_dir, task_id) for task_id in args.task_ids]
    if [row["task_id"] for row in official] != [row["task_id"] for row in version_d]:
        raise ValueError("Official and Version D task IDs do not match")
    if any(a["total_episodes"] != b["total_episodes"] for a, b in zip(official, version_d)):
        raise ValueError("Official and Version D episode counts do not match")

    official_total = sum(row["total_episodes"] for row in official)
    version_d_total = sum(row["total_episodes"] for row in version_d)
    official_successes = sum(row["successes"] for row in official)
    version_d_successes = sum(row["successes"] for row in version_d)
    paired = sum(
        b["successes"] - a["successes"] for a, b in zip(official, version_d)
    )
    payload = {
        "protocol": {
            "suite": "libero_spatial",
            "task_ids": args.task_ids,
            "paired": True,
            "selection": "none",
        },
        "official": {
            "successes": official_successes,
            "total_episodes": official_total,
            "success_rate": official_successes / official_total,
            "per_task": official,
        },
        "version_d": {
            "successes": version_d_successes,
            "total_episodes": version_d_total,
            "success_rate": version_d_successes / version_d_total,
            "per_task": version_d,
        },
        "delta_successes": paired,
        "delta_success_rate": version_d_successes / version_d_total - official_successes / official_total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
