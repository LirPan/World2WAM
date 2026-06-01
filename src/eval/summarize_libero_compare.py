#!/usr/bin/env python3
"""Compare two LIBERO eval output directories (official vs World2WAM)."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def _collect_task_results(output_dir: Path) -> dict[str, dict]:
    task_results: dict[str, dict] = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        suite_dir = output_dir / suite
        if not suite_dir.is_dir():
            continue
        for filename in os.listdir(suite_dir):
            if not filename.startswith("gpu") or not filename.endswith("_results.json"):
                continue
            with open(suite_dir / filename, encoding="utf-8") as f:
                result = json.load(f)
            parts = filename.split("_")
            task_id = int(parts[1].replace("task", ""))
            key = f"{suite}_{task_id}"
            total = int(result["total_episodes"])
            successes = int(result["successes"])
            task_results[key] = {
                "suite": suite,
                "task_id": task_id,
                "success_rate": successes / max(total, 1) * 100.0,
                "successes": successes,
                "total_episodes": total,
                "task_description": result.get("task_description", ""),
            }
    return task_results


def _suite_summary(task_results: dict[str, dict]) -> dict[str, dict]:
    suite_stats: dict[str, dict] = defaultdict(
        lambda: {"tasks": 0, "trials": 0, "successes": 0}
    )
    for row in task_results.values():
        s = row["suite"]
        suite_stats[s]["tasks"] += 1
        suite_stats[s]["trials"] += row["total_episodes"]
        suite_stats[s]["successes"] += row["successes"]
    out: dict[str, dict] = {}
    for suite, st in suite_stats.items():
        rate = st["successes"] / max(st["trials"], 1) * 100.0
        out[suite] = {
            "num_tasks": st["tasks"],
            "total_trials": st["trials"],
            "total_successes": st["successes"],
            "success_rate_pct": rate,
        }
    return out


def compare_libero_dirs(official_dir: Path, world2wam_dir: Path) -> dict:
    off_tasks = _collect_task_results(official_dir)
    w2_tasks = _collect_task_results(world2wam_dir)
    all_keys = sorted(set(off_tasks) | set(w2_tasks))

    per_task = []
    for key in all_keys:
        off = off_tasks.get(key)
        w2 = w2_tasks.get(key)
        off_rate = off["success_rate"] if off else None
        w2_rate = w2["success_rate"] if w2 else None
        delta = None
        if off_rate is not None and w2_rate is not None:
            delta = w2_rate - off_rate
        per_task.append(
            {
                "task_key": key,
                "official_success_rate_pct": off_rate,
                "world2wam_success_rate_pct": w2_rate,
                "delta_pct": delta,
                "task_description": (off or w2 or {}).get("task_description", ""),
            }
        )

    off_suites = _suite_summary(off_tasks)
    w2_suites = _suite_summary(w2_tasks)
    suite_compare = {}
    for suite in sorted(set(off_suites) | set(w2_suites)):
        o = off_suites.get(suite, {})
        w = w2_suites.get(suite, {})
        o_rate = o.get("success_rate_pct")
        w_rate = w.get("success_rate_pct")
        suite_compare[suite] = {
            "official": o,
            "world2wam": w,
            "delta_success_rate_pct": (
                (w_rate - o_rate) if o_rate is not None and w_rate is not None else None
            ),
        }

    def _overall(suites: dict[str, dict]) -> float | None:
        trials = sum(v.get("total_trials", 0) for v in suites.values())
        succ = sum(v.get("total_successes", 0) for v in suites.values())
        if trials == 0:
            return None
        return succ / trials * 100.0

    off_overall = _overall(off_suites)
    w2_overall = _overall(w2_suites)

    return {
        "official_dir": str(official_dir.resolve()),
        "world2wam_dir": str(world2wam_dir.resolve()),
        "overall": {
            "official_success_rate_pct": off_overall,
            "world2wam_success_rate_pct": w2_overall,
            "delta_success_rate_pct": (
                (w2_overall - off_overall)
                if off_overall is not None and w2_overall is not None
                else None
            ),
        },
        "suites": suite_compare,
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=str, required=True)
    parser.add_argument("--world2wam-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    official_dir = Path(args.official_dir)
    world2wam_dir = Path(args.world2wam_dir)
    if not official_dir.is_dir():
        raise FileNotFoundError(official_dir)
    if not world2wam_dir.is_dir():
        raise FileNotFoundError(world2wam_dir)

    summary = compare_libero_dirs(official_dir, world2wam_dir)
    text = json.dumps(summary, indent=2)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
