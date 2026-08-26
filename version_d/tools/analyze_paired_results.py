#!/usr/bin/env python3
"""Recompute fair R0-vs-Version-D comparisons using identical task sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_methods(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries = payload.get("summaries")
    if isinstance(summaries, dict):
        values = []
        for method_name, summary in summaries.items():
            item = dict(summary)
            item.setdefault("method", method_name)
            values.append(item)
    elif isinstance(summaries, list):
        values = summaries
    else:
        raise ValueError("expected a 'summaries' mapping or list")

    def name(item: dict[str, Any]) -> str:
        return str(item.get("method", item.get("label", ""))).lower()

    r0 = next((item for item in values if name(item).startswith("r0")), None)
    version_d = next(
        (
            item
            for item in values
            if name(item).startswith("r3") or "versiond" in name(item) or "version_d" in name(item)
        ),
        None,
    )
    if r0 is None or version_d is None:
        raise ValueError("could not identify both R0 and Version D summaries")
    return r0, version_d


def task_rates(summary: dict[str, Any], phase: str) -> dict[str, float | None]:
    key = f"{phase}_success_rate"
    return {
        str(row["task_name"]): None if row.get(key) is None else float(row[key])
        for row in summary.get("per_task", [])
    }


def analyze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    r0, version_d = load_methods(payload)
    episodes = int(payload.get("episodes", r0.get("episodes", 0)) or 0)
    result: dict[str, Any] = {
        "source": str(path),
        "protocol": payload.get("protocol"),
        "episodes_per_task": episodes,
        "phases": {},
    }
    for phase in ("clean", "random"):
        r0_rates = task_rates(r0, phase)
        vd_rates = task_rates(version_d, phase)
        all_tasks = sorted(set(r0_rates) | set(vd_rates))
        matched = [
            task
            for task in all_tasks
            if r0_rates.get(task) is not None and vd_rates.get(task) is not None
        ]
        r0_mean = sum(float(r0_rates[t]) for t in matched) / len(matched) if matched else None
        vd_mean = sum(float(vd_rates[t]) for t in matched) / len(matched) if matched else None
        result["phases"][phase] = {
            "matched_tasks": matched,
            "missing_tasks": [task for task in all_tasks if task not in matched],
            "matched_task_count": len(matched),
            "complete": len(matched) == len(all_tasks),
            "r0_success_rate": r0_mean,
            "version_d_success_rate": vd_mean,
            "delta_percentage_points": (
                None if r0_mean is None or vd_mean is None else 100.0 * (vd_mean - r0_mean)
            ),
            "matched_episodes_per_method": len(matched) * episodes if episodes else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.summary)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
