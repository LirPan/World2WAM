#!/usr/bin/env python3
"""Summarize bidirectional ablation offline head eval JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_run_name(name: str) -> dict[str, str]:
    """Parse experiments/ablations/{mode}_lc{lambda}_h{horizon}/ directory name."""
    out = {"run_name": name, "mode": "", "lambda_cycle": "", "future_horizon": ""}
    if "_lc" in name and "_h" in name:
        mode, rest = name.split("_lc", 1)
        lc, horizon = rest.split("_h", 1)
        out["mode"] = mode
        out["lambda_cycle"] = lc
        out["future_horizon"] = horizon
    return out


def summarize_ablations(ablations_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(ablations_root.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_json = run_dir / "eval" / "offline_head_eval.json"
        if not eval_json.is_file():
            continue
        with open(eval_json, encoding="utf-8") as f:
            metrics = json.load(f)
        meta = _parse_run_name(run_dir.name)
        row = {
            **meta,
            "output_dir": str(run_dir),
            "forward_mse": metrics.get("forward_mse"),
            "inverse_action_mse": metrics.get("inverse_action_mse"),
            "cycle_action_mse": metrics.get("cycle_action_mse"),
            "forward_cosine": metrics.get("forward_cosine"),
            "num_batches": metrics.get("num_batches"),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablations-root",
        type=str,
        default="experiments/ablations",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.ablations_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Ablations root not found: {root}")

    rows = summarize_ablations(root)
    csv_path = Path(args.output_csv or root / "summary.csv")
    json_path = Path(args.output_json or root / "summary.json")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "mode",
        "lambda_cycle",
        "future_horizon",
        "forward_mse",
        "inverse_action_mse",
        "cycle_action_mse",
        "forward_cosine",
        "num_batches",
        "output_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} rows to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
