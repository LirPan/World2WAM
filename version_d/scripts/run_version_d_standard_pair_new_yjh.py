#!/usr/bin/env python3
"""New_yjh-side runner for the same standard Version D pair."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/DATA/disk0/yjh/robotwin_w2wam")
FASTWAM = ROOT / "third_party/FastWAM_official"
PYTHON = ROOT / "env/bin/python"
STATS = FASTWAM / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
R0 = FASTWAM / "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
B5 = ROOT / "checkpoints/version_d_transfer/ablation14_B5_fic_project14.pt"
TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
)
PHASES = (("clean", "demo_clean"), ("random", "demo_randomized"))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_rate(path: Path) -> float | None:
    if not path.exists():
        return None
    value = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = float(line.strip())
        except ValueError:
            pass
    return value


def run_method(name: str, ckpt: Path, out_root: Path, episodes: int, gpu: int) -> dict:
    run_name = f"{name}_fixed5_n{episodes}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "driver.log").open("a", encoding="utf-8")
    rows = []
    for task in TASKS:
        row = {"task_name": task, "clean_success_rate": None, "random_success_rate": None}
        for phase, task_config in PHASES:
            cmd = [
                str(PYTHON),
                str(FASTWAM / "experiments/robotwin/eval_robotwin_single.py"),
                f"ckpt={ckpt}",
                f"gpu_id={gpu}",
                f"EVALUATION.task_name={task}",
                f"EVALUATION.task_config={task_config}",
                f"EVALUATION.eval_num_episodes={episodes}",
                f"EVALUATION.dataset_stats_path={STATS}",
                # The official evaluator derives its result namespace from the
                # basename of output_dir. Keep it unique per method/run so
                # concurrent evaluations cannot overwrite one another.
                f"EVALUATION.output_dir={out_dir / 'official_hydra' / run_name}",
            ]
            log.write(f"\n[{now()}] CMD {' '.join(cmd)}\n")
            log.flush()
            env = os.environ.copy()
            env["WORLD2WAM_GRAPH_LITE"] = "0"
            proc = subprocess.run(cmd, cwd=str(FASTWAM), env=env, stdout=log, stderr=subprocess.STDOUT)
            tag = ckpt.stem
            result = FASTWAM / "evaluate_results/robotwin" / tag / run_name / task / f"_result_{phase}.txt"
            key = f"{phase}_success_rate"
            row[key] = parse_rate(result)
            row[f"{phase}_return_code"] = proc.returncode
        rows.append(row)
    log.close()
    valid_clean = [r["clean_success_rate"] for r in rows if r["clean_success_rate"] is not None]
    valid_random = [r["random_success_rate"] for r in rows if r["random_success_rate"] is not None]
    summary = {
        "label": "validation",
        "method": name,
        "checkpoint": str(ckpt),
        "graph_lite": False,
        "episodes": episodes,
        "tasks": len(TASKS),
        "clean": sum(valid_clean) / len(valid_clean) if valid_clean else None,
        "random": sum(valid_random) / len(valid_random) if valid_random else None,
        "per_task": rows,
        "created_at": now(),
        "host": os.uname().nodename,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-root", default=str(ROOT / "runs/version_d_standard_pair_n3"))
    parser.add_argument("--version-d-ckpt", default=str(B5))
    args = parser.parse_args()
    version_d_ckpt = Path(args.version_d_ckpt)
    if not R0.exists() or not version_d_ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {R0} or {version_d_ckpt}")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": "official_robotwin_standard_no_graph_lite",
        "episodes": args.episodes,
        "gpu_id": args.gpu_id,
        "tasks": list(TASKS),
        "summaries": [
            run_method("R0_standard", R0, out_root, args.episodes, args.gpu_id),
            run_method("VersionD_standard", version_d_ckpt, out_root, args.episodes, args.gpu_id),
        ],
    }
    (out_root / f"pair_summary_n{args.episodes}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
