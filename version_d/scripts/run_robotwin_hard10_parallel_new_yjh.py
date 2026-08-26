#!/usr/bin/env python3
"""Parallel, same-protocol RoboTwin hard10 evaluation on New_yjh."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/DATA/disk0/yjh/robotwin_w2wam")
FASTWAM = ROOT / "third_party/FastWAM_official"
PYTHON = ROOT / "env/bin/python"
STATS = FASTWAM / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
CHECKPOINTS = {
    "R0_standard": FASTWAM / "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt",
    "R3_standard": ROOT / "checkpoints/R3_lora_fic_merged.pt",
}
TASKS = (
    "beat_block_hammer",
    "move_stapler_pad",
    "pick_dual_bottles",
    "place_dual_shoes",
    "press_stapler",
    "put_object_cabinet",
    "stack_blocks_three",
    "stack_bowls_three",
    "stamp_seal",
    "turn_switch",
)
PHASES = (("clean", "demo_clean"), ("random", "demo_randomized"))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def result_path(checkpoint: Path, run_name: str, task: str, phase: str) -> Path:
    return FASTWAM / "evaluate_results/robotwin" / checkpoint.stem / run_name / task / f"_result_{phase}.txt"


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


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gpus", default="2,3,4,5,6")
    parser.add_argument("--output-root", default=str(ROOT / "runs/robotwin_hard10_standard_pair_n10"))
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    out_root = Path(args.output_root)
    log_root = out_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for task in TASKS:
        for phase, task_config in PHASES:
            for method, checkpoint in CHECKPOINTS.items():
                run_name = f"{method}_hard10_n{args.episodes}"
                if parse_rate(result_path(checkpoint, run_name, task, phase)) is not None:
                    continue
                jobs.append((method, checkpoint, task, phase, task_config, run_name))

    meta = {
        "protocol": "official_robotwin_standard_no_graph_lite",
        "episodes": args.episodes,
        "gpus": gpus,
        "tasks": list(TASKS),
        "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
        "started_at": now(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    running: dict[int, tuple[subprocess.Popen, object, tuple]] = {}
    free = list(gpus)
    failures = []
    while jobs or running:
        while jobs and free:
            gpu = free.pop(0)
            method, checkpoint, task, phase, task_config, run_name = jobs.pop(0)
            # The official evaluator uses output_dir.name as the global result
            # namespace. A generic "hydra_out" basename causes methods/tasks
            # launched in parallel to collide.
            hydra_out = out_root / method / run_name
            hydra_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(PYTHON),
                str(FASTWAM / "experiments/robotwin/eval_robotwin_single.py"),
                f"ckpt={checkpoint}",
                f"gpu_id={gpu}",
                f"EVALUATION.task_name={task}",
                f"EVALUATION.task_config={task_config}",
                f"EVALUATION.eval_num_episodes={args.episodes}",
                f"EVALUATION.dataset_stats_path={STATS}",
                f"EVALUATION.output_dir={hydra_out}",
            ]
            log_path = log_root / f"{method}_{task}_{phase}.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"[{now()}] GPU={gpu} CMD {' '.join(cmd)}\n")
            handle.flush()
            env = os.environ.copy()
            env["WORLD2WAM_GRAPH_LITE"] = "0"
            proc = subprocess.Popen(cmd, cwd=str(FASTWAM), env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (proc, handle, (method, checkpoint, task, phase, run_name))
        if not running:
            break
        time.sleep(5)
        for gpu, (proc, handle, spec) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle.write(f"[{now()}] return_code={rc}\n")
            handle.close()
            method, checkpoint, task, phase, run_name = spec
            if rc != 0 or parse_rate(result_path(checkpoint, run_name, task, phase)) is None:
                failures.append({"method": method, "task": task, "phase": phase, "return_code": rc})
            del running[gpu]
            free.append(gpu)
            free.sort()

    summaries = {}
    for method, checkpoint in CHECKPOINTS.items():
        run_name = f"{method}_hard10_n{args.episodes}"
        rows = []
        for task in TASKS:
            rows.append({
                "task_name": task,
                "clean_success_rate": parse_rate(result_path(checkpoint, run_name, task, "clean")),
                "random_success_rate": parse_rate(result_path(checkpoint, run_name, task, "random")),
            })
        clean = [r["clean_success_rate"] for r in rows if r["clean_success_rate"] is not None]
        random = [r["random_success_rate"] for r in rows if r["random_success_rate"] is not None]
        summaries[method] = {
            "checkpoint": str(checkpoint),
            "graph_lite": False,
            "clean": sum(clean) / len(clean) if clean else None,
            "random": sum(random) / len(random) if random else None,
            "per_task": rows,
        }

    # Never compare aggregates with different denominators.  Failed simulator
    # jobs previously made the raw random means look favorable even though the
    # two methods had results for different task sets.
    paired = {}
    for phase in ("clean", "random"):
        key = f"{phase}_success_rate"
        r0_rows = {row["task_name"]: row[key] for row in summaries["R0_standard"]["per_task"]}
        r3_rows = {row["task_name"]: row[key] for row in summaries["R3_standard"]["per_task"]}
        matched_tasks = [
            task for task in TASKS
            if r0_rows.get(task) is not None and r3_rows.get(task) is not None
        ]
        r0_mean = (
            sum(float(r0_rows[task]) for task in matched_tasks) / len(matched_tasks)
            if matched_tasks else None
        )
        r3_mean = (
            sum(float(r3_rows[task]) for task in matched_tasks) / len(matched_tasks)
            if matched_tasks else None
        )
        paired[phase] = {
            "matched_tasks": matched_tasks,
            "missing_tasks": [task for task in TASKS if task not in matched_tasks],
            "matched_task_count": len(matched_tasks),
            "episodes_per_task": args.episodes,
            "r0": r0_mean,
            "version_d": r3_mean,
            "delta_pp": None if r0_mean is None or r3_mean is None else 100.0 * (r3_mean - r0_mean),
            "complete": len(matched_tasks) == len(TASKS),
        }
    payload = {
        **meta,
        "finished_at": now(),
        "failures": failures,
        "summaries": summaries,
        "paired_comparison": paired,
    }
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
