#!/usr/bin/env python3
"""Run FastWAM's official RoboTwin evaluator task-by-task on one physical GPU."""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def read_tasks(config_path: Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for RoboTwin evaluation") from exc
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if isinstance(config, dict):
        # Official RoboTwin _eval_step_limit.yml maps task name -> step limit.
        if config and all(isinstance(key, str) for key in config):
            return list(config.keys())
        for key in ("tasks", "task_names", "task_name"):
            value = config.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
    raise RuntimeError(f"Could not find task list in {config_path}")


def parse_success(result_file: Path) -> float | None:
    if not result_file.exists():
        return None
    numbers = re.findall(r"(?:success(?:_rate)?|Success(?: Rate)?)?[^0-9]*([0-9]+(?:\.[0-9]+)?)", result_file.read_text(errors="replace"))
    if not numbers:
        numbers = re.findall(r"([0-9]+(?:\.[0-9]+)?)", result_file.read_text(errors="replace"))
    if not numbers:
        return None
    value = float(numbers[-1])
    return value / 100 if value > 1 else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastwam-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Evaluate only the first N tasks (for a quick smoke result).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names. Overrides --max-tasks when provided.",
    )
    parser.add_argument(
        "--phase",
        choices=("both", "clean", "random"),
        default="both",
        help="Run both benchmark phases or only one phase.",
    )
    args = parser.parse_args()

    fastwam = Path(args.fastwam_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    task_config = fastwam / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
    available_tasks = read_tasks(task_config)
    tasks = available_tasks
    if args.tasks:
        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        if not tasks:
            raise SystemExit("--tasks must contain at least one task name")
        unknown = [task for task in tasks if task not in available_tasks]
        if unknown:
            raise SystemExit(f"Unknown task(s): {', '.join(unknown)}")
        if len(set(tasks)) != len(tasks):
            raise SystemExit("--tasks must not contain duplicates")
    elif args.max_tasks is not None:
        if args.max_tasks <= 0:
            raise SystemExit("--max-tasks must be positive")
        tasks = tasks[: args.max_tasks]
    evaluator = fastwam / "experiments" / "robotwin" / "eval_robotwin_single.py"
    result_root = fastwam / "evaluate_results" / "robotwin" / Path(args.checkpoint).stem / args.label
    rows = []

    phase_specs = (("clean", "demo_clean"), ("random", "demo_randomized"))
    if args.phase != "both":
        phase_specs = tuple(spec for spec in phase_specs if spec[0] == args.phase)
    for phase, config_name in phase_specs:
        for task in tasks:
            log_path = output / f"{task}_{phase}.log"
            command = [
                sys.executable, str(evaluator), f"ckpt={Path(args.checkpoint).resolve()}",
                f"gpu_id={args.gpu_id}", f"EVALUATION.task_name={task}",
                f"EVALUATION.task_config={config_name}", f"EVALUATION.output_dir={args.label}",
                f"EVALUATION.dataset_stats_path={Path(args.dataset_stats).resolve()}",
                f"EVALUATION.eval_num_episodes={args.episodes}",
            ]
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=fastwam, stdout=log, stderr=subprocess.STDOUT, check=False)
            result_file = result_root / task / ("_result_clean.txt" if phase == "clean" else "_result_random.txt")
            rows.append({
                "task": task, "phase": phase, "returncode": completed.returncode,
                "result_file": str(result_file), "success": parse_success(result_file),
                "log": str(log_path),
            })

    aggregate = {}
    for phase in ("clean", "random"):
        values = [row["success"] for row in rows if row["phase"] == phase and row["success"] is not None]
        aggregate[phase] = sum(values) / len(values) if values else None
    aggregate["rand"] = aggregate.pop("random")
    aggregate["avg"] = (aggregate["clean"] + aggregate["rand"]) / 2 if all(aggregate[phase] is not None for phase in ("clean", "rand")) else None
    summary = {
        "label": args.label, "checkpoint": str(Path(args.checkpoint).resolve()), "gpu_id": args.gpu_id,
        "episodes": args.episodes, "phase": args.phase, "tasks": tasks, "aggregate": aggregate, "rows": rows,
        "hard_metrics": None,
        "hard_metrics_note": "Not populated: this evaluator has no verified official hard-task partition.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "phase", "returncode", "success", "result_file", "log"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate))


if __name__ == "__main__":
    main()
