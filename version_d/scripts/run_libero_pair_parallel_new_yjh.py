#!/usr/bin/env python3
"""Resume-safe paired LIBERO-Spatial evaluation on opportunistically idle GPUs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    label: str
    task_id: int


@dataclass
class Running:
    job: Job
    process: subprocess.Popen
    log_handle: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--idle-memory-mb", type=int, default=1000)
    parser.add_argument("--idle-util", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fastwam-root", type=Path, default=Path("/DATA/disk0/yjh/libero_work_wj")
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=Path("/DATA/disk0/yjh/robotwin_w2wam/latest/code/policy_lora"),
    )
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=Path(
            "/DATA/disk0/yjh/world2wam/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO_fresh"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/DATA/disk0/yjh/libero_work_wj/env/libero_venv/bin/python"),
    )
    parser.add_argument(
        "--version-d-checkpoint",
        type=Path,
        default=Path(
            "/DATA/disk0/yjh/libero_work_wj/runs/libero_version_d_spatial/exported/version_d_libero.pt"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/DATA/disk0/yjh/libero_work_wj/runs/libero_version_d_spatial_eval_n10"),
    )
    return parser.parse_args()


def gpu_stats() -> dict[int, tuple[int, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(command, text=True).splitlines()
    result = {}
    for row in rows:
        gpu, memory, util = (int(item.strip()) for item in row.split(","))
        result[gpu] = (memory, util)
    return result


def result_path(run_root: Path, job: Job) -> Path:
    return run_root / job.label / "libero_spatial" / f"gpu0_task{job.task_id}_results.json"


def valid_result(path: Path, expected_trials: int) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    successes = int(payload.get("successes", -1))
    total = int(payload.get("total_episodes", -1))
    return total == expected_trials and 0 <= successes <= total


def checkpoint(args: argparse.Namespace, label: str) -> Path:
    if label == "official":
        return args.fastwam_root / "checkpoints/fastwam_release/libero_uncond_2cam224.pt"
    return args.version_d_checkpoint


def build_command(args: argparse.Namespace, job: Job) -> list[str]:
    evaluator = args.fastwam_root / "experiments/libero/eval_libero_single.py"
    stats = args.fastwam_root / "checkpoints/fastwam_release/dataset_stats.json"
    output = args.run_root / job.label
    return [
        "xvfb-run",
        "-a",
        str(args.python),
        str(evaluator),
        f"ckpt={checkpoint(args, job.label)}",
        f"seed={args.seed}",
        "gpu_id=0",
        "EVALUATION.device=cuda:0",
        "EVALUATION.task_suite_name=libero_spatial",
        f"EVALUATION.task_id={job.task_id}",
        f"EVALUATION.num_trials={args.num_trials}",
        f"EVALUATION.dataset_stats_path={stats}",
        f"EVALUATION.output_dir={output}",
    ]


def ordered_jobs(task_ids: list[int]) -> list[Job]:
    priority = [task for task in (0, 1, 2) if task in task_ids]
    remaining = [task for task in task_ids if task not in priority]
    return [Job(label, task) for task in priority + remaining for label in ("official", "version_d")]


def write_state(args: argparse.Namespace, pending: list[Job], running: dict[int, Running], failures: dict[Job, int]) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "suite": "libero_spatial",
            "task_ids": args.task_ids,
            "num_trials": args.num_trials,
            "seed": args.seed,
            "paired": True,
        },
        "pending": [job.__dict__ for job in pending],
        "running": {str(gpu): item.job.__dict__ for gpu, item in running.items()},
        "failures": [{**job.__dict__, "attempts": count} for job, count in failures.items()],
    }
    (args.run_root / "dispatcher_state.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    allowed_gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    args.run_root.mkdir(parents=True, exist_ok=True)
    lock_handle = open("/tmp/world2wam_libero_pair_parallel.lock", "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("another World2WAM LIBERO dispatcher is already active") from exc

    required = [
        args.python,
        args.version_d_checkpoint,
        args.fastwam_root / "checkpoints/fastwam_release/libero_uncond_2cam224.pt",
        args.fastwam_root / "checkpoints/fastwam_release/dataset_stats.json",
        args.fastwam_root / "experiments/libero/eval_libero_single.py",
        args.policy_root / "src/eval/summarize_libero_pair.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing required paths: {missing}")

    jobs = ordered_jobs(args.task_ids)
    pending = [job for job in jobs if not valid_result(result_path(args.run_root, job), args.num_trials)]
    attempts = {job: 0 for job in jobs}
    failures: dict[Job, int] = {}
    running: dict[int, Running] = {}
    env_base = os.environ.copy()
    pythonpath = [args.policy_root, args.libero_root, args.fastwam_root, args.fastwam_root / "src"]
    env_base["PYTHONPATH"] = ":".join(map(str, pythonpath)) + ":" + env_base.get("PYTHONPATH", "")
    env_base["MUJOCO_GL"] = env_base.get("MUJOCO_GL", "egl")

    manifest = {
        "jobs": [job.__dict__ for job in jobs],
        "commands": [{**job.__dict__, "command": build_command(args, job)} for job in jobs],
    }
    (args.run_root / "dispatcher_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    while pending or running:
        for gpu, item in list(running.items()):
            return_code = item.process.poll()
            if return_code is None:
                continue
            item.log_handle.close()
            del running[gpu]
            if return_code == 0 and valid_result(result_path(args.run_root, item.job), args.num_trials):
                print(f"complete gpu={gpu} job={item.job}", flush=True)
                continue
            attempts[item.job] += 1
            if attempts[item.job] < args.max_attempts:
                pending.append(item.job)
                print(f"retry job={item.job} attempt={attempts[item.job] + 1}", flush=True)
            else:
                failures[item.job] = attempts[item.job]
                print(f"failed job={item.job} attempts={attempts[item.job]}", flush=True)

        stats = gpu_stats()
        idle = [
            gpu
            for gpu in allowed_gpus
            if gpu not in running
            and gpu in stats
            and stats[gpu][0] <= args.idle_memory_mb
            and stats[gpu][1] <= args.idle_util
        ]
        for gpu in idle:
            if not pending:
                break
            job = pending.pop(0)
            if valid_result(result_path(args.run_root, job), args.num_trials):
                continue
            log_path = args.run_root / "logs" / f"{job.label}_task{job.task_id}_gpu{gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a")
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                build_command(args, job),
                cwd=args.fastwam_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = Running(job, process, log_handle)
            print(f"launch gpu={gpu} pid={process.pid} job={job}", flush=True)

        write_state(args, pending, running, failures)
        if pending or running:
            time.sleep(args.poll_seconds)

    if failures:
        raise SystemExit(f"LIBERO paired evaluation incomplete: {failures}")

    summarizer = args.policy_root / "src/eval/summarize_libero_pair.py"
    subprocess.run(
        [
            str(args.python),
            str(summarizer),
            "--official-dir",
            str(args.run_root / "official"),
            "--version-d-dir",
            str(args.run_root / "version_d"),
            "--task-ids",
            *(str(task) for task in args.task_ids),
            "--output",
            str(args.run_root / "libero_pair_summary.json"),
        ],
        check=True,
        env=env_base,
    )


if __name__ == "__main__":
    main()
