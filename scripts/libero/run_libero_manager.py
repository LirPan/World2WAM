#!/usr/bin/env python3
"""World2WAM fork of FastWAM LIBERO eval manager — uses minimal parallel launcher."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
_FASTWAM_ROOT = Path(os.environ.get("FASTWAM_ROOT", _MINIMAL_ROOT.parent / "code" / "FastWAM")).resolve()
_PARALLEL_SCRIPT = _MINIMAL_ROOT / "scripts/libero/run_libero_parallel_test.sh"


def create_task_file(
    output_file: Path,
    task_suite_names: list[str],
    task_limit: int | None = None,
) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            if task_limit is not None:
                n_tasks = min(n_tasks, int(task_limit))
            print(f"\n{suite_name}:")
            print(f"- Number of tasks: {n_tasks}")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {total_tasks}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides(overrides: list[str]) -> list[str]:
    return [ov for ov in overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice(cfg: DictConfig, overrides: list[str]) -> str:
    for ov in overrides:
        if ov.startswith("task="):
            return ov.split("=", 1)[1]
    if cfg.get("task") is not None:
        return str(cfg.task)
    return "libero_uncond_2cam224_1e-4"


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
) -> None:
    if not _PARALLEL_SCRIPT.is_file():
        raise FileNotFoundError(f"Evaluation script not found: {_PARALLEL_SCRIPT}")

    root_dir = str(_FASTWAM_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": root_dir,
            "RUN_ID": run_id,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
            "MINIMAL_ROOT": str(_MINIMAL_ROOT),
            "WORLD2WAM_LIBERO_TASK_SCRIPT": str(
                Path(
                    os.environ.get(
                        "WORLD2WAM_LIBERO_TASK_SCRIPT",
                        _MINIMAL_ROOT / "scripts/libero/libero_single_task.sh",
                    )
                )
            ),
            "USE_TMUX": os.environ.get("USE_TMUX", "1"),
        }
    )

    print("\nStarting evaluation (World2WAM LIBERO manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Parallel script: {_PARALLEL_SCRIPT}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"USE_TMUX: {env['USE_TMUX']}")

    try:
        subprocess.run(
            ["bash", str(_PARALLEL_SCRIPT), str(task_file)],
            env=env,
            cwd=root_dir,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}")
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}")
            print(failed_tasks.read_text(encoding="utf-8"))
        raise


def main() -> None:
    overrides = sys.argv[1:]
    config_dir = _FASTWAM_ROOT / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="sim_libero", overrides=overrides)

    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice(cfg, overrides)
    manager = cfg.MULTIRUN

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_limit = os.environ.get("TASK_LIMIT")
    task_limit_int = int(task_limit) if task_limit else None

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
        if not task_file.is_file():
            task_file = create_task_file(task_file, list(manager.task_suite_names), task_limit_int)
    else:
        task_file = output_dir / "tasks.txt"
        task_file = create_task_file(task_file, list(manager.task_suite_names), task_limit_int)

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(overrides),
    )


if __name__ == "__main__":
    main()
