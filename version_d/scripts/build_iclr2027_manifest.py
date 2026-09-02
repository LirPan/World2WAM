#!/usr/bin/env python3
"""Build the frozen New_yjh LIBERO/LIBERO-Plus paper-sprint queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ROOT = Path("/DATA/disk0/yjh/world2wam_iclr2027")


def job(
    job_id: str,
    command: list[str],
    *,
    resource: str = "gpu",
    depends_on: list[str] | None = None,
    expected: list[dict] | None = None,
    stage: str,
) -> dict:
    return {
        "id": job_id,
        "stage": stage,
        "resource": resource,
        "depends_on": depends_on or [],
        "command": command,
        "expected": expected or [],
        "max_attempts": 3,
    }


def eval_job(
    root: Path,
    *,
    job_id: str,
    method: str,
    benchmark: str,
    checkpoint: Path,
    stats: Path,
    sample_ratio: str,
    task_config: str,
    depends_on: list[str],
    stage: str,
) -> dict:
    output = root / "results" / benchmark / job_id
    return job(
        job_id,
        [
            "bash",
            str(root / "deploy" / "scripts" / "run_iclr2027_libero_eval.sh"),
            method,
            benchmark,
            str(checkpoint),
            str(stats),
            str(output),
            sample_ratio,
            task_config,
        ],
        depends_on=depends_on,
        expected=[
            {
                "path": str(output / "completion.json"),
                "type": "json",
                "required_keys": ["complete", "result_files"],
            }
        ],
        stage=stage,
    )


def build(root: Path, protocol_path: Path) -> dict:
    protocol_raw = protocol_path.read_bytes()
    protocol_hash = hashlib.sha256(protocol_raw).hexdigest()
    scripts = root / "deploy" / "scripts"
    status = root / "status"
    fast_ckpt = Path(
        "/DATA/disk0/yjh/libero_work_wj/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
    )
    fast_stats = Path(
        "/DATA/disk0/yjh/libero_work_wj/checkpoints/fastwam_release/dataset_stats.json"
    )
    faster_ckpt = root / "third_party/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt"
    faster_stats = root / "third_party/FasterWAM/checkpoints/fasterwam_release/libero/dataset_stats.json"
    current_d = Path(
        "/DATA/disk0/yjh/libero_work_wj/runs/libero_version_d_spatial/exported/version_d_libero.pt"
    )
    fast_robotwin_ckpt = Path(
        "/DATA/disk0/yjh/robotwin_w2wam/third_party/FastWAM_official/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
    )
    fast_robotwin_stats = Path(
        "/DATA/disk0/yjh/robotwin_w2wam/third_party/FastWAM_official/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
    )
    faster_robotwin_ckpt = root / "third_party/FasterWAM/checkpoints/fasterwam_release/robotwin/step_029355.pt"
    faster_robotwin_stats = root / "third_party/FasterWAM/checkpoints/fasterwam_release/robotwin/dataset_stats.json"

    jobs: list[dict] = []
    jobs.append(
        job(
            "bootstrap",
            ["bash", str(scripts / "bootstrap_iclr2027_new_yjh.sh")],
            resource="cpu",
            stage="P0",
            expected=[
                {
                    "path": str(status / "bootstrap.complete.json"),
                    "type": "json",
                    "required_keys": ["complete", "fasterwam_commit", "libero_plus_commit"],
                }
            ],
        )
    )
    jobs.append(
        job(
            "bootstrap_robotwin",
            ["bash", str(scripts / "bootstrap_iclr2027_robotwin_new_yjh.sh")],
            resource="gpu",
            depends_on=["bootstrap"],
            stage="P0",
            expected=[
                {
                    "path": str(status / "robotwin_bootstrap.complete.json"),
                    "type": "json",
                    "required_keys": ["complete"],
                }
            ],
        )
    )
    jobs.append(
        job(
            "download_libero_training_data",
            ["bash", str(scripts / "download_iclr2027_libero_data.sh")],
            resource="cpu",
            depends_on=["bootstrap"],
            stage="P0",
            expected=[
                {
                    "path": str(status / "libero_training_data.complete.json"),
                    "type": "json",
                    "required_keys": ["complete"],
                }
            ],
        )
    )

    exploratory = [
        ("plus15_fastwam", "FastWAM", fast_ckpt, fast_stats, "libero_fastwam_2cam224_1e-4"),
        (
            "plus15_version_d_spatial_exploratory",
            "VersionD_spatial_exploratory",
            current_d,
            fast_stats,
            "libero_fastwam_2cam224_1e-4",
        ),
        (
            "plus15_fasterwam",
            "FasterWAM",
            faster_ckpt,
            faster_stats,
            "libero_fasterwam_2cam224_1e-4",
        ),
    ]
    for job_id, method, checkpoint, stats_path, task_config in exploratory:
        jobs.append(
            eval_job(
                root,
                job_id=job_id,
                method=method,
                benchmark="libero_plus",
                checkpoint=checkpoint,
                stats=stats_path,
                sample_ratio="0.15",
                task_config=task_config,
                depends_on=["bootstrap"],
                stage="P1-smoke",
            )
        )

    cache_ids = []
    for shard in range(4):
        job_id = f"libero_cache_shard_{shard}of4"
        cache_ids.append(job_id)
        manifest = (
            root
            / "cache/libero_all_suites/world2wam_iclr2027_libero_all"
            / f"manifest_shard{shard}of4.json"
        )
        jobs.append(
            job(
                job_id,
                ["bash", str(scripts / "run_iclr2027_libero_cache_shard.sh"), str(shard), "4"],
                depends_on=["download_libero_training_data"],
                stage="P0-cache",
                expected=[
                    {
                        "path": str(manifest),
                        "type": "json",
                        "required_keys": ["complete", "counts", "records"],
                    }
                ],
            )
        )

    train_ids: dict[tuple[str, int], str] = {}
    for method in ("B1", "B2", "B3", "B4", "B5"):
        for seed in (42, 43, 44):
            job_id = f"train_{method}_s{seed}"
            train_ids[(method, seed)] = job_id
            dependencies = ["download_libero_training_data"] if method == "B1" else cache_ids
            checkpoint = root / "checkpoints/libero" / f"{method}_s{seed}.pt"
            jobs.append(
                job(
                    job_id,
                    ["bash", str(scripts / "run_iclr2027_libero_train.sh"), method, str(seed)],
                    depends_on=dependencies,
                    stage="P2-train",
                    expected=[{"path": str(checkpoint), "type": "file"}],
                )
            )

    standard_ids = []
    for job_id, method, checkpoint, stats_path, task_config in (
        ("libero_full_fastwam", "FastWAM", fast_ckpt, fast_stats, "libero_fastwam_2cam224_1e-4"),
        (
            "libero_full_fasterwam",
            "FasterWAM",
            faster_ckpt,
            faster_stats,
            "libero_fasterwam_2cam224_1e-4",
        ),
    ):
        standard_ids.append(job_id)
        jobs.append(
            eval_job(
                root,
                job_id=job_id,
                method=method,
                benchmark="libero",
                checkpoint=checkpoint,
                stats=stats_path,
                sample_ratio="null",
                task_config=task_config,
                depends_on=["bootstrap"],
                stage="P3-standard",
            )
        )

    for method in ("B1", "B2", "B3", "B4", "B5"):
        for seed in (42, 43, 44):
            job_id = f"libero_full_{method}_s{seed}"
            standard_ids.append(job_id)
            checkpoint = root / "checkpoints/libero" / f"{method}_s{seed}.pt"
            jobs.append(
                eval_job(
                    root,
                    job_id=job_id,
                    method=f"{method}_s{seed}",
                    benchmark="libero",
                    checkpoint=checkpoint,
                    stats=fast_stats,
                    sample_ratio="null",
                    task_config="libero_fastwam_2cam224_1e-4",
                    depends_on=[train_ids[(method, seed)]],
                    stage="P3-standard",
                )
            )

    plus15_formal_ids = []
    for method in ("B1", "B2", "B3", "B4"):
        seed = 42
        job_id = f"plus15_{method}_s{seed}"
        plus15_formal_ids.append(job_id)
        jobs.append(
            eval_job(
                root,
                job_id=job_id,
                method=f"{method}_s{seed}",
                benchmark="libero_plus",
                checkpoint=root / "checkpoints/libero" / f"{method}_s{seed}.pt",
                stats=fast_stats,
                sample_ratio="0.15",
                task_config="libero_fastwam_2cam224_1e-4",
                depends_on=[train_ids[(method, seed)]],
                stage="P1-formal",
            )
        )
    for seed in (42, 43, 44):
        job_id = f"plus15_B5_s{seed}"
        plus15_formal_ids.append(job_id)
        jobs.append(
            eval_job(
                root,
                job_id=job_id,
                method=f"B5_s{seed}",
                benchmark="libero_plus",
                checkpoint=root / "checkpoints/libero" / f"B5_s{seed}.pt",
                stats=fast_stats,
                sample_ratio="0.15",
                task_config="libero_fastwam_2cam224_1e-4",
                depends_on=[train_ids[("B5", seed)]],
                stage="P1-formal",
            )
        )

    full_gate = ["plus15_fastwam", "plus15_fasterwam", *plus15_formal_ids]
    for job_id, method, checkpoint, stats_path, task_config, extra_dep in (
        (
            "plus_full_fastwam",
            "FastWAM",
            fast_ckpt,
            fast_stats,
            "libero_fastwam_2cam224_1e-4",
            [],
        ),
        (
            "plus_full_fasterwam",
            "FasterWAM",
            faster_ckpt,
            faster_stats,
            "libero_fasterwam_2cam224_1e-4",
            [],
        ),
        *[
            (
                f"plus_full_B5_s{seed}",
                f"B5_s{seed}",
                root / "checkpoints/libero" / f"B5_s{seed}.pt",
                fast_stats,
                "libero_fastwam_2cam224_1e-4",
                [train_ids[("B5", seed)]],
            )
            for seed in (42, 43, 44)
        ],
    ):
        jobs.append(
            eval_job(
                root,
                job_id=job_id,
                method=method,
                benchmark="libero_plus",
                checkpoint=checkpoint,
                stats=stats_path,
                sample_ratio="null",
                task_config=task_config,
                depends_on=[*full_gate, *extra_dep],
                stage="P4-plus-full",
            )
        )

    robotwin_train_ids: dict[tuple[str, int], str] = {}
    for method in ("B1", "B2", "B3", "B4", "B5", "B5_no_hard"):
        for seed in (42, 43, 44):
            job_id = f"robotwin_train_{method}_s{seed}"
            robotwin_train_ids[(method, seed)] = job_id
            checkpoint = root / "checkpoints/robotwin" / f"{method}_s{seed}.pt"
            jobs.append(
                job(
                    job_id,
                    ["bash", str(scripts / "run_iclr2027_robotwin_train.sh"), method, str(seed)],
                    depends_on=["bootstrap_robotwin"],
                    stage="P2-robotwin-train",
                    expected=[{"path": str(checkpoint), "type": "file"}],
                )
            )

    robotwin_eval_specs = [
        (
            "robotwin_full_fastwam",
            "FastWAM",
            fast_robotwin_ckpt,
            fast_robotwin_stats,
            "robotwin_fastwam_3cam_384_1e-4",
            ["bootstrap_robotwin"],
        ),
        (
            "robotwin_full_fasterwam",
            "FasterWAM",
            faster_robotwin_ckpt,
            faster_robotwin_stats,
            "robotwin_fasterwam_3cam_384_1e-4",
            ["bootstrap_robotwin"],
        ),
    ]
    for method in ("B1", "B2", "B3", "B4", "B5", "B5_no_hard"):
        for seed in (42, 43, 44):
            robotwin_eval_specs.append(
                (
                    f"robotwin_full_{method}_s{seed}",
                    f"{method}_s{seed}",
                    root / "checkpoints/robotwin" / f"{method}_s{seed}.pt",
                    fast_robotwin_stats,
                    "robotwin_fastwam_3cam_384_1e-4",
                    [robotwin_train_ids[(method, seed)]],
                )
            )
    for job_id, method, checkpoint, stats_path, task_config, dependencies in robotwin_eval_specs:
        output = root / "results/robotwin" / job_id
        jobs.append(
            job(
                job_id,
                [
                    "bash",
                    str(scripts / "run_iclr2027_robotwin_eval.sh"),
                    method,
                    str(checkpoint),
                    str(stats_path),
                    str(output),
                    task_config,
                    "10",
                ],
                depends_on=dependencies,
                stage="P3-robotwin-full",
                expected=[
                    {
                        "path": str(output / "completion.json"),
                        "type": "json",
                        "required_keys": ["complete"],
                    },
                    {"path": str(output / "summary.json"), "type": "json"},
                ],
            )
        )

    protocol = json.loads(protocol_raw)
    gates = protocol["quality_gates"]
    return {
        "schema_version": 1,
        "run_id": "world2wam_iclr2027_20260901",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_hash,
        "allowed_gpus": protocol["allowed_gpus"],
        "minimum_free_disk_gb": gates["minimum_free_disk_gb"],
        "gpu_idle_memory_mb": gates["gpu_idle_memory_mb"],
        "gpu_idle_utilization_percent": gates["gpu_idle_utilization_percent"],
        "gpu_idle_consecutive_checks": gates["gpu_idle_consecutive_checks"],
        "gpu_idle_check_interval_seconds": 2,
        "retry_delays_seconds": gates["retry_delays_seconds"],
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "protocols/iclr2027_paper_sprint.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
