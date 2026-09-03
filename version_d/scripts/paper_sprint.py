#!/usr/bin/env python3
"""Resume-safe World2WAM paper-sprint scheduler.

The scheduler is intentionally benchmark agnostic.  A frozen JSON manifest
contains exact commands, dependencies, GPU policy, and expected artifacts.
Commands receive one physical GPU through ``CUDA_VISIBLE_DEVICES`` and see it
as CUDA device 0.  GPU 0 on New_yjh is excluded by the frozen protocol.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_expected(items: list[dict[str, Any]]) -> bool:
    for item in items:
        path = Path(item["path"])
        if not path.exists():
            return False
        kind = item.get("type", "file")
        if kind == "file" and (not path.is_file() or path.stat().st_size == 0):
            return False
        if kind == "json":
            try:
                payload = load_json(path)
            except (OSError, json.JSONDecodeError):
                return False
            required = item.get("required_keys", [])
            if any(key not in payload for key in required):
                return False
            # Incremental manifests contain the key from their first atomic
            # checkpoint onward. Presence alone is not job completion.
            if "complete" in required and payload.get("complete") is not True:
                return False
    return True


def gpu_snapshot() -> dict[int, tuple[int, int, int]]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    compute_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    process_counts: dict[str, int] = {}
    for row in compute_rows:
        if not row.strip():
            continue
        gpu_uuid = row.split(",", 1)[0].strip()
        process_counts[gpu_uuid] = process_counts.get(gpu_uuid, 0) + 1
    result = {}
    for row in rows:
        gpu_text, gpu_uuid, memory_text, util_text = (value.strip() for value in row.split(","))
        result[int(gpu_text)] = (
            int(memory_text),
            int(util_text),
            process_counts.get(gpu_uuid, 0),
        )
    return result


def confirmed_idle_gpus(
    allowed: list[int],
    *,
    max_memory: int,
    max_util: int,
    checks: int,
    interval: float,
) -> list[int]:
    counts = {gpu: 0 for gpu in allowed}
    for check in range(checks):
        snapshot = gpu_snapshot()
        for gpu in allowed:
            memory, util, process_count = snapshot.get(gpu, (10**9, 100, 1))
            counts[gpu] = (
                counts[gpu] + 1
                if memory <= max_memory and util <= max_util and process_count == 0
                else 0
            )
        if check + 1 < checks:
            time.sleep(interval)
    return [gpu for gpu, count in counts.items() if count == checks]


@dataclass
class Running:
    job: dict[str, Any]
    process: subprocess.Popen[Any]
    log_handle: Any
    gpu: int | None
    gpu_lock: Any | None
    started_at: float


class Scheduler:
    def __init__(self, manifest_path: Path, run_root: Path):
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_json(self.manifest_path)
        self.run_root = run_root.resolve()
        self.status_dir = self.run_root / "status"
        self.log_dir = self.run_root / "logs"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = {job["id"]: job for job in self.manifest["jobs"]}
        self.allowed_gpus = [int(gpu) for gpu in self.manifest["allowed_gpus"]]
        self.retry_delays = list(self.manifest.get("retry_delays_seconds", [60, 300, 900]))
        self.minimum_free_disk_gb = int(self.manifest.get("minimum_free_disk_gb", 200))
        self.max_memory = int(self.manifest.get("gpu_idle_memory_mb", 2000))
        self.max_util = int(self.manifest.get("gpu_idle_utilization_percent", 10))
        self.idle_checks = int(self.manifest.get("gpu_idle_consecutive_checks", 3))
        self.idle_interval = float(self.manifest.get("gpu_idle_check_interval_seconds", 2))
        self.poll_seconds = float(self.manifest.get("scheduler_poll_seconds", 3))
        self.running: dict[str, Running] = {}
        self.next_retry: dict[str, float] = {}
        self._validate_manifest()

    @staticmethod
    def process_group_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.killpg(int(pid), 0)
        except (ProcessLookupError, PermissionError, ValueError):
            return False
        return True

    @staticmethod
    def scheduling_priority(job: dict[str, Any]) -> tuple[int, str]:
        """Run paper-critical prerequisites and B5 before secondary ablations."""
        job_id = str(job["id"])
        if job_id.startswith("libero_cache_shard_"):
            rank = 0
        elif job_id.startswith("train_B5_s"):
            rank = 10
        elif job_id.startswith("train_B1_s"):
            rank = 20
        elif job_id.startswith(("train_B2_s", "train_B3_s", "train_B4_s")):
            rank = 30
        elif job_id.startswith("plus15_B5_s"):
            rank = 40
        elif job_id.startswith("libero_full_B5_s"):
            rank = 50
        else:
            rank = 100
        return rank, job_id

    def reconcile_artifacts(self) -> None:
        """Adopt complete artifacts after supervisor replacement or host reboot."""
        for job_id, job in self.jobs.items():
            state = self.state(job_id)
            if state.get("status") == "complete":
                continue
            expected = job.get("expected", [])
            if expected and validate_expected(expected):
                self.write_state(
                    job_id,
                    status="complete",
                    recovered_from_artifacts=True,
                    return_code=0,
                )
                continue
            if (
                state.get("status") == "running"
                and job_id not in self.running
                and not self.process_group_alive(state.get("pid"))
            ):
                self.write_state(
                    job_id,
                    status="retry_wait",
                    recovered_dead_process=True,
                    reason="recorded process group is no longer alive",
                )

    def _validate_manifest(self) -> None:
        if len(self.jobs) != len(self.manifest["jobs"]):
            raise ValueError("job ids must be unique")
        for job in self.jobs.values():
            unknown = set(job.get("depends_on", [])) - set(self.jobs)
            if unknown:
                raise ValueError(f"job {job['id']} has unknown dependencies: {sorted(unknown)}")
            if job.get("resource", "gpu") not in {"cpu", "gpu"}:
                raise ValueError(f"job {job['id']} has invalid resource")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(job_id: str) -> None:
            if job_id in visiting:
                raise ValueError(f"dependency cycle includes {job_id}")
            if job_id in visited:
                return
            visiting.add(job_id)
            for dependency in self.jobs[job_id].get("depends_on", []):
                visit(dependency)
            visiting.remove(job_id)
            visited.add(job_id)

        for job_id in self.jobs:
            visit(job_id)

    def state_path(self, job_id: str) -> Path:
        return self.status_dir / "jobs" / f"{job_id}.json"

    def state(self, job_id: str) -> dict[str, Any]:
        path = self.state_path(job_id)
        if path.is_file():
            return load_json(path)
        return {"id": job_id, "status": "pending", "attempts": 0}

    def write_state(self, job_id: str, **updates: Any) -> None:
        payload = self.state(job_id)
        payload.update(updates)
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(self.state_path(job_id), payload)

    def completed(self, job_id: str) -> bool:
        state = self.state(job_id)
        return state.get("status") == "complete" and validate_expected(
            self.jobs[job_id].get("expected", [])
        )

    def ready(self, job: dict[str, Any]) -> bool:
        job_id = job["id"]
        state = self.state(job_id)
        # A replacement supervisor may inherit jobs that are still running in
        # their own sessions. Keep those states reserved until the expected
        # artifact appears and reconcile_artifacts() marks them complete.
        # Paused/deferred jobs are deliberately excluded from scheduling. This
        # lets a completed smoke test yield its GPU to formal training without
        # discarding its partial results. Resume explicitly by setting pending.
        if state.get("status") in {
            "complete",
            "failed",
            "running",
            "paused",
            "deferred",
            "paused_after_protocol_smoke",
        }:
            return False
        if job_id in self.running or time.time() < self.next_retry.get(job_id, 0):
            return False
        return all(self.completed(dependency) for dependency in job.get("depends_on", []))

    def replace_tokens(self, values: list[str], gpu: int | None) -> list[str]:
        return [value.replace("{gpu}", "" if gpu is None else str(gpu)) for value in values]

    def launch(self, job: dict[str, Any], gpu: int | None = None) -> None:
        gpu_lock = None
        if gpu is not None:
            gpu_lock = open(f"/tmp/world2wam_iclr2027_gpu{gpu}.lock", "w", encoding="utf-8")
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close()
                return
        command = self.replace_tokens(list(job["command"]), gpu)
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in job.get("env", {}).items()})
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["WORLD2WAM_PHYSICAL_GPU"] = str(gpu)
        log_path = self.log_dir / f"{job['id']}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=job.get("workdir") or None,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        attempts = int(self.state(job["id"]).get("attempts", 0)) + 1
        self.write_state(
            job["id"],
            status="running",
            attempts=attempts,
            pid=process.pid,
            gpu=gpu,
            command=command,
            log=str(log_path),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        self.running[job["id"]] = Running(
            job=job,
            process=process,
            log_handle=log_handle,
            gpu=gpu,
            gpu_lock=gpu_lock,
            started_at=time.time(),
        )

    def reap(self) -> None:
        for job_id, running in list(self.running.items()):
            return_code = running.process.poll()
            if return_code is None:
                continue
            running.log_handle.close()
            if running.gpu_lock is not None:
                fcntl.flock(running.gpu_lock, fcntl.LOCK_UN)
                running.gpu_lock.close()
            del self.running[job_id]
            state = self.state(job_id)
            attempts = int(state.get("attempts", 1))
            valid = return_code == 0 and validate_expected(running.job.get("expected", []))
            if valid:
                self.write_state(
                    job_id,
                    status="complete",
                    return_code=return_code,
                    duration_seconds=round(time.time() - running.started_at, 3),
                )
                continue
            max_attempts = int(running.job.get("max_attempts", 3))
            if attempts >= max_attempts:
                self.write_state(job_id, status="failed", return_code=return_code)
            else:
                delay = self.retry_delays[min(attempts - 1, len(self.retry_delays) - 1)]
                self.next_retry[job_id] = time.time() + delay
                self.write_state(job_id, status="retry_wait", return_code=return_code, retry_in=delay)

    def heartbeat(self) -> None:
        states = {job_id: self.state(job_id).get("status", "pending") for job_id in self.jobs}
        counts = {status: list(states.values()).count(status) for status in sorted(set(states.values()))}
        running_payload: dict[str, dict[str, Any]] = {}
        for job_id, status in states.items():
            if status != "running":
                continue
            active = self.running.get(job_id)
            running_payload[job_id] = {
                "pid": active.process.pid if active is not None else self.state(job_id).get("pid"),
                "gpu": active.gpu if active is not None else self.state(job_id).get("gpu"),
            }
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "manifest_sha256": sha256_path(self.manifest_path),
            "counts": counts,
            "running": running_payload,
            "jobs": states,
        }
        atomic_json(self.status_dir / "heartbeat.json", payload)

    def run(self) -> None:
        lock_path = self.run_root / "scheduler.lock"
        lock_handle = lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another paper-sprint scheduler is active") from exc

        self.reconcile_artifacts()
        while True:
            self.reap()
            # A separate pre-start supervisor may finish a queued checkpoint
            # while this scheduler is alive. Adopt it before assigning a free
            # GPU so completed work is never repeated.
            self.reconcile_artifacts()
            for job_id, job in self.jobs.items():
                state = self.state(job_id)
                if state.get("status") not in {"pending", "retry_wait"}:
                    continue
                failed_dependencies = [
                    dependency
                    for dependency in job.get("depends_on", [])
                    if self.state(dependency).get("status") == "failed"
                ]
                if failed_dependencies:
                    self.write_state(
                        job_id,
                        status="failed",
                        reason="dependency_failed",
                        failed_dependencies=failed_dependencies,
                    )
            free_gb = shutil.disk_usage(self.run_root).free / (1024**3)
            if free_gb < self.minimum_free_disk_gb:
                atomic_json(
                    self.status_dir / "blocked.json",
                    {"reason": "low_disk", "free_gb": free_gb, "updated_at": time.time()},
                )
                time.sleep(30)
                continue

            cpu_busy = any(item.gpu is None for item in self.running.values())
            if not cpu_busy:
                for job in self.jobs.values():
                    if job.get("resource", "gpu") == "cpu" and self.ready(job):
                        self.launch(job)
                        break

            assigned = {item.gpu for item in self.running.values() if item.gpu is not None}
            # A restarted scheduler does not own Popen objects for inherited
            # process groups. Reserve their recorded GPUs while they are alive.
            for job_id in self.jobs:
                state = self.state(job_id)
                gpu = state.get("gpu")
                if (
                    state.get("status") == "running"
                    and gpu is not None
                    and self.process_group_alive(state.get("pid"))
                ):
                    assigned.add(int(gpu))
            candidate_gpus = [gpu for gpu in self.allowed_gpus if gpu not in assigned]
            if candidate_gpus:
                idle = confirmed_idle_gpus(
                    candidate_gpus,
                    max_memory=self.max_memory,
                    max_util=self.max_util,
                    checks=self.idle_checks,
                    interval=self.idle_interval,
                )
                gpu_jobs = sorted(
                    (
                        job
                        for job in self.jobs.values()
                        if job.get("resource", "gpu") == "gpu" and self.ready(job)
                    ),
                    key=self.scheduling_priority,
                )
                for gpu, job in zip(idle, gpu_jobs):
                    self.launch(job, gpu=gpu)

            self.heartbeat()
            terminal = [self.state(job_id).get("status") for job_id in self.jobs]
            if not self.running and all(status in {"complete", "failed"} for status in terminal):
                break
            time.sleep(self.poll_seconds)

        self.heartbeat()


def freeze_manifest(manifest: Path, run_root: Path) -> None:
    payload = load_json(manifest)
    required = {"run_id", "allowed_gpus", "jobs"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"manifest missing keys: {sorted(missing)}")
    run_root.mkdir(parents=True, exist_ok=True)
    frozen = run_root / "manifest.json"
    if frozen.exists() and sha256_path(frozen) != sha256_path(manifest):
        raise RuntimeError("run manifest is already frozen with different content")
    if not frozen.exists():
        shutil.copy2(manifest, frozen)
    (run_root / "manifest.sha256").write_text(f"{sha256_path(frozen)}  manifest.json\n", encoding="utf-8")
    Scheduler(frozen, run_root).heartbeat()
    print(frozen)


def print_status(manifest: Path, run_root: Path) -> None:
    scheduler = Scheduler(manifest, run_root)
    scheduler.heartbeat()
    print((scheduler.status_dir / "heartbeat.json").read_text(encoding="utf-8"))


def summarize(run_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("*results.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if "successes" not in payload or "total_episodes" not in payload:
            continue
        episodes = int(payload["total_episodes"])
        successes = int(payload["successes"])
        rows.append(
            {
                "path": str(path),
                "benchmark": payload.get("benchmark", ""),
                "suite": payload.get("task_suite", ""),
                "task_id": payload.get("task_id", ""),
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes if episodes else "",
                "duration_seconds": payload.get("duration", ""),
            }
        )
    destination = run_root / "summary_results.csv"
    fields = ["path", "benchmark", "suite", "task_id", "successes", "episodes", "success_rate", "duration_seconds"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "run", "resume", "status", "summarize"])
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "summarize":
        summarize(args.run_root.resolve())
        return
    if args.manifest is None:
        raise SystemExit("--manifest is required")
    manifest = args.manifest.resolve()
    run_root = args.run_root.resolve()
    if args.command == "plan":
        freeze_manifest(manifest, run_root)
        return
    frozen = run_root / "manifest.json"
    active_manifest = frozen if frozen.exists() else manifest
    if args.command in {"run", "resume"}:
        if not frozen.exists():
            freeze_manifest(manifest, run_root)
        Scheduler(active_manifest, run_root).run()
    else:
        print_status(active_manifest, run_root)


if __name__ == "__main__":
    main()
