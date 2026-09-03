#!/usr/bin/env python3
"""Reset scheduler states after an external dependency has been repaired."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reset-retry", action="append", default=[])
    parser.add_argument("--reset-stale-running", action="store_true")
    parser.add_argument(
        "--adopt",
        action="append",
        default=[],
        metavar="JOB_ID:PID:GPU",
        help="record a live job launched by another scheduler",
    )
    args = parser.parse_args()

    status_root = args.run_root / "status" / "jobs"
    retry_ids = set(args.reset_retry)
    changed: list[str] = []
    for path in sorted(status_root.glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        dependency_failure = (
            state.get("status") == "failed" and state.get("reason") == "dependency_failed"
        )
        selected_retry = path.stem in retry_ids and state.get("status") == "retry_wait"
        stale_running = False
        if args.reset_stale_running and state.get("status") == "running":
            try:
                os.kill(int(state.get("pid", -1)), 0)
            except (ProcessLookupError, ValueError):
                stale_running = True
            except PermissionError:
                stale_running = False
        if not dependency_failure and not selected_retry and not stale_running:
            continue
        atomic_write(
            path,
            {
                "id": path.stem,
                "status": "pending",
                "attempts": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "recovered_after_dependency_repair": True,
            },
        )
        changed.append(path.stem)

    adopted: list[str] = []
    for value in args.adopt:
        job_id, pid_raw, gpu_raw = value.split(":", 2)
        pid = int(pid_raw)
        os.kill(pid, 0)
        path = status_root / f"{job_id}.json"
        if not path.is_file():
            raise SystemExit(f"unknown job id: {job_id}")
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update(
            {
                "id": job_id,
                "status": "running",
                "pid": pid,
                "gpu": int(gpu_raw),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "adopted_external_process": True,
            }
        )
        atomic_write(path, state)
        adopted.append(job_id)

    print(
        json.dumps(
            {
                "reset_count": len(changed),
                "reset_jobs": changed,
                "adopt_count": len(adopted),
                "adopted_jobs": adopted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
