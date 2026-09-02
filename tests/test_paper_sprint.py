"""Small, dependency-free tests for the ICLR 2027 experiment scheduler."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


paper_sprint = _load("paper_sprint", ROOT / "version_d/scripts/paper_sprint.py")
manifest_builder = _load(
    "build_iclr2027_manifest",
    ROOT / "version_d/scripts/build_iclr2027_manifest.py",
)


def test_frozen_manifest_has_expected_jobs():
    protocol = ROOT / "version_d/protocols/iclr2027_paper_sprint.json"
    payload = manifest_builder.build(Path("/tmp/world2wam-test"), protocol)
    job_ids = [job["id"] for job in payload["jobs"]]
    assert len(job_ids) == 92
    assert len(set(job_ids)) == 92
    assert payload["allowed_gpus"] == [1, 2, 3, 4, 5, 6, 7]
    assert "plus_full_B5_s44" in job_ids
    assert "robotwin_full_B5_no_hard_s44" in job_ids


def test_validate_expected_rejects_missing_or_empty(tmp_path: Path):
    result = tmp_path / "result.json"
    expected = [{"path": str(result), "type": "json", "required_keys": ["complete"]}]
    assert not paper_sprint.validate_expected(expected)
    result.write_text("{}\n")
    assert not paper_sprint.validate_expected(expected)
    result.write_text(json.dumps({"complete": True}) + "\n")
    assert paper_sprint.validate_expected(expected)


def test_scheduler_recovers_completed_artifact(tmp_path: Path):
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "test",
                "allowed_gpus": [1],
                "jobs": [
                    {
                        "id": "train",
                        "resource": "gpu",
                        "depends_on": [],
                        "command": ["true"],
                        "expected": [{"path": str(artifact), "type": "file"}],
                    }
                ],
            }
        )
        + "\n"
    )
    scheduler = paper_sprint.Scheduler(manifest, tmp_path / "run")
    scheduler.reconcile_artifacts()
    state = scheduler.state("train")
    assert state["status"] == "complete"
    assert state["recovered_from_artifacts"] is True
