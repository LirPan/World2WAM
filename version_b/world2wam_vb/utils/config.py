from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from world2wam_vb.utils.import_paths import project_root


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = base or project_root()
    return (base / p).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path, project_root())
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty config: {path}")

    cfg_path = path.parent
    if "defaults" in raw:
        merged: dict[str, Any] = {}
        for entry in raw["defaults"]:
            if isinstance(entry, str):
                base_path = cfg_path / f"{entry}.yaml"
                if base_path.exists():
                    with open(base_path, encoding="utf-8") as bf:
                        base = yaml.safe_load(bf) or {}
                    merged.update(base)
        merged.update({k: v for k, v in raw.items() if k != "defaults"})
        cfg = merged
    else:
        cfg = raw

    return normalize_config(cfg)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    root = project_root()
    for key in (
        "fastwam_root",
        "libero_root",
        "cache_dir",
        "output_dir",
        "official_fastwam_checkpoint",
        "checkpoint_path",
        "dataset_stats_path",
    ):
        if key in out and out[key] is not None:
            out[key] = str(resolve_path(out[key], root))

    official = out.get("official_fastwam_checkpoint") or out.get("checkpoint_path")
    if official:
        out["official_fastwam_checkpoint"] = str(official)
        out.setdefault("checkpoint_path", str(official))

    dirs = out.get("lerobot_dataset_dirs")
    if dirs:
        out["lerobot_dataset_dirs"] = [str(resolve_path(d, root)) for d in dirs]

    return out
