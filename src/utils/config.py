from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .path_utils import minimal_project_root, resolve_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path, minimal_project_root())
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    root = minimal_project_root()
    for key in ("fastwam_root", "libero_root", "cache_dir", "output_dir"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = str(resolve_path(cfg[key], root))
    return cfg
