from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .checkpoint_utils import normalize_config
from .path_utils import minimal_project_root, resolve_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path, minimal_project_root())
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return normalize_config(cfg)
