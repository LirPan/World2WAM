from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_fastwam_path(fastwam_root: str | Path) -> Path:
    root = Path(fastwam_root).resolve()
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def add_version_b_to_path() -> None:
    root = project_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
