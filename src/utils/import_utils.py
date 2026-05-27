from __future__ import annotations

import sys
from pathlib import Path

from .path_utils import resolve_path


def add_repo_to_path(repo_root: str | Path, src_subdir: str | None = None) -> Path:
    """Insert repository root (and optional src parent) into sys.path if missing."""
    root = resolve_path(repo_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Repository path does not exist: {root}. "
            "Set fastwam_root / libero_root in config."
        )
    candidates = [root]
    if src_subdir:
        src = root / src_subdir
        if src.exists():
            candidates.append(src.parent)
    for p in candidates:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def add_fastwam_path(fastwam_root: str | Path) -> Path:
    """FastWAM imports as `fastwam.*` from repo root (contains src/fastwam)."""
    root = add_repo_to_path(fastwam_root)
    src = root / "src"
    if src.exists():
        add_repo_to_path(src.parent)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    return root


def add_libero_path(libero_root: str | Path) -> Path:
    """LIBERO package lives under libero/ subfolder."""
    root = add_repo_to_path(libero_root)
    pkg = root / "libero"
    if pkg.exists() and str(pkg.parent) not in sys.path:
        sys.path.insert(0, str(pkg.parent))
    return root
