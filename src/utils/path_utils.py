from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def resolve_path(path: PathLike, base: PathLike | None = None) -> Path:
    """Resolve path; if relative, anchor to base (default: minimal_world2wam root)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    if base is None:
        base = Path(__file__).resolve().parents[2]
    return (Path(base) / p).resolve()


def minimal_project_root() -> Path:
    return Path(__file__).resolve().parents[2]
