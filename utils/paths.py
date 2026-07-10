from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return project_root().parent


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    root = base or workspace_root()
    return (root / p).resolve()


def fastwam_root(cfg: dict) -> Path:
    fw = cfg.get("fastwam", {})
    root = fw.get("root") or cfg.get("fastwam_root")
    if not root:
        raise KeyError("Config missing fastwam.root")
    return resolve_path(root)


def fastwam_checkpoint(cfg: dict) -> Path:
    fw = cfg.get("fastwam", {})
    ckpt = fw.get("checkpoint") or cfg.get("official_fastwam_checkpoint")
    if not ckpt:
        raise KeyError("Config missing fastwam.checkpoint")
    root = fastwam_root(cfg)
    p = Path(ckpt)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def fastwam_dataset_stats(cfg: dict) -> Path:
    fw = cfg.get("fastwam", {})
    stats = fw.get("dataset_stats") or cfg.get("dataset_stats_path")
    if not stats:
        raise KeyError("Config missing fastwam.dataset_stats")
    root = fastwam_root(cfg)
    p = Path(stats)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def libero_dataset_dir(cfg: dict) -> Path:
    lib = cfg.get("libero", {})
    ds = lib.get("dataset_dir") or cfg.get("lerobot_dataset_dirs", [None])[0]
    if not ds:
        raise KeyError("Config missing libero.dataset_dir")
    fw_root = fastwam_root(cfg)
    p = Path(ds)
    if p.is_absolute():
        return p.resolve()
    return (fw_root / p).resolve()
