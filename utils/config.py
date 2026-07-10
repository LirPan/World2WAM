from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import fastwam_checkpoint, fastwam_dataset_stats, fastwam_root, project_root, resolve_path, workspace_root


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path, workspace_root())
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty config: {path}")
    return normalize_config(raw)


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    ws = workspace_root()

    fw = dict(out.get("fastwam", {}))
    if fw.get("root"):
        fw["root"] = str(resolve_path(fw["root"], ws))
    if fw.get("checkpoint"):
        root = Path(fw["root"]) if fw.get("root") else ws
        ckpt = Path(fw["checkpoint"])
        fw["checkpoint"] = str(ckpt if ckpt.is_absolute() else (root / ckpt).resolve())
    if fw.get("dataset_stats"):
        root = Path(fw["root"]) if fw.get("root") else ws
        stats = Path(fw["dataset_stats"])
        fw["dataset_stats"] = str(stats if stats.is_absolute() else (root / stats).resolve())
    out["fastwam"] = fw

    lib = dict(out.get("libero", {}))
    if lib.get("root"):
        lib["root"] = str(resolve_path(lib["root"], ws))
    if lib.get("dataset_dir"):
        fw_root = Path(fw["root"]) if fw.get("root") else ws
        ds = Path(lib["dataset_dir"])
        lib["dataset_dir"] = str(ds if ds.is_absolute() else (fw_root / ds).resolve())
    out["libero"] = lib

    cache = dict(out.get("cache", {}))
    if cache.get("output_dir"):
        cache["output_dir"] = str(resolve_path(cache["output_dir"], ws))
    out["cache"] = cache

    train = dict(out.get("train", {}))
    if train.get("output_dir"):
        train["output_dir"] = str(resolve_path(train["output_dir"], ws))
    out["train"] = train

    out["fastwam_root"] = str(fastwam_root(out))
    out["official_fastwam_checkpoint"] = str(fastwam_checkpoint(out))
    out["dataset_stats_path"] = str(fastwam_dataset_stats(out))
    out["fastwam_task_config"] = fw.get("task_config", "libero_uncond_2cam224_1e-4")
    out["lerobot_dataset_dirs"] = [str(resolve_path(lib["dataset_dir"], ws))] if lib.get("dataset_dir") else []
    out["horizon"] = int(out.get("horizon", 10))
    out["suite"] = out.get("suite", "libero_spatial")

    model = out.get("model", {})
    out["latent_dim"] = int(model.get("latent_dim", 48))
    out["text_dim"] = int(model.get("text_dim", 512))
    out["action_dim"] = int(model.get("action_dim", 7))

    return out


def save_config_copy(cfg: dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return path
