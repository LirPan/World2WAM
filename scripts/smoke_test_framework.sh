#!/usr/bin/env bash
# Layered smoke tests (no full Wan download required for tier 0-1).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "=== Tier 0: pure Python modules (no FastWAM load) ==="
python - <<'PY'
import torch
from src.models.future_latent_head import FutureLatentHead
from src.losses.world2wam_losses import compute_future_latent_loss, FUTURE_LATENT_MISSING_MSG
from src.data.future_latent_cache import FutureLatentCache
import tempfile
from pathlib import Path

h = FutureLatentHead(1024, 7, 48)
out = h(torch.randn(2, 1024), torch.randn(2, 7))
assert out.shape == (2, 48)
cache = FutureLatentCache(Path(tempfile.mkdtemp()))
cache.save_future_latent(0, 0, 1, torch.randn(48))
assert cache.has_future_latent(0, 0, 1)
print("Tier 0 OK")
PY

echo "=== Tier 1: FastWAM import path ==="
python - <<'PY'
from src.utils.config import load_config
from src.utils.import_utils import add_fastwam_path

cfg = load_config("configs/fastwam_future_distill.yaml")
add_fastwam_path(cfg["fastwam_root"])
import fastwam
from fastwam.runtime import create_fastwam
print("fastwam import OK", fastwam.__file__)
PY

echo "=== Tier 2: dataset (needs LeRobot LIBERO on disk) ==="
python - <<'PY'
import sys
from src.utils.config import load_config
from src.data.libero_dataset_adapter import build_fastwam_dataset, LiberoDatasetAdapter

cfg = load_config("configs/fastwam_future_distill.yaml")
try:
    ds, _ = build_fastwam_dataset(cfg)
    ad = LiberoDatasetAdapter(ds, future_horizon=1, cache=None)
    s = ad[0]
    print("dataset OK", "video", tuple(s["video"].shape), "action", tuple(s["action"].shape))
except Exception as e:
    print("SKIP dataset:", e)
    print("  -> Download LIBERO LeRobot data per FastWAM README / HuggingFace yuanty/LIBERO-fastwam")
PY

echo "=== Tier 3: FastWAM model load (needs Wan + ActionDiT checkpoints) ==="
python - <<'PY'
import sys
from src.utils.config import load_config

cfg = load_config("configs/fastwam_future_distill.yaml")
try:
    from src.wrappers.fastwam_wrapper import FastWAMWrapper
    w = FastWAMWrapper(
        fastwam_root=cfg["fastwam_root"],
        freeze_backbone=True,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    print("FastWAMWrapper OK", "hidden_dim", w.hidden_dim, "action_dim", w.action_dim)
except Exception as e:
    print("SKIP model load:", e)
    print("  -> Run FastWAM model prep: ActionDiT backbone + Wan2.2 weights in checkpoints/")
PY

echo "=== smoke_test_framework.sh finished ==="
