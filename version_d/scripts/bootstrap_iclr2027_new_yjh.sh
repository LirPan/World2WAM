#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTER="$ROOT/third_party/FasterWAM"
MARKER="$ROOT/status/bootstrap.complete.json"
FASTER_REV="83667817df0d4f823f39d90700e61ea2f432ac45"
LIBERO_PLUS_REV="4976dc3"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

mkdir -p "$ROOT"/{status,logs,third_party,checkpoints,results,runs,cache,data}
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
export PATH="$HOME/.local/bin:$PATH"

if [[ -s "$MARKER" ]]; then
  log "bootstrap marker already exists"
  exit 0
fi

if [[ ! -d "$FASTER/.git" ]]; then
  log "cloning FasterWAM"
  git clone https://github.com/hustvl/FasterWAM.git "$FASTER"
fi
git -C "$FASTER" fetch --all --tags
git -C "$FASTER" checkout --detach "$FASTER_REV"
[[ "$(git -C "$FASTER" rev-parse HEAD)" == "$FASTER_REV" ]]

ROBOTWIN_GPU_PATCH="$ROOT/deploy/patches/fasterwam_robotwin_visible_gpu.patch"
if git -C "$FASTER" apply --check "$ROBOTWIN_GPU_PATCH"; then
  log "patching RoboTwin manager to preserve physical CUDA device assignment"
  git -C "$FASTER" apply "$ROBOTWIN_GPU_PATCH"
elif git -C "$FASTER" apply --reverse --check "$ROBOTWIN_GPU_PATCH"; then
  log "RoboTwin physical GPU patch already applied"
else
  echo "RoboTwin GPU patch does not apply cleanly to pinned FasterWAM commit" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv in user site"
  UV_PYPI_INDEX="${UV_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}" \
    python3 -m pip install --user --upgrade --retries 10 --timeout 600 \
      --index-url "$UV_PYPI_INDEX" uv
fi
UV_BIN="$(command -v uv)"
log "uv=$UV_BIN"

log "installing isolated LIBERO-Plus environment"
PATH="$(dirname "$UV_BIN"):$PATH" bash "$FASTER/scripts/setup/install_libero_plus.sh"
[[ "$(git -C "$FASTER/third_party/LIBERO-plus" rev-parse --short=7 HEAD)" == "$LIBERO_PLUS_REV" ]]

log "installing isolated standard LIBERO environment"
PATH="$(dirname "$UV_BIN"):$PATH" bash "$FASTER/scripts/setup/install_libero.sh"

log "installing FasterWAM core environment"
PATH="$(dirname "$UV_BIN"):$PATH" bash "$FASTER/scripts/setup/install_core.sh"

mkdir -p "$FASTER/checkpoints/fasterwam_release"
if [[ ! -s "$FASTER/checkpoints/fasterwam_release/libero/step_021700.pt" ]]; then
  log "downloading FasterWAM release checkpoints"
  HF_ENDPOINT="$HF_ENDPOINT" "$FASTER/.venv/bin/huggingface-cli" download hustvl/FasterWAM \
    --local-dir "$FASTER/checkpoints/fasterwam_release"
fi

mkdir -p "$FASTER/data"
ln -sfn /DATA/disk0/yjh/libero_work_wj/data/libero_mujoco3.3.2 \
  "$FASTER/data/libero_mujoco3.3.2"
ln -sfn /DATA/disk0/yjh/robotwin_w2wam/data/robotwin2.0 \
  "$FASTER/data/robotwin2.0"

python3 - "$MARKER" "$FASTER_REV" "$LIBERO_PLUS_REV" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "complete": True,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "fasterwam_commit": sys.argv[2],
    "libero_plus_commit": sys.argv[3],
}, indent=2) + "\n")
PY
log "bootstrap complete"
