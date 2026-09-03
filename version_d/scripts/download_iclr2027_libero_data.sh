#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTER="$ROOT/third_party/FasterWAM"
DATA=/DATA/disk0/yjh/libero_work_wj/data/libero_mujoco3.3.2
MARKER="$ROOT/status/libero_training_data.complete.json"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

required=(
  libero_10_no_noops_lerobot
  libero_goal_no_noops_lerobot
  libero_object_no_noops_lerobot
  libero_spatial_no_noops_lerobot
)

mkdir -p "$DATA" "$ROOT/status"

dataset_ready() {
  local directory
  for directory in "${required[@]}"; do
    [[ -s "$DATA/$directory/meta/tasks.jsonl" ]] || return 1
  done
  return 0
}

extract_available_archives() {
  local directory archive target backup
  for directory in "${required[@]}"; do
    [[ -s "$DATA/$directory/meta/tasks.jsonl" ]] && continue
    archive="$DATA/$directory.tar.gz"
    [[ -s "$archive" ]] || continue
    target="$DATA/$directory"
    if [[ -L "$target" ]]; then
      unlink "$target"
    elif [[ -d "$target" ]]; then
      if rmdir "$target" 2>/dev/null; then
        :
      else
        backup="$target.incompatible.$(date +%Y%m%d%H%M%S)"
        mv "$target" "$backup"
      fi
    fi
    tar -xzf "$archive" -C "$DATA"
  done
}

extract_available_archives
if ! dataset_ready; then
  HF_ENDPOINT="$HF_ENDPOINT" HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=600 \
    "$FASTER/.venv/bin/huggingface-cli" download \
    yuanty/LIBERO-fastwam --repo-type dataset --local-dir "$DATA"
  extract_available_archives
fi

for directory in "${required[@]}"; do
  if [[ ! -s "$DATA/$directory/meta/tasks.jsonl" ]]; then
    echo "incompatible or incomplete dataset: $directory (meta/tasks.jsonl missing)" >&2
    exit 1
  fi
done

python3 - "$MARKER" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"complete": True, "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
PY
