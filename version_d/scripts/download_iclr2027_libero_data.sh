#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTER="$ROOT/third_party/FasterWAM"
DATA=/DATA/disk0/yjh/libero_work_wj/data/libero_mujoco3.3.2
MARKER="$ROOT/status/libero_training_data.complete.json"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$DATA" "$ROOT/status"
if [[ -s "$MARKER" ]]; then
  exit 0
fi

HF_ENDPOINT="$HF_ENDPOINT" "$FASTER/.venv/bin/huggingface-cli" download \
  yuanty/LIBERO-fastwam --repo-type dataset --local-dir "$DATA"

for archive in "$DATA"/*.tar.gz; do
  [[ -f "$archive" ]] || continue
  tar -xzf "$archive" -C "$DATA"
done

required=(
  libero_10_no_noops_lerobot
  libero_goal_no_noops_lerobot
  libero_object_no_noops_lerobot
  libero_spatial_no_noops_lerobot
)
for directory in "${required[@]}"; do
  count="$(find -L "$DATA/$directory" -type f 2>/dev/null | wc -l)"
  if (( count < 10 )); then
    echo "incomplete dataset: $directory files=$count" >&2
    exit 1
  fi
done

python3 - "$MARKER" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"complete": True, "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
PY
