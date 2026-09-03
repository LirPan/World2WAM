#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTER="$ROOT/third_party/FasterWAM"
ROBOTWIN="$FASTER/third_party/RoboTwin"
ASSET_SOURCE="${ROBOTWIN_ASSET_SOURCE:-/DATA/disk0/yjh/robotwin_w2wam/third_party/FastWAM_official/third_party/RoboTwin/assets}"
MARKER="$ROOT/status/robotwin_bootstrap.complete.json"
if [[ -s "$MARKER" ]]; then
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"

# The host already has the complete official RoboTwin assets. Reuse them to
# avoid a second multi-GB download and the installer's interactive path prompt.
mkdir -p "$ROBOTWIN/assets"
for asset in background_texture embodiments objects; do
  source_path="$ASSET_SOURCE/$asset"
  target_path="$ROBOTWIN/assets/$asset"
  if [[ -d "$source_path" && ! -e "$target_path" ]]; then
    ln -s "$source_path" "$target_path"
  fi
done

if [[ ! -x "$FASTER/.venvs/robotwin/bin/python" ]]; then
  bash "$FASTER/scripts/setup/install_robotwin.sh" </dev/null
fi

[[ -x "$FASTER/.venvs/robotwin/bin/python" ]]
[[ -d "$ROBOTWIN/assets/embodiments" ]]
[[ -d "$ROBOTWIN/assets/objects" ]]
[[ -s "$FASTER/checkpoints/fasterwam_release/robotwin/step_029355.pt" ]]
[[ -s "$FASTER/checkpoints/fasterwam_release/robotwin/dataset_stats.json" ]]
[[ -f /DATA/disk0/yjh/robotwin_w2wam/data/robotwin2.0/robotwin2.0/meta/tasks.jsonl ]]

python3 - "$MARKER" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"complete": True, "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
PY
