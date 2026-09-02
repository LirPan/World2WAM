#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTER="$ROOT/third_party/FasterWAM"
MARKER="$ROOT/status/robotwin_bootstrap.complete.json"
if [[ -s "$MARKER" ]]; then
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
bash "$FASTER/scripts/setup/install_robotwin.sh"

[[ -x "$FASTER/.venvs/robotwin/bin/python" ]]
[[ -s "$FASTER/checkpoints/fasterwam_release/robotwin/step_029355.pt" ]]
[[ -s "$FASTER/checkpoints/fasterwam_release/robotwin/dataset_stats.json" ]]
[[ -f /DATA/disk0/yjh/robotwin_w2wam/data/robotwin2.0/robotwin2.0/meta/tasks.jsonl ]]

python3 - "$MARKER" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"complete": True, "completed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
PY
