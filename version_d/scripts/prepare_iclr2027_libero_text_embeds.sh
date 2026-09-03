#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTWAM=/DATA/disk0/yjh/libero_work_wj/code/FastWAM_official
PY=/DATA/disk0/yjh/robotwin_w2wam/env/bin/python
DATA=/DATA/disk0/yjh/libero_work_wj/data/libero_mujoco3.3.2
MARKER="$ROOT/status/libero_text_embeds.complete.json"
CACHE="$FASTWAM/data/text_embeds_cache/libero"

datasets=(
  "$DATA/libero_10_no_noops_lerobot"
  "$DATA/libero_goal_no_noops_lerobot"
  "$DATA/libero_object_no_noops_lerobot"
  "$DATA/libero_spatial_no_noops_lerobot"
)

for dataset in "${datasets[@]}"; do
  test -s "$dataset/meta/tasks.jsonl"
done

dataset_override="data.train.dataset_dirs=[$(IFS=,; echo "${datasets[*]}")]"
cd "$FASTWAM"
"$PY" scripts/precompute_text_embeds.py \
  task=libero_uncond_2cam224_1e-4 \
  "$dataset_override" \
  +overwrite=false

mkdir -p "$(dirname "$MARKER")"
"$PY" - "$MARKER" "$CACHE" "${datasets[@]}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

marker = pathlib.Path(sys.argv[1])
cache = pathlib.Path(sys.argv[2])
datasets = [pathlib.Path(value) for value in sys.argv[3:]]
prompts = set()
for dataset in datasets:
    for line in (dataset / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            prompts.add(json.loads(line)["task"])
cache_files = list(cache.glob("*.t5_len128.wan22ti2v5b.pt"))
if len(cache_files) < len(prompts):
    raise RuntimeError(
        f"text cache incomplete: files={len(cache_files)} prompts={len(prompts)}"
    )
marker.write_text(
    json.dumps(
        {
            "complete": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "prompt_count": len(prompts),
            "cache_file_count": len(cache_files),
        },
        indent=2,
    )
    + "\n"
)
PY
