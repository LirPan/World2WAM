#!/usr/bin/env bash
set -euo pipefail

if (( $# != 6 )); then
  echo "usage: $0 METHOD CHECKPOINT STATS OUTPUT TASK_CONFIG EPISODES" >&2
  exit 2
fi
METHOD="$1"
CHECKPOINT="$2"
STATS="$3"
OUTPUT="$4"
TASK_CONFIG="$5"
EPISODES="$6"

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTER="$ROOT/third_party/FasterWAM"
PY="$FASTER/.venvs/robotwin/bin/python"
[[ -x "$PY" ]]
[[ -s "$CHECKPOINT" ]]
[[ -s "$STATS" ]]
mkdir -p "$OUTPUT"

export MPLCONFIGDIR="$ROOT/cache/matplotlib/robotwin"
export NUMBA_CACHE_DIR="$ROOT/cache/numba/robotwin"
export PYTHONPATH="$FASTER/src:$FASTER${PYTHONPATH:+:$PYTHONPATH}"
# Prefer the already validated shared DiffSynth/Wan cache used by the
# official FastWAM installation.  The FasterWAM checkout may contain only
# policy checkpoints; pointing the loader there otherwise triggers a fresh
# 10+ GB ModelScope download and fails in minimal environments.
COMMON_MODEL_BASE="${WORLD2WAM_COMMON_MODEL_BASE:-/DATA/disk0/yjh/robotwin_w2wam/third_party/FastWAM_official/checkpoints}"
if [[ -s "$COMMON_MODEL_BASE/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" ]]; then
  export DIFFSYNTH_MODEL_BASE_PATH="$COMMON_MODEL_BASE"
else
  export DIFFSYNTH_MODEL_BASE_PATH="$FASTER/checkpoints"
fi
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

"$PY" "$FASTER/experiments/robotwin/run_robotwin_manager.py" \
  "task=$TASK_CONFIG" \
  "ckpt=$CHECKPOINT" \
  "EVALUATION.dataset_stats_path=$STATS" \
  "EVALUATION.eval_num_episodes=$EPISODES" \
  "EVALUATION.replan_steps=28" \
  "EVALUATION.robotwin_root=$FASTER/third_party/RoboTwin" \
  "EVALUATION.output_dir=$OUTPUT" \
  "MULTIRUN.num_gpus=1" \
  "MULTIRUN.max_tasks_per_gpu=1"

TAG="$(basename "$CHECKPOINT" .pt)"
INTERNAL="$FASTER/evaluate_results/robotwin/$TAG/$(basename "$OUTPUT")"
[[ -s "$INTERNAL/summary.json" ]]
cp "$INTERNAL/summary.json" "$OUTPUT/summary.json"
cp "$INTERNAL/summary.csv" "$OUTPUT/summary.csv"
python3 - "$OUTPUT" "$METHOD" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
root = pathlib.Path(sys.argv[1])
(root / "completion.json").write_text(json.dumps({
    "complete": True,
    "method": sys.argv[2],
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
