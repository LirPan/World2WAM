#!/usr/bin/env bash
set -euo pipefail

if (( $# != 7 )); then
  echo "usage: $0 METHOD BENCHMARK CHECKPOINT STATS OUTPUT SAMPLE_RATIO TASK_CONFIG" >&2
  exit 2
fi
METHOD="$1"
BENCHMARK="$2"
CHECKPOINT="$3"
STATS="$4"
OUTPUT="$5"
SAMPLE_RATIO="$6"
TASK_CONFIG="$7"

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
FASTER="$ROOT/third_party/FasterWAM"
case "$BENCHMARK" in
  libero_plus)
    PY="$FASTER/.venvs/libero-plus/bin/python"
    CONFIG=sim_libero_plus
    TRIALS=1
    ;;
  libero)
    PY="$FASTER/.venvs/libero/bin/python"
    CONFIG=sim_libero
    TRIALS=50
    ;;
  *) echo "unsupported benchmark: $BENCHMARK" >&2; exit 2 ;;
esac

[[ -x "$PY" ]]
[[ -s "$CHECKPOINT" ]]
[[ -s "$STATS" ]]
mkdir -p "$OUTPUT"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export DIFFSYNTH_MODEL_BASE_PATH="$FASTER/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export LIBERO_CONFIG_PATH="$FASTER/.runtime/${BENCHMARK//_/-}"
export PYTHONPATH="$FASTER/src:$FASTER:$FASTER/third_party/LIBERO-plus:$FASTER/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"

ratio_override="MULTIRUN.task_sample_ratio=$SAMPLE_RATIO"
"$PY" "$FASTER/experiments/libero/run_libero_manager.py" \
  --config-name "$CONFIG" \
  "task=$TASK_CONFIG" \
  "ckpt=$CHECKPOINT" \
  "EVALUATION.dataset_stats_path=$STATS" \
  "EVALUATION.output_dir=$OUTPUT" \
  "EVALUATION.num_trials=$TRIALS" \
  "$ratio_override" \
  "MULTIRUN.task_sample_seed=42" \
  "MULTIRUN.num_gpus=1" \
  "MULTIRUN.max_tasks_per_gpu=1" \
  "MULTIRUN.chunk_size=20"

python3 - "$OUTPUT" "$METHOD" "$BENCHMARK" <<'PY'
import json, pathlib, sys
from datetime import datetime, timezone
root = pathlib.Path(sys.argv[1])
summary = root / "completion.json"
result_files = list(root.glob("**/*results.json"))
summary.write_text(json.dumps({
    "complete": True,
    "method": sys.argv[2],
    "benchmark": sys.argv[3],
    "result_files": len(result_files),
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
