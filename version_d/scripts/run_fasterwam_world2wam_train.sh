#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 SEED GPU_ID" >&2
  exit 2
fi
SEED="$1"
GPU_ID="$2"
case "$SEED" in 42|43|44) ;; *) echo "seed must be 42, 43, or 44" >&2; exit 2 ;; esac

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
POLICY="$ROOT/deploy/runtime/policy_lora"
CONFIG="$ROOT/deploy/configs/fasterwam_world2wam_robotwin.yaml"
PY=/DATA/disk0/yjh/robotwin_w2wam/env/bin/python
RUN_ID="FasterWAM_W2W_s${SEED}"
RUN_DIR="$ROOT/runs/robotwin_train/$RUN_ID"
OUT="$ROOT/checkpoints/robotwin/$RUN_ID.pt"

LOCKROOT="$ROOT/.locks/fasterwam_w2w_train"
mkdir -p "$LOCKROOT"
if ! mkdir "$LOCKROOT/seed${SEED}" 2>/dev/null; then
  echo "seed $SEED is already running; refusing duplicate launch" >&2
  exit 4
fi
trap 'rmdir "$LOCKROOT/seed${SEED}" 2>/dev/null || true' EXIT

mkdir -p "$ROOT/checkpoints/robotwin" "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$POLICY:$POLICY/src:$ROOT/third_party/FasterWAM/src:$ROOT/third_party/FasterWAM${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${WORLD2WAM_COMMON_MODEL_BASE:-/DATA/disk0/yjh/robotwin_w2wam/third_party/FastWAM_official/checkpoints}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# FasterWAM resolves its text cache relative to the current working directory
# (the policy_lora checkout).  Reuse the already validated shared RobotWin
# cache without copying hundreds of GB or relying on a fragile CWD.
CACHE_SRC="${WORLD2WAM_TEXT_CACHE:-/DATA/disk0/yjh/robotwin_w2wam/data/text_embeds_cache}"
mkdir -p "$POLICY/data"
POLICY_CACHE="$POLICY/data/text_embeds_cache_fasterwam"
if [[ ! -e "$POLICY_CACHE" ]]; then
  ln -s "$CACHE_SRC" "$POLICY_CACHE"
fi

# The FasterWAM checkout already contains a cache root in some deployments;
# only install the suite-level link so we never delete existing cache files.
FASTER_CACHE_ROOT="$ROOT/third_party/FasterWAM/data/text_embeds_cache_fasterwam"
mkdir -p "$FASTER_CACHE_ROOT"
if [[ ! -e "$FASTER_CACHE_ROOT/robotwin" ]]; then
  ln -s "$CACHE_SRC/robotwin" "$FASTER_CACHE_ROOT/robotwin"
fi

if [[ ! -f "$FASTER_CACHE_ROOT/robotwin/09ab47dde6f36892a65652afa760670bf79492d41d2b0abdd7ece2fc9b0b0baf.t5_len128.wan22ti2v5b.pt" ]]; then
  echo "Missing FasterWAM RobotWin text cache at $CACHE_SRC" >&2
  exit 3
fi

cd "$POLICY"
"$PY" src/train/train_lora_fic_hardtask.py \
  --config "$CONFIG" --backbone-mode lora --max-steps 6000 \
  --resume-from auto --run-id "$RUN_ID" --seed "$SEED" \
  --lambda-fwd 0.1 --lambda-inv 0.05 --lambda-cycle 0.05 \
  --enable-inverse true --enable-cycle true \
  --hard-sample-fraction 0.7 --gradient-mode project_conflicts

# Bake the LoRA delta into FasterWAM's native MoT checkpoint format.  The
# auxiliary F/I/C heads remain training-only and are intentionally absent
# from the exported inference checkpoint.
EXPORT_DEVICE=cuda "$PY" src/tools/export_libero_checkpoint.py \
  --bundle "$RUN_DIR/checkpoints/world2wam_final.pt" \
  --config "$CONFIG" --output "$OUT" --tag "$RUN_ID"
sha256sum "$OUT" > "$OUT.sha256"
