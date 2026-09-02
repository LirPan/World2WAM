#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 METHOD SEED" >&2
  exit 2
fi
METHOD="$1"
SEED="$2"
case "$METHOD" in B1|B2|B3|B4|B5|B5_no_hard) ;; *) echo "invalid method" >&2; exit 2 ;; esac
case "$SEED" in 42|43|44) ;; *) echo "invalid seed" >&2; exit 2 ;; esac

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
POLICY="$ROOT/deploy/runtime/policy_lora"
CONFIG="$ROOT/deploy/configs/robotwin_iclr2027.yaml"
PY=/DATA/disk0/yjh/robotwin_w2wam/env/bin/python
RUN_ID="${METHOD}_s${SEED}"
RUN_DIR="$ROOT/runs/robotwin_train/$RUN_ID"
OUT="$ROOT/checkpoints/robotwin/$RUN_ID.pt"
HARD=0.7
[[ "$METHOD" == B5_no_hard ]] && HARD=0.0

mkdir -p "$ROOT/checkpoints/robotwin" "$RUN_DIR"
cd "$POLICY"
if [[ "$METHOD" == B1 ]]; then
  "$PY" src/train/train_lora_action_hard.py \
    --config "$CONFIG" --backbone-mode lora --max-steps 6000 \
    --resume-from auto --run-id "$RUN_ID" --seed "$SEED" \
    --hard-sample-fraction "$HARD"
else
  extra=(--lambda-fwd 0.1 --hard-sample-fraction "$HARD")
  case "$METHOD" in
    B2) extra+=(--lambda-inv 0 --lambda-cycle 0 --enable-inverse false --enable-cycle false --gradient-mode naive) ;;
    B3) extra+=(--lambda-inv 0.05 --lambda-cycle 0 --enable-inverse true --enable-cycle false --gradient-mode naive) ;;
    B4) extra+=(--lambda-inv 0.05 --lambda-cycle 0.05 --enable-inverse true --enable-cycle true --gradient-mode naive) ;;
    B5|B5_no_hard) extra+=(--lambda-inv 0.05 --lambda-cycle 0.05 --enable-inverse true --enable-cycle true --gradient-mode project_conflicts) ;;
  esac
  "$PY" src/train/train_lora_fic_hardtask.py \
    --config "$CONFIG" --backbone-mode lora --max-steps 6000 \
    --resume-from auto --run-id "$RUN_ID" --seed "$SEED" "${extra[@]}"
fi

EXPORT_DEVICE=cuda "$PY" src/tools/export_libero_checkpoint.py \
  --bundle "$RUN_DIR/checkpoints/world2wam_final.pt" \
  --config "$CONFIG" --output "$OUT" --tag "$RUN_ID"
sha256sum "$OUT" > "$OUT.sha256"
