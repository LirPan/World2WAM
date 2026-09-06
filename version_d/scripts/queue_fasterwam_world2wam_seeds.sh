#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 GPU_ID" >&2
  exit 2
fi
GPU_ID="$1"
ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
TRAIN="$ROOT/deploy/run_fasterwam_world2wam_train.sh"
LOGDIR="$ROOT/runs/paper_sprint_v2/logs"
mkdir -p "$LOGDIR"

# Seed 42 is started independently; wait for its resumable bundle before
# occupying the same GPU with the remaining seeds.
while [[ ! -s "$ROOT/runs/robotwin_train/FasterWAM_W2W_s42/checkpoints/world2wam_final.pt" ]]; do
  sleep 60
done

for seed in 43 44; do
  if [[ -s "$ROOT/checkpoints/robotwin/FasterWAM_W2W_s${seed}.pt" ]]; then
    continue
  fi
  "$TRAIN" "$seed" "$GPU_ID" >"$LOGDIR/fasterwam_world2wam_s${seed}.log" 2>&1
done
