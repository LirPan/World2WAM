#!/usr/bin/env bash
set -u

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
EVAL="$ROOT/deploy/run_iclr2027_robotwin_eval.sh"
FASTER="$ROOT/third_party/FasterWAM"
STATS="$FASTER/checkpoints/fasterwam_release/robotwin/dataset_stats.json"
LOGDIR="$ROOT/runs/paper_sprint_v2/logs"
LOCKROOT="$ROOT/.locks/fasterwam_w2w_eval"
mkdir -p "$LOGDIR" "$LOCKROOT"

gpu_is_free() {
  local gpu="$1" mem util pids
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 999999)"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 999999)"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')"
  [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && -z "$pids" ]] || return 1
  (( mem < 1000 && util < 5 ))
}

acquire_gpu() {
  while :; do
    for gpu in 1 2 4 5 6 7; do
      if ! mkdir "$LOCKROOT/gpu${gpu}" 2>/dev/null; then continue; fi
      if gpu_is_free "$gpu"; then
        sleep 5
        if gpu_is_free "$gpu"; then
          echo "$gpu"
          return 0
        fi
      fi
      rmdir "$LOCKROOT/gpu${gpu}" 2>/dev/null || true
    done
    sleep 60
  done
}

for seed in 42 43 44; do
  ckpt="$ROOT/checkpoints/robotwin/FasterWAM_W2W_s${seed}.pt"
  out="$ROOT/results/robotwin/robotwin_full_fasterwam_w2w_s${seed}"
  marker="$out/.complete"
  if [[ -f "$marker" ]]; then
    echo "[$(date)] skip completed seed=$seed" >>"$LOGDIR/fasterwam_world2wam_eval.log"
    continue
  fi
  while [[ ! -s "$ckpt" || ! -s "$ckpt.sha256" ]]; do sleep 60; done
  gpu="$(acquire_gpu)"
  echo "[$(date)] start seed=$seed gpu=$gpu" >>"$LOGDIR/fasterwam_world2wam_eval.log"
  (
    trap 'rmdir "$LOCKROOT/gpu${gpu}" 2>/dev/null || true' EXIT
    "$EVAL" "FasterWAM_W2W_s${seed}" "$ckpt" "$STATS" "$out" \
      robotwin_fasterwam_3cam_384_1e-4 10 >"$LOGDIR/robotwin_full_fasterwam_w2w_s${seed}.log" 2>&1
    rc=$?
    if (( rc == 0 )); then touch "$marker"; fi
    echo "[$(date)] finish seed=$seed gpu=$gpu rc=$rc" >>"$LOGDIR/fasterwam_world2wam_eval.log"
    exit "$rc"
  )
done
echo "[$(date)] all FasterWAM-World2WAM evaluations complete" >>"$LOGDIR/fasterwam_world2wam_eval.log"
