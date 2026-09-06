#!/usr/bin/env bash
set -u

# Resource-safe queue for FasterWAM + World2WAM seeds.  It never assumes a
# particular GPU is free and never starts while nvidia-smi reports a compute
# process on the candidate device.
ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
TRAIN="$ROOT/deploy/run_fasterwam_world2wam_train.sh"
LOGDIR="$ROOT/runs/paper_sprint_v2/logs"
LOCKROOT="$ROOT/.locks/fasterwam_w2w_train"
mkdir -p "$LOGDIR" "$LOCKROOT"

gpu_is_free() {
  local gpu="$1" mem util
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 999999)"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 999999)"
  [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')" ]] || return 1
  (( mem < 1000 && util < 5 ))
}

acquire_gpu() {
  while :; do
    for gpu in 1 2 4 5 6 7; do
      if ! mkdir "$LOCKROOT/gpu${gpu}" 2>/dev/null; then continue; fi
      # Recheck after the lock to close the process-start race.
      if gpu_is_free "$gpu"; then
        sleep 5
      fi
      if gpu_is_free "$gpu"; then
        echo "$gpu"
        return 0
      fi
      rmdir "$LOCKROOT/gpu${gpu}"
    done
    sleep 60
  done
}

for seed in 42 43 44; do
  out="$ROOT/checkpoints/robotwin/FasterWAM_W2W_s${seed}.pt"
  if [[ -s "$out" && -s "$out.sha256" ]]; then
    echo "[$(date)] skip completed seed=$seed" >>"$LOGDIR/fasterwam_world2wam_auto.log"
    continue
  fi
  # A manually started seed owns this lock; wait for it and then continue.
  if [[ -d "$LOCKROOT/seed${seed}" ]]; then
    while [[ ! -s "$out" || ! -s "$out.sha256" ]]; do sleep 60; done
    continue
  fi
  gpu="$(acquire_gpu)"
  echo "[$(date)] start seed=$seed gpu=$gpu" >>"$LOGDIR/fasterwam_world2wam_auto.log"
  (
    trap 'rmdir "$LOCKROOT/gpu${gpu}" 2>/dev/null || true' EXIT
    "$TRAIN" "$seed" "$gpu" >>"$LOGDIR/fasterwam_world2wam_s${seed}.log" 2>&1
    rc=$?
    echo "[$(date)] finish seed=$seed gpu=$gpu rc=$rc" >>"$LOGDIR/fasterwam_world2wam_auto.log"
    exit "$rc"
  )
  if [[ ! -s "$out" || ! -s "$out.sha256" ]]; then
    echo "[$(date)] seed=$seed did not produce a verified export; stop queue" >>"$LOGDIR/fasterwam_world2wam_auto.log"
    exit 1
  fi
done
echo "[$(date)] all FasterWAM-World2WAM seeds complete" >>"$LOGDIR/fasterwam_world2wam_auto.log"
