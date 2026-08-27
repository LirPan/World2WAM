#!/usr/bin/env bash
# Claim the first genuinely idle GPU on New_yjh, finish the missing RoboTwin
# hard10 jobs, then run the small LIBERO-Spatial Version D pilot. Every stage
# is resume-safe and may be restarted after preemption.
set -u

ROOT="${WORLD2WAM_ROOT:-/DATA/disk0/yjh/robotwin_w2wam}"
POLICY_ROOT="${POLICY_ROOT:-$ROOT/latest/code/policy_lora}"
LIBERO_WORK="${LIBERO_WORK:-/DATA/disk0/yjh/libero_work_wj}"
PY="${ROBOTWIN_PYTHON:-$ROOT/env/bin/python}"
HARD_RUNNER="${HARD_RUNNER:-$ROOT/run_robotwin_hard10_parallel_new_yjh.py}"
HARD_OUT="${HARD_OUT:-$ROOT/runs/robotwin_hard10_standard_pair_n10_v2}"
LIBERO_RUNNER="${LIBERO_RUNNER:-$LIBERO_WORK/scripts/run_libero_version_d_new_yjh.sh}"
PILOT_CONFIG="${PILOT_CONFIG:-$POLICY_ROOT/configs/libero_version_d_new_yjh_pilot.yaml}"
PILOT_RUN="${PILOT_RUN:-$LIBERO_WORK/runs/libero_version_d_spatial_pilot}"
LOG="${PRIORITY_LOG:-$LIBERO_WORK/runs/world2wam_priority_queue.log}"
LOCK="${PRIORITY_LOCK:-/tmp/world2wam_priority_new_yjh.lock}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1000}"
IDLE_UTIL="${IDLE_UTIL:-5}"
POLL_SECONDS="${POLL_SECONDS:-45}"

mkdir -p "$(dirname "$LOG")"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '[%s] another priority queue is already active\n' "$(date -Is)" | tee -a "$LOG"
  exit 0
fi

exec > >(tee -a "$LOG") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' -v max_mem="$IDLE_MEM_MB" -v max_util="$IDLE_UTIL" '
    {
      gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
      if (($2 + 0) <= max_mem && ($3 + 0) <= max_util) { print $1; exit }
    }'
}

wait_for_gpu() {
  local gpu=""
  while [[ -z "$gpu" ]]; do
    gpu="$(idle_gpu)"
    if [[ -z "$gpu" ]]; then
      log "all GPUs busy; retry in ${POLL_SECONDS}s" >&2
      sleep "$POLL_SECONDS"
    fi
  done
  printf '%s' "$gpu"
}

hard10_complete() {
  [[ -s "$HARD_OUT/summary.json" ]] || return 1
  "$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); c=p.get("paired_comparison",{}); raise SystemExit(0 if c.get("clean",{}).get("complete") and c.get("random",{}).get("complete") else 1)' "$HARD_OUT/summary.json"
}

log "priority queue started"

if hard10_complete; then
  log "RoboTwin hard10 pair is already complete"
else
  hard_attempt=1
  while (( hard_attempt <= 3 )); do
    gpu="$(wait_for_gpu)"
    log "claim GPU${gpu}: RoboTwin missing hard10 jobs, attempt=${hard_attempt}/3"
    "$PY" "$HARD_RUNNER" --episodes 10 --gpus "$gpu" --output-root "$HARD_OUT"
    rc=$?
    if hard10_complete; then
      log "RoboTwin hard10 pair completed"
      break
    fi
    log "RoboTwin attempt=${hard_attempt} incomplete (rc=${rc}); remaining jobs will resume"
    hard_attempt=$((hard_attempt + 1))
    sleep 30
  done
fi

if [[ -s "$PILOT_RUN/libero_pair_summary.json" ]]; then
  log "LIBERO Version D pilot already complete"
else
  gpu="$(wait_for_gpu)"
  log "claim GPU${gpu}: LIBERO Version D pilot"
  POLICY_ROOT="$POLICY_ROOT" \
  LIBERO_VERSION_D_CONFIG="$PILOT_CONFIG" \
  LIBERO_VERSION_D_RUN="$PILOT_RUN" \
  LIBERO_CACHE_MAX_SAMPLES="${LIBERO_CACHE_MAX_SAMPLES:-1000}" \
  LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-0 1 2}" \
  LIBERO_NUM_TRIALS="${LIBERO_NUM_TRIALS:-10}" \
  LIBERO_TRAIN_GPU="$gpu" \
  LIBERO_EVAL_GPU="$gpu" \
  bash "$LIBERO_RUNNER"
  rc=$?
  log "LIBERO Version D pilot exited rc=${rc}"
fi

log "priority queue finished"
