#!/usr/bin/env bash
# Wait for a genuinely idle allowed GPU on FiveAges and run the resume-safe
# LIBERO-Spatial Version D pilot.
set -u

POLICY_ROOT="${POLICY_ROOT:-/DATA/disk0/jianhua/latest/code/policy_lora}"
PIPELINE="${LIBERO_PIPELINE:-/DATA/disk0/jianhua/latest/scripts/libero_version_d_after_robotwin.sh}"
PY="${PYTHON_BIN:-/DATA/disk0/jianhua/_shared/miniconda3/envs/world2wam/bin/python}"
CONFIG="${LIBERO_VERSION_D_CONFIG:-$POLICY_ROOT/configs/libero_version_d_fiveages_pilot.yaml}"
RUN_ROOT="${LIBERO_VERSION_D_RUN:-/DATA/disk0/jianhua/latest/experiments/iclr_2027/libero_version_d_pilot}"
LOG="${FIVEAGES_QUEUE_LOG:-$RUN_ROOT/logs/waiter.log}"
LOCK="${FIVEAGES_QUEUE_LOCK:-/tmp/world2wam_libero_version_d_fiveages.lock}"
GPU_IDS="${FIVEAGES_GPU_IDS:-0 1 2 3 4 5 6 7}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1000}"
IDLE_UTIL="${IDLE_UTIL:-5}"
POLL_SECONDS="${POLL_SECONDS:-10}"

mkdir -p "$(dirname "$LOG")"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '[%s] another FiveAges Version D waiter is active\n' "$(date -Is)" | tee -a "$LOG"
  exit 0
fi

exec > >(tee -a "$LOG") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

idle_gpu() {
  local row gpu mem util allowed
  while IFS=',' read -r gpu mem util; do
    gpu="${gpu// /}"; mem="${mem// /}"; util="${util// /}"
    allowed=0
    for candidate in $GPU_IDS; do
      [[ "$gpu" == "$candidate" ]] && allowed=1
    done
    if (( allowed == 1 && mem <= IDLE_MEM_MB && util <= IDLE_UTIL )); then
      printf '%s' "$gpu"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
}

wait_for_gpu() {
  local gpu=""
  while [[ -z "$gpu" ]]; do
    gpu="$(idle_gpu)"
    if [[ -z "$gpu" ]]; then
      log "allowed GPUs (${GPU_IDS}) busy; retry in ${POLL_SECONDS}s" >&2
      sleep "$POLL_SECONDS"
    fi
  done
  printf '%s' "$gpu"
}

log "FiveAges Version D waiter started"
if [[ -s "$RUN_ROOT/libero_pair_summary.json" ]]; then
  log "pilot already complete: $RUN_ROOT/libero_pair_summary.json"
else
  attempt=1
  while (( attempt <= 3 )); do
    gpu="$(wait_for_gpu)"
    log "claim GPU${gpu}: LIBERO Version D pilot, attempt=${attempt}/3"
    POLICY_ROOT="$POLICY_ROOT" \
    PYTHON_BIN="$PY" \
    LIBERO_VERSION_D_CONFIG="$CONFIG" \
    LIBERO_VERSION_D_RUN="$RUN_ROOT" \
    LIBERO_CACHE_MAX_SAMPLES="${LIBERO_CACHE_MAX_SAMPLES:-1000}" \
    LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-0 1 2}" \
    LIBERO_NUM_TRIALS="${LIBERO_NUM_TRIALS:-10}" \
    LIBERO_TRAIN_GPU="$gpu" \
    LIBERO_EVAL_GPU="$gpu" \
    WAIT_FOR_ROBOTWIN=0 \
    bash "$PIPELINE"
    rc=$?
    if [[ -s "$RUN_ROOT/libero_pair_summary.json" ]]; then
      log "LIBERO Version D pilot complete"
      break
    fi
    log "pilot attempt=${attempt} incomplete (rc=${rc}); resume after next idle slot"
    attempt=$((attempt + 1))
    sleep 30
  done
fi

if [[ ! -s "$RUN_ROOT/libero_pair_summary.json" ]]; then
  log "pilot remains incomplete after 3 attempts; inspect pipeline log"
  exit 1
fi

FULL_CONFIG="${FULL_CONFIG:-$POLICY_ROOT/configs/libero_version_d_fiveages.yaml}"
FULL_RUN="${FULL_RUN:-/DATA/disk0/jianhua/latest/experiments/iclr_2027/libero_version_d}"
if [[ -s "$FULL_RUN/libero_pair_summary.json" ]]; then
  log "LIBERO Version D 10-task run already complete"
else
  gpu="$(wait_for_gpu)"
  log "claim GPU${gpu}: LIBERO Version D 10-task matched run"
  POLICY_ROOT="$POLICY_ROOT" \
  PYTHON_BIN="$PY" \
  LIBERO_VERSION_D_CONFIG="$FULL_CONFIG" \
  LIBERO_VERSION_D_RUN="$FULL_RUN" \
  LIBERO_CACHE_MAX_SAMPLES="${LIBERO_FULL_CACHE_MAX_SAMPLES:-12000}" \
  LIBERO_TASK_IDS="${LIBERO_FULL_TASK_IDS:-0 1 2 3 4 5 6 7 8 9}" \
  LIBERO_NUM_TRIALS="${LIBERO_FULL_NUM_TRIALS:-10}" \
  LIBERO_TRAIN_GPU="$gpu" \
  LIBERO_EVAL_GPU="$gpu" \
  WAIT_FOR_ROBOTWIN=0 \
  bash "$PIPELINE"
  log "LIBERO Version D 10-task run exited rc=$?"
fi
