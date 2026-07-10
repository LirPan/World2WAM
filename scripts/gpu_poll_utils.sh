#!/usr/bin/env bash
# GPU polling helpers for Version A background jobs.
# Source: source minimal_world2wam/scripts/gpu_poll_utils.sh

GPU_POLL_SEC="${GPU_POLL_SEC:-120}"
GPU_INDEX_MAX="${GPU_INDEX_MAX:-7}"

_gpu_status_line() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "no-nvidia-smi"
    return 0
  fi
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "GPU%s:%sMiB/%s%% ", $1, $2, $3}' \
    | tr -d ' '
}

# Args: min_free_mb max_util_percent
# Prints one GPU index per line (most free memory first).
_pick_gpus() {
  local min_free="${1:-40000}"
  local max_util="${2:-30}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v min="${min_free}" -v maxu="${max_util}" -v idxmax="${GPU_INDEX_MAX}" \
      '$1+0 <= idxmax && $2+0 >= min && $3+0 <= maxu {print $2+0, $1+0}' \
    | sort -nr \
    | awk '{print $2}'
}

# Args: count min_free_mb max_util_percent
# Sets CUDA_VISIBLE_DEVICES to best GPU and exports PICKED_GPU.
wait_for_gpus() {
  local need="${1:-1}"
  local min_free="${2:-40000}"
  local max_util="${3:-30}"
  while true; do
    mapfile -t _picked < <(_pick_gpus "${min_free}" "${max_util}" | head -n "${need}")
    local status
    status="$(_gpu_status_line)"
    if ((${#_picked[@]} >= need)); then
      export PICKED_GPU="${_picked[0]}"
      export CUDA_VISIBLE_DEVICES="${PICKED_GPU}"
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      echo "$(date -Iseconds) picked GPU ${PICKED_GPU} (need=${need}, free>=${min_free}MiB, util<=${max_util}%) ${status}"
      return 0
    fi
    echo "$(date -Iseconds) waiting GPU: need=${need} free>=${min_free}MiB util<=${max_util}% have=${#_picked[@]} ${status}"
    sleep "${GPU_POLL_SEC}"
  done
}

run_stage_on_gpu() {
  local stage="$1"
  local min_free="$2"
  local max_util="$3"
  shift 3
  echo ""
  echo "== [${stage}] $(date -Iseconds) =="
  wait_for_gpus 1 "${min_free}" "${max_util}"
  echo "[${stage}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "$@"
}
