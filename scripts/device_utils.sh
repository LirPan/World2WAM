#!/usr/bin/env bash
# Pick compute device: prefer GPU when enough free memory, else CPU.
# Source this file: source minimal_world2wam/scripts/device_utils.sh

GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-20000}"
GPU_POLL_SEC="${GPU_POLL_SEC:-120}"

pick_compute_device() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return 0
  fi

  local best_gpu=-1
  local best_free=0
  local idx used total free

  while IFS=',' read -r idx used total; do
    idx="$(echo "${idx}" | tr -d ' ')"
    used="$(echo "${used}" | tr -d ' ')"
    total="$(echo "${total}" | tr -d ' ')"
    free=$((total - used))
    if (( free >= GPU_FREE_MIN_MB && free > best_free )); then
      best_free="${free}"
      best_gpu="${idx}"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

  if (( best_gpu >= 0 )); then
    export CUDA_VISIBLE_DEVICES="${best_gpu}"
    echo "cuda"
  else
    unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
    echo "cpu"
  fi
}

log_device_pick() {
  local stage="$1"
  local device="$2"
  if [[ "${device}" == "cuda" ]]; then
    echo "[device] ${stage}: cuda (GPU ${CUDA_VISIBLE_DEVICES:-0}, free>=${GPU_FREE_MIN_MB}MB)"
  else
    echo "[device] ${stage}: cpu (no GPU with >=${GPU_FREE_MIN_MB}MB free)"
  fi
}

run_with_auto_device() {
  local stage="$1"
  shift
  local device
  device="$(pick_compute_device)"
  log_device_pick "${stage}" "${device}"
  DEVICE="${device}" "$@"
}
