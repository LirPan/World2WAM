#!/usr/bin/env bash
# Resume latent precompute into an existing cache dir (skip existing *.pt via --resume).
# Polls GPU 0-7 and uses any card with enough free memory.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
CONFIG="${CONFIG:-configs/world2wam_libero_spatial_h10_paper.yaml}"
CACHE_OUT="${CACHE_OUT:-cache/libero_spatial_h10_full_fastwam}"
MAX_SAMPLES="${MAX_SAMPLES:-600000}"
PRECOMPUTE_SHARDS="${PRECOMPUTE_SHARDS:-1}"
GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-70000}"
GPU_POLL_SEC="${GPU_POLL_SEC:-120}"
GPU_INDEX_MAX="${GPU_INDEX_MAX:-7}"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
cd "${WORKSPACE}"

_gpu_status_line() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{printf "GPU%s:%sMiB/%s%% ", $1, $2, $3}' \
    | tr -d ' '
}

_pick_free_gpus() {
  # Prints one GPU index per line, sorted by most free memory first.
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v min="${GPU_FREE_MIN_MB}" -v max="${GPU_INDEX_MAX}" \
      '$1 <= max && $2+0 >= min {print $2+0, $1+0}' \
    | sort -nr \
    | awk '{print $2}'
}

_wait_for_gpus() {
  local need="$1"
  while true; do
    mapfile -t SELECTED_GPUS < <(_pick_free_gpus | head -n "${need}")
    local status
    status="$(_gpu_status_line)"
    if ((${#SELECTED_GPUS[@]} >= need)); then
      echo "$(date -Iseconds) picked ${need} GPU(s): ${SELECTED_GPUS[*]} (${status})"
      return 0
    fi
    echo "$(date -Iseconds) need>=${need} free GPU(s) (>=${GPU_FREE_MIN_MB} MiB each), have ${#SELECTED_GPUS[@]}. ${status}"
    sleep "${GPU_POLL_SEC}"
  done
}

_run_shard() {
  local gpu="$1"
  local shard_id="$2"
  local num_shards="$3"
  # Isolate to one physical GPU; FastWAM load needs ~66GB on an otherwise empty card.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/cache/precompute_fastwam_latents.py \
    --config "${CONFIG}" \
    --output "${CACHE_OUT}" \
    --max_samples "${MAX_SAMPLES}" \
    --device cuda \
    --resume \
    --shard_id "${shard_id}" \
    --num_shards "${num_shards}"
}

echo "== resume_precompute: $(date -Iseconds) =="
echo "CONFIG=${CONFIG} CACHE_OUT=${CACHE_OUT} MAX_SAMPLES=${MAX_SAMPLES} poll GPU 0-${GPU_INDEX_MAX}"

desired_shards="${PRECOMPUTE_SHARDS}"
if ((desired_shards < 1)); then
  desired_shards=1
fi

_wait_for_gpus 1
mapfile -t ALL_FREE < <(_pick_free_gpus)
num_shards=$(( desired_shards < ${#ALL_FREE[@]} ? desired_shards : ${#ALL_FREE[@]} ))
if ((num_shards < 1)); then
  num_shards=1
fi
# Re-pick exactly num_shards GPUs (most free first).
mapfile -t SELECTED_GPUS < <(_pick_free_gpus | head -n "${num_shards}")
echo "Using ${num_shards} shard(s) on GPU(s): ${SELECTED_GPUS[*]}"

if ((num_shards == 1)); then
  _run_shard "${SELECTED_GPUS[0]}" 0 1
else
  pids=()
  for ((s = 0; s < num_shards; s++)); do
    _run_shard "${SELECTED_GPUS[$s]}" "${s}" "${num_shards}" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  CUDA_VISIBLE_DEVICES="${SELECTED_GPUS[0]}" python minimal_world2wam/cache/precompute_fastwam_latents.py \
    --config "${CONFIG}" \
    --output "${CACHE_OUT}" \
    --max_samples "${MAX_SAMPLES}" \
    --finalize_only
fi

echo "== resume_precompute done: $(date -Iseconds) =="
