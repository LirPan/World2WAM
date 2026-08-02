#!/usr/bin/env bash
# Version A: multi-GPU precompute to 1M, then train + LIBERO eval.
# Uses all currently free A100s (shard parallel). Survives Cursor exit (nohup).
#
#   bash scripts/run_version_a_1m_multigpu.sh          # foreground (prefer nohup wrapper)
#   bash scripts/run_version_a_1m_multigpu.sh status
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/jianhua}"
CONFIG="${CONFIG:-configs/world2wam_physics_flow_dit_main.yaml}"
CACHE_OUT="${CACHE_OUT:-cache/libero_spatial_h10_full_fastwam}"
OUT_DIR="${OUT_DIR:-experiments/world2wam_physics_flow_dit_main}"
MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
FLOW_SAMPLE_STEPS="${FLOW_SAMPLE_STEPS:-10}"
GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-65000}"
GPU_INDEX_MAX="${GPU_INDEX_MAX:-7}"
PRECOMPUTE_SHARDS="${PRECOMPUTE_SHARDS:-8}"  # capped by free GPUs
JOB_DIR="${WORKSPACE}/cache/bg_jobs"
LOG="${JOB_DIR}/version_a_1m_multigpu.log"
CKPT="${OUT_DIR}/physics_world2wam_final.pt"

mkdir -p "${JOB_DIR}" "${CACHE_OUT}" "${OUT_DIR}"
cd "${WORKSPACE}"

# shellcheck disable=SC1091
source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh"
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
export MUJOCO_GL=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID

_log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

_pick_free_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v min="${GPU_FREE_MIN_MB}" -v max="${GPU_INDEX_MAX}" \
      '$1+0 <= max && $2+0 >= min {print $2+0, $1+0}' \
    | sort -nr \
    | awk '{print $2}'
}

_cache_count() {
  find "${CACHE_OUT}" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l
}

_run_shard() {
  local gpu="$1" shard_id="$2" num_shards="$3"
  local slog="${JOB_DIR}/precompute_shard${shard_id}_gpu${gpu}.log"
  _log "start shard ${shard_id}/${num_shards} on GPU ${gpu} -> ${slog}"
  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/cache/precompute_fastwam_latents.py \
    --config "${CONFIG}" \
    --output "${CACHE_OUT}" \
    --max_samples "${MAX_SAMPLES}" \
    --device cuda \
    --resume \
    --shard_id "${shard_id}" \
    --num_shards "${num_shards}" \
    >> "${slog}" 2>&1
  _log "done shard ${shard_id}/${num_shards} on GPU ${gpu}"
}

_precompute_multigpu() {
  local have
  have="$(_cache_count)"
  _log "======== PRECOMPUTE start: have=${have} target=${MAX_SAMPLES} ========"
  if (( have >= MAX_SAMPLES )); then
    _log "Cache already >= ${MAX_SAMPLES}; skip precompute"
    CUDA_VISIBLE_DEVICES=0 python minimal_world2wam/cache/precompute_fastwam_latents.py \
      --config "${CONFIG}" --output "${CACHE_OUT}" --finalize_only || true
    return 0
  fi

  mapfile -t FREE < <(_pick_free_gpus)
  if ((${#FREE[@]} < 1)); then
    _log "ERROR: no free GPU with >=${GPU_FREE_MIN_MB}MiB"
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv | tee -a "${LOG}"
    exit 1
  fi

  local n="${PRECOMPUTE_SHARDS}"
  if (( n > ${#FREE[@]} )); then n=${#FREE[@]}; fi
  if (( n < 1 )); then n=1; fi
  mapfile -t GPUS < <(printf '%s\n' "${FREE[@]}" | head -n "${n}")
  _log "Using ${n} shard(s) on GPU(s): ${GPUS[*]}"

  local pids=()
  local s
  for ((s = 0; s < n; s++)); do
    _run_shard "${GPUS[$s]}" "${s}" "${n}" &
    pids+=($!)
  done

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      _log "WARN: shard pid ${pid} exited non-zero"
      fail=1
    fi
  done

  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python minimal_world2wam/cache/precompute_fastwam_latents.py \
    --config "${CONFIG}" \
    --output "${CACHE_OUT}" \
    --finalize_only
  have="$(_cache_count)"
  _log "======== PRECOMPUTE done: have=${have} fail=${fail} ========"
  if (( have < MAX_SAMPLES )); then
    _log "ERROR: cache ${have} < target ${MAX_SAMPLES}"
    exit 1
  fi
}

_train() {
  if [[ -f "${CKPT}" ]]; then
    _log "Checkpoint exists ${CKPT} — skip train"
    return 0
  fi
  # Prefer a free GPU with enough memory for training (~45GB)
  local gpu
  gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' '$2+0>=45000{print $2+0,$1+0}' | sort -nr | awk 'NR==1{print $2}')"
  if [[ -z "${gpu}" ]]; then
    _log "No free GPU for train yet; waiting..."
    while true; do
      gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$2+0>=45000{print $2+0,$1+0}' | sort -nr | awk 'NR==1{print $2}')"
      [[ -n "${gpu}" ]] && break
      sleep 120
    done
  fi
  _log "======== TRAIN on GPU ${gpu} ========"
  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/train/train_physics_world2wam.py \
    --config "${CONFIG}" \
    --cache_dir "${CACHE_OUT}" \
    --output_dir "${OUT_DIR}" \
    --device cuda
  _log "======== TRAIN done ========"
}

_eval() {
  if [[ ! -f "${CKPT}" ]]; then
    _log "ERROR: missing ${CKPT}"
    exit 1
  fi
  local gpu
  gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' '$2+0>=30000{print $2+0,$1+0}' | sort -nr | awk 'NR==1{print $2}')"
  [[ -z "${gpu}" ]] && gpu=0
  _log "======== EVAL on GPU ${gpu} ========"

  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/eval/eval_offline_cache_only.py \
    --config "${CONFIG}" --cache_dir "${CACHE_OUT}" --adapter_ckpt "${CKPT}" \
    --use_physics true --device cuda \
    --output experiments/eval_offline_physics_flow_dit_main.json

  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/eval/eval_libero_world2wam.py \
    --config "${CONFIG}" --mode baseline \
    --max_tasks "${MAX_TASKS}" --num_trials "${NUM_TRIALS}" --device cuda \
    --output experiments/eval_baseline_version_a_main.json

  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/eval/eval_libero_world2wam.py \
    --config "${CONFIG}" --mode ours_onestep_physics_flow_dit \
    --adapter_ckpt "${CKPT}" --flow_sample_steps "${FLOW_SAMPLE_STEPS}" \
    --max_tasks "${MAX_TASKS}" --num_trials "${NUM_TRIALS}" --device cuda \
    --output experiments/eval_ours_onestep_physics_flow_dit_main.json

  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/eval/eval_libero_world2wam.py \
    --config "${CONFIG}" --mode ours_onestep_flow_dit \
    --adapter_ckpt "${CKPT}" --flow_sample_steps "${FLOW_SAMPLE_STEPS}" \
    --max_tasks "${MAX_TASKS}" --num_trials "${NUM_TRIALS}" --device cuda \
    --output experiments/eval_ours_onestep_flow_dit_main.json

  python minimal_world2wam/scripts/summarize_version_a_results.py \
    --workspace "${WORKSPACE}" --output experiments/VERSION_A_SUMMARY.json || true
  _log "======== EVAL done ========"
}

case "${1:-run}" in
  status)
    echo "log: ${LOG}"
    echo "cache: $(_cache_count) / ${MAX_SAMPLES}"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
    tail -n 20 "${LOG}" 2>/dev/null || true
    ;;
  run|*)
    _log "======== Version A 1M multi-GPU pipeline START ========"
    _precompute_multigpu
    _train
    _eval
    _log "======== Version A 1M multi-GPU pipeline DONE ========"
    ;;
esac
