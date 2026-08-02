#!/usr/bin/env bash
# Version C pipeline: FastWAM teacher actions -> residual train -> offline -> LIBERO sweep.
#
#   bash scripts/run_version_c_pipeline.sh                 # full
#   bash scripts/run_version_c_pipeline.sh precompute       # only FW actions
#   bash scripts/run_version_c_pipeline.sh train
#   bash scripts/run_version_c_pipeline.sh eval
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/jianhua}"
CONFIG="${CONFIG:-configs/world2wam_physics_residual_flow_dit_vc.yaml}"
CACHE_OUT="${CACHE_OUT:-cache/libero_spatial_h10_full_fastwam}"
OUT_DIR="${OUT_DIR:-experiments/world2wam_physics_residual_flow_dit_vc}"
CKPT="${CKPT:-${OUT_DIR}/physics_world2wam_final.pt}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
PRECOMPUTE_SHARDS="${PRECOMPUTE_SHARDS:-4}"
GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-65000}"
JOB_DIR="${WORKSPACE}/cache/bg_jobs"
LOG="${JOB_DIR}/version_c_pipeline.log"

mkdir -p "${JOB_DIR}" "${OUT_DIR}"
cd "${WORKSPACE}"

# shellcheck disable=SC1091
source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
export MUJOCO_GL=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID

_log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

_pick_free_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v min="${GPU_FREE_MIN_MB}" '$2+0 >= min {print $2+0, $1+0}' \
    | sort -nr | awk '{print $2}'
}

_precompute_fw_actions() {
  _log "======== precompute FastWAM teacher actions ========"
  mapfile -t FREE < <(_pick_free_gpus)
  local n="${PRECOMPUTE_SHARDS}"
  if ((${#FREE[@]} < 1)); then
    _log "ERROR: no free GPU >= ${GPU_FREE_MIN_MB}MiB"
    exit 1
  fi
  if ((n > ${#FREE[@]})); then n=${#FREE[@]}; fi
  _log "Using ${n} shard(s) on GPUs: ${FREE[*]:0:${n}}"
  local pids=()
  local s
  for ((s = 0; s < n; s++)); do
    local gpu="${FREE[$s]}"
    local slog="${JOB_DIR}/precompute_fw_action_shard${s}_gpu${gpu}.log"
    _log "shard ${s}/${n} GPU ${gpu} -> ${slog}"
    CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/scripts/precompute_fastwam_actions.py \
      --config "${CONFIG}" \
      --cache_dir "${CACHE_OUT}" \
      --device cuda \
      --resume \
      --shard_id "${s}" \
      --num_shards "${n}" \
      >> "${slog}" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then fail=1; fi
  done
  if ((fail)); then
    _log "ERROR: one or more FW-action shards failed"
    exit 1
  fi
  _log "FW-action precompute done"
}

_train() {
  mapfile -t FREE < <(_pick_free_gpus)
  local gpu="${FREE[0]:-0}"
  _log "======== train Version C on GPU ${gpu} ========"
  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/train/train_physics_world2wam.py \
    --config "${CONFIG}" \
    --cache_dir "${CACHE_OUT}" \
    --output_dir "${OUT_DIR}" \
    --device cuda
  _log "train done -> ${CKPT}"
}

_eval_sweep() {
  if [[ ! -f "${CKPT}" ]]; then
    _log "ERROR: missing ${CKPT}"
    exit 1
  fi
  mapfile -t FREE < <(_pick_free_gpus)
  local gpu="${FREE[0]:-0}"
  _log "======== LIBERO Version C residual sweep on GPU ${gpu} ========"
  local alpha
  for alpha in 0.0 0.25 0.5 1.0; do
    local out="experiments/eval_vc_residual_a${alpha}_10x${NUM_TRIALS}.json"
    _log "eval alpha=${alpha} -> ${out}"
    CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/eval/eval_libero_world2wam.py \
      --config "${CONFIG}" \
      --mode ours_residual_physics_flow_dit_vc \
      --adapter_ckpt "${CKPT}" \
      --residual_mode additive \
      --residual_alpha "${alpha}" \
      --residual_gate confidence_soft \
      --zero_alpha_on_uncertain \
      --max_tasks "${MAX_TASKS}" \
      --num_trials "${NUM_TRIALS}" \
      --device cuda \
      --output "${out}"
  done
  _log "eval sweep done"
}

case "${1:-all}" in
  precompute) _precompute_fw_actions ;;
  train) _train ;;
  eval) _eval_sweep ;;
  all)
    _precompute_fw_actions
    _train
    _eval_sweep
    ;;
  *)
    echo "Usage: $0 {all|precompute|train|eval}"
    exit 1
    ;;
esac
