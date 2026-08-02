#!/usr/bin/env bash
# Version A full experiment: precompute -> physics FlowDiT train -> offline + LIBERO eval.
# Designed for GPU polling wrapper (poll_gpu_version_a.sh) or direct run on a free GPU.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/jianhua}"
CONFIG="${CONFIG:-configs/world2wam_physics_flow_dit_main.yaml}"
CACHE_OUT="${CACHE_OUT:-cache/libero_spatial_h10_full_fastwam}"
OUT_DIR="${OUT_DIR:-experiments/world2wam_physics_flow_dit_main}"
CKPT="${CKPT:-${OUT_DIR}/physics_world2wam_final.pt}"
MAX_SAMPLES="${MAX_SAMPLES:-600000}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
FLOW_SAMPLE_STEPS="${FLOW_SAMPLE_STEPS:-10}"
SKIP_PRECOMPUTE="${SKIP_PRECOMPUTE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
USE_GPU_POLL="${USE_GPU_POLL:-1}"

# GPU thresholds per stage
PRECOMPUTE_GPU_FREE_MIN_MB="${PRECOMPUTE_GPU_FREE_MIN_MB:-70000}"
PRECOMPUTE_GPU_MAX_UTIL="${PRECOMPUTE_GPU_MAX_UTIL:-15}"
TRAIN_GPU_FREE_MIN_MB="${TRAIN_GPU_FREE_MIN_MB:-45000}"
TRAIN_GPU_MAX_UTIL="${TRAIN_GPU_MAX_UTIL:-25}"
EVAL_GPU_FREE_MIN_MB="${EVAL_GPU_FREE_MIN_MB:-30000}"
EVAL_GPU_MAX_UTIL="${EVAL_GPU_MAX_UTIL:-30}"

source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
export MUJOCO_GL=egl

if [[ -f "${WORKSPACE}/use_proxy.sh" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE}/use_proxy.sh" || true
fi

# shellcheck disable=SC1091
source "${WORKSPACE}/minimal_world2wam/scripts/gpu_poll_utils.sh"

cd "${WORKSPACE}"
mkdir -p "${CACHE_OUT}" "${OUT_DIR}" experiments cache/bg_jobs

LOG_TAG="version_a_full"
MAIN_LOG="${WORKSPACE}/cache/bg_jobs/${LOG_TAG}.log"

_log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${MAIN_LOG}"
}

_run_on_gpu() {
  local stage="$1"
  local min_free="$2"
  local max_util="$3"
  shift 3
  if [[ "${USE_GPU_POLL}" == "1" ]]; then
    run_stage_on_gpu "${stage}" "${min_free}" "${max_util}" "$@"
  else
    _log "== [${stage}] direct run (no poll) =="
    "$@"
  fi
}

_cache_count() {
  if [[ -f "${CACHE_OUT}/meta.json" ]]; then
    python3 -c "import json; print(json.load(open('${CACHE_OUT}/meta.json'))['num_samples'])" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

_log "======== Version A full pipeline start ========"
_log "CONFIG=${CONFIG} CACHE=${CACHE_OUT} OUT=${OUT_DIR}"
_log "EVAL: ${MAX_TASKS} tasks x ${NUM_TRIALS} trials | MAX_SAMPLES=${MAX_SAMPLES}"

_log "== smoke test =="
bash minimal_world2wam/scripts/smoke_test.sh 2>&1 | tee -a "${MAIN_LOG}"

current_samples="$(_cache_count)"
_log "Cache samples: ${current_samples} (target ${MAX_SAMPLES})"

if [[ "${SKIP_PRECOMPUTE}" != "1" ]] && (( current_samples < MAX_SAMPLES )); then
  _log "== precompute resume -> ${MAX_SAMPLES} =="
  _run_on_gpu precompute "${PRECOMPUTE_GPU_FREE_MIN_MB}" "${PRECOMPUTE_GPU_MAX_UTIL}" \
    python minimal_world2wam/cache/precompute_fastwam_latents.py \
      --config "${CONFIG}" \
      --output "${CACHE_OUT}" \
      --max_samples "${MAX_SAMPLES}" \
      --device cuda \
      --resume
  current_samples="$(_cache_count)"
  _log "Precompute done. Cache samples: ${current_samples}"
else
  _log "Skip precompute (SKIP_PRECOMPUTE=${SKIP_PRECOMPUTE}, have ${current_samples})"
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ -f "${CKPT}" ]]; then
    _log "Checkpoint exists: ${CKPT} — skip training (set SKIP_TRAIN=0 and remove ckpt to retrain)"
  else
    _log "== train physics FlowDiT (Version A main) =="
    _run_on_gpu train_physics "${TRAIN_GPU_FREE_MIN_MB}" "${TRAIN_GPU_MAX_UTIL}" \
      python minimal_world2wam/train/train_physics_world2wam.py \
        --config "${CONFIG}" \
        --cache_dir "${CACHE_OUT}" \
        --output_dir "${OUT_DIR}" \
        --device cuda
  fi
else
  _log "Skip train (SKIP_TRAIN=1)"
fi

if [[ ! -f "${CKPT}" ]]; then
  _log "ERROR: missing checkpoint ${CKPT}"
  exit 1
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  _log "== offline eval (latent verification) =="
  _run_on_gpu eval_offline "${EVAL_GPU_FREE_MIN_MB}" "${EVAL_GPU_MAX_UTIL}" \
    python minimal_world2wam/eval/eval_offline_cache_only.py \
      --config "${CONFIG}" \
      --cache_dir "${CACHE_OUT}" \
      --adapter_ckpt "${CKPT}" \
      --use_physics true \
      --device cuda \
      --output experiments/eval_offline_physics_flow_dit_main.json

  _log "== LIBERO baseline (FastWAM official) ${MAX_TASKS}x${NUM_TRIALS} =="
  _run_on_gpu eval_baseline "${EVAL_GPU_FREE_MIN_MB}" "${EVAL_GPU_MAX_UTIL}" \
    python minimal_world2wam/eval/eval_libero_world2wam.py \
      --config "${CONFIG}" \
      --mode baseline \
      --max_tasks "${MAX_TASKS}" \
      --num_trials "${NUM_TRIALS}" \
      --device cuda \
      --output experiments/eval_baseline_version_a_main.json

  _log "== LIBERO ours_onestep_physics_flow_dit ${MAX_TASKS}x${NUM_TRIALS} =="
  _run_on_gpu eval_ours_physics "${EVAL_GPU_FREE_MIN_MB}" "${EVAL_GPU_MAX_UTIL}" \
    python minimal_world2wam/eval/eval_libero_world2wam.py \
      --config "${CONFIG}" \
      --mode ours_onestep_physics_flow_dit \
      --adapter_ckpt "${CKPT}" \
      --flow_sample_steps "${FLOW_SAMPLE_STEPS}" \
      --max_tasks "${MAX_TASKS}" \
      --num_trials "${NUM_TRIALS}" \
      --device cuda \
      --output experiments/eval_ours_onestep_physics_flow_dit_main.json

  _log "== LIBERO ours_onestep_flow_dit (no physics ablation) ${MAX_TASKS}x${NUM_TRIALS} =="
  _run_on_gpu eval_ours_flow_dit "${EVAL_GPU_FREE_MIN_MB}" "${EVAL_GPU_MAX_UTIL}" \
    python minimal_world2wam/eval/eval_libero_world2wam.py \
      --config "${CONFIG}" \
      --mode ours_onestep_flow_dit \
      --adapter_ckpt "${CKPT}" \
      --flow_sample_steps "${FLOW_SAMPLE_STEPS}" \
      --max_tasks "${MAX_TASKS}" \
      --num_trials "${NUM_TRIALS}" \
      --device cuda \
      --output experiments/eval_ours_onestep_flow_dit_main.json

  _log "== summarize results =="
  python minimal_world2wam/scripts/summarize_version_a_results.py \
    --workspace "${WORKSPACE}" \
    --output experiments/VERSION_A_SUMMARY.json 2>&1 | tee -a "${MAIN_LOG}"
else
  _log "Skip eval (SKIP_EVAL=1)"
fi

_log "======== Version A full pipeline DONE ========"
