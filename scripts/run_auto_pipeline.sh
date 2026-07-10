#!/usr/bin/env bash
# Full World2WAM experiment with per-stage GPU/CPU auto-pick and precompute GPU upgrade.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
CONFIG="${CONFIG:-configs/world2wam_libero_spatial_h10.yaml}"
CACHE_OUT="${CACHE_OUT:-cache/debug_libero_spatial_h10}"
PRE_MAX="${PRE_MAX:-100}"
MAX_TASKS="${MAX_TASKS:-1}"
NUM_TRIALS="${NUM_TRIALS:-1}"
GPU_UPGRADE_POLL_SEC="${GPU_UPGRADE_POLL_SEC:-120}"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
if [[ -f /DATA/disk0/jianhua/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk0/jianhua/use_proxy.sh || true
fi
# shellcheck disable=SC1091
source "${WORKSPACE}/minimal_world2wam/scripts/device_utils.sh"

cd "${WORKSPACE}"
mkdir -p "${CACHE_OUT}" experiments

_run_precompute() {
  local device="$1"
  python minimal_world2wam/cache/precompute_fastwam_latents.py \
    --config "${CONFIG}" \
    --output "${CACHE_OUT}" \
    --max_samples "${PRE_MAX}" \
    --device "${device}" \
    --resume
}

run_precompute_with_gpu_upgrade() {
  local device pid new_dev
  device="$(pick_compute_device)"
  log_device_pick "precompute" "${device}"

  if [[ "${device}" == "cuda" ]]; then
    _run_precompute cuda
    return 0
  fi

  echo "[precompute] GPU busy -> start on CPU; will upgrade to GPU when free (poll=${GPU_UPGRADE_POLL_SEC}s)"
  _run_precompute cpu &
  pid=$!

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${GPU_UPGRADE_POLL_SEC}"
    new_dev="$(pick_compute_device)"
    if [[ "${new_dev}" == "cuda" ]]; then
      echo "[precompute] GPU now free -> upgrading to CUDA (resume)"
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      device=cuda
      log_device_pick "precompute_upgrade" "${device}"
      _run_precompute cuda &
      pid=$!
    fi
  done
  wait "${pid}"
}

run_train_heads() {
  local device
  device="$(pick_compute_device)"
  log_device_pick "train_heads" "${device}"
  python minimal_world2wam/train/train_world2wam_heads.py \
    --config "${CONFIG}" \
    --cache_dir "${CACHE_OUT}" \
    --use_fwd true --use_inv true --use_cycle true \
    --device "${device}"
}

run_train_adapter() {
  local device
  device="$(pick_compute_device)"
  log_device_pick "train_adapter" "${device}"
  python minimal_world2wam/train/train_world2wam_adapter.py \
    --config "${CONFIG}" \
    --cache_dir "${CACHE_OUT}" \
    --use_act true --use_fwd true --use_inv true --use_cycle true \
    --device "${device}"
}

echo "== run_auto_pipeline: $(date -Iseconds) =="
echo "CONFIG=${CONFIG} CACHE_OUT=${CACHE_OUT} PRE_MAX=${PRE_MAX}"

echo "== smoke test =="
bash minimal_world2wam/scripts/smoke_test.sh

echo "== precompute =="
run_precompute_with_gpu_upgrade

echo "== train heads =="
run_train_heads

echo "== train adapter =="
run_train_adapter

echo "== eval compare (offline + baseline + ours) =="
EVAL_DEVICE="$(pick_compute_device)"
log_device_pick "eval_compare" "${EVAL_DEVICE}"
EVAL_DEVICE="${EVAL_DEVICE}" MAX_TASKS="${MAX_TASKS}" NUM_TRIALS="${NUM_TRIALS}" \
  CACHE_OUT="${CACHE_OUT}" CONFIG="${CONFIG}" \
  bash minimal_world2wam/scripts/bg_launch.sh eval_compare

echo "== run_auto_pipeline done: $(date -Iseconds) =="
