#!/usr/bin/env bash
# Export full-policy merged ckpt, then official vs World2WAM LIBERO compare (FULL_RUN).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="${ROOT}/experiments/bg_jobs/export_compare.log"
mkdir -p "${ROOT}/experiments/bg_jobs"

exec > >(tee -a "${LOG}") 2>&1
echo "======== export_compare started $(date -Iseconds) ========"

if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

BUNDLE="${ROOT}/experiments/world2wam_policy_improve_full/checkpoints/world2wam_final.pt"
if [[ ! -f "${BUNDLE}" ]]; then
  echo "ERROR: missing ${BUNDLE}"
  exit 1
fi

# Eval GPU(s) chosen after CPU export completes.
export FULL_RUN=1
export NUM_TRIALS=50
export RUN_TAG="${RUN_TAG:-policy_full_compare}"
export WORLD2WAM_BUNDLE="${BUNDLE}"

echo "FULL_RUN=${FULL_RUN} RUN_TAG=${RUN_TAG}"
echo "Bundle: ${BUNDLE}"

echo "======== [1/2] Export merged checkpoint (CPU, avoids GPU contention) ========"
MERGED="${ROOT}/experiments/exported_ckpts/world2wam_merged_policy_full.pt"
if [[ -f "${MERGED}" ]]; then
  echo "SKIP export: ${MERGED} already exists ($(du -h "${MERGED}" | cut -f1))"
else
  unset CUDA_VISIBLE_DEVICES
  EXPORT_DEVICE=cpu CONFIG=configs/world2wam_policy_improve_full.yaml EXPORT_TAG=policy_full \
    WORLD2WAM_BUNDLE="${BUNDLE}" bash "${ROOT}/scripts/08_export_libero_checkpoint.sh"
fi
if [[ ! -f "${MERGED}" ]]; then
  echo "ERROR: export failed, missing ${MERGED}"
  exit 1
fi
echo "Merged ckpt OK: ${MERGED} ($(du -h "${MERGED}" | cut -f1))"

# Re-pick GPUs after export (most free memory).
if [[ -z "${EVAL_CUDA_DEVICES:-}" ]]; then
  EVAL_CUDA_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -2 | cut -d',' -f1 | tr '\n' ',' | sed 's/,$//')"
fi
export CUDA_DEVICES="${EVAL_CUDA_DEVICES}"
export USE_TMUX=0
export MAX_TASKS_PER_GPU=1

echo "======== [2/2] LIBERO compare (official vs World2WAM) on GPU ${CUDA_DEVICES} ========"
bash "${ROOT}/scripts/run_compare_libero_success.sh"

echo "======== export_compare finished $(date -Iseconds) ========"
echo "Compare JSON: ${ROOT}/experiments/libero_eval/${RUN_TAG}/compare_summary.json"
