#!/usr/bin/env bash
# Compare LIBERO sim success: official FastWAM vs World2WAM (merged policy ckpt).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
OFFICIAL_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"

RUN_TAG="${RUN_TAG:-compare_$(date +%Y%m%d_%H%M%S)}"
NUM_TRIALS="${NUM_TRIALS:-5}"
TASK_LIMIT="${TASK_LIMIT:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-2}"
USE_TMUX="${USE_TMUX:-0}"
FULL_RUN="${FULL_RUN:-0}"

WORLD2WAM_BUNDLE="${WORLD2WAM_BUNDLE:-${ROOT}/experiments/world2wam_policy_improve_full/checkpoints/world2wam_final.pt}"
COMPARE_ROOT="${ROOT}/experiments/libero_eval/${RUN_TAG}"

if [[ "${FULL_RUN}" == "1" ]]; then
  NUM_TRIALS="${NUM_TRIALS:-50}"
  TASK_LIMIT=""
  USE_TMUX="${USE_TMUX:-1}"
  CUDA_DEVICES="${CUDA_DEVICES:-2,3,6,7}"
fi

mkdir -p "${COMPARE_ROOT}"

echo "==> [A] Official baseline LIBERO eval..."
CKPT="${OFFICIAL_CKPT}" WORLD2WAM_BUNDLE="" OUTPUT_DIR="${COMPARE_ROOT}/official" \
  NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" \
  CUDA_DEVICES="${CUDA_DEVICES}" USE_TMUX="${USE_TMUX}" RUN_TAG="${RUN_TAG}_official" \
  bash "${ROOT}/scripts/run_libero_spatial_success.sh"

echo "==> [B] World2WAM merged LIBERO eval..."
MERGED_CKPT="${ROOT}/experiments/exported_ckpts/world2wam_merged_policy_full.pt"
if [[ -f "${MERGED_CKPT}" ]]; then
  CKPT="${MERGED_CKPT}" WORLD2WAM_BUNDLE="" OUTPUT_DIR="${COMPARE_ROOT}/world2wam" \
    NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" \
    CUDA_DEVICES="${CUDA_DEVICES}" USE_TMUX="${USE_TMUX}" RUN_TAG="${RUN_TAG}_world2wam" \
    bash "${ROOT}/scripts/run_libero_spatial_success.sh"
else
  WORLD2WAM_BUNDLE="${WORLD2WAM_BUNDLE}" CKPT="" OUTPUT_DIR="${COMPARE_ROOT}/world2wam" \
    NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" \
    CUDA_DEVICES="${CUDA_DEVICES}" USE_TMUX="${USE_TMUX}" RUN_TAG="${RUN_TAG}_world2wam" \
    EXPORT_DEVICE=cpu EXPORT_TAG="${RUN_TAG}" bash "${ROOT}/scripts/run_libero_spatial_success.sh"
fi

OFFICIAL_DIR="${COMPARE_ROOT}/official"
WORLD2WAM_DIR="${COMPARE_ROOT}/world2wam"

if [[ ! -d "${OFFICIAL_DIR}" || ! -d "${WORLD2WAM_DIR}" ]]; then
  echo "Could not locate eval output dirs: ${OFFICIAL_DIR} ${WORLD2WAM_DIR}"
  exit 1
fi

echo "==> Summarize compare..."
python -m src.eval.summarize_libero_compare \
  --official-dir "${OFFICIAL_DIR}" \
  --world2wam-dir "${WORLD2WAM_DIR}" \
  --output "${COMPARE_ROOT}/compare_summary.json" | tee "${COMPARE_ROOT}/compare_summary.txt"

echo "==> Done: ${COMPARE_ROOT}/compare_summary.json"
