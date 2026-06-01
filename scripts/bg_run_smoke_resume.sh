#!/usr/bin/env bash
# Resume smoke from step 4 (LIBERO sim) when policy train + export already done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="${ROOT}/experiments/bg_jobs/smoke_resume.log"
mkdir -p "${ROOT}/experiments/bg_jobs"

exec > >(tee -a "${LOG}") 2>&1
echo "======== smoke_resume started $(date -Iseconds) ========"

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/libero_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}"
export USE_TMUX=0
export NUM_TRIALS="${NUM_TRIALS:-5}"
export TASK_LIMIT="${TASK_LIMIT:-2}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"

POLICY_BUNDLE="${ROOT}/experiments/world2wam_policy_improve/checkpoints/world2wam_final.pt"
MERGED_CKPT="${ROOT}/experiments/exported_ckpts/world2wam_merged_smoke.pt"

if [[ ! -f "${MERGED_CKPT}" ]]; then
  echo "Missing merged ckpt; running export..."
  WORLD2WAM_BUNDLE="${POLICY_BUNDLE}" EXPORT_TAG="smoke" \
    bash "${ROOT}/scripts/08_export_libero_checkpoint.sh"
fi

echo "======== [4/6] LIBERO sim — official baseline ========"
CKPT="" OUTPUT_DIR="${ROOT}/experiments/libero_eval/smoke_official" RUN_TAG="smoke_official" \
  bash "${ROOT}/scripts/run_libero_spatial_success.sh"

echo "======== [5/6] LIBERO sim — World2WAM merged ========"
CKPT="${MERGED_CKPT}" OUTPUT_DIR="${ROOT}/experiments/libero_eval/smoke_world2wam" RUN_TAG="smoke_world2wam" \
  bash "${ROOT}/scripts/run_libero_spatial_success.sh"

echo "======== [6/6] Ablation smoke + compare ========"
SMOKE=1 bash "${ROOT}/scripts/sweep_bidirectional_ablations.sh"

WORLD2WAM_BUNDLE="${POLICY_BUNDLE}" RUN_TAG="smoke_compare" \
  OUTPUT_DIR="${ROOT}/experiments/libero_eval/smoke_compare" \
  NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" \
  bash "${ROOT}/scripts/run_compare_libero_success.sh"

echo "======== smoke_resume finished $(date -Iseconds) ========"
