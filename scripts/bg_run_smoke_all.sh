#!/usr/bin/env bash
# Full smoke validation: framework tiers + quick policy + LIBERO sim + ablation + compare.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="${ROOT}/experiments/bg_jobs/smoke_all.log"
mkdir -p "${ROOT}/experiments/bg_jobs"

exec > >(tee -a "${LOG}") 2>&1
echo "======== smoke_all started $(date -Iseconds) ========"

if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/libero_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}"
export USE_TMUX=0
export NUM_TRIALS="${NUM_TRIALS:-5}"
export TASK_LIMIT="${TASK_LIMIT:-2}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_TRIALS=${NUM_TRIALS} TASK_LIMIT=${TASK_LIMIT}"

echo ""
if [[ "${SKIP_FRAMEWORK:-0}" == "1" ]]; then
  echo "======== [1/6] SKIP framework smoke (SKIP_FRAMEWORK=1) ========"
else
  echo "======== [1/6] Framework smoke (tier 0-3) ========"
  bash "${ROOT}/scripts/smoke_test_framework.sh"
fi

echo ""
echo "======== [1/6] Quick Policy train (20 steps, LoRA smoke) ========"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}"
if [[ "${SKIP_POLICY:-0}" == "1" && -f "${ROOT}/experiments/world2wam_policy_improve/checkpoints/world2wam_final.pt" ]]; then
  echo "SKIP policy train (SKIP_POLICY=1)"
else
  bash "${ROOT}/scripts/07_train_world2wam_policy_improve.sh" --max-steps 20
fi

POLICY_BUNDLE="${ROOT}/experiments/world2wam_policy_improve/checkpoints/world2wam_final.pt"
if [[ ! -f "${POLICY_BUNDLE}" ]]; then
  echo "ERROR: missing ${POLICY_BUNDLE} after quick policy train"
  exit 1
fi
echo "Policy bundle OK: ${POLICY_BUNDLE}"

echo ""
echo "======== [3/6] Export merged checkpoint ========"
WORLD2WAM_BUNDLE="${POLICY_BUNDLE}" EXPORT_TAG="smoke" \
  bash "${ROOT}/scripts/08_export_libero_checkpoint.sh"
MERGED_CKPT="${ROOT}/experiments/exported_ckpts/world2wam_merged_smoke.pt"
if [[ ! -f "${MERGED_CKPT}" ]]; then
  MERGED_CKPT="$(ls -t "${ROOT}/experiments/exported_ckpts/"world2wam_merged_*.pt 2>/dev/null | head -1)"
fi
echo "Merged ckpt: ${MERGED_CKPT}"

echo ""
echo "======== [4/6] LIBERO sim smoke — official baseline ========"
CKPT="" OUTPUT_DIR="${ROOT}/experiments/libero_eval/smoke_official" RUN_TAG="smoke_official" \
  bash "${ROOT}/scripts/run_libero_spatial_success.sh"

echo ""
echo "======== [5/6] LIBERO sim smoke — World2WAM (merged) ========"
CKPT="${MERGED_CKPT}" OUTPUT_DIR="${ROOT}/experiments/libero_eval/smoke_world2wam" RUN_TAG="smoke_world2wam" \
  bash "${ROOT}/scripts/run_libero_spatial_success.sh"

echo ""
echo "======== [6/6] Ablation sweep smoke + compare ========"
SMOKE=1 bash "${ROOT}/scripts/sweep_bidirectional_ablations.sh"

WORLD2WAM_BUNDLE="${POLICY_BUNDLE}" RUN_TAG="smoke_compare" \
  OUTPUT_DIR="${ROOT}/experiments/libero_eval/smoke_compare" \
  NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" \
  bash "${ROOT}/scripts/run_compare_libero_success.sh" || {
  echo "WARN: compare step failed (see log); continuing."
}

echo ""
echo "======== smoke_all finished $(date -Iseconds) ========"
echo "Logs: ${LOG}"
echo "Official summary: ${ROOT}/experiments/libero_eval/smoke_official/summary.txt"
echo "World2WAM summary: ${ROOT}/experiments/libero_eval/smoke_world2wam/summary.txt"
echo "Compare JSON: ${ROOT}/experiments/libero_eval/smoke_compare/compare_summary.json"
echo "Ablations: ${ROOT}/experiments/ablations/summary.csv"
