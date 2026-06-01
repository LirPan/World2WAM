#!/usr/bin/env bash
# Policy full training only (no smoke / no LIBERO eval).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="${ROOT}/experiments/bg_jobs/policy_train.log"
mkdir -p "${ROOT}/experiments/bg_jobs"

exec > >(tee -a "${LOG}") 2>&1
echo "======== policy_train started $(date -Iseconds) ========"

if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

# Pick GPU with most free memory if not set.
if [[ -z "${CUDA_DEVICE:-}" ]]; then
  CUDA_DEVICE="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')"
fi
export CUDA_DEVICE
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
echo "Using CUDA_DEVICE=${CUDA_DEVICE}"

SKIP_LIBERO=1 bash "${ROOT}/scripts/run_full_pipeline_policy.sh"

echo "======== policy_train finished $(date -Iseconds) ========"
echo "Checkpoints: ${ROOT}/experiments/world2wam_policy_improve_full/checkpoints/"
