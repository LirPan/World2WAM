#!/usr/bin/env bash
# Bidirectional ablation sweep (validate 3 modes, then full grid).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="${ROOT}/experiments/bg_jobs/ablations.log"
mkdir -p "${ROOT}/experiments/bg_jobs"

exec > >(tee -a "${LOG}") 2>&1
echo "======== ablations started $(date -Iseconds) ========"

if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

if [[ -z "${CUDA_DEVICE:-}" && -z "${ABLATION_DEVICE:-}" ]]; then
  CUDA_DEVICE="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '$2 >= 18000 {gsub(/ /,"",$1); print $1; exit}')"
fi

if [[ -n "${CUDA_DEVICE:-}" ]]; then
  export CUDA_DEVICE
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  export ABLATION_DEVICE="${ABLATION_DEVICE:-cuda}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  echo "CUDA_DEVICE=${CUDA_DEVICE} ABLATION_DEVICE=cuda"
else
  export ABLATION_DEVICE="${ABLATION_DEVICE:-cpu}"
  export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
  unset CUDA_VISIBLE_DEVICES
  echo "ABLATION_DEVICE=cpu MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} (no GPU with >=18GB free)"
fi
export ABLATION_DEVICE

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
  echo "======== [1/2] Validate forward_only + bidirectional + cycle ========"
  VALIDATE=1 bash "${ROOT}/scripts/sweep_bidirectional_ablations.sh"
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "VALIDATE_ONLY=1: skipping full grid."
  echo "======== ablations finished $(date -Iseconds) ========"
  exit 0
fi

echo "======== [2/2] Full ablation grid (mode x lambda x horizon) ========"
# Default HORIZONS=1 if unset: h=2/4 need separate full precompute (~53k samples each).
export HORIZONS="${HORIZONS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
echo "HORIZONS=${HORIZONS} BATCH_SIZE=${BATCH_SIZE}"
bash "${ROOT}/scripts/sweep_bidirectional_ablations.sh"

echo "======== ablations finished $(date -Iseconds) ========"
echo "Summary: ${ROOT}/experiments/ablations/summary.csv"
