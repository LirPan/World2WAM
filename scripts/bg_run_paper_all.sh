#!/usr/bin/env bash
# Full paper pipeline (FULL_RUN=1): policy train -> export -> LIBERO compare -> ablations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG="${ROOT}/experiments/bg_jobs/paper_all.log"
mkdir -p "${ROOT}/experiments/bg_jobs"

exec > >(tee -a "${LOG}") 2>&1
echo "======== paper_all started $(date -Iseconds) ========"

if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/activate_env.sh"

# GPU 6/7 have ~60GB free; avoid 0-3 (occupied ~58GB each).
export FULL_RUN=1
export USE_TMUX=0
export MAX_TASKS_PER_GPU=1
export CUDA_DEVICE="${CUDA_DEVICE:-6}"
export CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
export NUM_GPUS="${NUM_GPUS:-2}"
export NUM_TRIALS="${NUM_TRIALS:-50}"

echo "FULL_RUN=${FULL_RUN} CUDA_DEVICE=${CUDA_DEVICE} CUDA_DEVICES=${CUDA_DEVICES} NUM_GPUS=${NUM_GPUS}"

bash "${ROOT}/scripts/run_paper_libero_all.sh"

echo "======== paper_all finished $(date -Iseconds) ========"
