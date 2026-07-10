#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/jianhua}"
VB_ROOT="${WORKSPACE}/Physics-Aligned World2WAM/version_b"
LOG="${WORKSPACE}/cache/bg_jobs/version_b_full.log"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${VB_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
cd "${VB_ROOT}"

_log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

# shellcheck disable=SC1091
source "${WORKSPACE}/minimal_world2wam/scripts/gpu_poll_utils.sh"

_log "Version B pipeline start WORKSPACE=${WORKSPACE}"

if [[ "${USE_GPU_POLL:-1}" == "1" ]]; then
  run_stage_on_gpu precompute_vb 70000 15 \
    python scripts/01_precompute_future_latents.py --config configs/precompute_latents.yaml
  run_stage_on_gpu train_vb 45000 25 \
    python scripts/train_physics_mot.py --config configs/physics_mot_train.yaml
else
  python scripts/01_precompute_future_latents.py --config configs/precompute_latents.yaml
  python scripts/train_physics_mot.py --config configs/physics_mot_train.yaml
fi

_log "Version B pipeline DONE"
