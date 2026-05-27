#!/usr/bin/env bash
# Source this file: source scripts/activate_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/DATA/disk1/yjh_space/idea2_workspace/code/FastWAM/checkpoints}"
echo "Activated ${WORLD2WAM_CONDA_ENV:-world2wam}"
echo "DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}"
