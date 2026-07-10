#!/usr/bin/env bash
# Run on REMOTE server after rsync migration.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/yjh/world2wam}"
cd "${WORKSPACE}"

echo "== symlinks =="
ln -sfn "Physics-Aligned World2WAM" minimal_world2wam
mkdir -p configs
ln -sfn "${WORKSPACE}/Physics-Aligned World2WAM/configs/world2wam_physics_flow_dit_main.yaml" \
  configs/world2wam_physics_flow_dit_main.yaml

echo "== conda env =="
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx world2wam; then
    echo "world2wam env already exists"
  else
    conda env create -f "${WORKSPACE}/Physics-Aligned World2WAM/scripts/world2wam_env.yaml" || \
      echo "WARN: conda create failed — install miniconda and retry"
  fi
else
  echo "WARN: conda not found. Install miniconda, then:"
  echo "  conda env create -f Physics-Aligned World2WAM/scripts/world2wam_env.yaml"
fi

export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
export MUJOCO_GL=egl

echo "== smoke test =="
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate world2wam 2>/dev/null || true
bash minimal_world2wam/scripts/smoke_test.sh

echo ""
echo "== Version A: start full pipeline =="
echo "  bash minimal_world2wam/scripts/poll_gpu_version_a.sh start"
echo ""
echo "== Version B: precompute future latents then train =="
echo "  cd version_b && python scripts/01_precompute_future_latents.py --config configs/precompute_latents.yaml"
echo "  python scripts/train_physics_mot.py --config configs/physics_mot_train.yaml"
echo ""
echo "REMOTE SETUP OK"
