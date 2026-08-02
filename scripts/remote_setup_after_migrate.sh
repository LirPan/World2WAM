#!/usr/bin/env bash
# Run on REMOTE server after rsync migration.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/yjh/world2wam}"
REPO="${WORKSPACE}/Physics-Aligned World2WAM"
cd "${WORKSPACE}"

echo "== symlinks =="
ln -sfn "${REPO}" minimal_world2wam
mkdir -p configs cache experiments cache/bg_jobs

echo "== patch config paths for ${WORKSPACE} =="
bash "${REPO}/scripts/patch_workspace_paths.sh" "${WORKSPACE}"

echo "== conda env =="
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx world2wam; then
    echo "world2wam env already exists — skip create"
  else
    conda env create -f "${REPO}/scripts/world2wam_env.yaml" || {
      echo "WARN: conda create failed — install miniconda/anaconda and retry:"
      echo "  conda env create -f ${REPO}/scripts/world2wam_env.yaml"
    }
  fi
else
  echo "WARN: conda not found. Install miniconda, then:"
  echo "  conda env create -f ${REPO}/scripts/world2wam_env.yaml"
fi

export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
export MUJOCO_GL=egl

echo "== smoke test =="
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate world2wam 2>/dev/null || true
bash minimal_world2wam/scripts/smoke_test.sh

echo ""
echo "== Ready. Version A full pipeline =="
echo "  export WORKSPACE=${WORKSPACE}"
echo "  bash minimal_world2wam/scripts/poll_gpu_version_a.sh start"
echo ""
echo "REMOTE SETUP OK"
