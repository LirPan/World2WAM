#!/usr/bin/env bash
# Deploy Version A + Version B on THIS machine (current server).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/jianhua}"
REPO="${WORKSPACE}/Physics-Aligned World2WAM"
export WORKSPACE

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
cd "${WORKSPACE}"

echo "== [local] patch paths =="
bash "${REPO}/scripts/patch_workspace_paths.sh" "${WORKSPACE}"

echo "== [local] smoke test =="
bash minimal_world2wam/scripts/smoke_test.sh

echo "== [local] Version A GPU poll (background) =="
bash minimal_world2wam/scripts/poll_gpu_version_a.sh start || true

echo "== [local] Version B poll (background) =="
bash minimal_world2wam/scripts/poll_gpu_version_b.sh start || true

echo "Local deploy done. Check:"
echo "  bash minimal_world2wam/scripts/poll_gpu_version_a.sh status"
echo "  bash minimal_world2wam/scripts/poll_gpu_version_b.sh status"
