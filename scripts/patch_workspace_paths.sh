#!/usr/bin/env bash
# Patch deploy config for a given workspace root (local or remote).
set -euo pipefail

WORKSPACE="${1:?Usage: patch_workspace_paths.sh /DATA/disk0/yjh/world2wam}"
REPO="${WORKSPACE}/Physics-Aligned World2WAM"
DEPLOY_TMPL="${REPO}/configs/world2wam_physics_flow_dit_deploy.yaml"
OUT_CFG="${WORKSPACE}/configs/world2wam_physics_flow_dit_main.yaml"
OUT_MAIN="${REPO}/configs/world2wam_physics_flow_dit_main.yaml"

mkdir -p "${WORKSPACE}/configs"
sed "s|__WORKSPACE__|${WORKSPACE}|g" "${DEPLOY_TMPL}" > "${OUT_CFG}"
cp "${OUT_CFG}" "${OUT_MAIN}"

# Version A main yaml absolute paths (legacy copy)
if [[ -f "${REPO}/configs/world2wam_physics_flow_dit_main.yaml" ]]; then
  sed -i "s|/DATA/disk0/jianhua|${WORKSPACE}|g" \
    "${REPO}/configs/world2wam_libero_spatial_h10_paper.yaml" \
    "${REPO}/configs/world2wam_libero_spatial_h10_physics_flow_dit_v1.yaml" 2>/dev/null || true
fi

ln -sfn "${REPO}" "${WORKSPACE}/minimal_world2wam"

echo "Patched configs for WORKSPACE=${WORKSPACE}"
echo "  main config -> ${OUT_CFG}"
