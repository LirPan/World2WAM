#!/usr/bin/env bash
# Rsync Version A + Version B full-training assets to remote server.
#
# Usage (from source machine):
#   bash minimal_world2wam/scripts/migrate_to_remote.sh check          # dry-run sizes
#   bash minimal_world2wam/scripts/migrate_to_remote.sh tier1        # code+weights+sim+env (~72G, no cache)
#   bash minimal_world2wam/scripts/migrate_to_remote.sh tier2        # tier1 + Version A cache 300k (~660G)
#   bash minimal_world2wam/scripts/migrate_to_remote.sh tier3        # tier2 + raw LIBERO data for Version B precompute
#   bash minimal_world2wam/scripts/migrate_to_remote.sh all          # everything listed below
#
# Env overrides:
#   REMOTE_USER_HOST  default yjh@120.92.211.106
#   REMOTE_PORT       default 22
#   REMOTE_BASE       default /DATA/disk0/yjh/world2wam  (must exist on remote)
#   LOCAL_BASE        default /DATA/disk0/jianhua
#
# Background (survives Cursor logout):
#   bash minimal_world2wam/scripts/migrate_to_remote_bg.sh start tier3
#   bash minimal_world2wam/scripts/migrate_to_remote_bg.sh status
set -euo pipefail

REMOTE_USER_HOST="${REMOTE_USER_HOST:-yjh@120.92.211.106}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_BASE="${REMOTE_BASE:-/DATA/disk0/yjh/world2wam}"
LOCAL_BASE="${LOCAL_BASE:-/DATA/disk0/jianhua}"

RSYNC_OPTS=(-avz --partial --info=progress2 -e "ssh -p ${REMOTE_PORT} -o ServerAliveInterval=30 -o ServerAliveCountMax=120 -o TCPKeepAlive=yes")

PLR="${LOCAL_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code"
FASTWAM="${PLR}/FastWAM"

_rsync() {
  local label="$1"
  shift
  echo ""
  echo "======== ${label} ========"
  rsync "${RSYNC_OPTS[@]}" "$@"
}

_remote_mkdir() {
  ssh -p "${REMOTE_PORT}" "${REMOTE_USER_HOST}" "mkdir -p ${REMOTE_BASE}/{configs,cache,experiments,plr/yjh_space_backup_20250602/idea2_workspace/code}"
}

sync_code() {
  _rsync "1/7 Project code (Version A + Version B)" \
    "${LOCAL_BASE}/Physics-Aligned World2WAM/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/Physics-Aligned World2WAM/" \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.git'
}

sync_configs_link() {
  _rsync "2/7 Workspace configs" \
    "${LOCAL_BASE}/configs/world2wam_physics_flow_dit_main.yaml" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/configs/" 2>/dev/null || true
}

sync_fastwam_code() {
  _rsync "3a/7 FastWAM code" \
    "${FASTWAM}/src" "${FASTWAM}/configs" "${FASTWAM}/experiments" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/"
}

sync_fastwam_weights() {
  _rsync "3b/7 FastWAM weights (~62G)" \
    "${FASTWAM}/checkpoints/fastwam_release/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/fastwam_release/"
  _rsync "3c/7 Wan VAE+T5 (~27G)" \
    "${FASTWAM}/checkpoints/DiffSynth-Studio/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/DiffSynth-Studio/"
  _rsync "3d/7 Wan2.2 backbone (~19G)" \
    "${FASTWAM}/checkpoints/Wan-AI/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/Wan-AI/"
  _rsync "3e/7 ActionDiT pretrained (~3.9G)" \
    "${FASTWAM}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/"
}

sync_libero_sim() {
  _rsync "4/7 LIBERO sim (Version A eval)" \
    "${PLR}/LIBERO_fresh/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO_fresh/"
  _rsync "4b/7 LIBERO (Version B config path)" \
    "${PLR}/LIBERO/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO/"
  _rsync "4c/7 MuJoCo deps (robosuite/bddl)" \
    "${LOCAL_BASE}/cache/src/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/cache/src/"
}

sync_version_a_cache() {
  _rsync "5/7 Version A latent cache 300k (~588G) — LONG" \
    "${LOCAL_BASE}/cache/libero_spatial_h10_full_fastwam/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/cache/libero_spatial_h10_full_fastwam/"
}

sync_raw_lerobot() {
  _rsync "6/7 Raw LIBERO LeRobot data for Version B precompute (~1.9G)" \
    "${FASTWAM}/data/libero_mujoco3.3.2/" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/data/libero_mujoco3.3.2/"
}

sync_conda_env() {
  _rsync "7/7 Conda env export (recreate on remote)" \
    "${LOCAL_BASE}/Physics-Aligned World2WAM/scripts/world2wam_env.yaml" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/Physics-Aligned World2WAM/scripts/"
  _rsync "7b/7 remote setup script" \
    "${LOCAL_BASE}/Physics-Aligned World2WAM/scripts/remote_setup_after_migrate.sh" \
    "${REMOTE_USER_HOST}:${REMOTE_BASE}/Physics-Aligned World2WAM/scripts/"
}

cmd_check() {
  echo "REMOTE=${REMOTE_USER_HOST}:${REMOTE_BASE} (port ${REMOTE_PORT})"
  du -sh \
    "${LOCAL_BASE}/Physics-Aligned World2WAM" \
    "${LOCAL_BASE}/cache/libero_spatial_h10_full_fastwam" \
    "${FASTWAM}/checkpoints/fastwam_release" \
    "${FASTWAM}/checkpoints/DiffSynth-Studio" \
    "${FASTWAM}/checkpoints/Wan-AI" \
    "${FASTWAM}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" \
    "${FASTWAM}/src" "${FASTWAM}/data/libero_mujoco3.3.2" \
    "${PLR}/LIBERO_fresh" "${PLR}/LIBERO" \
    "${LOCAL_BASE}/cache/src" \
    "${LOCAL_BASE}/miniconda3/envs/world2wam" 2>/dev/null || true
}

tier1() { _remote_mkdir; sync_code; sync_configs_link; sync_fastwam_code; sync_fastwam_weights; sync_libero_sim; sync_conda_env; }
tier2() { tier1; sync_version_a_cache; }
tier3() { tier2; sync_raw_lerobot; }

case "${1:-check}" in
  check) cmd_check ;;
  tier1) tier1; echo "Done tier1 (~72G without cache)" ;;
  tier2) tier2; echo "Done tier2 (~660G with Version A cache)" ;;
  tier3|all) tier3; echo "Done tier3/all" ;;
  code) _remote_mkdir; sync_code; sync_configs_link; sync_conda_env ;;
  cache) _remote_mkdir; sync_version_a_cache ;;
  weights) _remote_mkdir; sync_fastwam_weights ;;
  data) _remote_mkdir; sync_raw_lerobot ;;
  *)
    echo "Usage: $0 {check|tier1|tier2|tier3|all|code|cache|weights|data}"
    exit 1
    ;;
esac
