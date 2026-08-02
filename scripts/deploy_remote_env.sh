#!/usr/bin/env bash
# Deploy Version A runtime environment to remote server.
#
# tier1 (~72G): code + FastWAM weights + LIBERO sim + conda env yaml
# +data (~1.9G): raw LIBERO dataset for precompute on remote
# +cache (~658G): optional latent cache (SYNC_CACHE=1)
#
# Direct SSH:
#   bash scripts/deploy_remote_env.sh
#
# Via laptop reverse tunnel (this server cannot reach 120.92.211.106 directly):
#   [laptop] FIVEAGES_HOST=<fiveages-ip> bash scripts/local_reverse_tunnel.sh
#   [this server] bash scripts/deploy_remote_env.sh tunnel
#
# Env: REMOTE_USER_HOST REMOTE_PORT REMOTE_BASE LOCAL_BASE SYNC_DATA SYNC_CACHE
set -euo pipefail

LOCAL_BASE="${LOCAL_BASE:-/DATA/disk0/jianhua}"
REMOTE_USER_HOST="${REMOTE_USER_HOST:-yjh@120.92.211.106}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_BASE="${REMOTE_BASE:-/DATA/disk0/yjh/world2wam}"
SYNC_DATA="${SYNC_DATA:-1}"
SYNC_CACHE="${SYNC_CACHE:-0}"

REPO="${LOCAL_BASE}/Physics-Aligned World2WAM"
MIGRATE="${REPO}/scripts/migrate_to_remote.sh"

export REMOTE_USER_HOST REMOTE_PORT REMOTE_BASE LOCAL_BASE

_ssh() {
  ssh -p "${REMOTE_PORT}" -o ConnectTimeout=15 -o ServerAliveInterval=30 \
    -o StrictHostKeyChecking=accept-new "${REMOTE_USER_HOST}" "$@"
}

_preflight() {
  echo "== Preflight =="
  echo "Local:  ${LOCAL_BASE}"
  echo "Remote: ${REMOTE_USER_HOST}:${REMOTE_BASE} (port ${REMOTE_PORT})"
  bash "${MIGRATE}" check | head -15
  echo ""
  if ! _ssh "echo SSH_OK; df -h $(dirname "${REMOTE_BASE}") 2>/dev/null | tail -1; echo -n 'GPUs: '; nvidia-smi -L 2>/dev/null | wc -l"; then
    echo ""
    echo "SSH failed. Fix firewall or use laptop tunnel:"
    echo "  [laptop] FIVEAGES_HOST=<this-server-ip> bash scripts/local_reverse_tunnel.sh"
    echo "  [here]   bash scripts/deploy_remote_env.sh tunnel"
    exit 1
  fi
}

_sync() {
  echo ""
  echo "== [1/3] tier1: code + weights + LIBERO sim (~72G) =="
  bash "${MIGRATE}" tier1

  if [[ "${SYNC_DATA}" == "1" ]]; then
    echo ""
    echo "== [2/3] raw LIBERO dataset (~1.9G) =="
    bash "${MIGRATE}" data
  fi

  if [[ "${SYNC_CACHE}" == "1" ]]; then
    echo ""
    echo "== [3/3] Version A latent cache (~658G) — hours =="
    bash "${MIGRATE}" cache
  else
    echo ""
    echo "== [skip cache] set SYNC_CACHE=1 to copy 300k cache later =="
  fi
}

_remote_setup() {
  echo ""
  echo "== Remote setup: conda + paths + smoke test =="
  _ssh "WORKSPACE=${REMOTE_BASE} bash '${REMOTE_BASE}/Physics-Aligned World2WAM/scripts/remote_setup_after_migrate.sh'"
}

_print_next() {
  echo ""
  echo "======== Remote env deploy DONE ========"
  echo "Remote workspace: ${REMOTE_BASE}"
  echo ""
  echo "On remote, when GPUs are free:"
  echo "  export WORKSPACE=${REMOTE_BASE}"
  echo "  bash minimal_world2wam/scripts/poll_gpu_version_a.sh start"
  echo ""
  if [[ "${SYNC_CACHE}" != "1" ]]; then
    echo "No cache yet — remote will precompute from scratch, OR copy cache later:"
    echo "  SYNC_CACHE=1 bash scripts/deploy_remote_env.sh sync-only"
  fi
  echo ""
  echo "16-GPU strategy: run precompute/train on both machines in parallel with different MAX_SAMPLES ranges."
}

case "${1:-direct}" in
  tunnel)
    REMOTE_USER_HOST="yjh@127.0.0.1"
    REMOTE_PORT="${TUNNEL_PORT:-2222}"
    echo "Using reverse tunnel ${REMOTE_USER_HOST}:${REMOTE_PORT}"
    _preflight
    _sync
    _remote_setup
    _print_next
    ;;
  sync-only)
    _preflight
    _sync
    ;;
  setup-only)
    _remote_setup
    ;;
  direct|*)
    _preflight
    _sync
    _remote_setup
    _print_next
    ;;
esac
