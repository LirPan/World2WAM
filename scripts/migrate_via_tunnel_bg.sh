#!/usr/bin/env bash
# Start migration via reverse SSH tunnel (laptop must run local_reverse_tunnel.sh first).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/jianhua}"
TUNNEL_PORT="${TUNNEL_PORT:-2222}"

echo "Testing tunnel localhost:${TUNNEL_PORT} ..."
if ! ssh -p "${TUNNEL_PORT}" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  yjh@127.0.0.1 "echo tunnel_ok; hostname" 2>/dev/null; then
  echo ""
  echo "ERROR: Cannot reach remote through tunnel."
  echo "On your LAPTOP (IP allowed by 120.92.211.106), run:"
  echo "  FIVEAGES_HOST=<this-server-ip> bash minimal_world2wam/scripts/local_reverse_tunnel.sh"
  echo ""
  echo "Keep that laptop terminal open, then re-run this script."
  exit 1
fi

export REMOTE_USER_HOST="yjh@127.0.0.1"
export REMOTE_PORT="${TUNNEL_PORT}"
export REMOTE_BASE="${REMOTE_BASE:-/DATA/disk0/yjh/world2wam}"
export LOCAL_BASE="${WORKSPACE}"

tier="${1:-tier1}"
echo "Tunnel OK. Starting background migrate ${tier} via 127.0.0.1:${TUNNEL_PORT} ..."
exec bash "${WORKSPACE}/Physics-Aligned World2WAM/scripts/migrate_to_remote_bg.sh" start "${tier}"
