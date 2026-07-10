#!/usr/bin/env bash
# Run on YOUR LAPTOP (the machine whose IP can SSH to 120.92.211.106).
# Keeps a reverse tunnel so the data server (fiveages) can reach remote via localhost:2222.
#
# Prerequisites:
#   - Laptop can:  ssh yjh@<FIVEAGES_HOST>          (this data server)
#   - Laptop can:  ssh yjh@120.92.211.106           (destination)
#
# Usage on laptop:
#   FIVEAGES_HOST=<ip-or-host-of-fiveages-A100-2> \
#     bash local_reverse_tunnel.sh
#
# Then on THIS server (fiveages), in another terminal:
#   REMOTE_USER_HOST=yjh@127.0.0.1 REMOTE_PORT=2222 \
#     bash minimal_world2wam/scripts/migrate_to_remote_bg.sh start tier1
#
# Keep this laptop terminal open (or use autossh). Ctrl+C stops the tunnel.
set -euo pipefail

FIVEAGES_HOST="${FIVEAGES_HOST:?Set FIVEAGES_HOST to this data server's SSH address}"
FIVEAGES_USER="${FIVEAGES_USER:-yjh}"
REMOTE_HOST="${REMOTE_HOST:-120.92.211.106}"
REMOTE_PORT_SSH="${REMOTE_PORT_SSH:-22}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-2222}"

echo "Opening reverse tunnel on ${FIVEAGES_HOST}..."
echo "  ${FIVEAGES_HOST}:127.0.0.1:${LOCAL_TUNNEL_PORT}  ->  laptop  ->  ${REMOTE_HOST}:${REMOTE_PORT_SSH}"
echo ""
echo "On the data server, run:"
echo "  ssh -p ${LOCAL_TUNNEL_PORT} yjh@127.0.0.1"
echo "  REMOTE_USER_HOST=yjh@127.0.0.1 REMOTE_PORT=${LOCAL_TUNNEL_PORT} \\"
echo "    bash minimal_world2wam/scripts/migrate_to_remote_bg.sh start tier1"
echo ""

exec ssh -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -o ExitOnForwardFailure=yes \
  -R "127.0.0.1:${LOCAL_TUNNEL_PORT}:${REMOTE_HOST}:${REMOTE_PORT_SSH}" \
  "${FIVEAGES_USER}@${FIVEAGES_HOST}"
