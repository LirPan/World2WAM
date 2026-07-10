#!/usr/bin/env bash
# Poll until SSH proxy (7890/17890) is up, then resume dataset download + extract.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-86400}"

echo "== wait_proxy_and_download: waiting for outbound proxy =="
elapsed=0
while true; do
  if source /DATA/disk0/jianhua/use_proxy.sh 2>/dev/null; then
    echo "$(date -Iseconds) proxy OK: ${http_proxy}"
    break
  fi
  echo "$(date -Iseconds) proxy not ready (lab 10.11.0.110:1080 or SSH tunnel) elapsed=${elapsed}s"
  if [[ "${elapsed}" -ge "${MAX_WAIT_SEC}" ]]; then
    echo "Timeout waiting for proxy"
    exit 1
  fi
  sleep "${POLL_SEC}"
  elapsed=$((elapsed + POLL_SEC))
done

cd "${WORKSPACE}"
bash minimal_world2wam/scripts/setup_deps.sh data
echo "DOWNLOAD AND EXTRACT DONE"
