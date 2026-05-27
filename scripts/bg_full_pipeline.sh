#!/usr/bin/env bash
# Chained: assets download -> pipeline B. Intended for nohup (see bg_launch.sh).
set -euo pipefail
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_TAG="[bg_full_pipeline $(date -Iseconds)]"

echo "${LOG_TAG} START"
bash "${ROOT}/scripts/download_assets.sh"
echo "${LOG_TAG} download_assets DONE"
bash "${ROOT}/scripts/run_pipeline_b.sh"
echo "${LOG_TAG} pipeline B DONE"
