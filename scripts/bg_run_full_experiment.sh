#!/usr/bin/env bash
# Background entry: full distill pipeline then LIBERO spatial success eval.
set -euo pipefail
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/experiments/bg_jobs"
mkdir -p "${LOG_DIR}"

{
  echo "[$(date -Iseconds)] === Full Pipeline B (precompute + train) ==="
  bash "${ROOT}/scripts/run_full_pipeline_b.sh"
  echo "[$(date -Iseconds)] === LIBERO spatial success eval ==="
  bash "${ROOT}/scripts/run_libero_spatial_success.sh"
  echo "[$(date -Iseconds)] === All finished ==="
} 2>&1 | tee -a "${LOG_DIR}/full_experiment.log"
