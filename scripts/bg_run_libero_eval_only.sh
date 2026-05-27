#!/usr/bin/env bash
set -euo pipefail
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${ROOT}/scripts/run_libero_spatial_success.sh" 2>&1 | tee -a "${ROOT}/experiments/bg_jobs/libero_eval.log"
