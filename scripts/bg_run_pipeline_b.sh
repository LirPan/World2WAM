#!/usr/bin/env bash
# Run pipeline B only (02->03->04), assumes download_assets already done.
set -euo pipefail
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${ROOT}/scripts/run_pipeline_b.sh"
