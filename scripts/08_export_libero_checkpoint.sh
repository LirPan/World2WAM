#!/usr/bin/env bash
# Export World2WAM bundle -> FastWAM-compatible merged checkpoint for LIBERO sim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/activate_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/activate_env.sh"
fi

BUNDLE="${WORLD2WAM_BUNDLE:-${1:-}}"
CONFIG="${CONFIG:-configs/world2wam_policy_improve.yaml}"
EXPORT_TAG="${EXPORT_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-${ROOT}/experiments/exported_ckpts/world2wam_merged_${EXPORT_TAG}.pt}"

if [[ -z "${BUNDLE}" ]]; then
  echo "Usage: WORLD2WAM_BUNDLE=/path/to/world2wam_final.pt bash scripts/08_export_libero_checkpoint.sh"
  echo "   or: bash scripts/08_export_libero_checkpoint.sh /path/to/world2wam_final.pt"
  exit 1
fi

EXPORTED_CKPT="$(python -m src.tools.export_libero_checkpoint \
  --bundle "${BUNDLE}" \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --tag "${EXPORT_TAG}")"

export EXPORTED_CKPT
echo "EXPORTED_CKPT=${EXPORTED_CKPT}"
