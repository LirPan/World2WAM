#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/activate_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/activate_env.sh"
fi

python -m src.train.train_bidirectional_world2wam \
  --config configs/bidirectional_world2wam_smoke.yaml \
  --mode cycle \
  "$@"
