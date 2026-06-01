#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/activate_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/activate_env.sh"
fi

python src/eval/eval_action_only_fastwam.py \
  --config configs/fastwam_future_distill.yaml \
  "$@"
