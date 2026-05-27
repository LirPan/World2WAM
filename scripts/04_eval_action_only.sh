#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python src/eval/eval_action_only_fastwam.py \
  --config configs/fastwam_future_distill.yaml \
  "$@"
