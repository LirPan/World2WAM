#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python src/train/train_fastwam_future_distill.py \
  --config configs/fastwam_future_distill.yaml \
  --mode future_distill \
  "$@"
