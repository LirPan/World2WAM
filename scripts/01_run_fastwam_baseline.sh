#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Option A: delegate full training to FastWAM (no repo edits)
python src/train/train_fastwam_future_distill.py \
  --config configs/fastwam_libero_baseline.yaml \
  --mode baseline \
  --delegate-baseline \
  "$@"

# Option B (sanity only): comment Option A and run:
# python src/train/train_fastwam_future_distill.py \
#   --config configs/fastwam_libero_baseline.yaml \
#   --mode baseline
