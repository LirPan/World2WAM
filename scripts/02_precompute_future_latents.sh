#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python src/data/precompute_future_latents.py \
  --config configs/fastwam_future_distill.yaml \
  "$@"
