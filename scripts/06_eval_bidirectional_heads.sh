#!/usr/bin/env bash
set -e

cd /DATA/disk1/yjh_space/idea2_workspace/minimal_world2wam

python -m src.eval.eval_bidirectional_heads \
  --config configs/bidirectional_world2wam_smoke.yaml \
  --max-batches 10 \
  "$@"
