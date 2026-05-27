#!/usr/bin/env bash
set -e

cd /DATA/disk1/yjh_space/idea2_workspace/minimal_world2wam

python -m src.train.train_bidirectional_world2wam \
  --config configs/bidirectional_world2wam.yaml \
  --mode cycle \
  "$@"
