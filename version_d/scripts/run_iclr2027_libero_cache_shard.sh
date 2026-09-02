#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 SHARD_ID NUM_SHARDS" >&2
  exit 2
fi
SHARD="$1"
SHARDS="$2"
ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
POLICY="$ROOT/deploy/runtime/policy_lora"
CONFIG="$ROOT/deploy/configs/libero_iclr2027_all_suites.yaml"
PY=/DATA/disk0/yjh/robotwin_w2wam/env/bin/python
MANIFEST="$ROOT/cache/libero_all_suites/world2wam_iclr2027_libero_all/manifest_shard${SHARD}of${SHARDS}.json"

mkdir -p "$(dirname "$MANIFEST")"
cd "$POLICY"
"$PY" src/data/precompute_future_latents.py \
  --config "$CONFIG" --device cuda \
  --samples-per-dataset 3000 --selection-seed 42 \
  --shard-id "$SHARD" --num-shards "$SHARDS" \
  --manifest "$MANIFEST" --manifest-every 50
