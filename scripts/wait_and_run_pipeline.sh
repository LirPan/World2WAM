#!/usr/bin/env bash
# Wait until LIBERO-Spatial dataset is extracted, then run debug experiment pipeline.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
DATA_DIR="${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"
CONFIG="${CONFIG:-configs/world2wam_libero_spatial_h10.yaml}"
CACHE_OUT="${CACHE_OUT:-cache/debug_libero_spatial_h10}"
PRE_MAX="${PRE_MAX:-100}"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-7200}"

# shellcheck disable=SC1091
source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME

echo "== wait_and_run_pipeline: polling dataset at ${DATA_DIR} =="
elapsed=0
while true; do
  n_parquet=$(find "${DATA_DIR}/data" -name '*.parquet' 2>/dev/null | wc -l)
  n_main=$(find "${DATA_DIR}/videos/chunk-000/observation.images.image" -name '*.mp4' 2>/dev/null | wc -l)
  n_wrist=$(find "${DATA_DIR}/videos/chunk-000/observation.images.wrist_image" -name '*.mp4' 2>/dev/null | wc -l)
  echo "$(date -Iseconds) parquet=${n_parquet} main=${n_main} wrist=${n_wrist} elapsed=${elapsed}s"
  if [[ "${n_parquet}" -ge 434 && "${n_main}" -ge 434 && "${n_wrist}" -ge 434 ]]; then
    echo "Dataset ready."
    break
  fi
  if [[ "${elapsed}" -ge "${MAX_WAIT_SEC}" ]]; then
    echo "Timeout waiting for dataset after ${MAX_WAIT_SEC}s"
    exit 1
  fi
  sleep "${POLL_SEC}"
  elapsed=$((elapsed + POLL_SEC))
done

cd "${WORKSPACE}"
echo "== smoke test =="
bash minimal_world2wam/scripts/smoke_test.sh

echo "== precompute (max=${PRE_MAX}) =="
python minimal_world2wam/cache/precompute_fastwam_latents.py \
  --config "${CONFIG}" --output "${CACHE_OUT}" --max_samples "${PRE_MAX}"

echo "== train heads =="
python minimal_world2wam/train/train_world2wam_heads.py \
  --config "${CONFIG}" --cache_dir "${CACHE_OUT}" \
  --use_fwd true --use_inv true --use_cycle true

echo "== offline eval =="
python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config "${CONFIG}" --mode offline --latent_verification \
  --cache_dir "${CACHE_OUT}" --heads_ckpt experiments/world2wam_heads/heads_final.pt

echo "== baseline sim (1 task) =="
python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config "${CONFIG}" --mode baseline --max_tasks 1 || echo "baseline sim skipped/failed (check LIBERO)"

echo "PIPELINE DONE"
