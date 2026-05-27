#!/usr/bin/env bash
# Resume from 03+04 after 02 cache exists.
set -euo pipefail
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  source /DATA/disk1/yjh_space/use_proxy.sh
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
RELEASE_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"
cd "${ROOT}"

PRECOMPUTE_MAX="${PRECOMPUTE_MAX:-128}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STATS_JSON="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"

RUN_CFG="${ROOT}/configs/fastwam_future_distill_run.yaml"
# shellcheck disable=SC1091
[[ -f "${RUN_CFG}" ]] || bash "${ROOT}/scripts/run_pipeline_b.sh" 2>/dev/null || true

python src/train/train_fastwam_future_distill.py \
  --config configs/fastwam_future_distill_run.yaml \
  --mode future_distill

python src/eval/eval_action_only_fastwam.py \
  --config configs/fastwam_future_distill_run.yaml \
  --checkpoint "${RELEASE_CKPT}" \
  --max-batches 10 \
  --output experiments/future_latent_distill/eval_results.json

echo "==> train+eval done"
