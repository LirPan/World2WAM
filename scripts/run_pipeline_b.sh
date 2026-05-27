#!/usr/bin/env bash
# Pipeline B: official ckpt + data -> text cache -> 02 -> 03 -> 04
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
RELEASE_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"

cd "${ROOT}"

PRECOMPUTE_MAX="${PRECOMPUTE_MAX:-128}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ ! -f "${RELEASE_CKPT}" ]]; then
  echo "Missing ${RELEASE_CKPT}. Run: bash scripts/download_assets.sh"
  exit 1
fi

STATS_JSON="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
TEXT_CACHE_DIR="${FASTWAM_ROOT}/data/text_embeds_cache/libero"
mkdir -p "${FASTWAM_ROOT}/runs"

if [[ ! -f "${STATS_JSON}" ]]; then
  echo "Missing ${STATS_JSON}"
  exit 1
fi

if [[ ! -d "${TEXT_CACHE_DIR}" ]] || [[ "$(find "${TEXT_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l)" -lt 1 ]]; then
  echo "==> Precompute T5 text embeddings (FastWAM)..."
  cd "${FASTWAM_ROOT}"
  python scripts/precompute_text_embeds.py \
    task=libero_uncond_2cam224_1e-4 \
    data.train.dataset_dirs="['${FASTWAM_ROOT}/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot']"
else
  echo "==> SKIP text embed precompute (cache exists: ${TEXT_CACHE_DIR})"
fi

cd "${ROOT}"

# Write run-specific config overlay
RUN_CFG="${ROOT}/configs/fastwam_future_distill_run.yaml"
cat > "${RUN_CFG}" <<EOF
# Auto-generated for pipeline B quick run
project_name: world2wam_minimal
fastwam_root: ${FASTWAM_ROOT}
libero_root: ${ROOT}/../code/LIBERO
fastwam_task_config: libero_uncond_2cam224_1e-4

use_future_latent_distill: true
lambda_fwd: 0.1
future_horizon: 1
future_latent_dim: 48
hidden_dim: 1024
action_dim: 7
use_gt_action_for_future_head: true
freeze_fastwam_backbone: true
anchor_action_idx: 0

batch_size: ${BATCH_SIZE}
num_epochs: 1
lr: 1.0e-4
seed: 42
num_workers: 2
device: cuda

lerobot_dataset_dirs:
  - ${FASTWAM_ROOT}/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
cache_dir: ./data/future_latents
output_dir: ./experiments/future_latent_distill
checkpoint_path: ${RELEASE_CKPT}
dataset_stats_path: ${STATS_JSON}

mixed_precision: bf16
log_every: 5
save_every: 1000000
max_train_steps: ${TRAIN_MAX_STEPS}
precompute_max_samples: ${PRECOMPUTE_MAX}
EOF

CACHE_COUNT=$(find "${ROOT}/data/future_latents/world2wam_minimal" -name '*.pt' 2>/dev/null | wc -l)
if [[ "${CACHE_COUNT}" -lt "${PRECOMPUTE_MAX}" ]]; then
  echo "==> [02] Precompute future latents (max ${PRECOMPUTE_MAX})..."
  python src/data/precompute_future_latents.py \
    --config configs/fastwam_future_distill_run.yaml \
    --max-samples "${PRECOMPUTE_MAX}"
else
  echo "==> SKIP [02] future latent cache already has ${CACHE_COUNT} files"
fi

echo "==> [03] Train FutureLatentHead..."
python src/train/train_fastwam_future_distill.py \
  --config configs/fastwam_future_distill_run.yaml \
  --mode future_distill

echo "==> [04] Action-only eval..."
python src/eval/eval_action_only_fastwam.py \
  --config configs/fastwam_future_distill_run.yaml \
  --checkpoint "${RELEASE_CKPT}" \
  --max-batches 10 \
  --output experiments/future_latent_distill/eval_results.json

echo "==> Pipeline B complete. See experiments/future_latent_distill/"
