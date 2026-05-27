#!/usr/bin/env bash
# Full Pipeline B: all spatial clips -> future latent cache -> train head (1 epoch) -> action-only smoke eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
RELEASE_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
STATS_JSON="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"

cd "${ROOT}"

# Full run defaults (override via env).
PRECOMPUTE_MAX="${PRECOMPUTE_MAX:-0}"   # 0 = entire dataset
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-0}" # 0 = full epoch(s) per num_epochs
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"

if [[ ! -f "${RELEASE_CKPT}" ]]; then
  echo "Missing ${RELEASE_CKPT}. Run: bash scripts/download_assets.sh"
  exit 1
fi

DATASET_LEN="$(python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT}")
from src.utils.config import load_config
from src.data.libero_dataset_adapter import build_fastwam_dataset
cfg = {
    "fastwam_root": "${FASTWAM_ROOT}",
    "fastwam_task_config": "libero_uncond_2cam224_1e-4",
    "lerobot_dataset_dirs": ["${FASTWAM_ROOT}/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"],
    "dataset_stats_path": "${STATS_JSON}",
}
base, _ = build_fastwam_dataset(cfg)
print(len(base))
PY
)"

echo "==> Dataset length: ${DATASET_LEN}"

TEXT_CACHE_DIR="${FASTWAM_ROOT}/data/text_embeds_cache/libero"
mkdir -p "${FASTWAM_ROOT}/runs"
if [[ ! -d "${TEXT_CACHE_DIR}" ]] || [[ "$(find "${TEXT_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l)" -lt 1 ]]; then
  echo "==> Precompute T5 text embeddings..."
  cd "${FASTWAM_ROOT}"
  python scripts/precompute_text_embeds.py \
    task=libero_uncond_2cam224_1e-4 \
    data.train.dataset_dirs="['${FASTWAM_ROOT}/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot']"
  cd "${ROOT}"
fi

RUN_CFG="${ROOT}/configs/fastwam_future_distill_run.yaml"
if [[ "${TRAIN_MAX_STEPS}" == "0" ]]; then
  MAX_STEPS_YAML="null"
else
  MAX_STEPS_YAML="${TRAIN_MAX_STEPS}"
fi

cat > "${RUN_CFG}" <<EOF
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
num_epochs: ${NUM_EPOCHS}
lr: 1.0e-4
seed: 42
num_workers: 4
device: cuda

lerobot_dataset_dirs:
  - ${FASTWAM_ROOT}/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
cache_dir: ./data/future_latents
output_dir: ./experiments/future_latent_distill_full
checkpoint_path: ${RELEASE_CKPT}
dataset_stats_path: ${STATS_JSON}

mixed_precision: bf16
log_every: 50
save_every: 2000
max_train_steps: ${MAX_STEPS_YAML}
precompute_max_samples: 0
EOF

CACHE_DIR="${ROOT}/data/future_latents/world2wam_minimal"
mkdir -p "${CACHE_DIR}"
CACHE_COUNT="$(find "${CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"

TARGET_N="${DATASET_LEN}"
if [[ "${PRECOMPUTE_MAX}" != "0" ]]; then
  TARGET_N="${PRECOMPUTE_MAX}"
fi

if [[ "${CACHE_COUNT}" -lt "${TARGET_N}" ]]; then
  echo "==> [02] Precompute future latents (${CACHE_COUNT}/${TARGET_N} cached)..."
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  PRECOMPUTE_ARGS=(--config configs/fastwam_future_distill_run.yaml)
  if [[ "${PRECOMPUTE_MAX}" != "0" ]]; then
    PRECOMPUTE_ARGS+=(--max-samples "${PRECOMPUTE_MAX}")
  fi
  python src/data/precompute_future_latents.py "${PRECOMPUTE_ARGS[@]}"
else
  echo "==> SKIP [02] cache complete (${CACHE_COUNT} files)"
fi

echo "==> [03] Train FutureLatentHead (full dataset, ${NUM_EPOCHS} epoch(s))..."
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
python src/train/train_fastwam_future_distill.py \
  --config configs/fastwam_future_distill_run.yaml \
  --mode future_distill

echo "==> [04] Action-only smoke eval..."
python src/eval/eval_action_only_fastwam.py \
  --config configs/fastwam_future_distill_run.yaml \
  --checkpoint "${RELEASE_CKPT}" \
  --max-batches 10 \
  --output experiments/future_latent_distill_full/eval_results.json

echo "==> Full Pipeline B done. Output: experiments/future_latent_distill_full/"
