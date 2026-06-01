#!/usr/bin/env bash
# Full Policy pipeline: cache check -> train LoRA policy -> export merged ckpt -> LIBERO smoke eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
RELEASE_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
STATS_JSON="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"
export MINIMAL_ROOT="${ROOT}"
export FASTWAM_ROOT

cd "${ROOT}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CUDA_DEVICE="${CUDA_DEVICE:-6}"
NUM_TRIALS="${NUM_TRIALS:-5}"
TASK_LIMIT="${TASK_LIMIT:-2}"

if [[ ! -f "${RELEASE_CKPT}" ]]; then
  echo "Missing ${RELEASE_CKPT}. Run: bash scripts/download_assets.sh"
  exit 1
fi

CACHE_DIR="${ROOT}/data/future_latents/world2wam_minimal"
CACHE_COUNT="$(find "${CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${CACHE_COUNT}" -lt 10 ]]; then
  echo "==> [02] Precompute future latents (cache sparse: ${CACHE_COUNT} files)..."
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  python src/data/precompute_future_latents.py --config configs/world2wam_policy_improve_full.yaml
else
  echo "==> SKIP [02] future latent cache looks populated (${CACHE_COUNT} files)"
fi

echo "==> [07] Train World2WAM Policy (full)..."
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
python -m src.train.train_fastwam_future_distill \
  --config "${CONFIG:-configs/world2wam_policy_improve_full.yaml}" \
  --mode future_distill \
  --backbone-mode lora \
  "$@"

BUNDLE="${ROOT}/experiments/world2wam_policy_improve_full/checkpoints/world2wam_final.pt"
if [[ ! -f "${BUNDLE}" ]]; then
  echo "Missing policy bundle: ${BUNDLE}"
  exit 1
fi

echo "==> [08] Export merged checkpoint for LIBERO..."
if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  echo "==> SKIP export merged ckpt (SKIP_EXPORT=1)"
elif [[ "${SKIP_LIBERO:-0}" == "1" ]]; then
  echo "==> SKIP export (train-only mode; run 08_export_libero_checkpoint.sh later for sim)"
else
  WORLD2WAM_BUNDLE="${BUNDLE}" CONFIG=configs/world2wam_policy_improve_full.yaml \
    EXPORT_TAG="policy_full" bash scripts/08_export_libero_checkpoint.sh
fi

if [[ "${SKIP_LIBERO:-0}" == "1" ]]; then
  echo "==> SKIP LIBERO eval (SKIP_LIBERO=1)"
else
  echo "==> LIBERO smoke eval (World2WAM merged)..."
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  WORLD2WAM_BUNDLE="${BUNDLE}" NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" \
    USE_TMUX="${USE_TMUX:-0}" RUN_TAG="policy_full_smoke" \
    bash scripts/run_libero_spatial_success.sh
fi

echo "==> Policy full pipeline done."
echo "    Bundle: ${BUNDLE}"
echo "    Merged: ${ROOT}/experiments/exported_ckpts/world2wam_merged_policy_full.pt"
