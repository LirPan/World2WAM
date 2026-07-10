#!/usr/bin/env bash
# Paper-level full World2WAM: ~1.17M latent cache, 10-epoch train, LIBERO 10x50 eval.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
CONFIG="${CONFIG:-configs/world2wam_libero_spatial_h10_paper.yaml}"
CACHE_OUT="${CACHE_OUT:-cache/libero_spatial_h10_full_fastwam}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
PAPER_GPU="${PAPER_GPU:-6}"
PAPER_GPU2="${PAPER_GPU2:-7}"
PRECOMPUTE_SHARDS="${PRECOMPUTE_SHARDS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
HEADS_CKPT="${HEADS_CKPT:-experiments/world2wam_heads_paper/heads_final.pt}"
ADAPTER_CKPT="${ADAPTER_CKPT:-experiments/world2wam_adapter_paper/adapter_final.pt}"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
export CUDA_VISIBLE_DEVICES="${PAPER_GPU}"
export EVAL_DEVICE=cuda

if [[ -f /DATA/disk0/jianhua/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk0/jianhua/use_proxy.sh || true
fi

cd "${WORKSPACE}"
mkdir -p "${CACHE_OUT}" experiments

echo "== paper_pipeline: $(date -Iseconds) =="
echo "CONFIG=${CONFIG} CACHE_OUT=${CACHE_OUT} GPUs=${PAPER_GPU},${PAPER_GPU2} shards=${PRECOMPUTE_SHARDS}"
echo "EVAL: ${MAX_TASKS} tasks x ${NUM_TRIALS} trials"

echo "== smoke test =="
bash minimal_world2wam/scripts/smoke_test.sh

echo "== precompute full (~1.17M, ${PRECOMPUTE_SHARDS}-GPU parallel, resume enabled) =="
_precompute_shard() {
  local gpu="$1"
  local shard_id="$2"
  local extra_args=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    extra_args+=(--max_samples "${MAX_SAMPLES}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python minimal_world2wam/cache/precompute_fastwam_latents.py \
    --config "${CONFIG}" \
    --output "${CACHE_OUT}" \
    --device cuda \
    --resume \
    --shard_id "${shard_id}" \
    --num_shards "${PRECOMPUTE_SHARDS}" \
    "${extra_args[@]}"
}

if [[ "${PRECOMPUTE_SHARDS}" -le 1 ]]; then
  _precompute_shard "${PAPER_GPU}" 0
else
  _precompute_shard "${PAPER_GPU}" 0 &
  pid0=$!
  _precompute_shard "${PAPER_GPU2}" 1 &
  pid1=$!
  wait "${pid0}"
  wait "${pid1}"
  CUDA_VISIBLE_DEVICES="${PAPER_GPU}" python minimal_world2wam/cache/precompute_fastwam_latents.py \
    --config "${CONFIG}" \
    --output "${CACHE_OUT}" \
    --finalize_only
fi

export CUDA_VISIBLE_DEVICES="${PAPER_GPU}"

echo "== train heads (10 epochs) =="
python minimal_world2wam/train/train_world2wam_heads.py \
  --config "${CONFIG}" \
  --cache_dir "${CACHE_OUT}" \
  --use_fwd true --use_inv true --use_cycle true \
  --device cuda

echo "== train adapter (10 epochs) =="
python minimal_world2wam/train/train_world2wam_adapter.py \
  --config "${CONFIG}" \
  --cache_dir "${CACHE_OUT}" \
  --use_act true --use_fwd true --use_inv true --use_cycle true \
  --device cuda

echo "== eval compare (offline + baseline + ours) =="
python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config "${CONFIG}" \
  --mode offline \
  --latent_verification \
  --cache_dir "${CACHE_OUT}" \
  --heads_ckpt "${HEADS_CKPT}" \
  --device cuda \
  --output experiments/eval_offline_paper.json

python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config "${CONFIG}" \
  --mode baseline \
  --max_tasks "${MAX_TASKS}" \
  --num_trials "${NUM_TRIALS}" \
  --device cuda \
  --output experiments/eval_baseline_paper.json

python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config "${CONFIG}" \
  --mode ours_adapter \
  --adapter_ckpt "${ADAPTER_CKPT}" \
  --max_tasks "${MAX_TASKS}" \
  --num_trials "${NUM_TRIALS}" \
  --device cuda \
  --output experiments/eval_ours_adapter_paper.json

python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config "${CONFIG}" \
  --mode ours_dit \
  --max_tasks "${MAX_TASKS}" \
  --num_trials "${NUM_TRIALS}" \
  --device cuda \
  --output experiments/eval_ours_dit_paper.json

echo "== paper_pipeline done: $(date -Iseconds) =="
