#!/usr/bin/env bash
# LIBERO spatial sim success rate with official FastWAM checkpoint (action-only infer_action).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/libero_env.sh"

CKPT="${CKPT:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
STATS_JSON="${STATS_JSON:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3,6,7}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/libero_eval/spatial_${RUN_TAG}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

# Ensure sim dependencies (idempotent).
if ! python -c "import mujoco, robosuite" 2>/dev/null; then
  echo "==> Installing mujoco + robosuite for LIBERO sim..."
  pip install 'mujoco>=3.1.0' 'robosuite==1.4.0' 'bddl==1.0.1' 'gym==0.25.2' 'easydict' -q
  pip install 'numpy==1.26.4' -q || true
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
NUM_GPUS="$(echo "${CUDA_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')"

cd "${FASTWAM_ROOT}"

echo "==> LIBERO spatial eval: ckpt=${CKPT}"
echo "    trials/task=${NUM_TRIALS}, gpus=${NUM_GPUS} (${CUDA_DEVICES}), out=${OUTPUT_DIR}"

python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt="${CKPT}" \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.num_trials="${NUM_TRIALS}" \
  EVALUATION.output_dir="${OUTPUT_DIR}" \
  EVALUATION.dataset_stats_path="${STATS_JSON}" \
  MULTIRUN.task_suite_names="[libero_spatial]" \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU}"

echo "==> Summarize results..."
python experiments/libero/summarize_results.py --output_dir "${OUTPUT_DIR}" | tee "${OUTPUT_DIR}/summary.txt"

echo "==> Done. Summary: ${OUTPUT_DIR}/summary.txt"
