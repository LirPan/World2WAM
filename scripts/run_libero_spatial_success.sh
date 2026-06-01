#!/usr/bin/env bash
# LIBERO spatial sim success (official or merged World2WAM ckpt).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
export MINIMAL_ROOT="${ROOT}"
export FASTWAM_ROOT

# shellcheck disable=SC1091
source "${ROOT}/scripts/libero_env.sh"

NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3,6,7}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/libero_eval/spatial_${RUN_TAG}}"
TASK_LIMIT="${TASK_LIMIT:-}"
USE_TMUX="${USE_TMUX:-1}"
STATS_JSON="${STATS_JSON:-${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OFFICIAL_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ROOT}/checkpoints"

# Resolve checkpoint: explicit CKPT wins; else WORLD2WAM_BUNDLE export; else official.
if [[ -n "${CKPT:-}" ]]; then
  :
elif [[ -n "${WORLD2WAM_BUNDLE:-}" ]]; then
  echo "==> Exporting merged checkpoint from bundle: ${WORLD2WAM_BUNDLE}"
  EXPORT_TAG="${EXPORT_TAG:-${RUN_TAG}}"
  WORLD2WAM_BUNDLE="${WORLD2WAM_BUNDLE}" EXPORT_TAG="${EXPORT_TAG}" \
    bash "${ROOT}/scripts/08_export_libero_checkpoint.sh"
  CKPT="${EXPORTED_CKPT:-${ROOT}/experiments/exported_ckpts/world2wam_merged_${EXPORT_TAG}.pt}"
elif [[ -z "${CKPT:-}" ]]; then
  CKPT="${OFFICIAL_CKPT}"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}"
  exit 1
fi

if ! python -c "import mujoco, robosuite" 2>/dev/null; then
  echo "==> Installing mujoco + robosuite for LIBERO sim..."
  pip install 'mujoco>=3.1.0' 'robosuite==1.4.0' 'bddl==1.0.1' 'gym==0.25.2' 'easydict' -q
  pip install 'numpy==1.26.4' -q || true
fi
if ! python -c "import future, matplotlib" 2>/dev/null; then
  echo "==> Installing LIBERO Python deps (future, matplotlib)..."
  pip install future matplotlib -q
fi

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
NUM_GPUS="$(echo "${CUDA_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')"

# Optional smoke task subset
TASK_FILE_ARG=()
if [[ -n "${TASK_LIMIT}" ]]; then
  TASK_FILE="${OUTPUT_DIR}/tasks_smoke.txt"
  : > "${TASK_FILE}"
  for ((tid = 0; tid < TASK_LIMIT; tid++)); do
    echo "libero_spatial,${tid}" >> "${TASK_FILE}"
  done
  TASK_FILE_ARG=("MULTIRUN.task_file=${TASK_FILE}")
  echo "==> Smoke mode: TASK_LIMIT=${TASK_LIMIT} -> ${TASK_FILE}"
fi

chmod +x "${ROOT}/scripts/libero/libero_single_task.sh"
chmod +x "${ROOT}/scripts/libero/run_libero_parallel_test.sh"

cd "${FASTWAM_ROOT}"

echo "==> LIBERO spatial eval: ckpt=${CKPT}"
echo "    trials/task=${NUM_TRIALS}, gpus=${NUM_GPUS} (${CUDA_DEVICES}), out=${OUTPUT_DIR}"
echo "    USE_TMUX=${USE_TMUX}"

export CKPT
python "${ROOT}/scripts/libero/run_libero_manager.py" \
  task=libero_uncond_2cam224_1e-4 \
  "ckpt=${CKPT}" \
  EVALUATION.task_suite_name=libero_spatial \
  "EVALUATION.num_trials=${NUM_TRIALS}" \
  "EVALUATION.output_dir=${OUTPUT_DIR}" \
  "EVALUATION.dataset_stats_path=${STATS_JSON}" \
  "MULTIRUN.task_suite_names=[libero_spatial]" \
  "MULTIRUN.num_gpus=${NUM_GPUS}" \
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
  "${TASK_FILE_ARG[@]}"

echo "==> Summarize results..."
python experiments/libero/summarize_results.py --output_dir "${OUTPUT_DIR}" | tee "${OUTPUT_DIR}/summary.txt"

echo "==> Done. Summary: ${OUTPUT_DIR}/summary.txt"
