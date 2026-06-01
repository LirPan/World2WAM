#!/usr/bin/env bash
# Run one LIBERO eval task with world2wam conda + libero env (used by parallel launcher).
set -euo pipefail

suite="${1:?suite}"
task_id="${2:?task_id}"
gpu_id="${3:?gpu_id}"

require_non_empty() {
  local var_name="$1"
  if [[ -z "${!var_name:-}" ]]; then
    echo "Error: required variable ${var_name} is not set" >&2
    exit 1
  fi
}

require_non_empty ROOT_DIR
require_non_empty CKPT
require_non_empty CONFIG
require_non_empty OUTPUT_DIR
require_non_empty NUM_TRIALS
require_non_empty STATUS_FILE
require_non_empty LOG_FILE
require_non_empty RESULT_FILE

MINIMAL_ROOT="${MINIMAL_ROOT:?MINIMAL_ROOT not set}"
# shellcheck disable=SC1091
source "${MINIMAL_ROOT}/scripts/activate_env.sh"
# shellcheck disable=SC1091
source "${MINIMAL_ROOT}/scripts/libero_env.sh"

cd "${ROOT_DIR}"
export EXP_NAME="${EXP_NAME:-}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

set +e
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python experiments/libero/eval_libero_single.py \
    "task=${CONFIG}" \
    "ckpt=${CKPT}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.task_id=${task_id}" \
    "gpu_id=${gpu_id}" \
    "EVALUATION.num_trials=${NUM_TRIALS}" \
    "EVALUATION.output_dir=${OUTPUT_DIR}" \
    ${EXTRA_ARGS} \
    > "${LOG_FILE}" 2>&1
else
  python experiments/libero/eval_libero_single.py \
    "task=${CONFIG}" \
    "ckpt=${CKPT}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.task_id=${task_id}" \
    "gpu_id=${gpu_id}" \
    "EVALUATION.num_trials=${NUM_TRIALS}" \
    "EVALUATION.output_dir=${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1
fi
rc=$?
set -e

if [[ "${rc}" -eq 0 ]] && [[ -f "${RESULT_FILE}" ]]; then
  echo "SUCCESS|${gpu_id}|${rc}|$(date +%s)|${LOG_FILE}" > "${STATUS_FILE}"
else
  echo "FAILED|${gpu_id}|${rc}|$(date +%s)|${LOG_FILE}" > "${STATUS_FILE}"
  exit "${rc}"
fi
