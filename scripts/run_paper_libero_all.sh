#!/usr/bin/env bash
# End-to-end LIBERO paper pipeline (stages 0-4, resumable via SKIP_* env vars).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STAGE="${STAGE:-all}"
NUM_TRIALS="${NUM_TRIALS:-5}"
TASK_LIMIT="${TASK_LIMIT:-2}"
FULL_RUN="${FULL_RUN:-0}"

run_stage_0() {
  echo "======== Stage 0: smoke official LIBERO sim ========"
  CKPT="" NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" USE_TMUX="${USE_TMUX:-0}" \
    RUN_TAG="paper_s0_official" OUTPUT_DIR="${ROOT}/experiments/libero_eval/paper_stage0_official" \
    bash scripts/run_libero_spatial_success.sh
}

run_stage_1() {
  echo "======== Stage 1: policy full train + export ========"
  bash scripts/run_full_pipeline_policy.sh
}

run_stage_2() {
  echo "======== Stage 2: smoke compare official vs world2wam ========"
  RUN_TAG="paper_s2_smoke" NUM_TRIALS="${NUM_TRIALS}" TASK_LIMIT="${TASK_LIMIT}" \
    bash scripts/run_compare_libero_success.sh
}

run_stage_3() {
  echo "======== Stage 3: full LIBERO compare ========"
  FULL_RUN=1 RUN_TAG="paper_s3_full" bash scripts/run_compare_libero_success.sh
}

run_stage_4() {
  echo "======== Stage 4: ablation sweep ========"
  if [[ "${FULL_RUN}" == "1" ]]; then
    bash scripts/sweep_bidirectional_ablations.sh
  else
    SMOKE=1 bash scripts/sweep_bidirectional_ablations.sh
  fi
}

case "${STAGE}" in
  0) run_stage_0 ;;
  1) run_stage_1 ;;
  2) run_stage_2 ;;
  3) run_stage_3 ;;
  4) run_stage_4 ;;
  all)
    [[ "${SKIP_STAGE_0:-0}" == "1" ]] || run_stage_0
    [[ "${SKIP_STAGE_1:-0}" == "1" ]] || run_stage_1
    [[ "${SKIP_STAGE_2:-0}" == "1" ]] || run_stage_2
    if [[ "${FULL_RUN}" == "1" ]]; then
      [[ "${SKIP_STAGE_3:-0}" == "1" ]] || run_stage_3
      [[ "${SKIP_STAGE_4:-0}" == "1" ]] || run_stage_4
    else
      echo "FULL_RUN=0: skipping stage 3 (full compare) and full ablation grid."
      [[ "${SKIP_STAGE_4:-0}" == "1" ]] || SMOKE=1 bash scripts/sweep_bidirectional_ablations.sh
    fi
    ;;
  *)
    echo "Unknown STAGE=${STAGE}. Use 0|1|2|3|4|all"
    exit 1
    ;;
esac

echo "==> run_paper_libero_all stage=${STAGE} done."
