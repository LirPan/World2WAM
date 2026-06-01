#!/usr/bin/env bash
# World2WAM policy improvement: LoRA/adapter on action path + L_action + warmup L_future.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/activate_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/activate_env.sh"
fi

MAX_STEPS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-steps)
      MAX_STEPS_ARGS=(--max-steps "$2")
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

python -m src.train.train_fastwam_future_distill \
  --config configs/world2wam_policy_improve.yaml \
  --mode future_distill \
  --backbone-mode lora \
  "${MAX_STEPS_ARGS[@]}" \
  "$@"
