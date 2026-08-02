#!/usr/bin/env bash
# Action-only LoRA + hard-task reweight (no future_latent required).
# Usage:
#   bash scripts/train_policy_lora_action.sh
#   bash scripts/train_policy_lora_action.sh --max-steps 200
set -euo pipefail
PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_LORA_ROOT="${POLICY_LORA_ROOT:-${PA_ROOT}/policy_lora}"
CONFIG="${CONFIG:-configs/world2wam_policy_lora_action_hard.yaml}"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
cd "${POLICY_LORA_ROOT}"
export PYTHONPATH="${POLICY_LORA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -u -m src.train.train_lora_action_hard \
  --config "${CONFIG}" \
  --backbone-mode lora \
  "$@"
