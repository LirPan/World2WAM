#!/usr/bin/env bash
# LoRA + Forward/Inverse/Cycle + hard-task reweight (vendored policy_lora/).
# Usage:
#   bash scripts/train_policy_lora_fic.sh                 # full
#   bash scripts/train_policy_lora_fic.sh --max-steps 200  # smoke
set -euo pipefail

PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_LORA_ROOT="${POLICY_LORA_ROOT:-${PA_ROOT}/policy_lora}"
# Backward-compatible override
IDEA2_ROOT="${IDEA2_ROOT:-${POLICY_LORA_ROOT}}"
CONFIG="${CONFIG:-configs/world2wam_policy_lora_fic_hard.yaml}"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam

cd "${IDEA2_ROOT}"
export PYTHONPATH="${IDEA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${IDEA2_ROOT}/${CONFIG}" ]]; then
  if [[ -f "${PA_ROOT}/configs/world2wam_policy_lora_fic_hard.yaml" ]]; then
    mkdir -p "$(dirname "${IDEA2_ROOT}/${CONFIG}")"
    cp -f "${PA_ROOT}/configs/world2wam_policy_lora_fic_hard.yaml" "${IDEA2_ROOT}/${CONFIG}"
  else
    echo "ERROR: missing config ${IDEA2_ROOT}/${CONFIG}"
    exit 1
  fi
fi

exec python -u -m src.train.train_lora_fic_hardtask \
  --config "${CONFIG}" \
  --backbone-mode lora \
  "$@"
