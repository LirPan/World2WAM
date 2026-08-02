#!/usr/bin/env bash
# Merge LoRA bundle -> official FastWAM .pt for eval_libero_single.py
set -euo pipefail
PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_LORA_ROOT="${POLICY_LORA_ROOT:-${PA_ROOT}/policy_lora}"
IDEA2_ROOT="${IDEA2_ROOT:-${POLICY_LORA_ROOT}}"
BUNDLE="${BUNDLE:-/DATA/disk0/jianhua/experiments/world2wam_policy_lora_fic_hard/checkpoints/world2wam_final.pt}"
OUT="${OUT:-/DATA/disk0/jianhua/experiments/exported_ckpts/world2wam_lora_fic_hard_merged.pt}"
CONFIG="${CONFIG:-configs/world2wam_policy_lora_fic_hard.yaml}"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
cd "${IDEA2_ROOT}"
export PYTHONPATH="${IDEA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$(dirname "${OUT}")"

exec python -u -m src.tools.export_libero_checkpoint \
  --bundle "${BUNDLE}" \
  --config "${CONFIG}" \
  --output "${OUT}" \
  "$@"
