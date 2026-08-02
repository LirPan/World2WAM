#!/usr/bin/env bash
# Precompute future latents for LoRA+F/I/C training.
set -euo pipefail
PA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_LORA_ROOT="${POLICY_LORA_ROOT:-${PA_ROOT}/policy_lora}"
IDEA2_ROOT="${IDEA2_ROOT:-${POLICY_LORA_ROOT}}"
source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
cd "${IDEA2_ROOT}"
export PYTHONPATH="${IDEA2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -u src/data/precompute_future_latents.py \
  --config configs/world2wam_policy_lora_fic_hard.yaml \
  "$@"
