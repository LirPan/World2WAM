#!/usr/bin/env bash
# Repair corrupted FastWAM release ckpt + regenerate ActionDiT, then auto-run experiment pipeline.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
FASTWAM_ROOT="${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM"
RELEASE_DIR="${FASTWAM_ROOT}/checkpoints/fastwam_release"
ACTION_DIT_OUT="${FASTWAM_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
HF_REPO="https://huggingface.co/yuanty/fastwam/resolve/main"
LIBERO_PT_EXPECTED_BYTES="${LIBERO_PT_EXPECTED_BYTES:-12041735140}"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
if [[ -f /DATA/disk0/jianhua/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk0/jianhua/use_proxy.sh || true
fi

# shellcheck disable=SC1091
source "${WORKSPACE}/minimal_world2wam/scripts/device_utils.sh"

mkdir -p "${RELEASE_DIR}"

download_hf_file() {
  local fname="$1"
  local expected="${2:-0}"
  local dest="${3:-${RELEASE_DIR}/${fname}}"
  local url="${HF_REPO}/${fname}"
  echo "== Download ${fname} =="
  if [[ -f "${dest}" ]]; then
    local sz
    sz=$(stat -c%s "${dest}" 2>/dev/null || echo 0)
    if [[ "${expected}" -gt 0 && "${sz}" -ge "${expected}" ]]; then
      echo "Already present: ${dest} (${sz} bytes)"
      return 0
    fi
    echo "Removing incomplete ${dest} (${sz} bytes, expected ${expected})"
    rm -f "${dest}"
  fi
  if [[ -z "${http_proxy:-}" ]]; then
    echo "ERROR: lab proxy required (source use_proxy.sh)" >&2
    exit 1
  fi
  curl -fL --retry 8 --retry-delay 20 -x "${http_proxy}" -o "${dest}.part" "${url}"
  mv -f "${dest}.part" "${dest}"
  ls -lh "${dest}"
}

validate_libero_ckpt() {
  python - <<'PY'
import sys
import torch
from pathlib import Path

p = Path("/DATA/disk0/jianhua/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt")
expected = int(__import__("os").environ.get("LIBERO_PT_EXPECTED_BYTES", "12041735140"))
size = p.stat().st_size
if size < expected * 0.99:
    raise SystemExit(f"libero ckpt too small: {size} < {expected}")
obj = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(obj, dict):
    raise SystemExit(f"unexpected payload type: {type(obj)}")
print(f"OK libero_uncond_2cam224.pt size={size} keys={len(obj)}")
PY
}

validate_action_dit() {
  python - <<'PY'
import torch
from pathlib import Path

p = Path("/DATA/disk0/jianhua/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt")
payload = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(payload, dict) or "backbone_state_dict" not in payload:
    raise SystemExit("ActionDiT payload missing backbone_state_dict")
n = len(payload["backbone_state_dict"])
if n < 100:
    raise SystemExit(f"ActionDiT too few keys: {n}")
print(f"OK ActionDiT keys={n} size={p.stat().st_size}")
PY
}

preprocess_action_dit() {
  if [[ -f "${ACTION_DIT_OUT}" ]] && validate_action_dit 2>/dev/null; then
    echo "ActionDiT already valid, skip preprocess"
    return 0
  fi
  rm -f "${ACTION_DIT_OUT}"

  local device dtype
  device="$(pick_compute_device)"
  if [[ "${device}" == "cuda" ]]; then
    dtype="bfloat16"
  else
    dtype="float32"
  fi
  log_device_pick "action_dit_preprocess" "${device}"
  cd "${FASTWAM_ROOT}"
  python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output "${ACTION_DIT_OUT}" \
    --device "${device}" \
    --dtype "${dtype}"
}

echo "== repair_fastwam_ckpts: $(date -Iseconds) =="

# Download libero ckpt (~12GB) in background while regenerating ActionDiT.
download_hf_file "libero_uncond_2cam224.pt" "${LIBERO_PT_EXPECTED_BYTES}" &
DL_PID=$!

echo "== Regenerate ActionDiT (parallel with libero download) =="
preprocess_action_dit

echo "== Wait for libero checkpoint download =="
wait "${DL_PID}"

echo "== Validate checkpoints =="
validate_libero_ckpt
validate_action_dit

echo "== Launch auto experiment pipeline =="
cd "${WORKSPACE}"
bash minimal_world2wam/scripts/run_auto_pipeline.sh

echo "== repair_fastwam_ckpts_and_run done: $(date -Iseconds) =="
