#!/usr/bin/env bash
# Repair corrupted VAE/T5 weights by downloading .pth from Wan-AI and converting to safetensors.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
FASTWAM_ROOT="${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM"
DS_DIR="${FASTWAM_ROOT}/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
WAN_REPO="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
if [[ -f /DATA/disk0/jianhua/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk0/jianhua/use_proxy.sh || true
fi

mkdir -p "${DS_DIR}"
cd "${DS_DIR}"

download_pth() {
  local fname="$1"
  local url="${WAN_REPO}/${fname}"
  echo "== Download ${fname} =="
  rm -f "${fname}"
  if [[ -n "${http_proxy:-}" ]]; then
    curl -fL --retry 8 --retry-delay 20 -x "${http_proxy}" -o "${fname}.part" "${url}"
  else
    curl -fL --retry 8 --retry-delay 20 -o "${fname}.part" "${url}"
  fi
  mv -f "${fname}.part" "${fname}"
  ls -lh "${fname}"
}

convert_to_safetensors() {
  python - <<'PY'
from pathlib import Path
import torch
from safetensors.torch import save_file

base = Path("/DATA/disk0/jianhua/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors")
pairs = [
    ("Wan2.2_VAE.pth", "Wan2.2_VAE.safetensors"),
    ("models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-bf16.safetensors"),
]

def flatten_state_dict(payload):
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(payload)}")
    if len(payload) == 1:
        for key in ("state_dict", "module", "model_state"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    return {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}

for src_name, dst_name in pairs:
    src = base / src_name
    dst = base / dst_name
    if not src.is_file():
        raise FileNotFoundError(src)
    print(f"== Convert {src_name} -> {dst_name} ==")
    state = torch.load(src, map_location="cpu", weights_only=True)
    tensors = flatten_state_dict(state)
    if not tensors:
        raise ValueError(f"No tensors found in {src_name}")
    save_file(tensors, str(dst))
    print(f"saved {dst_name} tensors={len(tensors)} size={dst.stat().st_size}")
PY
}

validate_files() {
  python - <<'PY'
from pathlib import Path
from safetensors import safe_open

base = Path("/DATA/disk0/jianhua/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors")
for name in ("Wan2.2_VAE.safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"):
    p = base / name
    with safe_open(str(p), framework="pt", device="cpu") as f:
        keys = len(f.keys())
    print(f"OK {name} keys={keys}")
PY
}

rm -f Wan2.2_VAE.safetensors models_t5_umt5-xxl-enc-bf16.safetensors
download_pth "Wan2.2_VAE.pth"
download_pth "models_t5_umt5-xxl-enc-bf16.pth"
convert_to_safetensors
echo "== Validate VAE/T5 =="
validate_files

echo "== Restart eval_compare =="
cd "${WORKSPACE}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}" NUM_TRIALS="${NUM_TRIALS:-1}" MAX_TASKS="${MAX_TASKS:-1}" \
  bash minimal_world2wam/scripts/bg_launch.sh eval_compare

echo "== repair_vae_t5_and_eval done: $(date -Iseconds) =="
