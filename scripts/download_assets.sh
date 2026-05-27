#!/usr/bin/env bash
# Download minimal assets for pipeline B (spatial subset + official ckpt + Wan base).
set -euo pipefail

# 走本机 SSH RemoteForward 7890 代理（见 /DATA/disk1/yjh_space/use_proxy.sh）
if [[ -f /DATA/disk1/yjh_space/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk1/yjh_space/use_proxy.sh
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../code/FastWAM" && pwd)"
MINIMAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${FASTWAM_ROOT}/data/libero_mujoco3.3.2"
CKPT_DIR="${FASTWAM_ROOT}/checkpoints"
RELEASE_DIR="${CKPT_DIR}/fastwam_release"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${WORLD2WAM_CONDA_ENV:-world2wam}"

export DIFFSYNTH_MODEL_BASE_PATH="${CKPT_DIR}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"

mkdir -p "${DATA_DIR}" "${RELEASE_DIR}" "${CKPT_DIR}"

log() { echo "[$(date -Iseconds)] $*"; }

RELEASE_PT="${RELEASE_DIR}/libero_uncond_2cam224.pt"
if [[ -f "${RELEASE_PT}" ]] && [[ "$(stat -c%s "${RELEASE_PT}" 2>/dev/null || echo 0)" -gt 1000000000 ]]; then
  log "SKIP [1/5] official ckpt already present: ${RELEASE_PT}"
else
  log "==> [1/5] FastWAM official LIBERO checkpoint (~12GB)..."
  huggingface-cli download yuanty/fastwam \
    libero_uncond_2cam224.pt \
    libero_uncond_2cam224_dataset_stats.json \
    --local-dir "${RELEASE_DIR}"
fi

log "==> [2/5] LIBERO spatial LeRobot tar (~1GB)..."
SPATIAL_TAR="${DATA_DIR}/libero_spatial_no_noops_lerobot.tar.gz"
if [[ ! -f "${SPATIAL_TAR}" ]]; then
  huggingface-cli download yuanty/LIBERO-fastwam \
    libero_spatial_no_noops_lerobot.tar.gz \
    --repo-type dataset \
    --local-dir "${DATA_DIR}"
fi
if [[ ! -d "${DATA_DIR}/libero_spatial_no_noops_lerobot/data" ]]; then
  echo "    Extracting spatial dataset..."
  tar -xzf "${SPATIAL_TAR}" -C "${DATA_DIR}"
fi

DS_VAE="${CKPT_DIR}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
if [[ -f "${DS_VAE}" ]]; then
  log "SKIP [3/5] VAE already present"
else
log "==> [3/5] Wan VAE + T5 (DiffSynth via ModelScope, with proxy + retries)..."
python - <<'PY'
import os
import time
from modelscope import snapshot_download

base = os.environ["DIFFSYNTH_MODEL_BASE_PATH"]
root = os.path.join(base, "DiffSynth-Studio/Wan-Series-Converted-Safetensors")
os.makedirs(root, exist_ok=True)
patterns = ["Wan2.2_VAE.safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"]

last_err = None
for attempt in range(5):
    try:
        snapshot_download(
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            local_dir=root,
            allow_file_pattern=patterns,
        )
        last_err = None
        break
    except Exception as e:
        last_err = e
        print(f"modelscope attempt {attempt + 1}/5 failed: {e}")
        time.sleep(30)

if last_err is not None:
    raise last_err

missing = [p for p in patterns if not os.path.isfile(os.path.join(root, p))]
if missing:
    raise FileNotFoundError(f"Missing after download: {missing}")
print("DiffSynth OK:", root, os.listdir(root))
PY
fi

DIT_GLOB="${CKPT_DIR}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model"*
if compgen -G "${DIT_GLOB}" > /dev/null; then
  log "SKIP [4/5] Wan DiT already present"
else
log "==> [4/5] Wan2.2-TI2V-5B DiT weights (large, may take a while)..."
python - <<PY
from huggingface_hub import snapshot_download
import os
base = os.environ["DIFFSYNTH_MODEL_BASE_PATH"]
local_dir = os.path.join(base, "Wan-AI/Wan2.2-TI2V-5B")
snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B",
    local_dir=local_dir,
    allow_patterns=["diffusion_pytorch_model*.safetensors"],
)
print("Wan DiT OK:", local_dir)
PY
fi

if [[ -d "${CKPT_DIR}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" ]]; then
  log "SKIP [5/5] Tokenizer already present"
else
log "==> [5/5] Tokenizer (Wan2.1-T2V-1.3B umt5)..."
python - <<PY
from huggingface_hub import snapshot_download
import os
base = os.environ["DIFFSYNTH_MODEL_BASE_PATH"]
local_dir = os.path.join(base, "Wan-AI/Wan2.1-T2V-1.3B")
snapshot_download(
    "Wan-AI/Wan2.1-T2V-1.3B",
    local_dir=local_dir,
    allow_patterns=["google/umt5-xxl/**"],
)
print("Tokenizer OK:", local_dir)
PY
fi

ACTION_DIT="${CKPT_DIR}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
if [[ ! -f "${ACTION_DIT}" ]]; then
  echo "==> ActionDiT backbone (interpolate from Wan DiT)..."
  cd "${FASTWAM_ROOT}"
  python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output "${ACTION_DIT}" \
    --device cuda \
    --dtype bfloat16
fi

log "==> All asset downloads finished."
log "RELEASE_CKPT=${RELEASE_PT}"
log "DATA=${DATA_DIR}/libero_spatial_no_noops_lerobot"
