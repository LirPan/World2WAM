#!/usr/bin/env bash
# Create conda env `world2wam` with FastWAM + minimal_world2wam deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_ROOT="$(cd "${ROOT}/../code/FastWAM" && pwd)"
ENV_NAME="${WORLD2WAM_CONDA_ENV:-world2wam}"

echo "==> Project root: ${ROOT}"
echo "==> FastWAM root: ${FASTWAM_ROOT}"
echo "==> Conda env: ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Load miniconda/anaconda first."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Env ${ENV_NAME} exists; reusing (set RECREATE=1 to remove first)."
  if [[ "${RECREATE:-0}" == "1" ]]; then
    conda env remove -n "${ENV_NAME}" -y
    conda create -n "${ENV_NAME}" python=3.10 -y
  fi
else
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"
pip install -U pip wheel setuptools

echo "==> Installing PyTorch (cu128 preferred; fallback PyPI cu12 bundle)..."
if ! pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --default-timeout=600 2>/dev/null; then
  echo "WARN: cu128 wheel unavailable or timed out; installing PyPI torch 2.7.1 (bundled CUDA 12.x)."
  pip install torch==2.7.1 torchvision==0.22.1 --default-timeout=600
fi

echo "==> Installing FastWAM (editable, deps without re-pinning torch)..."
pip install -e "${FASTWAM_ROOT}" --no-deps
pip install \
  "accelerate==1.12.0" "av==16.0.1" "boto3==1.35.99" "datasets==3.6.0" "deepspeed==0.18.5" \
  "einops==0.8.1" "gitpython==3.1.45" "huggingface-hub==0.29.2" "hydra-core==1.3.2" \
  "imageio==2.37.0" "imageio-ffmpeg==0.6.0" "jsonlines==4.0.0" "modelscope==1.34.0" \
  "numpy==1.26.4" "omegaconf==2.3.0" "packaging==25.0" "pandas==2.2.3" "pillow>=12.0.0" \
  "pyarrow==23.0.0" "regex==2025.11.3" "rich==14.2.0" "safetensors==0.5.3" \
  "termcolor==2.5.0" "torchcodec==0.5" "tqdm==4.66.5" "transformers==4.49.0" \
  "peft>=0.10.0" "future" "matplotlib" \
  "wandb==0.23.1" h5py pyyaml --default-timeout=300

# Optional: LIBERO for sim eval only (not required for LeRobot pipeline)
LIBERO_ROOT="$(cd "${ROOT}/../code/LIBERO" && pwd)"
if [[ -f "${LIBERO_ROOT}/requirements.txt" ]]; then
  echo "==> Installing LIBERO requirements (eval)..."
  pip install -r "${LIBERO_ROOT}/requirements.txt" || echo "WARN: LIBERO pip install had issues; sim eval may need manual fix."
fi

echo "==> Verifying imports..."
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import fastwam
print("fastwam", fastwam.__file__)
import hydra
print("hydra OK")
PY

cat <<EOF

================================================================================
Done. Activate with:

  conda activate ${ENV_NAME}

Quick test:

  cd ${ROOT}
  bash scripts/smoke_test_framework.sh

See docs/RUN_EXPERIMENTS.md for full experiment workflow.
================================================================================
EOF
