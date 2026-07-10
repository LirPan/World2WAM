#!/usr/bin/env bash
# Repair corrupted Wan2.2 DiT shards, then run full debug pipeline.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
WAN_DIR="${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B"

source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
unset PYTHONHOME
if [[ -f /DATA/disk0/jianhua/use_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source /DATA/disk0/jianhua/use_proxy.sh || true
fi

mkdir -p "${WAN_DIR}"
cd "${WAN_DIR}"

download_one() {
  local fname="$1"
  local url="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/${fname}"
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

validate_shards() {
  python - <<'PY'
from pathlib import Path
from safetensors import safe_open
base = Path("/DATA/disk0/jianhua/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B")
bad = []
for p in sorted(base.glob("diffusion_pytorch_model-*.safetensors")):
    try:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            _ = list(f.keys())[:1]
        print("OK", p.name)
    except Exception as e:
        print("BAD", p.name, repr(e))
        bad.append(p.name)
if bad:
    raise SystemExit(1)
PY
}

# 00001 and 00002 are known corrupted; redownload from scratch.
download_one "diffusion_pytorch_model-00001-of-00003.safetensors"
download_one "diffusion_pytorch_model-00002-of-00003.safetensors"

echo "== Validate Wan shards =="
validate_shards

echo "== Wan repaired, start pipeline =="
cd "${WORKSPACE}"
bash minimal_world2wam/scripts/wait_and_run_pipeline.sh
