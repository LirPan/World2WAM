#!/usr/bin/env bash
# Grid sweep: bidirectional ablation (mode x lambda_cycle x future_horizon).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/scripts/activate_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/activate_env.sh"
fi

if [[ -n "${CUDA_DEVICE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
fi

SMOKE="${SMOKE:-0}"
ABLATION_DEVICE="${ABLATION_DEVICE:-cuda}"
MODES="${MODES:-forward_only,bidirectional,cycle}"
LAMBDAS="${LAMBDAS:-0,0.1,0.5,1.0}"
HORIZONS="${HORIZONS:-1,2,4}"
ABLATIONS_ROOT="${ABLATIONS_ROOT:-${ROOT}/experiments/ablations}"
GEN_CFG_DIR="${ROOT}/configs/ablation"
mkdir -p "${ABLATIONS_ROOT}" "${GEN_CFG_DIR}"

PRECOMPUTE_MAX="${PRECOMPUTE_MAX:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ "${SMOKE}" == "1" ]]; then
  MODES="cycle"
  LAMBDAS="0.1"
  HORIZONS="1"
  PRECOMPUTE_MAX=128
  MAX_TRAIN_STEPS=50
  NUM_EPOCHS=1
  BATCH_SIZE=8
fi

# Quick end-to-end validation: all three modes, one lambda/horizon.
if [[ "${VALIDATE:-0}" == "1" ]]; then
  MODES="forward_only,bidirectional,cycle"
  LAMBDAS="0.1"
  HORIZONS="1"
  PRECOMPUTE_MAX=256
  MAX_TRAIN_STEPS=80
  NUM_EPOCHS=1
  BATCH_SIZE=8
fi

IFS=',' read -r -a MODE_ARR <<< "${MODES}"
IFS=',' read -r -a LAMBDA_ARR <<< "${LAMBDAS}"
IFS=',' read -r -a HORIZON_ARR <<< "${HORIZONS}"

ensure_horizon_cache() {
  local horizon="$1"
  local cfg="${GEN_CFG_DIR}/precompute_h${horizon}.yaml"
  cat > "${cfg}" <<EOF
project_name: world2wam_minimal
fastwam_root: ../code/FastWAM
libero_root: ../code/LIBERO
fastwam_task_config: libero_uncond_2cam224_1e-4
future_horizon: ${horizon}
cache_dir: ./data/future_latents
lerobot_dataset_dirs:
  - ../code/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
dataset_stats_path: ../code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
device: cuda
mixed_precision: bf16
EOF
  python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT}")
from src.utils.config import load_config
from src.utils.checkpoint_utils import verify_future_latent_cache
from src.utils.path_utils import resolve_path
cfg = load_config(resolve_path("${cfg}", Path("${ROOT}")))
try:
    verify_future_latent_cache(cfg)
    print("cache ok horizon=${horizon}")
except FileNotFoundError:
    print("cache missing horizon=${horizon}")
    raise SystemExit(1)
PY
  if [[ $? -ne 0 ]]; then
    echo "==> Precompute future latents for horizon=${horizon}"
    python src/data/precompute_future_latents.py --config "${cfg}"
  fi
}

for horizon in "${HORIZON_ARR[@]}"; do
  ensure_horizon_cache "${horizon}" || true
  if ! python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT}")
from src.utils.config import load_config
from src.utils.checkpoint_utils import verify_future_latent_cache
from src.utils.path_utils import resolve_path
cfg_path = Path("${GEN_CFG_DIR}/precompute_h${horizon}.yaml")
cfg = load_config(resolve_path(str(cfg_path), Path("${ROOT}")))
verify_future_latent_cache(cfg)
PY
  then
    python src/data/precompute_future_latents.py --config "${GEN_CFG_DIR}/precompute_h${horizon}.yaml"
  fi
done

for mode in "${MODE_ARR[@]}"; do
  for lc in "${LAMBDA_ARR[@]}"; do
    for horizon in "${HORIZON_ARR[@]}"; do
      run_name="${mode}_lc${lc}_h${horizon}"
      out_dir="${ABLATIONS_ROOT}/${run_name}"
      cfg_path="${GEN_CFG_DIR}/${run_name}.yaml"
      MAX_STEPS_YAML="${MAX_TRAIN_STEPS:-null}"

      if [[ -f "${out_dir}/eval/offline_head_eval.json" ]]; then
        echo "==> SKIP existing ${run_name}"
        continue
      fi

      cat > "${cfg_path}" <<EOF
project_name: world2wam_minimal
fastwam_root: ../code/FastWAM
libero_root: ../code/LIBERO
fastwam_task_config: libero_uncond_2cam224_1e-4
experiment_role: bidirectional_analysis
official_fastwam_checkpoint: ../code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt
backbone_mode: frozen
freeze_fastwam_backbone: true
future_horizon: ${horizon}
future_latent_dim: 48
hidden_dim: 1024
action_dim: null
use_gt_action_for_forward_head: true
use_target_future_for_inverse_head: true
lambda_fwd: 1.0
lambda_inv: 1.0
lambda_cycle: ${lc}
batch_size: ${BATCH_SIZE}
num_epochs: ${NUM_EPOCHS}
lr: 1.0e-4
seed: 42
num_workers: 4
device: ${ABLATION_DEVICE}
lerobot_dataset_dirs:
  - ../code/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
cache_dir: ./data/future_latents
output_dir: ./experiments/ablations/${run_name}
dataset_stats_path: ../code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
mixed_precision: bf16
log_every: 10
save_every: 500
precompute_max_samples: ${PRECOMPUTE_MAX}
max_train_steps: ${MAX_STEPS_YAML:-null}
inverse_hidden_size: 1024
inverse_dropout: 0.0
EOF

      echo "==> Train ${run_name} mode=${mode}"
      python -m src.train.train_bidirectional_world2wam \
        --config "${cfg_path}" \
        --mode "${mode}"

      echo "==> Eval ${run_name}"
      python -m src.eval.eval_bidirectional_heads \
        --config "${cfg_path}" \
        --max-batches 20
    done
  done
done

python -m src.eval.summarize_ablations \
  --ablations-root "${ABLATIONS_ROOT}"

echo "==> Ablation sweep done: ${ABLATIONS_ROOT}/summary.csv"
