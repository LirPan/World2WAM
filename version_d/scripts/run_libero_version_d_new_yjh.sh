#!/usr/bin/env bash
# Resume-safe LIBERO-Spatial Version D train/export/paired-evaluation pipeline
# for the New_yjh server layout.
set -euo pipefail

POLICY_ROOT="${POLICY_ROOT:-/DATA/disk0/yjh/robotwin_w2wam/latest/code/policy_lora}"
FASTWAM_ROOT="${FASTWAM_ROOT:-/DATA/disk0/yjh/libero_work_wj}"
LIBERO_ROOT="${LIBERO_ROOT:-/DATA/disk0/yjh/world2wam/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO_fresh}"
PY="${PYTHON_BIN:-/DATA/disk0/yjh/libero_work_wj/env/libero_venv/bin/python}"
CONFIG="${LIBERO_VERSION_D_CONFIG:-$POLICY_ROOT/configs/libero_version_d_new_yjh.yaml}"
RUN_ROOT="${LIBERO_VERSION_D_RUN:-/DATA/disk0/yjh/libero_work_wj/runs/libero_version_d_spatial}"
CACHE_MAX_SAMPLES="${LIBERO_CACHE_MAX_SAMPLES:-12000}"
TRAIN_GPU="${LIBERO_TRAIN_GPU:-0}"
EVAL_GPU="${LIBERO_EVAL_GPU:-1}"
NUM_TRIALS="${LIBERO_NUM_TRIALS:-10}"
SEED="${LIBERO_SEED:-42}"
TASK_IDS="${LIBERO_TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"

TRAIN_BUNDLE="$RUN_ROOT/train/checkpoints/world2wam_final.pt"
MERGED_CKPT="$RUN_ROOT/exported/version_d_libero.pt"
OFFICIAL_CKPT="$FASTWAM_ROOT/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
STATS="${LIBERO_DATASET_STATS:-/DATA/disk0/yjh/libero_work_wj/checkpoints/fastwam_release/dataset_stats.json}"
EVAL_SCRIPT="$FASTWAM_ROOT/experiments/libero/eval_libero_single.py"
SUMMARIZER="$POLICY_ROOT/src/eval/summarize_libero_pair.py"

mkdir -p "$RUN_ROOT"/{logs,exported,eval/official,eval/version_d}
exec > >(tee -a "$RUN_ROOT/logs/pipeline.log") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
need() { [[ -e "$1" ]] || { log "missing required path: $1"; exit 2; }; }

need "$PY"
need "$CONFIG"
need "$OFFICIAL_CKPT"
need "$STATS"
need "$EVAL_SCRIPT"
need "$SUMMARIZER"

export PYTHONPATH="$POLICY_ROOT:$LIBERO_ROOT:$FASTWAM_ROOT:$FASTWAM_ROOT/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ ! -f "$RUN_ROOT/cache_manifest.complete" ]]; then
  log "precomputing future latents (max_samples=$CACHE_MAX_SAMPLES, gpu=$TRAIN_GPU)"
  cd "$POLICY_ROOT"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PY" src/data/precompute_future_latents.py \
    --config "$CONFIG" --max-samples "$CACHE_MAX_SAMPLES" --device cuda \
    --manifest "$RUN_ROOT/cache_manifest.json"
  touch "$RUN_ROOT/cache_manifest.complete"
fi

if [[ ! -f "$TRAIN_BUNDLE" ]]; then
  log "training Version D (F/I/C + project_conflicts, gpu=$TRAIN_GPU)"
  cd "$POLICY_ROOT"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PY" src/train/train_lora_fic_hardtask.py \
    --config "$CONFIG" --backbone-mode lora --gradient-mode project_conflicts
else
  log "reusing training bundle: $TRAIN_BUNDLE"
fi

if [[ ! -f "$MERGED_CKPT" ]]; then
  log "exporting merged LIBERO checkpoint"
  cd "$POLICY_ROOT"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" EXPORT_DEVICE=cuda "$PY" \
    src/tools/export_libero_checkpoint.py \
    --bundle "$TRAIN_BUNDLE" --config "$CONFIG" --output "$MERGED_CKPT"
fi

run_eval() {
  local label="$1" ckpt="$2" out="$3" task_id result
  for task_id in $TASK_IDS; do
    result="$out/libero_spatial/gpu0_task${task_id}_results.json"
    if [[ -s "$result" ]]; then
      log "skip completed ${label} task=${task_id}: $result"
      continue
    fi
    log "evaluate ${label} task=${task_id}, trials=$NUM_TRIALS, seed=$SEED, gpu=$EVAL_GPU"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" xvfb-run -a "$PY" "$EVAL_SCRIPT" \
      "ckpt=$ckpt" \
      "seed=$SEED" \
      "EVALUATION.task_suite_name=libero_spatial" \
      "EVALUATION.task_id=$task_id" \
      "EVALUATION.num_trials=$NUM_TRIALS" \
      "gpu_id=0" \
      "EVALUATION.device=cuda" \
      "EVALUATION.dataset_stats_path=$STATS" \
      "EVALUATION.output_dir=$out"
  done
}

run_eval official "$OFFICIAL_CKPT" "$RUN_ROOT/eval/official"
run_eval version_d "$MERGED_CKPT" "$RUN_ROOT/eval/version_d"

cd "$POLICY_ROOT"
"$PY" "$SUMMARIZER" \
  --official-dir "$RUN_ROOT/eval/official" \
  --version-d-dir "$RUN_ROOT/eval/version_d" \
  --task-ids $TASK_IDS \
  --output "$RUN_ROOT/libero_pair_summary.json"

log "complete: $RUN_ROOT/libero_pair_summary.json"
