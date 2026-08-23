#!/usr/bin/env bash
# Train/export/evaluate LIBERO Version D after the RobotWin run is complete.
# The evaluation uses the same task IDs, trials, seed, evaluator, and success
# criterion for official FastWAM and Version D. It never selects favorable rows.
set -euo pipefail

ROOT="${POLICY_ROOT:-/DATA/disk0/jianhua/latest/code/policy_lora}"
FASTWAM_ROOT="${FASTWAM_ROOT:-/DATA/disk0/jianhua/_shared/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM}"
LIBERO_ROOT="${LIBERO_ROOT:-/DATA/disk0/jianhua/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO_fresh}"
PY="${PYTHON_BIN:-/DATA/disk0/jianhua/miniconda3/envs/world2wam/bin/python}"
CONFIG="${LIBERO_VERSION_D_CONFIG:-$ROOT/configs/libero_version_d_fiveages.yaml}"
RUN_ROOT="${LIBERO_VERSION_D_RUN:-/DATA/disk0/jianhua/latest/experiments/iclr_2027/libero_version_d}"
CACHE_MAX_SAMPLES="${LIBERO_CACHE_MAX_SAMPLES:-12000}"
TRAIN_GPU="${LIBERO_TRAIN_GPU:-0}"
EVAL_GPU="${LIBERO_EVAL_GPU:-1}"
NUM_TRIALS="${LIBERO_NUM_TRIALS:-10}"
SEED="${LIBERO_SEED:-42}"
TASK_IDS="${LIBERO_TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
ROBOTWIN_DONE_MARKER="${ROBOTWIN_DONE_MARKER:-/DATA/disk0/jianhua/latest/experiments/iclr_2027/robotwin_lora_fic_aligned/status/robotwin_complete}"
ROBOTWIN_R3_CHECKPOINT="${ROBOTWIN_R3_CHECKPOINT:-/DATA/disk0/jianhua/latest/checkpoints/world2wam_r3_merged.pt}"
WAIT_FOR_ROBOTWIN="${WAIT_FOR_ROBOTWIN:-1}"

TRAIN_BUNDLE="$RUN_ROOT/train/checkpoints/world2wam_final.pt"
MERGED_CKPT="$RUN_ROOT/exported/version_d_libero.pt"
OFFICIAL_CKPT="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
STATS="${FASTWAM_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
EVAL_SCRIPT="${FASTWAM_ROOT}/experiments/libero/eval_libero_single.py"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/eval/official" "$RUN_ROOT/eval/version_d" "$RUN_ROOT/exported"
exec > >(tee -a "$RUN_ROOT/logs/libero_version_d_pipeline.log") 2>&1

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

if [[ "$WAIT_FOR_ROBOTWIN" == "1" ]]; then
  log "Waiting for RobotWin completion marker or idle state: $ROBOTWIN_DONE_MARKER"
  while true; do
    if [[ -f "$ROBOTWIN_DONE_MARKER" ]]; then
      log "RobotWin completion marker found"
      break
    fi
    active_jobs="$(pgrep -af 'train_lora_fic_hardtask.py|eval_robotwin_single.py|run_paired_(fixed5|hard10)' || true)"
    if [[ -f "$ROBOTWIN_R3_CHECKPOINT" && -z "$active_jobs" ]]; then
      log "No active RobotWin train/eval process and R3 checkpoint exists; continuing"
      break
    fi
    sleep 60
  done
fi

[[ -f "$CONFIG" ]] || { log "Missing config: $CONFIG"; exit 1; }
[[ -f "$OFFICIAL_CKPT" ]] || { log "Missing official checkpoint: $OFFICIAL_CKPT"; exit 1; }
[[ -f "$STATS" ]] || { log "Missing dataset stats: $STATS"; exit 1; }
[[ -f "$EVAL_SCRIPT" ]] || { log "Missing evaluator: $EVAL_SCRIPT"; exit 1; }

export PYTHONPATH="${LIBERO_ROOT}:${FASTWAM_ROOT}:${FASTWAM_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ ! -f "$RUN_ROOT/cache_manifest.complete" ]]; then
  log "Precomputing LIBERO future latents"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PY" src/data/precompute_future_latents.py \
    --config "$CONFIG" --max-samples "$CACHE_MAX_SAMPLES" --device cuda \
    --manifest "$RUN_ROOT/cache_manifest.json"
  touch "$RUN_ROOT/cache_manifest.complete"
fi

if [[ ! -f "$TRAIN_BUNDLE" ]]; then
  log "Training LIBERO Version D (7D, F/I/C, project_conflicts)"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PY" src/train/train_lora_fic_hardtask.py \
    --config "$CONFIG" --backbone-mode lora --gradient-mode project_conflicts
else
  log "Reusing existing Version D bundle: $TRAIN_BUNDLE"
fi

if [[ ! -f "$MERGED_CKPT" ]]; then
  log "Exporting Version D to official LIBERO checkpoint format"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" EXPORT_DEVICE=cuda "$PY" src/tools/export_libero_checkpoint.py \
    --bundle "$TRAIN_BUNDLE" --config "$CONFIG" --output "$MERGED_CKPT"
fi

run_eval() {
  local label="$1" ckpt="$2" out="$3" task_id
  for task_id in $TASK_IDS; do
    log "Evaluating ${label}: task=${task_id}, trials=${NUM_TRIALS}, seed=${SEED}"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" xvfb-run -a "$PY" "$EVAL_SCRIPT" \
      "ckpt=$ckpt" \
      "seed=$SEED" \
      "EVALUATION.task_suite_name=libero_spatial" \
      "EVALUATION.task_id=$task_id" \
      "EVALUATION.num_trials=$NUM_TRIALS" \
      "EVALUATION.gpu_id=0" \
      "EVALUATION.device=cuda" \
      "EVALUATION.dataset_stats_path=$STATS" \
      "EVALUATION.output_dir=$out"
  done
}

run_eval official "$OFFICIAL_CKPT" "$RUN_ROOT/eval/official"
run_eval version_d "$MERGED_CKPT" "$RUN_ROOT/eval/version_d"

cd "$ROOT"
"$PY" "$ROOT/src/eval/summarize_libero_pair.py" \
  --official-dir "$RUN_ROOT/eval/official" \
  --version-d-dir "$RUN_ROOT/eval/version_d" \
  --task-ids $TASK_IDS \
  --output "$RUN_ROOT/libero_pair_summary.json"

touch "$RUN_ROOT/libero_version_d_complete"
log "LIBERO Version D pipeline complete: $RUN_ROOT/libero_pair_summary.json"
