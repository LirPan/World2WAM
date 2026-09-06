#!/usr/bin/env bash
set -u

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
EVAL="$ROOT/deploy/run_iclr2027_robotwin_eval.sh"
FASTER="$ROOT/third_party/FasterWAM"
FAST_STATS=/DATA/disk0/yjh/robotwin_w2wam/third_party/FastWAM_official/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json
FAST_CKPT=/DATA/disk0/yjh/robotwin_w2wam/third_party/FastWAM_official/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
FASTER_CKPT=$FASTER/checkpoints/fasterwam_release/robotwin/step_029355.pt
FASTER_STATS=$FASTER/checkpoints/fasterwam_release/robotwin/dataset_stats.json
LOGDIR=$ROOT/runs/paper_sprint_v2/logs
LOCKROOT=$ROOT/.locks/robotwin_retry
mkdir -p "$LOGDIR" "$LOCKROOT"

gpu_is_free() {
  local gpu="$1" mem util pids
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 999999)"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 999999)"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')"
  [[ "$mem" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && -z "$pids" ]] || return 1
  (( mem < 1000 && util < 5 ))
}

run_one() {
  local name="$1" method="$2" ckpt="$3" stats="$4" task="$5"
  local out="$ROOT/results/robotwin/$name"
  [[ -s "$out/completion.json" ]] && return 0
  while :; do
    # GPU1 is reserved for the FasterWAM-World2WAM training queue.
    # Do not race the trainer even during its model-loading window, when
    # nvidia-smi may temporarily report low memory.
    for gpu in 2 4 6 7; do
      if ! mkdir "$LOCKROOT/gpu${gpu}" 2>/dev/null; then continue; fi
      if ! gpu_is_free "$gpu"; then rmdir "$LOCKROOT/gpu${gpu}"; continue; fi
      sleep 5
      if ! gpu_is_free "$gpu"; then rmdir "$LOCKROOT/gpu${gpu}"; continue; fi
      echo "[$(date)] start $name gpu=$gpu" >>"$LOGDIR/robotwin_retry.log"
      (
        trap 'rmdir "$LOCKROOT/gpu${gpu}" 2>/dev/null || true' EXIT
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$EVAL" "$method" "$ckpt" "$stats" "$out" "$task" 10 >"$LOGDIR/${name}.log" 2>&1
        rc=$?
        echo "[$(date)] finish $name gpu=$gpu rc=$rc" >>"$LOGDIR/robotwin_retry.log"
        exit "$rc"
      )
      return $?
    done
    sleep 60
  done
}

for spec in \
  "robotwin_full_B5_s43_v2|VersionD_s43|$ROOT/checkpoints/robotwin/B5_s43.pt|$FAST_STATS|robotwin_fastwam_3cam_384_1e-4" \
  "robotwin_full_B5_s44_v2|VersionD_s44|$ROOT/checkpoints/robotwin/B5_s44.pt|$FAST_STATS|robotwin_fastwam_3cam_384_1e-4" \
  "robotwin_full_fasterwam_v2|FasterWAM|$FASTER_CKPT|$FASTER_STATS|robotwin_fasterwam_3cam_384_1e-4"; do
  IFS='|' read -r name method ckpt stats task <<<"$spec"
  while [[ ! -s "$ckpt" ]]; do sleep 60; done
  run_one "$name" "$method" "$ckpt" "$stats" "$task" || true
done
