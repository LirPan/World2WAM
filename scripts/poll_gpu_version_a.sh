#!/usr/bin/env bash
# Poll GPUs until free, then run Version A full pipeline in background (survives Cursor exit).
#
# Usage:
#   bash minimal_world2wam/scripts/poll_gpu_version_a.sh          # start (nohup)
#   bash minimal_world2wam/scripts/poll_gpu_version_a.sh status
#   bash minimal_world2wam/scripts/poll_gpu_version_a.sh tail
#   bash minimal_world2wam/scripts/poll_gpu_version_a.sh stop
#
# Env overrides:
#   CONFIG, CACHE_OUT, OUT_DIR, MAX_SAMPLES, MAX_TASKS, NUM_TRIALS
#   GPU_POLL_SEC (default 120), TRAIN_GPU_FREE_MIN_MB, PRECOMPUTE_GPU_FREE_MIN_MB
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
JOB_DIR="${WORKSPACE}/cache/bg_jobs"
JOB_NAME="version_a_poll"
PIDFILE="${JOB_DIR}/${JOB_NAME}.pid"
LOGFILE="${JOB_DIR}/${JOB_NAME}.log"
PIPELINE="${WORKSPACE}/minimal_world2wam/scripts/run_version_a_full_pipeline.sh"

mkdir -p "${JOB_DIR}"

_cmd_status() {
  if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "[RUNNING] ${JOB_NAME} pid=$(cat "${PIDFILE}")"
    echo "  log: ${LOGFILE}"
    echo "  pipeline log: ${JOB_DIR}/version_a_full.log"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
  elif [[ -f "${LOGFILE}" ]]; then
    echo "[STOPPED/DONE] ${JOB_NAME}"
    echo "  log: ${LOGFILE}"
    tail -n 5 "${LOGFILE}" 2>/dev/null | sed 's/^/  /'
    if [[ -f "${WORKSPACE}/experiments/VERSION_A_SUMMARY.json" ]]; then
      echo "  summary: ${WORKSPACE}/experiments/VERSION_A_SUMMARY.json"
      python3 "${WORKSPACE}/minimal_world2wam/scripts/summarize_version_a_results.py" 2>/dev/null || true
    fi
  else
    echo "[NOT STARTED] ${JOB_NAME}"
  fi
}

_cmd_stop() {
  if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    kill "$(cat "${PIDFILE}")" 2>/dev/null || true
    echo "Sent SIGTERM to pid=$(cat "${PIDFILE}")"
  else
    echo "Not running"
  fi
}

_run_poll_loop() {
  local pipeline="${WORKSPACE}/Physics-Aligned World2WAM/scripts/run_version_a_full_pipeline.sh"
  if [[ ! -f "${pipeline}" ]]; then
    pipeline="${WORKSPACE}/minimal_world2wam/scripts/run_version_a_full_pipeline.sh"
  fi
  if [[ ! -f "${pipeline}" ]]; then
    echo "$(date -Iseconds) ERROR: pipeline script not found under WORKSPACE=${WORKSPACE}" >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "${WORKSPACE}/Physics-Aligned World2WAM/scripts/gpu_poll_utils.sh" 2>/dev/null || \
    source "${WORKSPACE}/minimal_world2wam/scripts/gpu_poll_utils.sh"

  echo "$(date -Iseconds) version_a_poll: waiting for any GPU stage threshold..."
  echo "  poll every ${GPU_POLL_SEC}s | precompute>=${PRECOMPUTE_GPU_FREE_MIN_MB:-70000}MiB train>=${TRAIN_GPU_FREE_MIN_MB:-45000}MiB"
  echo "  pipeline=${pipeline}"

  # Wait until at least one GPU can run the least demanding upcoming stage (eval).
  while true; do
    mapfile -t gpus < <(_pick_gpus "${EVAL_GPU_FREE_MIN_MB:-30000}" "${EVAL_GPU_MAX_UTIL:-30}" | head -n 1)
    status="$(_gpu_status_line)"
    if ((${#gpus[@]} >= 1)); then
      echo "$(date -Iseconds) GPU available (eval threshold met on GPU ${gpus[0]}). Starting pipeline. ${status}"
      break
    fi
    echo "$(date -Iseconds) all GPUs busy. ${status}"
    sleep "${GPU_POLL_SEC:-120}"
  done

  export USE_GPU_POLL=1
  bash "${pipeline}"
}

case "${1:-start}" in
  status)
    _cmd_status
    ;;
  tail)
    tail -f "${LOGFILE}"
    ;;
  stop)
    _cmd_stop
    ;;
  start|poll)
    if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "Already running: pid=$(cat "${PIDFILE}") log=${LOGFILE}"
      exit 0
    fi
    echo "Starting ${JOB_NAME} in background -> ${LOGFILE}"
    echo "Pipeline detail log -> ${JOB_DIR}/version_a_full.log"
    nohup bash -c "
      set -euo pipefail
      WORKSPACE='${WORKSPACE}'
      CONFIG='${CONFIG:-configs/world2wam_physics_flow_dit_main.yaml}'
      CACHE_OUT='${CACHE_OUT:-cache/libero_spatial_h10_full_fastwam}'
      OUT_DIR='${OUT_DIR:-experiments/world2wam_physics_flow_dit_main}'
      MAX_SAMPLES='${MAX_SAMPLES:-600000}'
      MAX_TASKS='${MAX_TASKS:-10}'
      NUM_TRIALS='${NUM_TRIALS:-50}'
      GPU_POLL_SEC='${GPU_POLL_SEC:-120}'
      PRECOMPUTE_GPU_FREE_MIN_MB='${PRECOMPUTE_GPU_FREE_MIN_MB:-70000}'
      TRAIN_GPU_FREE_MIN_MB='${TRAIN_GPU_FREE_MIN_MB:-45000}'
      EVAL_GPU_FREE_MIN_MB='${EVAL_GPU_FREE_MIN_MB:-30000}'
      export CONFIG CACHE_OUT OUT_DIR MAX_SAMPLES MAX_TASKS NUM_TRIALS
      export GPU_POLL_SEC PRECOMPUTE_GPU_FREE_MIN_MB TRAIN_GPU_FREE_MIN_MB EVAL_GPU_FREE_MIN_MB
      source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
      conda activate world2wam
      export PYTHONPATH=\"\${WORKSPACE}:\${PYTHONPATH:-}\"
      $(declare -f _run_poll_loop)
      _run_poll_loop
    " >> "${LOGFILE}" 2>&1 &
    echo $! > "${PIDFILE}"
    disown -h $! 2>/dev/null || true
    echo "pid=$(cat "${PIDFILE}")"
    echo "Check: bash minimal_world2wam/scripts/poll_gpu_version_a.sh status"
    echo "Tail:  bash minimal_world2wam/scripts/poll_gpu_version_a.sh tail"
    ;;
  *)
    echo "Usage: $0 {start|status|tail|stop}"
    exit 1
    ;;
esac
