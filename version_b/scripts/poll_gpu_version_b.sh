#!/usr/bin/env bash
# Poll GPU then run Version B full pipeline (precompute future latents + physics MoT train).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/DATA/disk0/jianhua}"
VB_ROOT="${WORKSPACE}/Physics-Aligned World2WAM/version_b"
JOB_DIR="${WORKSPACE}/cache/bg_jobs"
JOB_NAME="version_b_poll"
PIDFILE="${JOB_DIR}/${JOB_NAME}.pid"
LOGFILE="${JOB_DIR}/${JOB_NAME}.log"
PIPELINE="${VB_ROOT}/scripts/run_version_b_full_pipeline.sh"

mkdir -p "${JOB_DIR}"

_run_poll_loop() {
  local pipeline="${WORKSPACE}/Physics-Aligned World2WAM/version_b/scripts/run_version_b_full_pipeline.sh"
  if [[ ! -f "${pipeline}" ]]; then
    echo "$(date -Iseconds) ERROR: Version B pipeline not found: ${pipeline}" >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "${WORKSPACE}/Physics-Aligned World2WAM/scripts/gpu_poll_utils.sh" 2>/dev/null || \
    source "${WORKSPACE}/minimal_world2wam/scripts/gpu_poll_utils.sh"

  echo "$(date -Iseconds) version_b_poll: waiting for GPU..."
  echo "  pipeline=${pipeline}"
  while true; do
    mapfile -t gpus < <(_pick_gpus "${EVAL_GPU_FREE_MIN_MB:-45000}" "${EVAL_GPU_MAX_UTIL:-30}" | head -n 1)
    if ((${#gpus[@]} >= 1)); then
      echo "$(date -Iseconds) GPU ${gpus[0]} available, starting Version B pipeline"
      break
    fi
    echo "$(date -Iseconds) waiting... $(_gpu_status_line)"
    sleep "${GPU_POLL_SEC:-120}"
  done
  export USE_GPU_POLL=1 WORKSPACE
  bash "${pipeline}"
}

case "${1:-start}" in
  status)
    if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "[RUNNING] ${JOB_NAME} pid=$(cat "${PIDFILE}") log=${LOGFILE}"
    else
      echo "[STOPPED] ${JOB_NAME} log=${LOGFILE}"
      tail -n 3 "${LOGFILE}" 2>/dev/null | sed 's/^/  /' || true
    fi
    ;;
  stop)
    [[ -f "${PIDFILE}" ]] && kill "$(cat "${PIDFILE}")" 2>/dev/null || true
    ;;
  start)
    if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "Already running pid=$(cat "${PIDFILE}")"
      exit 0
    fi
    nohup bash -c "
      set -euo pipefail
      WORKSPACE='${WORKSPACE}'
      export WORKSPACE
      GPU_POLL_SEC='${GPU_POLL_SEC:-120}'
      EVAL_GPU_FREE_MIN_MB='${EVAL_GPU_FREE_MIN_MB:-45000}'
      EVAL_GPU_MAX_UTIL='${EVAL_GPU_MAX_UTIL:-30}'
      export GPU_POLL_SEC EVAL_GPU_FREE_MIN_MB EVAL_GPU_MAX_UTIL
      source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
      conda activate world2wam
      export PYTHONPATH=\"\${WORKSPACE}:\${PYTHONPATH:-}\"
      $(declare -f _run_poll_loop)
      _run_poll_loop
    " >> "${LOGFILE}" 2>&1 &
    echo $! > "${PIDFILE}"
    echo "Started ${JOB_NAME} pid=$(cat "${PIDFILE}") log=${LOGFILE}"
    ;;
  *) echo "Usage: $0 {start|status|stop}"; exit 1 ;;
esac
