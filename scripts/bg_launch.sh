#!/usr/bin/env bash
# Launch a long-running job detached from Cursor/SSH session.
# Usage:
#   bash scripts/bg_launch.sh download_assets
#   bash scripts/bg_launch.sh full_pipeline
#   bash scripts/bg_launch.sh status
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_DIR="${ROOT}/experiments/bg_jobs"
mkdir -p "${JOB_DIR}"

usage() {
  cat <<EOF
Usage: bash scripts/bg_launch.sh <command>

Commands:
  download_assets   Download ckpt, data, Wan weights (nohup)
  full_pipeline     download_assets + pipeline B (nohup)
  pipeline_b        pipeline B only: 02->03->04 (nohup, assets must exist)
  train_eval        resume 03->04 only (nohup, cache must exist)
  full_experiment   full 02->03->04 + LIBERO spatial success (nohup)
  libero_eval       LIBERO spatial success only (nohup)
  status            Show running jobs and last log lines
  tail <job>        tail -f log (download_assets|full_pipeline|pipeline_b|full_experiment)

Logs/PIDs: ${JOB_DIR}/
EOF
}

cmd_status() {
  for name in download_assets full_pipeline pipeline_b train_eval full_experiment libero_eval; do
    pidfile="${JOB_DIR}/${name}.pid"
    logfile="${JOB_DIR}/${name}.log"
    if [[ -f "${pidfile}" ]]; then
      pid=$(cat "${pidfile}")
      if kill -0 "${pid}" 2>/dev/null; then
        echo "[RUNNING] ${name} pid=${pid} log=${logfile}"
      else
        echo "[DONE/DEAD] ${name} pid=${pid} (see ${logfile})"
      fi
    else
      echo "[NOT STARTED] ${name}"
    fi
    if [[ -f "${logfile}" ]]; then
      echo "  --- last 3 lines ---"
      tail -n 3 "${logfile}" | sed 's/^/  /'
    fi
  done
}

cmd_tail() {
  local name="${1:-full_pipeline}"
  logfile="${JOB_DIR}/${name}.log"
  [[ -f "${logfile}" ]] || { echo "No log: ${logfile}"; exit 1; }
  tail -f "${logfile}"
}

cmd_launch() {
  local name="$1"
  local script_name="${name}"
  case "${name}" in
    full_pipeline) script_name="bg_full_pipeline" ;;
    pipeline_b) script_name="bg_run_pipeline_b" ;;
    train_eval) script_name="bg_train_eval_only" ;;
    full_experiment) script_name="bg_run_full_experiment" ;;
    libero_eval) script_name="bg_run_libero_eval_only" ;;
  esac
  local script="${ROOT}/scripts/${script_name}.sh"
  [[ -f "${script}" ]] || { echo "Missing ${script}"; exit 1; }

  local pidfile="${JOB_DIR}/${name}.pid"
  local logfile="${JOB_DIR}/${name}.log"

  if [[ -f "${pidfile}" ]]; then
    oldpid=$(cat "${pidfile}")
    if kill -0 "${oldpid}" 2>/dev/null; then
      echo "Already running: ${name} pid=${oldpid}"
      echo "Log: ${logfile}"
      exit 0
    fi
  fi

  echo "Starting ${name} in background..."
  echo "Log: ${logfile}"
  PROXY_SH="/DATA/disk1/yjh_space/use_proxy.sh"
  nohup bash -c "
    set -euo pipefail
    if [[ -f '${PROXY_SH}' ]]; then source '${PROXY_SH}'; fi
    exec bash '${script}'
  " >> "${logfile}" 2>&1 &
  echo $! > "${pidfile}"
  disown -h $! 2>/dev/null || true
  echo "Started pid=$(cat "${pidfile}")"
  echo "Monitor: bash scripts/bg_launch.sh status"
  echo "         bash scripts/bg_launch.sh tail ${name}"
}

main() {
  local sub="${1:-}"
  case "${sub}" in
    download_assets|full_pipeline|pipeline_b|train_eval|full_experiment|libero_eval)
      cmd_launch "${sub}"
      ;;
    status)
      cmd_status
      ;;
    tail)
      cmd_tail "${2:-pipeline_b}"
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      echo "Unknown command: ${sub}"
      usage
      exit 1
      ;;
  esac
}

main "$@"
