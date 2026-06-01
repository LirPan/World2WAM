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
  policy_full       Policy full train + export + LIBERO smoke (nohup)
  policy_train      Policy full train + export only, no LIBERO (nohup)
  libero_compare    Official vs World2WAM LIBERO compare (nohup)
  export_compare    Export policy_full + FULL_RUN LIBERO compare (nohup)
  ablations         Bidirectional ablation validate + full grid (nohup)
  paper_all         Full paper pipeline FULL_RUN=1 (nohup)
  smoke_all         Full smoke: framework + policy20 + LIBERO + ablation + compare (nohup)
  smoke_resume      Resume smoke from LIBERO sim step 4 (nohup)
  status            Show running jobs and last log lines
  tail <job>        tail -f log (download_assets|full_pipeline|policy_full|...)

Logs/PIDs: ${JOB_DIR}/
EOF
}

cmd_status() {
  for name in download_assets full_pipeline pipeline_b train_eval full_experiment libero_eval policy_full policy_train libero_compare export_compare ablations paper_all smoke_all smoke_resume; do
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
    policy_full) script_name="bg_run_policy_full" ;;
    policy_train) script_name="bg_run_policy_train_only" ;;
    libero_compare) script_name="bg_run_libero_compare" ;;
    export_compare) script_name="bg_run_export_compare" ;;
    ablations) script_name="bg_run_ablations" ;;
    paper_all) script_name="bg_run_paper_all" ;;
    smoke_all) script_name="bg_run_smoke_all" ;;
    smoke_resume) script_name="bg_run_smoke_resume" ;;
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
    download_assets|full_pipeline|pipeline_b|train_eval|full_experiment|libero_eval|policy_full|policy_train|libero_compare|export_compare|ablations|paper_all|smoke_all|smoke_resume)
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
