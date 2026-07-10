#!/usr/bin/env bash
# Start data migration to remote in background (nohup). Survives Cursor / SSH disconnect from IDE.
#
#   bash minimal_world2wam/scripts/migrate_to_remote_bg.sh start tier3
#   bash minimal_world2wam/scripts/migrate_to_remote_bg.sh status
#   bash minimal_world2wam/scripts/migrate_to_remote_bg.sh tail
#   bash minimal_world2wam/scripts/migrate_to_remote_bg.sh stop
#
# Env: REMOTE_USER_HOST REMOTE_PORT REMOTE_BASE LOCAL_BASE
set -euo pipefail

WORKSPACE="${LOCAL_BASE:-/DATA/disk0/jianhua}"
JOB_DIR="${WORKSPACE}/cache/bg_jobs"
JOB_NAME="migrate_remote"
PIDFILE="${JOB_DIR}/${JOB_NAME}.pid"
LOGFILE="${JOB_DIR}/${JOB_NAME}.log"
MIGRATE="${WORKSPACE}/Physics-Aligned World2WAM/scripts/migrate_to_remote.sh"

mkdir -p "${JOB_DIR}"

_cmd_status() {
  if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    echo "[RUNNING] migrate pid=$(cat "${PIDFILE}")"
    echo "  log: ${LOGFILE}"
    tail -n 5 "${LOGFILE}" 2>/dev/null | sed 's/^/  /'
  elif [[ -f "${LOGFILE}" ]]; then
    echo "[STOPPED/DONE] migrate"
    echo "  log: ${LOGFILE}"
    if rg -q "Done tier|REMOTE SETUP OK|error|failed|timed out" "${LOGFILE}" 2>/dev/null; then
      tail -n 8 "${LOGFILE}" | sed 's/^/  /'
    fi
  else
    echo "[NOT STARTED]"
  fi
}

case "${1:-status}" in
  start)
    tier="${2:-tier3}"
    if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "Already running pid=$(cat "${PIDFILE}")"
      exit 0
    fi
    echo "Starting background migrate (${tier}) -> ${LOGFILE}"
    echo "Remote: ${REMOTE_USER_HOST:-yjh@120.92.211.106}:${REMOTE_BASE:-/DATA/disk0/yjh/world2wam}"
    nohup bash -c "
      set -euo pipefail
      export REMOTE_USER_HOST='${REMOTE_USER_HOST:-yjh@120.92.211.106}'
      export REMOTE_PORT='${REMOTE_PORT:-22}'
      export REMOTE_BASE='${REMOTE_BASE:-/DATA/disk0/yjh/world2wam}'
      export LOCAL_BASE='${WORKSPACE}'
      echo '=== migrate start \$(date -Iseconds) tier=${tier} ===' >> '${LOGFILE}'
      bash '${MIGRATE}' '${tier}' >> '${LOGFILE}' 2>&1
      ec=\$?
      echo '=== migrate end \$(date -Iseconds) exit='\$ec' ===' >> '${LOGFILE}'
      exit \$ec
    " >> "${LOGFILE}" 2>&1 &
    echo $! > "${PIDFILE}"
    disown -h $! 2>/dev/null || true
    echo "pid=$(cat "${PIDFILE}")"
    echo "Check: bash minimal_world2wam/scripts/migrate_to_remote_bg.sh status"
    ;;
  status) _cmd_status ;;
  tail) tail -f "${LOGFILE}" ;;
  stop)
    if [[ -f "${PIDFILE}" ]]; then
      kill "$(cat "${PIDFILE}")" 2>/dev/null || true
      echo "Stopped pid $(cat "${PIDFILE}")"
    fi
    ;;
  *)
    echo "Usage: $0 {start [tier1|tier2|tier3]|status|tail|stop}"
    exit 1
    ;;
esac
