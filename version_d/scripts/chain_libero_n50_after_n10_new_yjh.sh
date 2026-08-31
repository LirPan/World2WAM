#!/usr/bin/env bash
# Launch the formal 50-trial paired evaluation after the 10-trial sanity run.
set -u

PY="${PYTHON_BIN:-/DATA/disk0/yjh/libero_work_wj/env/libero_venv/bin/python}"
RUNNER="${LIBERO_PARALLEL_RUNNER:-/DATA/disk0/yjh/libero_work_wj/scripts/run_libero_pair_parallel_new_yjh.py}"
N10_ROOT="${LIBERO_N10_ROOT:-/DATA/disk0/yjh/libero_work_wj/runs/libero_version_d_spatial_eval_n10}"
N50_ROOT="${LIBERO_N50_ROOT:-/DATA/disk0/yjh/libero_work_wj/runs/libero_version_d_spatial_eval_n50}"
GPUS="${LIBERO_EVAL_GPUS:-1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG="${LIBERO_CHAIN_LOG:-/DATA/disk0/yjh/libero_work_wj/runs/libero_eval_chain.log}"
LOCK="${LIBERO_CHAIN_LOCK:-/tmp/world2wam_libero_eval_chain.lock}"

mkdir -p "$(dirname "$LOG")" "$N50_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf '[%s] another LIBERO evaluation chain is active\n' "$(date -Is)" >> "$LOG"
  exit 0
fi

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

if [[ -s "$N50_ROOT/libero_pair_summary.json" ]]; then
  log "formal n50 evaluation already complete"
  exit 0
fi

log "waiting for n10 paired summary"
while [[ ! -s "$N10_ROOT/libero_pair_summary.json" ]]; do
  if ! pgrep -f '[r]un_libero_pair_parallel_new_yjh.py.*--num-trials 10' >/dev/null 2>&1; then
    log "n10 dispatcher is not active and summary is missing; stop chain for inspection"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

log "n10 complete; launching formal n50 paired evaluation"
"$PY" -u "$RUNNER" \
  --gpus "$GPUS" \
  --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --num-trials 50 \
  --seed 42 \
  --run-root "$N50_ROOT" >> "$LOG" 2>&1
rc=$?
log "formal n50 dispatcher exited rc=$rc"
exit "$rc"
