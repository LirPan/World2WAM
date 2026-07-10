#!/usr/bin/env bash
# Full tonight pipeline in background (nohup / tmux). Survives Cursor disconnect.
#
#   bash minimal_world2wam/scripts/bg_launch.sh setup_all    # download data + LIBERO (long)
#   bash minimal_world2wam/scripts/bg_launch.sh full_pipeline
#   bash minimal_world2wam/scripts/bg_launch.sh tmux full_pipeline
#   bash minimal_world2wam/scripts/bg_launch.sh status
#   tmux attach -t world2wam
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${ROOT}/.." && pwd)"
JOB_DIR="${WORKSPACE}/cache/bg_jobs"
TMUX_SESSION="${TMUX_SESSION:-world2wam}"
CONFIG="${CONFIG:-configs/world2wam_libero_spatial_h10.yaml}"
CONDA_BASE="${CONDA_BASE:-/DATA/disk0/jianhua/miniconda3}"
CACHE_OUT="${CACHE_OUT:-cache/debug_libero_spatial_h10}"
PRE_MAX="${PRE_MAX:-100}"
PAPER_GPU="${PAPER_GPU:-6}"
PAPER_GPU2="${PAPER_GPU2:-7}"
PRECOMPUTE_SHARDS="${PRECOMPUTE_SHARDS:-2}"
HEADS_CKPT="${HEADS_CKPT:-experiments/world2wam_heads/heads_final.pt}"
ADAPTER_CKPT="${ADAPTER_CKPT:-experiments/world2wam_adapter/adapter_final.pt}"

mkdir -p "${JOB_DIR}"

_activate_env() {
  export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
  unset PYTHONHOME
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-world2wam}"
  fi
  if [[ -f /DATA/disk0/jianhua/use_proxy.sh ]]; then
    # shellcheck disable=SC1091
    source /DATA/disk0/jianhua/use_proxy.sh || true
  fi
  cd "${WORKSPACE}"
}

_run_nohup() {
  local name="$1"
  shift
  local pidfile="${JOB_DIR}/${name}.pid"
  local logfile="${JOB_DIR}/${name}.log"
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "Already running: ${name} pid=$(cat "${pidfile}") log=${logfile}"
    return 0
  fi
  echo "Starting ${name} (nohup) -> ${logfile}"
  nohup bash -c "
    set -euo pipefail
    WORKSPACE='${WORKSPACE}'
    ROOT='${ROOT}'
    JOB_DIR='${JOB_DIR}'
    CONDA_BASE='${CONDA_BASE}'
    CONFIG='${CONFIG}'
    CACHE_OUT='${CACHE_OUT}'
    PRE_MAX='${PRE_MAX}'
    $(declare -f _activate_env)
    _activate_env
    $*
  " >> "${logfile}" 2>&1 &
  echo $! > "${pidfile}"
  disown -h $! 2>/dev/null || true
  echo "pid=$(cat "${pidfile}")"
}

cmd_status() {
  for name in setup_all setup_data setup_libero wait_download redownload_pipeline repair_wan_pipeline repair_vae_eval repair_fastwam_ckpts auto_pipeline paper_pipeline version_a_poll version_a_full wait_pipeline smoke precompute train_heads train_adapter eval_baseline eval_offline eval_ours eval_ours_dit eval_compare full_pipeline tonight; do
    pidfile="${JOB_DIR}/${name}.pid"
    logfile="${JOB_DIR}/${name}.log"
    if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
      echo "[RUNNING] ${name} pid=$(cat "${pidfile}") log=${logfile}"
    elif [[ -f "${logfile}" ]]; then
      echo "[DONE/DEAD] ${name} log=${logfile}"
      tail -n 2 "${logfile}" 2>/dev/null | sed 's/^/  /'
    else
      echo "[NOT STARTED] ${name}"
    fi
  done
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "[TMUX] session ${TMUX_SESSION} — attach: tmux attach -t ${TMUX_SESSION}"
  fi
}

_launch_job() {
  local name="$1"
  case "${name}" in
    setup_data)
      _run_nohup setup_data "bash minimal_world2wam/scripts/setup_deps.sh data"
      ;;
    setup_libero)
      _run_nohup setup_libero "bash minimal_world2wam/scripts/setup_deps.sh libero"
      ;;
    setup_all)
      _run_nohup setup_all "bash minimal_world2wam/scripts/setup_deps.sh all"
      ;;
    smoke)
      _run_nohup smoke "bash minimal_world2wam/scripts/smoke_test.sh"
      ;;
    precompute)
      _run_nohup precompute "python minimal_world2wam/cache/precompute_fastwam_latents.py --config ${CONFIG} --output ${CACHE_OUT} --max_samples ${PRE_MAX}"
      ;;
    train_heads)
      _run_nohup train_heads "python minimal_world2wam/train/train_world2wam_heads.py --config ${CONFIG} --cache_dir ${CACHE_OUT} --use_fwd true --use_inv true --use_cycle true"
      ;;
    train_adapter)
      _run_nohup train_adapter "python minimal_world2wam/train/train_world2wam_adapter.py --config ${CONFIG} --cache_dir ${CACHE_OUT} --use_act true --use_fwd true --use_inv true --use_cycle true"
      ;;
    eval_baseline)
      _run_nohup eval_baseline "python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode baseline --max_tasks ${MAX_TASKS:-1} --num_trials ${NUM_TRIALS:-1} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}}"
      ;;
    eval_offline)
      _run_nohup eval_offline "python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode offline --latent_verification --cache_dir ${CACHE_OUT} --heads_ckpt ${HEADS_CKPT} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}}"
      ;;
    eval_ours)
      _run_nohup eval_ours "python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode ours_adapter --adapter_ckpt ${ADAPTER_CKPT:-experiments/world2wam_adapter/adapter_final.pt} --max_tasks ${MAX_TASKS:-1} --num_trials ${NUM_TRIALS:-1} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}}"
      ;;
    eval_ours_dit)
      _run_nohup eval_ours_dit "python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode ours_dit --max_tasks ${MAX_TASKS:-1} --num_trials ${NUM_TRIALS:-1} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}}"
      ;;
    eval_compare)
      _run_nohup eval_compare "
        python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode offline --latent_verification --cache_dir ${CACHE_OUT} --heads_ckpt ${HEADS_CKPT} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}} &&
        python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode baseline --max_tasks ${MAX_TASKS:-1} --num_trials ${NUM_TRIALS:-1} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}} &&
        python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode ours_adapter --adapter_ckpt ${ADAPTER_CKPT} --max_tasks ${MAX_TASKS:-1} --num_trials ${NUM_TRIALS:-1} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}} &&
        python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode ours_dit --max_tasks ${MAX_TASKS:-1} --num_trials ${NUM_TRIALS:-1} ${EVAL_DEVICE:+--device ${EVAL_DEVICE}}
      "
      ;;
    full_pipeline)
      _run_nohup full_pipeline "
        python minimal_world2wam/cache/precompute_fastwam_latents.py --config ${CONFIG} --output ${CACHE_OUT} --max_samples ${PRE_MAX} &&
        python minimal_world2wam/train/train_world2wam_heads.py --config ${CONFIG} --cache_dir ${CACHE_OUT} --use_fwd true --use_inv true --use_cycle true &&
        python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode offline --latent_verification --cache_dir ${CACHE_OUT} --heads_ckpt experiments/world2wam_heads/heads_final.pt
      "
      ;;
    tonight)
      # setup -> smoke -> precompute -> train -> offline eval -> baseline sim
      _run_nohup tonight "
        bash minimal_world2wam/scripts/setup_deps.sh all &&
        bash minimal_world2wam/scripts/smoke_test.sh &&
        python minimal_world2wam/cache/precompute_fastwam_latents.py --config ${CONFIG} --output ${CACHE_OUT} --max_samples ${PRE_MAX} &&
        python minimal_world2wam/train/train_world2wam_heads.py --config ${CONFIG} --cache_dir ${CACHE_OUT} --use_fwd true --use_inv true --use_cycle true &&
        python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode offline --latent_verification --cache_dir ${CACHE_OUT} --heads_ckpt experiments/world2wam_heads/heads_final.pt &&
        python minimal_world2wam/eval/eval_libero_world2wam.py --config ${CONFIG} --mode baseline --max_tasks 1
      "
      ;;
    wait_pipeline)
      _run_nohup wait_pipeline "bash minimal_world2wam/scripts/wait_and_run_pipeline.sh"
      ;;
    wait_download)
      _run_nohup wait_download "bash minimal_world2wam/scripts/wait_proxy_and_download.sh"
      ;;
    redownload_pipeline)
      _run_nohup redownload_pipeline "bash minimal_world2wam/scripts/redownload_and_pipeline.sh"
      ;;
    repair_wan_pipeline)
      _run_nohup repair_wan_pipeline "bash minimal_world2wam/scripts/repair_wan_and_run_pipeline.sh"
      ;;
    repair_vae_eval)
      _run_nohup repair_vae_eval "bash minimal_world2wam/scripts/repair_vae_t5_and_eval.sh"
      ;;
    repair_fastwam_ckpts)
      _run_nohup repair_fastwam_ckpts "bash minimal_world2wam/scripts/repair_fastwam_ckpts_and_run.sh"
      ;;
    auto_pipeline)
      _run_nohup auto_pipeline "bash minimal_world2wam/scripts/run_auto_pipeline.sh"
      ;;
    paper_pipeline)
      _run_nohup paper_pipeline "PAPER_GPU=${PAPER_GPU} PAPER_GPU2=${PAPER_GPU2} PRECOMPUTE_SHARDS=${PRECOMPUTE_SHARDS} CONFIG=${CONFIG} CACHE_OUT=${CACHE_OUT} MAX_TASKS=${MAX_TASKS:-10} NUM_TRIALS=${NUM_TRIALS:-50} HEADS_CKPT=${HEADS_CKPT} ADAPTER_CKPT=${ADAPTER_CKPT} bash minimal_world2wam/scripts/run_paper_pipeline.sh"
      ;;
    version_a_poll)
      _run_nohup version_a_poll "bash minimal_world2wam/scripts/poll_gpu_version_a.sh start"
      ;;
    version_a_full)
      _run_nohup version_a_full "USE_GPU_POLL=${USE_GPU_POLL:-1} CONFIG=${CONFIG:-configs/world2wam_physics_flow_dit_main.yaml} CACHE_OUT=${CACHE_OUT:-cache/libero_spatial_h10_full_fastwam} OUT_DIR=${OUT_DIR:-experiments/world2wam_physics_flow_dit_main} MAX_SAMPLES=${MAX_SAMPLES:-600000} MAX_TASKS=${MAX_TASKS:-10} NUM_TRIALS=${NUM_TRIALS:-50} bash minimal_world2wam/scripts/run_version_a_full_pipeline.sh"
      ;;
    parallel_prep)
      echo "Starting setup_data + setup_libero in parallel..."
      rm -f "${JOB_DIR}/setup_data.pid" "${JOB_DIR}/setup_libero.pid" 2>/dev/null || true
      _launch_job setup_data
      _launch_job setup_libero
      _launch_job wait_pipeline
      echo "Launched: setup_data, setup_libero, wait_pipeline (auto-runs experiment when data ready)"
      ;;
    *)
      echo "Unknown job: ${name}"; exit 1
      ;;
  esac
}

case "${1:-}" in
  setup_data|setup_libero|setup_all|smoke|precompute|train_heads|train_adapter|eval_baseline|eval_offline|eval_ours|eval_ours_dit|eval_compare|full_pipeline|tonight|wait_pipeline|wait_download|redownload_pipeline|repair_wan_pipeline|repair_vae_eval|repair_fastwam_ckpts|auto_pipeline|paper_pipeline|version_a_poll|version_a_full|parallel_prep)
    _launch_job "${1}"
    ;;
  tmux)
    job="${2:-tonight}"
    if ! command -v tmux >/dev/null 2>&1; then
      echo "tmux not found; using nohup"
      _launch_job "${job}"
      exit 0
    fi
    inner="CACHE_OUT=${CACHE_OUT} PRE_MAX=${PRE_MAX} bash minimal_world2wam/scripts/bg_launch.sh ${job}"
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      tmux send-keys -t "${TMUX_SESSION}" "cd ${WORKSPACE} && ${inner}" C-m
    else
      tmux new-session -d -s "${TMUX_SESSION}" "cd ${WORKSPACE} && ${inner}; echo JOB_DONE; exec bash"
    fi
    echo "tmux attach -t ${TMUX_SESSION}"
    echo "log: ${JOB_DIR}/${job}.log (if nohup fallback) or scroll tmux"
    ;;
  status) cmd_status ;;
  tail) tail -f "${JOB_DIR}/${2:-tonight}.log" ;;
  *)
    cat <<EOF
Usage: bash minimal_world2wam/scripts/bg_launch.sh <job> | tmux <job> | status | tail [job]

Jobs:
  setup_data / setup_libero / setup_all   download HF dataset + LIBERO sim (proxy via use_proxy.sh)
  smoke / precompute / train_heads / train_adapter
  paper_pipeline   full ~1.17M cache + 10-epoch train + LIBERO 10x50 eval (GPU)
  version_a_poll   GPU-poll then Version A physics FlowDiT full pipeline (background)
  version_a_full   Version A full pipeline directly (with per-stage GPU poll inside)
  eval_baseline / eval_offline / eval_ours / eval_ours_dit / eval_compare / full_pipeline / tonight

Paper-level (GPU 6, background):
  CONFIG=configs/world2wam_libero_spatial_h10_paper.yaml \\
  CACHE_OUT=cache/libero_spatial_h10_full_fastwam \\
  PAPER_GPU=6 MAX_TASKS=10 NUM_TRIALS=50 \\
  bash minimal_world2wam/scripts/bg_launch.sh paper_pipeline

Env: CONFIG CACHE_OUT PRE_MAX CONDA_ENV MAX_TASKS NUM_TRIALS HEADS_CKPT ADAPTER_CKPT EVAL_DEVICE PAPER_GPU GPU_FREE_MIN_MB
  bash minimal_world2wam/scripts/bg_launch.sh tmux tonight
  tmux attach -t world2wam

Env: CONFIG CACHE_OUT PRE_MAX CONDA_ENV MAX_TASKS NUM_TRIALS HEADS_CKPT ADAPTER_CKPT EVAL_DEVICE GPU_FREE_MIN_MB
Logs: ${JOB_DIR}/
EOF
    ;;
esac
