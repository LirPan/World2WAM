#!/usr/bin/env bash
# One-time / repair setup: download LIBERO-Spatial data + clone LIBERO sim assets.
# Large downloads run via nohup/tmux — see bg_launch.sh setup_all.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
FASTWAM_ROOT="${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM"
DATA_DIR="${FASTWAM_ROOT}/data/libero_mujoco3.3.2"
LIBERO_DEST="${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO_fresh"
LOG_DIR="${WORKSPACE}/cache/bg_jobs"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

_activate() {
  # shellcheck disable=SC1091
  source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
  conda activate world2wam
  export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
  unset PYTHONHOME
  if [[ -f /DATA/disk0/jianhua/use_proxy.sh ]]; then
    # shellcheck disable=SC1091
    source /DATA/disk0/jianhua/use_proxy.sh 2>/dev/null || echo "WARN: proxy unavailable, continuing without proxy"
  fi
}

download_spatial_dataset() {
  echo "== Download LIBERO-Spatial from HuggingFace =="
  cd "${DATA_DIR}"
  local tar_name="libero_spatial_no_noops_lerobot.tar.gz"
  local hf_url="https://huggingface.co/datasets/yuanty/LIBERO-fastwam/resolve/main/${tar_name}"
  local fresh="${FRESH_DOWNLOAD:-0}"

  if [[ "${fresh}" == "1" ]]; then
    echo "FRESH_DOWNLOAD=1: removing old tar, HF cache partials, extracted dir"
    rm -f "${tar_name}"
    rm -rf .cache/huggingface/download/*.incomplete 2>/dev/null || true
    rm -rf libero_spatial_no_noops_lerobot
  fi

  _download_tar() {
    local attempt="$1"
    echo "Download attempt ${attempt}/10"
    if [[ -n "${http_proxy:-}" ]]; then
      echo "curl via proxy ${http_proxy}"
      # Fresh download: no -C -. Resume only when FRESH_DOWNLOAD=0 and tar exists.
      if [[ -f "${tar_name}" && "${fresh}" != "1" ]]; then
        curl -fL --retry 5 --retry-delay 10 -C - -x "${http_proxy}" \
          -o "${tar_name}" "${hf_url}" || true
      else
        curl -fL --retry 5 --retry-delay 10 -x "${http_proxy}" \
          -o "${tar_name}.part" "${hf_url}" && mv -f "${tar_name}.part" "${tar_name}"
      fi
    else
      echo "ERROR: proxy required for HF download on this server"
      return 1
    fi
    local sz
    sz=$(stat -c%s "${tar_name}" 2>/dev/null || echo 0)
    echo "tar size=${sz}"
    if gzip -t "${tar_name}" 2>/dev/null; then
      echo "gzip integrity: OK"
      return 0
    fi
    echo "gzip integrity: FAIL (corrupt or incomplete)"
    return 1
  }

  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -f "${tar_name}" ]] && gzip -t "${tar_name}" 2>/dev/null; then
      echo "Valid tar already present"
      break
    fi
    if [[ "${attempt}" -gt 1 && "${fresh}" != "1" ]]; then
      rm -f "${tar_name}"
      fresh=1
    fi
    _download_tar "${attempt}" || true
    if [[ -f "${tar_name}" ]] && gzip -t "${tar_name}" 2>/dev/null; then
      break
    fi
    sleep 30
  done

  if [[ ! -f "${tar_name}" ]] || ! gzip -t "${tar_name}" 2>/dev/null; then
    echo "Download failed: tar missing or gzip corrupt"
    exit 1
  fi

  echo "== Extract =="
  rm -rf libero_spatial_no_noops_lerobot
  tar -xzf "${tar_name}"
  n_parquet=$(find libero_spatial_no_noops_lerobot/data -name '*.parquet' 2>/dev/null | wc -l)
  n_main=$(find libero_spatial_no_noops_lerobot/videos/chunk-000/observation.images.image -name '*.mp4' 2>/dev/null | wc -l)
  n_wrist=$(find libero_spatial_no_noops_lerobot/videos/chunk-000/observation.images.wrist_image -name '*.mp4' 2>/dev/null | wc -l)
  echo "parquet=${n_parquet} main_videos=${n_main} wrist_videos=${n_wrist}"
  [[ "${n_parquet}" -ge 434 && "${n_main}" -ge 434 && "${n_wrist}" -ge 434 ]] || {
    echo "Dataset incomplete after extract"
    exit 1
  }
}

setup_libero_sim() {
  echo "== Clone LIBERO =="
  if [[ ! -d "${LIBERO_DEST}/.git" ]]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DEST}"
  else
    git -C "${LIBERO_DEST}" fetch --depth 1 origin main 2>/dev/null || git -C "${LIBERO_DEST}" fetch --depth 1 origin master 2>/dev/null || true
    git -C "${LIBERO_DEST}" reset --hard FETCH_HEAD 2>/dev/null || true
  fi
  n_bddl=$(find "${LIBERO_DEST}/libero/libero/bddl_files/libero_spatial" -name '*.bddl' 2>/dev/null | wc -l)
  echo "bddl_spatial=${n_bddl}"
  [[ "${n_bddl}" -gt 0 ]] || { echo "LIBERO bddl missing"; exit 1; }

  echo "== pip install LIBERO + mujoco =="
  pip uninstall -y libero 2>/dev/null || true
  pip install -e "${LIBERO_DEST}" mujoco==3.3.2 --no-build-isolation
  export PYTHONPATH="${LIBERO_DEST}:${WORKSPACE}:${PYTHONPATH:-}"
  mkdir -p "${HOME}/.libero"
  if [[ ! -f "${HOME}/.libero/config.yaml" ]]; then
    cat > "${HOME}/.libero/config.yaml" <<EOF
benchmark_root: ${LIBERO_DEST}/libero/libero
bddl_files: ${LIBERO_DEST}/libero/libero/bddl_files
init_states: ${LIBERO_DEST}/libero/libero/init_files
datasets: ${LIBERO_DEST}/libero/libero/datasets
assets: ${LIBERO_DEST}/libero/libero/assets
EOF
  fi
  python -c "from libero.libero import benchmark; print('LIBERO OK')"
}

main() {
  _activate
  case "${1:-all}" in
    data) download_spatial_dataset ;;
    data_fresh) FRESH_DOWNLOAD=1 download_spatial_dataset ;;
    libero) setup_libero_sim ;;
    all)
      download_spatial_dataset
      setup_libero_sim
      ;;
    *) echo "Usage: $0 [data|libero|all]"; exit 1 ;;
  esac
  echo "SETUP DONE"
}

main "$@"
