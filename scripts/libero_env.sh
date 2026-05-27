#!/usr/bin/env bash
# Source before LIBERO sim eval (does not modify upstream repos).
set -euo pipefail

_MINIMAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_LIBERO_REPO="$(cd "${_MINIMAL_ROOT}/../code/LIBERO" && pwd)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/DATA/disk1/yjh_space/.libero}"
export PYTHONPATH="${_MINIMAL_ROOT}/third_party/libero_ns:${PYTHONPATH:-}"

if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${_LIBERO_REPO}/libero/libero
bddl_files: ${_LIBERO_REPO}/libero/libero/bddl_files
init_states: ${_LIBERO_REPO}/libero/libero/init_files
datasets: ${_MINIMAL_ROOT}/data/libero_sim_assets/datasets
assets: ${_LIBERO_REPO}/libero/libero/assets
EOF
fi
